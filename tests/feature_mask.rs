use graph_matching::Detector;
use opencv::{
    core::{self, Mat, Point, Point2f, RotatedRect, Scalar, Size2f, Vector},
    imgcodecs, imgproc,
};
use testresult::TestResult;

fn rotated_corners(cx: f32, cy: f32, w: f32, h: f32, angle: f32) -> TestResult<Vector<Point>> {
    let mut pts = [Point2f::default(); 4];
    let (hw, hh) = ((w / 2.).round() as i32, (h / 2.).round() as i32);
    RotatedRect::new(Point2f::new(cx, cy), Size2f::new(w, h), angle)?.points(&mut pts)?;
    Ok(pts
        .iter()
        .map(|p| Point::new(hw + p.x.round() as i32, hh + p.y.round() as i32))
        .collect())
}
#[derive(Clone, Copy)]
struct RectDesc {
    w: i32,
    h: i32,
    c: f64,
}

fn create_rect_image(desc: &[RectDesc], angle: f32, typ: i32) -> TestResult<Mat> {
    let mut iter = desc.into_iter();
    let &RectDesc { w, h, c: color, .. } = iter.next().unwrap();
    let (center_x, center_y) = (w / 2, h / 2);

    let mut img = Mat::new_rows_cols_with_default(h, w, typ, Scalar::all(color))?;
    for rect in iter {
        let (cx, cy) = (
            (center_x - rect.w / 2) as f32,
            (center_y - rect.h / 2) as f32,
        );
        let corners = rotated_corners(cx, cy, rect.w as f32, rect.h as f32, angle)?;
        imgproc::fill_poly(
            &mut img,
            &corners,
            Scalar::all(rect.c),
            imgproc::LINE_8,
            0,
            Point::new(0, 0),
        )?;
    }

    Ok(img)
}

#[test]
fn mask_rotated() -> TestResult {
    let outer = RectDesc {
        w: 400,
        h: 400,
        c: 255.,
    };
    let inner = RectDesc {
        w: 100,
        h: 100,
        c: 0.,
    };
    let inner_hole = RectDesc {
        w: 50,
        h: 50,
        c: 255.,
    };
    let cover_inner_hole_mask = RectDesc {
        w: 70,
        h: 70,
        c: 0.,
    };
    let train_img = create_rect_image(&[outer, inner, inner_hole], 0.0, core::CV_8UC3)?;
    let search_img = create_rect_image(&[outer, inner], 45.0, core::CV_8UC3)?;
    let mask_img = create_rect_image(&[outer, cover_inner_hole_mask], 0., core::CV_8UC1)?;
    let center = Point2f::new((outer.w / 2) as f32, (outer.h / 2) as f32);

    let mut encoded_bytes = core::Vector::<u8>::new();
    imgcodecs::imencode_def(".png", &mask_img, &mut encoded_bytes)?;
    let output_path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::write(output_path.join("mask_rotated.png"), &encoded_bytes)?;

    let det = Detector::builder()
        .with_template("r", &train_img, |mut c| {
            c.use_mask(mask_img);
            c.add_rotated(45.0, center);
        })
        .build()?;

    // Match against same image - should find it
    let matches = det.match_templates(&search_img, 0.1, None, None)?;
    let Some(best) = matches.into_iter().max() else {
        panic!("Expected pattern to be found");
    };
    assert!(best.similarity > 0.99, "sim={:.2}", best.similarity);

    Ok(())
}

#[test]
fn mask_size_mismatch_errors() {
    let img = Mat::new_rows_cols_with_default(100, 100, core::CV_8UC3, Scalar::all(0.0)).unwrap();
    let wrong_mask =
        Mat::new_rows_cols_with_default(50, 50, core::CV_8UC1, Scalar::all(255.0)).unwrap();
    let center = Point2f::new(50.0, 50.0);

    let Err(e) = Detector::builder()
        .with_template("r", &img, |mut c| {
            c.use_mask(wrong_mask);
            c.add_rotated(0.0, center);
        })
        .build()
    else {
        panic!("Counld unexpectedly build the detector")
    };
    let msg = format!("{e}");
    assert!(
        msg.contains("feature_mask") && msg.contains("does not match template size"),
        "Didn't contain pattern: {e}"
    );
}
