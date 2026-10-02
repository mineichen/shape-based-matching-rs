use graph_matching::{Detector, Point2i};
use opencv::{
    core::{self as cv, Mat},
    imgcodecs, imgproc,
};
use testresult::TestResult;

fn rotated_corners(
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    angle: f32,
) -> TestResult<cv::Vector<cv::Point>> {
    let mut pts = [cv::Point2f::default(); 4];
    let (hw, hh) = ((w / 2.).round() as i32, (h / 2.).round() as i32);
    cv::RotatedRect::new(cv::Point2f::new(cx, cy), cv::Size2f::new(w, h), angle)?
        .points(&mut pts)?;
    Ok(pts
        .iter()
        .map(|p| cv::Point::new(hw + p.x.round() as i32, hh + p.y.round() as i32))
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

    let mut img = Mat::new_rows_cols_with_default(h, w, typ, cv::Scalar::all(color))?;
    for rect in iter {
        let (cx, cy) = (
            (center_x - rect.w / 2) as f32,
            (center_y - rect.h / 2) as f32,
        );
        let corners = rotated_corners(cx, cy, rect.w as f32, rect.h as f32, angle)?;
        imgproc::fill_poly(
            &mut img,
            &corners,
            cv::Scalar::all(rect.c),
            imgproc::LINE_8,
            0,
            cv::Point::new(0, 0),
        )?;
    }

    Ok(img)
}

#[test]
fn mask_rotated() -> TestResult {
    const OUTER_SIZE: i32 = 400;
    let outer = RectDesc {
        w: OUTER_SIZE,
        h: OUTER_SIZE,
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
    let train_img = create_rect_image(&[outer, inner, inner_hole], 0.0, cv::CV_8UC3)?;
    let search_img = create_rect_image(&[outer, inner], 45.0, cv::CV_8UC3)?;
    let mask_img = create_rect_image(&[outer, cover_inner_hole_mask], 0., cv::CV_8UC1)?;
    // Pivot is the center pixel of the template image.
    let center = Point2i::splat(OUTER_SIZE);

    let mut encoded_bytes = cv::Vector::<u8>::new();
    imgcodecs::imencode_def(".png", &mask_img, &mut encoded_bytes)?;
    let output_path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::write(output_path.join("mask_rotated.png"), &encoded_bytes)?;

    let mut det = Detector::builder()
        .with_template("r", &train_img, |mut c| {
            c.use_mask(mask_img);
            c.add_rotated(45.0, center);
        })
        .build()?;

    // Match against same image - should find it
    let matches = det.match_templates(&search_img, 0.85, None)?;

    let Some(best) = matches.into_iter().max() else {
        panic!("Expected pattern to be found");
    };
    assert!(best.similarity > 0.99, "sim={:.2}", best.similarity);

    Ok(())
}

#[test]
fn mask_size_mismatch_errors() {
    // Deliberately mismatched mask: half the image edge, must fail.
    const IMG_SIZE: i32 = 100;
    let img =
        Mat::new_rows_cols_with_default(IMG_SIZE, IMG_SIZE, cv::CV_8UC3, cv::Scalar::all(0.0))
            .unwrap();
    let wrong_mask = Mat::new_rows_cols_with_default(
        IMG_SIZE / 2,
        IMG_SIZE / 2,
        cv::CV_8UC1,
        cv::Scalar::all(255.0),
    )
    .unwrap();
    // Pivot is the center pixel of the template image.
    let center = Point2i::splat(IMG_SIZE / 2);

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
