use graph_matching::{Detector, Match};
use opencv::{
    core::{self, Mat, Scalar},
    imgcodecs, imgproc,
};
use testresult::TestResult;

const IMAGE_WIDTH: i32 = 400;
const IMAGE_HEIGHT: i32 = 400;
const ELLIPSE_WIDTH: i32 = 80;
const ELLIPSE_HEIGHT: i32 = 50;
const ELLIPSE_THICKNESS: i32 = 3;

#[test]
fn ellipse_detection() -> TestResult {
    // Draw an ellipse as the template
    let center = core::Point::new(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2);
    let train_canvas = create_ellipse_image(center, 0.0)?;

    // Create detector
    let center_f = core::Point2f::new(center.x as _, center.y as _);
    let detector = Detector::builder()
        .with_template("ellipse", &train_canvas, |mut cfg| {
            cfg.add_rotated(0.0, center_f); // Explicitly add zero angle
            cfg.add_rotated(45.0, center_f);
        })
        .build()?;
    assert_eq!(detector.num_templates("ellipse"), 2);

    // Create a test image with the same ellipse rotated 45 degrees
    let test_canvas = create_ellipse_image(center, 45.0)?;
    let mut result = detector.match_templates(&test_canvas, 0.80, None)?;
    result.sort();

    let best_match = result.last().expect("Expected at least one match").clone();
    println!(
        "Best match at ({}, {}) with similarity {:.2}",
        best_match.x, best_match.y, best_match.similarity
    );

    // Always generate debug image with red dotted ellipse overlay
    let mut debug_image = result.debug_visual(test_canvas.clone(), None)?;
    draw_found_ellipse(&best_match, &mut debug_image);

    let mut encoded_bytes = core::Vector::<u8>::new();
    imgcodecs::imencode_def(".png", &debug_image, &mut encoded_bytes)?;
    let output_path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::write(output_path.join("ellipse_detection.png"), &encoded_bytes)?;

    assert!(dbg!(best_match.similarity) > 0.95);
    assert!(dbg!(best_match.angle()) == 45.0);

    Ok(())
}

fn create_ellipse_image(center: core::Point, angle: f64) -> TestResult<Mat> {
    let mut canvas = core::Mat::new_rows_cols_with_default(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        core::CV_8UC3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    let axes = core::Size::new(ELLIPSE_WIDTH, ELLIPSE_HEIGHT);
    let color = Scalar::new(0.0, 0.0, 0.0, 0.0);

    imgproc::ellipse(
        &mut canvas,
        center,
        axes,
        angle,
        0.0,
        360.0,
        color,
        ELLIPSE_THICKNESS,
        imgproc::LINE_8,
        0,
    )?;
    Ok(canvas)
}

fn draw_found_ellipse(best_match: &Match, debug_image: &mut Mat) -> TestResult {
    // Draw red 3px ellipse arcs at detected position with detected angle
    // Draw only 90-degree arcs so original ellipse remains visible
    let detected_angle = best_match.angle();
    let center = best_match.center_point();

    // Use same axes as original ellipse
    let axes = core::Size::new(ELLIPSE_WIDTH, ELLIPSE_HEIGHT);
    let red = Scalar::new(0.0, 0.0, 255.0, 0.0);

    // Draw 4 short arcs (30 degrees each, spaced 90 degrees apart)
    for arc_start in (0..360).step_by(30) {
        imgproc::ellipse(
            debug_image,
            center,
            axes,
            detected_angle as f64,
            arc_start as f64,
            (arc_start + 20) as f64,
            red,
            3,
            imgproc::LINE_AA,
            0,
        )?;
    }
    Ok(())
}

#[test]
fn rotated_range() -> TestResult {
    // Create a simple shape
    let width = 200;
    let height = 200;
    let mut canvas = core::Mat::new_rows_cols_with_default(
        height,
        width,
        core::CV_8UC3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;

    // Draw a rectangle
    imgproc::rectangle(
        &mut canvas,
        core::Rect::new(50, 50, 100, 80),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

    // Test add_rotated_range with builder
    let center = core::Point2f::new((width / 2) as f32, (height / 2) as f32);
    let detector = Detector::builder()
        .with_template("rectangle", &canvas, |mut cfg| {
            cfg.add_rotated_range((0..=90u16).step_by(30), center);
        })
        .build()?;

    assert_eq!(detector.num_templates("rectangle"), 4);

    Ok(())
}

#[test]
fn multiple_rotations() -> TestResult {
    // Create template with a distinctive shape (triangle)
    let width = 304;
    let height = 304;
    let mut template_canvas = core::Mat::new_rows_cols_with_default(
        height,
        width,
        core::CV_8UC3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;

    // Draw triangle using lines instead of polylines
    imgproc::line(
        &mut template_canvas,
        core::Point::new(150, 50),
        core::Point::new(50, 250),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::line(
        &mut template_canvas,
        core::Point::new(50, 250),
        core::Point::new(250, 250),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::line(
        &mut template_canvas,
        core::Point::new(250, 250),
        core::Point::new(150, 50),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

    // Create detector and add all rotations via builder
    let center = core::Point2f::new((width / 2) as f32, (height / 2) as f32);
    let detector = Detector::builder()
        .with_template("triangle", &template_canvas, |mut cfg| {
            cfg.add_rotated_range((0..=180u16).step_by(45), center);
        })
        .build()?;
    assert_eq!(detector.num_templates("triangle"), 5);

    // Test matching with a 90-degree rotated triangle
    let mut test_canvas = core::Mat::new_rows_cols_with_default(
        height,
        width,
        core::CV_8UC3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;

    // Draw rotated triangle using lines
    imgproc::line(
        &mut test_canvas,
        core::Point::new(250, 150),
        core::Point::new(50, 50),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::line(
        &mut test_canvas,
        core::Point::new(50, 50),
        core::Point::new(50, 250),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::line(
        &mut test_canvas,
        core::Point::new(50, 250),
        core::Point::new(250, 150),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

    let best = detector
        .match_templates(&test_canvas, 0.3, None)?
        .into_iter()
        .max()
        .unwrap();

    println!("Best match: similarity={:.2}", best.similarity);
    assert!(best.similarity >= 0.3);

    Ok(())
}
