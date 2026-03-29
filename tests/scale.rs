use std::path::Path;

use graph_matching::Detector;
use opencv::{
    core::{self, Scalar},
    imgcodecs, imgproc,
};
use testresult::TestResult;

const IMG_SIZE: i32 = 400;
const RECT_W: i32 = 60;
const RECT_H: i32 = 40;
const THICKNESS: i32 = 2;

fn create_rect_image(cx: i32, cy: i32, w: i32, h: i32) -> TestResult<core::Mat> {
    let mut canvas = core::Mat::new_rows_cols_with_default(
        IMG_SIZE,
        IMG_SIZE,
        core::CV_8UC3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    let tl = core::Point::new(cx - w / 2, cy - h / 2);
    imgproc::rectangle(
        &mut canvas,
        core::Rect::new(tl.x, tl.y, w, h),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        THICKNESS,
        imgproc::LINE_8,
        0,
    )?;
    Ok(canvas)
}

#[test]
fn scaled_detection() -> TestResult {
    let center = core::Point::new(IMG_SIZE / 2, IMG_SIZE / 2);
    let template_img = create_rect_image(center.x, center.y, RECT_W, RECT_H)?;
    let scale = 2.0;

    let center_f = core::Point2f::new(center.x as f32, center.y as f32);
    let detector = Detector::builder()
        .with_template("rect", &template_img, |mut cfg| {
            cfg.add_scaled(scale, center_f);
        })
        .build()?;

    assert_eq!(detector.num_templates("rect"), 2);

    // Create test image with a slightly smaller rectangle (0.8 scale)
    let scaled_w = (RECT_W as f32 * scale).round() as i32;
    let scaled_h = (RECT_H as f32 * scale).round() as i32;
    let test_img = create_rect_image(center.x, center.y, scaled_w, scaled_h)?;

    let result = detector.match_templates(&test_img, 0.5, None)?;
    let debug_img = result.debug_visual(test_img, None)?;
    let mut encoded_bytes = core::Vector::<u8>::new();
    imgcodecs::imencode_def(".png", &debug_img, &mut encoded_bytes)?;

    // Save result
    let output_file = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("debug_images")
        .join("find_scaled.png");
    std::fs::write(&output_file, &encoded_bytes)?;
    let best = result.iter().max().unwrap();

    println!(
        "Best match: similarity={:.2}, scale={:.2}",
        best.similarity,
        best.scale()
    );
    assert!(
        best.similarity > 0.9,
        "Expected similarity > 0.5, got {:.2}",
        best.similarity
    );
    assert!(
        (best.scale() - scale).abs() < 0.01,
        "Expected scale {scale:.2}, got {:.2}",
        best.scale()
    );

    Ok(())
}

#[test]
fn scaled_range() -> TestResult {
    let center = core::Point::new(IMG_SIZE / 2, IMG_SIZE / 2);
    let template_img = create_rect_image(center.x, center.y, RECT_W, RECT_H)?;

    let center_f = core::Point2f::new(center.x as f32, center.y as f32);
    let detector = Detector::builder()
        .with_template("rect", &template_img, |mut cfg| {
            cfg.add_scaled_range(
                (80u16..=120).step_by(10).map(|s| s as f32 / 100.0),
                center_f,
            );
        })
        .build()?;

    // base(1.0) + 0.8 + 0.9 + 1.1 + 1.2 = 5 templates (1.0 scale is skipped)
    assert_eq!(detector.num_templates("rect"), 5);

    Ok(())
}
