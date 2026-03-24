use graph_matching::Detector;
use opencv::{
    core::{self, Scalar},
    imgcodecs, imgproc,
    prelude::*,
};
use testresult::TestResult;

const IMG_SIZE: i32 = 200;
const RECT_SIZE: i32 = 80;
const RECT_THICKNESS: i32 = 3;

#[test]
fn rect_position_offset() -> TestResult {
    let center = core::Point::new(IMG_SIZE / 2, IMG_SIZE / 2);
    let rect_tl = core::Point::new(center.x - RECT_SIZE / 2, center.y - RECT_SIZE / 2);
    let rect = core::Rect::new(rect_tl.x, rect_tl.y, RECT_SIZE, RECT_SIZE);

    // Create template image: white with black rect outline centered
    let mut template = core::Mat::new_rows_cols_with_default(
        IMG_SIZE,
        IMG_SIZE,
        core::CV_8UC3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    imgproc::rectangle(
        &mut template,
        rect,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        RECT_THICKNESS,
        imgproc::LINE_8,
        0,
    )?;

    // Build detector with center at image center
    let center_f = core::Point2f::new(center.x as f32, center.y as f32);
    let detector = Detector::builder()
        .with_template("rect", &template, |mut cfg| {
            cfg.add_rotated(0.0, center_f);
        })
        .build()?;

    // Match against the same image
    let mut result = detector.match_templates(&template, 0.05, None, None)?;
    result.sort();
    let best = result.last().expect("Expected at least one match").clone();

    let found_center = best.center_point();
    let templ = best.match_template();
    println!(
        "Match at ({}, {}), center ({}, {}), expected center ({}, {}), templ tl=({}, {}) size=({}, {})",
        best.x,
        best.y,
        found_center.x,
        found_center.y,
        center.x,
        center.y,
        templ.tl_x,
        templ.tl_y,
        templ.width,
        templ.height
    );

    // Draw found rect in green onto a copy of the test image
    let mut overlay = template.clone();
    let found_tl = core::Point::new(
        found_center.x - RECT_SIZE / 2,
        found_center.y - RECT_SIZE / 2,
    );
    let found_rect = core::Rect::new(found_tl.x, found_tl.y, RECT_SIZE, RECT_SIZE);
    imgproc::rectangle(
        &mut overlay,
        found_rect,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        RECT_THICKNESS,
        imgproc::LINE_8,
        0,
    )?;

    // Check for remaining black pixels (pixels that were black in template but not painted green)
    let mut remaining_black = 0i32;
    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let orig: core::Vec3b = *template.at_2d(y, x)?;
            let over: core::Vec3b = *overlay.at_2d(y, x)?;
            let was_black = orig[0] == 0 && orig[1] == 0 && orig[2] == 0;
            let now_green = over[0] == 0 && over[1] == 255 && over[2] == 0;
            if was_black && !now_green {
                remaining_black += 1;
            }
        }
    }

    if remaining_black > 0 || found_center != center {
        // Save debug image
        let mut encoded_bytes = core::Vector::<u8>::new();
        imgcodecs::imencode_def(".png", &overlay, &mut encoded_bytes)?;
        let output_path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
        std::fs::write(output_path.join("rect_offset.png"), &encoded_bytes)?;
        println!("Debug image saved, remaining black pixels: {remaining_black}");
    }

    assert_eq!(
        found_center,
        center,
        "Center offset: expected ({}, {}), got ({}, {}), delta ({}, {})",
        center.x,
        center.y,
        found_center.x,
        found_center.y,
        found_center.x - center.x,
        found_center.y - center.y,
    );

    Ok(())
}
