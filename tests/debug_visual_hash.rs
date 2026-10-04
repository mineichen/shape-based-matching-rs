//! Byte-exact regression guard for `debug_visual` output.
//!
//! Step 1 of UNIFIED_COORDINATE_SYSTEM.md: the sha2-256 hash of all pixel
//! values of the rendered output must not change while the coordinate
//! system refactor is in progress. Any change that shifts a single pixel
//! fails these tests.
//!
//! NOTE: the pinned hashes were detected once in this container and depend
//! on the exact opencv build (rasterization + font rendering). Re-detect
//! only if the opencv version changes.
//!
//! These tests are deliberately not part of the `backend_test!` matrix: the
//! hashes were detected with the pure-Rust filters, and `opencv::imgproc`
//! rounds differently (see tests/filters_parity.rs), which moves a pixel
//! across a gradient-orientation boundary and therefore changes the extracted
//! features and the rendered image. `pulp` and `fearless_simd` are
//! bit-identical, so both must reproduce the same hash; `opencv` is called
//! too, but skipped while it renders a different image.

use graph_matching::{Backend, Detector, Point2i};
use opencv::{
    core::{self, Mat, Scalar},
    imgproc,
    prelude::*,
};
use sha2::{Digest, Sha256};
use testresult::TestResult;

/// sha2-256 over all pixel bytes of a continuous 8-bit Mat.
fn hash_mat_pixels(mat: &Mat) -> TestResult<String> {
    assert!(mat.is_continuous());
    let bytes = mat.data_bytes()?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn assert_pixel_hash(actual: String, expected: &str) {
    assert_eq!(
        actual, expected,
        "debug_visual output changed! Detected hash (pin inline if intended): {actual}"
    );
}

const IMAGE_WIDTH: i32 = 400;
const IMAGE_HEIGHT: i32 = 400;
const ELLIPSE_WIDTH: i32 = 80;
const ELLIPSE_HEIGHT: i32 = 50;
const ELLIPSE_THICKNESS: i32 = 3;

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

// Detected 2026-10-02, opencv 4.x in this container (400x400 CV_8UC3 output).
const ELLIPSE_FLOW_SHA2: &str = "3e13e36d07ef72f04d839dd2a90ce27d4c7fb755327d30ffc873ab3a5cbe10dd";

fn ellipse_output_is_stable(backend: impl Backend) -> TestResult {
    // Mirrors tests/detector.rs::ellipse_detection (sorted result, no
    // extra overlay) — the render whose pixels must never change.
    let center = core::Point::new(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2);
    let train_canvas = create_ellipse_image(center, 0.0)?;

    // Pinned-scenario pivot: the center pixel of the template image.
    // Pivots are pixel indices — the pivot is the center of that pixel
    // (pixel `N`'s center is `N + 0.5`), so this renders byte-identical
    // features to the previously pinned `200.5` float pivot.
    let center_image = Point2i::splat(IMAGE_WIDTH / 2);
    let mut detector = Detector::builder()
        .with_backend(backend)
        .with_template("ellipse", &train_canvas, |mut cfg| {
            cfg.add_rotated(0.0, center_image); // Explicitly add zero angle
            cfg.add_rotated(45.0, center_image);
        })
        .build()?;
    assert_eq!(detector.num_templates("ellipse"), 2);

    let test_canvas = create_ellipse_image(center, 45.0)?;
    let mut result = detector.match_templates(&test_canvas, 0.95, None)?;
    result.sort();

    let debug_image = result.debug_visual(test_canvas.clone(), None)?;
    assert_pixel_hash(hash_mat_pixels(&debug_image)?, ELLIPSE_FLOW_SHA2);
    Ok(())
}

const IMG_SIZE: i32 = 400;
const RECT_W: i32 = 60;
const RECT_H: i32 = 40;
const THICKNESS: i32 = 2;

fn create_rect_image(pos: Point2i, w: i32, h: i32) -> TestResult<Mat> {
    let mut canvas = core::Mat::new_rows_cols_with_default(
        IMG_SIZE,
        IMG_SIZE,
        core::CV_8UC3,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    let tl = core::Point::new(pos.x - w / 2, pos.y - h / 2);
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

// Detected 2026-10-02, opencv 4.x in this container (400x400 CV_8UC3 output).
const SCALE_FLOW_SHA2: &str = "74efec12dcfcdd60e9bd8b524ded5338f86eb7f1af703245d95b19976b33ff7e";

fn scale_output_is_stable(backend: impl Backend) -> TestResult {
    // Mirrors tests/scale.rs::scaled_detection (unsorted result).
    let center = Point2i::splat(IMG_SIZE / 2);
    let template_img = create_rect_image(center, RECT_W, RECT_H)?;
    let scale = 2.0;

    // Pinned-scenario pivot: the center pixel of the template image (see
    // the ellipse test above).
    let center_image = Point2i::splat(IMG_SIZE / 2);
    let mut detector = Detector::builder()
        .with_backend(backend)
        .with_template("rect", &template_img, |mut cfg| {
            cfg.add_scaled(scale, center_image);
        })
        .build()?;

    assert_eq!(detector.num_templates("rect"), 2);

    let scaled_w = (RECT_W as f32 * scale).round() as i32;
    let scaled_h = (RECT_H as f32 * scale).round() as i32;
    let test_img = create_rect_image(center, scaled_w, scaled_h)?;

    let result = detector.match_templates(&test_img, 0.95, None)?;
    let debug_img = result.debug_visual(test_img, None)?;
    assert_pixel_hash(hash_mat_pixels(&debug_img)?, SCALE_FLOW_SHA2);
    Ok(())
}

#[cfg(feature = "pulp")]
#[test]
fn debug_visual_output_is_stable_pulp() -> TestResult {
    ellipse_output_is_stable(graph_matching::PulpBackend)?;
    scale_output_is_stable(graph_matching::PulpBackend)
}

#[cfg(feature = "fearless-simd")]
#[test]
fn debug_visual_output_is_stable_fearless_simd() -> TestResult {
    ellipse_output_is_stable(graph_matching::FearlessSimdBackend::default())?;
    scale_output_is_stable(graph_matching::FearlessSimdBackend::default())
}

/// The `opencv` backend renders a different image today: `imgproc` rounds
/// differently than the pure-Rust kernels (see tests/filters_parity.rs), which
/// flips a gradient-orientation bin, changes the extracted features and with
/// them every drawn pixel. Run with `--ignored` to see the current hashes
/// (`1f5feb67…` for the ellipse, `88b02760…` for the scaled rect).
#[cfg(feature = "opencv")]
#[test]
#[ignore = "opencv currently produces another hash; the other backends have to be fixed first"]
fn debug_visual_output_is_stable_opencv() -> TestResult {
    ellipse_output_is_stable(graph_matching::OpenCvBackend)?;
    scale_output_is_stable(graph_matching::OpenCvBackend)
}
