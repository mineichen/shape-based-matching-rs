//! Verifies the pluggable filter `Backend` abstraction: the `OpenCv` backend
//! (old `imgproc` implementation) must produce the same detector results as
//! the default `Native` (pure-Rust) backend, since both filters are
//! bit-compatible (gauss/pyr within rounding, sobel exact).
#![cfg(feature = "opencv")]

use graph_matching::{Detector, Native, OpenCv, Point2i};
use opencv::{
    core::{self, Mat, Scalar},
    imgproc,
};
use testresult::TestResult;

const IMG_SIZE: i32 = 200;
const RECT_SIZE: i32 = 80;

fn test_image() -> TestResult<Mat> {
    let mut canvas =
        Mat::new_rows_cols_with_default(IMG_SIZE, IMG_SIZE, core::CV_8UC1, Scalar::all(255.0))?;
    let tl = (IMG_SIZE - RECT_SIZE) / 2;
    imgproc::rectangle(
        &mut canvas,
        core::Rect::new(tl, tl, RECT_SIZE, RECT_SIZE),
        Scalar::all(0.0),
        3,
        imgproc::LINE_8,
        0,
    )?;
    Ok(canvas)
}

#[test]
fn opencv_backend_matches_native() -> TestResult {
    // 63 exercises the u8 accumulator branch (< 64 features),
    // 70 exercises the u16 branch (>= 64 features).
    for num_features in [63, 70] {
        check_backend_parity(num_features)?;
    }
    Ok(())
}

fn check_backend_parity(num_features: usize) -> TestResult {
    let img = test_image()?;
    // Pivot is the center pixel of the template image.
    let center = Point2i::splat(IMG_SIZE / 2);

    let build = || {
        Detector::builder()
            .num_features(num_features)
            .with_template("rect", &img, |mut cfg| {
                cfg.add_rotated(0.0, center);
                cfg.add_rotated(45.0, center);
            })
    };

    let mut native = build().build()?;
    let mut ocv = build().with_backend(OpenCv).build()?;

    // Guard against a weak test image: if fewer candidates than requested
    // exist, extraction silently returns fewer features and the 70-case
    // would still take the u8 (< 64) branch. Note `>=` (not `==`):
    // extraction currently returns num_features + 1 when candidates allow.
    let native_len = native
        .base_template("rect")
        .map(|t| t.features.len())
        .unwrap_or(0);
    let ocv_len = ocv
        .base_template("rect")
        .map(|t| t.features.len())
        .unwrap_or(0);
    assert!(
        native_len >= num_features,
        "native detector did not use all {num_features} features (got {native_len})"
    );
    assert!(
        ocv_len >= num_features,
        "opencv detector did not use all {num_features} features (got {ocv_len})"
    );
    if num_features >= 64 {
        assert!(
            native_len >= 64 && ocv_len >= 64,
            "expected u16 branch (>= 64 features) for num_features={num_features}, got native={native_len} opencv={ocv_len}"
        );
    }

    assert_eq!(native.num_templates("rect"), ocv.num_templates("rect"));

    let native_matches = native.match_templates(&img, 0.5, None)?;
    let ocv_matches = ocv.match_templates(&img, 0.5, None)?;
    assert!(!native_matches.is_empty());
    assert!(!ocv_matches.is_empty());
    // The backends agree on parity up to the documented rounding differences
    // (gauss <= 1 LSB), so the *count* of raw matches above the threshold may
    // differ slightly; the best match must be identical.
    let native_best = native_matches.iter().max().unwrap();
    let ocv_best = ocv_matches.iter().max().unwrap();

    assert_eq!(
        native_best.pos.x, ocv_best.pos.x,
        "best x mismatch (num_features={num_features})"
    );
    assert_eq!(
        native_best.pos.y, ocv_best.pos.y,
        "best y mismatch (num_features={num_features})"
    );
    assert!(
        (native_best.similarity - ocv_best.similarity).abs() < 0.01,
        "best similarity mismatch (num_features={num_features}): native {} vs opencv {}",
        native_best.similarity,
        ocv_best.similarity
    );
    assert!(
        native_best.similarity > 0.9,
        "native sim={native_best:?} (num_features={num_features})"
    );
    Ok(())
}

#[test]
fn native_backend_is_default() -> TestResult {
    let img = test_image()?;
    // Pivot is the center pixel of the template image.
    let center = Point2i::splat(IMG_SIZE / 2);
    // Default build must be usable without naming Native explicitly.
    let mut detector = Detector::builder()
        .with_template("rect", &img, |mut cfg| {
            cfg.add_rotated(0.0, center);
        })
        .build()?;
    let matches = detector.match_templates(&img, 0.5, None)?;
    assert!(!matches.is_empty());

    // Explicit Native must behave identically.
    let mut explicit = Detector::builder()
        .with_backend(Native)
        .with_template("rect", &img, |mut cfg| {
            cfg.add_rotated(0.0, center);
        })
        .build()?;
    let explicit_matches = explicit.match_templates(&img, 0.5, None)?;
    assert_eq!(matches.len(), explicit_matches.len());
    Ok(())
}
