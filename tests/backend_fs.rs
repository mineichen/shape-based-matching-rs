//! End-to-end check that the `fearless_simd` backend is a drop-in replacement
//! for the `pulp` backend: the two must produce identical detector features and
//! matches, so callers can switch backends without changing results.
#![cfg(feature = "fearless-simd")]

use graph_matching::{Detector, FearlessSimd, Point2i};
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
fn fearless_simd_backend_matches_native() -> TestResult {
    // 63 exercises the u8 accumulator branch (< 64 features),
    // 70 exercises the u16 branch (>= 64 features).
    for num_features in [63, 70] {
        check(num_features)?;
    }
    Ok(())
}

fn check(num_features: usize) -> TestResult {
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
    let mut fs = build().with_backend(FearlessSimd::default()).build()?;

    let native_len = native
        .base_template("rect")
        .map(|t| t.features.len())
        .unwrap_or(0);
    let fs_len = fs
        .base_template("rect")
        .map(|t| t.features.len())
        .unwrap_or(0);
    assert_eq!(
        native_len, fs_len,
        "feature count differs (num_features={num_features})"
    );
    assert!(native_len >= num_features, "detector used too few features");

    assert_eq!(native.num_templates("rect"), fs.num_templates("rect"));

    let a = native.match_templates(&img, 0.5, None)?;
    let b = fs.match_templates(&img, 0.5, None)?;
    assert!(!a.is_empty());
    // Bit-identical filters must give exactly the same matches.
    assert_eq!(a.len(), b.len(), "match count differs");
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.pos.x, y.pos.x);
        assert_eq!(x.pos.y, y.pos.y);
        assert_eq!(x.similarity, y.similarity);
        assert_eq!(x.class_id, y.class_id);
    }
    Ok(())
}
