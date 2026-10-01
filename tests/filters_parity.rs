//! Parity tests: pure-Rust `graph_matching::filters` vs `opencv::imgproc`.
//!
//! Allowed to use `imgproc` here (dev-dependency); production code must not.

use graph_matching::filters;
use opencv::{
    core::{self, Mat, Scalar, Size},
    imgproc,
    prelude::*,
};
use testresult::TestResult;

// Deterministic xorshift64 fill, no extra deps.
fn fill_random(mat: &mut Mat, channels: i32, seed: u64) -> TestResult {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut s = seed.max(1);
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 33) as u8
    };
    for r in 0..rows {
        let row_ptr = mat.ptr_mut(r)?;
        unsafe {
            for c in 0..(cols * channels) as usize {
                *row_ptr.add(c) = next();
            }
        }
    }
    Ok(())
}

fn make_gray(rows: i32, cols: i32, seed: u64) -> TestResult<Mat> {
    let mut m = Mat::new_rows_cols_with_default(rows, cols, core::CV_8UC1, Scalar::all(0.0))?;
    fill_random(&mut m, 1, seed)?;
    Ok(m)
}

fn make_color(rows: i32, cols: i32, seed: u64) -> TestResult<Mat> {
    let mut m = Mat::new_rows_cols_with_default(rows, cols, core::CV_8UC3, Scalar::all(0.0))?;
    fill_random(&mut m, 3, seed)?;
    Ok(m)
}

fn max_abs_diff_u8(a: &Mat, b: &Mat) -> TestResult<i32> {
    assert_eq!((a.rows(), a.cols(), a.typ()), (b.rows(), b.cols(), b.typ()));
    let da = a.data_bytes()?;
    let db = b.data_bytes()?;
    assert_eq!(da.len(), db.len());
    Ok(da
        .iter()
        .zip(db.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).abs())
        .max()
        .unwrap_or(0))
}

fn opencv_gaussian(src: &Mat) -> TestResult<Mat> {
    let mut dst = Mat::default();
    imgproc::gaussian_blur(
        src,
        &mut dst,
        Size::new(7, 7),
        0.0,
        0.0,
        core::BORDER_REPLICATE,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(dst)
}

fn opencv_sobel_gray(src: &Mat) -> TestResult<(Mat, Mat)> {
    let mut dx = Mat::default();
    let mut dy = Mat::default();
    imgproc::sobel(
        src,
        &mut dx,
        core::CV_32F,
        1,
        0,
        3,
        1.0,
        0.0,
        core::BORDER_REPLICATE,
    )?;
    imgproc::sobel(
        src,
        &mut dy,
        core::CV_32F,
        0,
        1,
        3,
        1.0,
        0.0,
        core::BORDER_REPLICATE,
    )?;
    Ok((dx, dy))
}

fn opencv_sobel_color(src: &Mat) -> TestResult<(Mat, Mat)> {
    let mut dx = Mat::default();
    let mut dy = Mat::default();
    imgproc::sobel(
        src,
        &mut dx,
        core::CV_16S,
        1,
        0,
        3,
        1.0,
        0.0,
        core::BORDER_REPLICATE,
    )?;
    imgproc::sobel(
        src,
        &mut dy,
        core::CV_16S,
        0,
        1,
        3,
        1.0,
        0.0,
        core::BORDER_REPLICATE,
    )?;
    Ok((dx, dy))
}

fn opencv_pyr_down(src: &Mat) -> TestResult<Mat> {
    let mut dst = Mat::default();
    imgproc::pyr_down_def(src, &mut dst)?;
    Ok(dst)
}

#[test]
fn gaussian_gray_matches_opencv() -> TestResult {
    for (rows, cols) in [(16, 16), (64, 48), (65, 63), (128, 128)] {
        let src = make_gray(rows, cols, 0x1234 + rows as u64 * 100 + cols as u64)?;
        let expected = opencv_gaussian(&src)?;
        let mut actual = Mat::default();
        filters::gaussian_blur_7x7(&src, &mut actual)?;
        assert_eq!(
            (actual.rows(), actual.cols(), actual.typ()),
            (expected.rows(), expected.cols(), expected.typ()),
            "size/type mismatch at {rows}x{cols}"
        );
        let d = max_abs_diff_u8(&expected, &actual)?;
        assert!(d <= 1, "gaussian gray {rows}x{cols}: max_abs_diff={d} > 1");
    }
    Ok(())
}

#[test]
fn gaussian_color_matches_opencv() -> TestResult {
    for (rows, cols) in [(16, 16), (64, 48), (65, 63)] {
        let src = make_color(rows, cols, 0xABCD + rows as u64 * 100 + cols as u64)?;
        let expected = opencv_gaussian(&src)?;
        let mut actual = Mat::default();
        filters::gaussian_blur_7x7(&src, &mut actual)?;
        assert_eq!(
            (actual.rows(), actual.cols(), actual.typ()),
            (expected.rows(), expected.cols(), expected.typ()),
            "size/type mismatch at {rows}x{cols}"
        );
        let d = max_abs_diff_u8(&expected, &actual)?;
        assert!(d <= 1, "gaussian color {rows}x{cols}: max_abs_diff={d} > 1");
    }
    Ok(())
}

#[test]
fn sobel_gray_matches_opencv_exact() -> TestResult {
    for (rows, cols) in [(16, 16), (64, 48), (65, 63)] {
        let src = make_gray(rows, cols, 0x5555 + rows as u64)?;
        // Sobel runs on blurred input in the real pipeline; compare on both.
        for input in [src, opencv_gaussian(&make_gray(rows, cols, 0x7777)?)?] {
            let (ex, ey) = opencv_sobel_gray(&input)?;
            let mut ax = Mat::default();
            let mut ay = Mat::default();
            filters::sobel_grayscale(&input, &mut ax, &mut ay)?;
            assert_eq!(
                (ax.rows(), ax.cols(), ax.typ()),
                (ex.rows(), ex.cols(), ex.typ())
            );
            assert_eq!(
                (ay.rows(), ay.cols(), ay.typ()),
                (ey.rows(), ey.cols(), ey.typ())
            );
            assert_eq!(ax.data_bytes()?, ex.data_bytes()?, "sobel gray dx differs");
            assert_eq!(ay.data_bytes()?, ey.data_bytes()?, "sobel gray dy differs");
        }
    }
    Ok(())
}

#[test]
fn sobel_color_matches_opencv_exact() -> TestResult {
    for (rows, cols) in [(16, 16), (64, 48), (65, 63)] {
        let src = make_color(rows, cols, 0x9999 + cols as u64)?;
        let blurred = opencv_gaussian(&src)?;
        let (ex, ey) = opencv_sobel_color(&blurred)?;
        let mut ax = Mat::default();
        let mut ay = Mat::default();
        filters::sobel_color_i16(&blurred, &mut ax, &mut ay)?;
        assert_eq!(
            (ax.rows(), ax.cols(), ax.typ()),
            (ex.rows(), ex.cols(), ex.typ())
        );
        assert_eq!(
            (ay.rows(), ay.cols(), ay.typ()),
            (ey.rows(), ey.cols(), ey.typ())
        );
        assert_eq!(ax.data_bytes()?, ex.data_bytes()?, "sobel color dx differs");
        assert_eq!(ay.data_bytes()?, ey.data_bytes()?, "sobel color dy differs");
    }
    Ok(())
}

#[test]
fn pyr_down_matches_opencv() -> TestResult {
    // gray + color + single-channel mask-like inputs, even and odd sizes
    for (rows, cols) in [(16, 16), (64, 48), (65, 63), (128, 128)] {
        let gray = make_gray(rows, cols, 0x2468 + rows as u64)?;
        let expected = opencv_pyr_down(&gray)?;
        let mut actual = Mat::default();
        filters::pyr_down(&gray, &mut actual)?;
        assert_eq!(
            (actual.rows(), actual.cols(), actual.typ()),
            (expected.rows(), expected.cols(), expected.typ()),
            "pyr_down gray size/type mismatch at {rows}x{cols}"
        );
        let d = max_abs_diff_u8(&expected, &actual)?;
        assert!(d <= 1, "pyr_down gray {rows}x{cols}: max_abs_diff={d} > 1");

        let color = make_color(rows, cols, 0x1357 + cols as u64)?;
        let expected = opencv_pyr_down(&color)?;
        let mut actual = Mat::default();
        filters::pyr_down(&color, &mut actual)?;
        assert_eq!(
            (actual.rows(), actual.cols(), actual.typ()),
            (expected.rows(), expected.cols(), expected.typ()),
            "pyr_down color size/type mismatch at {rows}x{cols}"
        );
        let d = max_abs_diff_u8(&expected, &actual)?;
        assert!(d <= 1, "pyr_down color {rows}x{cols}: max_abs_diff={d} > 1");
    }
    Ok(())
}
