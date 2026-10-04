//! Parity tests: pure-Rust `graph_matching::filters::fsimd` vs `opencv::imgproc`.
//!
//! Allowed to use `imgproc` here (dev-dependency); production code must not.
//! The `fearless-simd` backend is feature-gated; without the feature this file
//! compiles to nothing.
//!
//! The Gaussian blur and `pyr_down` are compared with a one-LSB tolerance,
//! because `imgproc` rounds the separable filter differently than the
//! integer kernels here. The Sobel results must be *byte-identical*: same
//! integer arithmetic, same rounding.
//!
//! Every filter runs on all of [`SIZES`] and both channel counts, so the SIMD
//! interior, the SIMD tail and the border frames are all compared.
#![cfg(feature = "fearless-simd")]

use graph_matching::filters::fsimd::{self, Scratch};
use opencv::{
    core::{self, Mat, Scalar, Size},
    imgproc,
    prelude::*,
};
use testresult::TestResult;

/// Sizes chosen to cover: tiny (all-border), odd/even, wide/tall, a size where
/// the SIMD interior is smaller than one vector, and two realistic sizes with
/// odd/even row and column counts (the SIMD tail and the row/column borders must
/// line up the same way there as in the small sizes).
const SIZES: [(i32, i32); 9] = [
    (1, 1),
    (2, 3),
    (7, 7),
    (16, 16),
    (63, 65),
    (64, 64),
    (129, 127),
    (479, 640),
    (480, 641),
];

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
    make_color(rows, cols, 1, seed)
}

fn make_color(rows: i32, cols: i32, channels: i32, seed: u64) -> TestResult<Mat> {
    let typ = if channels == 1 {
        core::CV_8UC1
    } else {
        core::CV_8UC3
    };
    let mut m = Mat::new_rows_cols_with_default(rows, cols, typ, Scalar::all(0.0))?;
    fill_random(&mut m, channels, seed)?;
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

fn assert_bytes_eq(a: &Mat, b: &Mat, what: &str) -> TestResult {
    assert_eq!(
        (a.rows(), a.cols(), a.typ()),
        (b.rows(), b.cols(), b.typ()),
        "{what}: size/type mismatch"
    );
    let da = a.data_bytes()?;
    let db = b.data_bytes()?;
    assert_eq!(da.len(), db.len(), "{what}: byte length mismatch");
    if da != db {
        let first = da
            .iter()
            .zip(db.iter())
            .position(|(x, y)| x != y)
            .unwrap_or(0);
        let n_diff = da.iter().zip(db.iter()).filter(|(x, y)| x != y).count();
        panic!(
            "{what}: outputs differ (byte-identical required), {n_diff}/{} bytes differ, \
             first at byte {first}",
            da.len()
        );
    }
    Ok(())
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
fn gaussian_matches_opencv() -> TestResult {
    for channels in [1, 3] {
        for (rows, cols) in SIZES {
            let src = make_color(
                rows,
                cols,
                channels,
                0x1234 + (rows * 31 + cols) as u64 * 7 + channels as u64,
            )?;
            let expected = opencv_gaussian(&src)?;
            let mut actual = Mat::default();
            fsimd::gaussian_blur_7x7(&mut Scratch::default(), &src, &mut actual)?;
            let what = format!("gaussian ch={channels} {rows}x{cols}");
            assert_bytes_eq(&expected, &actual, &what)?;
        }
    }
    Ok(())
}

#[test]
fn sobel_gray_matches_opencv_exact() -> TestResult {
    for (rows, cols) in SIZES {
        let src = make_gray(rows, cols, 0x5555 + rows as u64 * 3 + cols as u64)?;
        // Sobel runs on blurred input in the real pipeline; compare on both.
        for input in [src, opencv_gaussian(&make_gray(rows, cols, 0x7777)?)?] {
            let (ex, ey) = opencv_sobel_gray(&input)?;
            let mut ax = Mat::default();
            let mut ay = Mat::default();
            fsimd::sobel_grayscale(&mut Scratch::default(), &input, &mut ax, &mut ay)?;
            let what = format!("sobel gray {rows}x{cols}");
            assert_bytes_eq(&ax, &ex, &format!("{what} dx"))?;
            assert_bytes_eq(&ay, &ey, &format!("{what} dy"))?;
        }
    }
    Ok(())
}

#[test]
fn sobel_color_matches_opencv_exact() -> TestResult {
    for (rows, cols) in SIZES {
        let src = make_color(rows, cols, 3, 0x9999 + rows as u64 * 5 + cols as u64)?;
        let blurred = opencv_gaussian(&src)?;
        let (ex, ey) = opencv_sobel_color(&blurred)?;
        let mut ax = Mat::default();
        let mut ay = Mat::default();
        fsimd::sobel_color_i16(&mut Scratch::default(), &blurred, &mut ax, &mut ay)?;
        let what = format!("sobel color {rows}x{cols}");
        assert_bytes_eq(&ax, &ex, &format!("{what} dx"))?;
        assert_bytes_eq(&ay, &ey, &format!("{what} dy"))?;
    }
    Ok(())
}

#[test]
fn pyr_down_matches_opencv() -> TestResult {
    // gray + color + odd and even sizes.
    for channels in [1, 3] {
        for (rows, cols) in SIZES {
            let src = make_color(
                rows,
                cols,
                channels,
                0x2468 + (rows * 11 + cols) as u64 * 3 + channels as u64,
            )?;
            let expected = opencv_pyr_down(&src)?;
            let mut actual = Mat::default();
            fsimd::pyr_down(&mut Scratch::default(), &src, &mut actual)?;
            let what = format!("pyr_down ch={channels} {rows}x{cols}");
            assert_bytes_eq(&expected, &actual, &what)?;
        }
    }
    Ok(())
}
