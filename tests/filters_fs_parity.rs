//! Parity tests for the `fearless_simd` filter backend vs the `pulp` backend
//! (`graph_matching::filters`) and vs `opencv::imgproc`.
//!
//! The `fearless_simd` backend is a different SIMD implementation of exactly
//! the same integer kernels, so it must be **bit-identical** to the `pulp`
//! backend. Parity with OpenCV keeps the documented tolerances (gauss/pyr
//! within one LSB because of the different rounding order, sobel exact).
//!
//! The backend is feature-gated; without the feature this file compiles to
//! nothing.
#![cfg(feature = "fearless-simd")]

use graph_matching::backend::{Backend, FearlessSimd, Native};
use opencv::{
    core::{self, Mat},
    prelude::*,
};
use testresult::TestResult;

/// Deterministic xorshift64 fill, no extra deps.
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

fn make(rows: i32, cols: i32, channels: i32, seed: u64) -> TestResult<Mat> {
    let typ = if channels == 1 {
        core::CV_8UC1
    } else {
        core::CV_8UC3
    };
    let mut m = Mat::new_rows_cols_with_default(rows, cols, typ, core::Scalar::all(0.0))?;
    fill_random(&mut m, channels, seed)?;
    Ok(m)
}

fn assert_same_shape(a: &Mat, b: &Mat, what: &str) {
    assert_eq!(
        (a.rows(), a.cols(), a.typ()),
        (b.rows(), b.cols(), b.typ()),
        "{what}: shape/type mismatch"
    );
}

fn assert_bytes_eq(a: &Mat, b: &Mat, what: &str) -> TestResult {
    assert_same_shape(a, b, what);
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
            "{what}: outputs differ (bit-identical required), {n_diff}/{} bytes differ, \
             first at byte {first}",
            da.len()
        );
    }
    Ok(())
}

fn max_abs_diff_bytes(a: &Mat, b: &Mat) -> TestResult<i32> {
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
    opencv::imgproc::gaussian_blur(
        src,
        &mut dst,
        core::Size::new(7, 7),
        0.0,
        0.0,
        core::BORDER_REPLICATE,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(dst)
}

fn opencv_sobel(src: &Mat, depth: i32) -> TestResult<(Mat, Mat)> {
    let mut dx = Mat::default();
    let mut dy = Mat::default();
    for (m, dx_sel, dy_sel) in [(&mut dx, 1, 0), (&mut dy, 0, 1)] {
        opencv::imgproc::sobel(
            src,
            m,
            depth,
            dx_sel,
            dy_sel,
            3,
            1.0,
            0.0,
            core::BORDER_REPLICATE,
        )?;
    }
    Ok((dx, dy))
}

fn opencv_pyr_down(src: &Mat) -> TestResult<Mat> {
    let mut dst = Mat::default();
    opencv::imgproc::pyr_down_def(src, &mut dst)?;
    Ok(dst)
}

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

#[test]
fn gaussian_bit_identical_to_pulp() -> TestResult {
    for channels in [1, 3] {
        for (rows, cols) in SIZES {
            let src = make(
                rows,
                cols,
                channels,
                0x1234 + (rows * 31 + cols) as u64 * 7 + channels as u64,
            )?;
            let mut want = Mat::default();
            let mut got = Mat::default();
            Native.gaussian_blur_7x7(&src, &mut want)?;
            FearlessSimd::default().gaussian_blur_7x7(&src, &mut got)?;
            assert_bytes_eq(
                &want,
                &got,
                &format!("gaussian ch={channels} {rows}x{cols}"),
            )?;
        }
    }
    Ok(())
}

#[test]
fn gaussian_within_one_lsb_of_opencv() -> TestResult {
    for channels in [1, 3] {
        for (rows, cols) in SIZES {
            let src = make(
                rows,
                cols,
                channels,
                0x4321 + (rows * 17 + cols) as u64 + channels as u64,
            )?;
            let want = opencv_gaussian(&src)?;
            let mut got = Mat::default();
            FearlessSimd::default().gaussian_blur_7x7(&src, &mut got)?;
            let d = max_abs_diff_bytes(&want, &got)?;
            assert!(
                d <= 1,
                "gaussian ch={channels} {rows}x{cols}: max_abs_diff={d} > 1"
            );
        }
    }
    Ok(())
}

#[test]
fn sobel_gray_bit_identical_to_pulp() -> TestResult {
    for (rows, cols) in SIZES {
        let src = make(rows, cols, 1, 0x5555 + rows as u64 * 3 + cols as u64)?;
        let (wx, wy) = opencv_sobel(&src, core::CV_32F)?;
        let mut ax = Mat::default();
        let mut ay = Mat::default();
        Native.sobel_grayscale(&src, &mut ax, &mut ay)?;
        let mut bx = Mat::default();
        let mut by = Mat::default();
        FearlessSimd::default().sobel_grayscale(&src, &mut bx, &mut by)?;
        assert_bytes_eq(&ax, &bx, &format!("sobel gray dx {rows}x{cols}"))?;
        assert_bytes_eq(&ay, &by, &format!("sobel gray dy {rows}x{cols}"))?;
        assert_bytes_eq(&wx, &bx, &format!("sobel gray dx vs opencv {rows}x{cols}"))?;
        assert_bytes_eq(&wy, &by, &format!("sobel gray dy vs opencv {rows}x{cols}"))?;
    }
    Ok(())
}

#[test]
fn sobel_color_bit_identical_to_pulp() -> TestResult {
    for (rows, cols) in SIZES {
        let src = make(rows, cols, 3, 0x9999 + rows as u64 * 5 + cols as u64)?;
        let blurred = opencv_gaussian(&src)?;
        let mut ax = Mat::default();
        let mut ay = Mat::default();
        Native.sobel_color_i16(&blurred, &mut ax, &mut ay)?;
        let mut bx = Mat::default();
        let mut by = Mat::default();
        FearlessSimd::default().sobel_color_i16(&blurred, &mut bx, &mut by)?;
        assert_bytes_eq(&ax, &bx, &format!("sobel color dx {rows}x{cols}"))?;
        assert_bytes_eq(&ay, &by, &format!("sobel color dy {rows}x{cols}"))?;
        let (wx, wy) = opencv_sobel(&blurred, core::CV_16S)?;
        assert_bytes_eq(&wx, &bx, &format!("sobel color dx vs opencv {rows}x{cols}"))?;
        assert_bytes_eq(&wy, &by, &format!("sobel color dy vs opencv {rows}x{cols}"))?;
    }
    Ok(())
}

#[test]
fn pyr_down_bit_identical_to_pulp() -> TestResult {
    for channels in [1, 3] {
        for (rows, cols) in SIZES {
            let src = make(
                rows,
                cols,
                channels,
                0x2468 + (rows * 11 + cols) as u64 * 3 + channels as u64,
            )?;
            let mut want = Mat::default();
            let mut got = Mat::default();
            Native.pyr_down(&src, &mut want)?;
            FearlessSimd::default().pyr_down(&src, &mut got)?;
            assert_bytes_eq(
                &want,
                &got,
                &format!("pyr_down ch={channels} {rows}x{cols}"),
            )?;
            let want_cv = opencv_pyr_down(&src)?;
            let d = max_abs_diff_bytes(&want_cv, &got)?;
            assert!(
                d <= 1,
                "pyr_down ch={channels} {rows}x{cols} vs opencv: max_abs_diff={d} > 1"
            );
        }
    }
    Ok(())
}

#[test]
fn dst_mats_are_reused() -> TestResult {
    // The production calling pattern reuses the destination `Mat`s: the backend
    // must not reallocate when the shape already matches.
    let src = make(37, 41, 3, 99)?;
    let mut dst = Mat::default();
    FearlessSimd::default().gaussian_blur_7x7(&src, &mut dst)?;
    let first = dst.data_bytes()?;
    let ptr = first.as_ptr();
    let first = first.to_vec();
    FearlessSimd::default().gaussian_blur_7x7(&src, &mut dst)?;
    assert!(
        std::ptr::eq(dst.data_bytes()?.as_ptr(), ptr),
        "destination Mat was reallocated"
    );
    assert_eq!(dst.data_bytes()?, first.as_slice());
    Ok(())
}

#[test]
fn rejects_bad_input() -> TestResult {
    // Empty and wrongly typed sources must be reported, not panic.
    let empty = Mat::default();
    let mut fs = FearlessSimd::default();
    assert!(fs.gaussian_blur_7x7(&empty, &mut Mat::default()).is_err());
    assert!(fs.pyr_down(&empty, &mut Mat::default()).is_err());
    assert!(
        fs.sobel_grayscale(&empty, &mut Mat::default(), &mut Mat::default())
            .is_err()
    );

    let color = make(20, 20, 3, 7)?;
    assert!(
        fs.sobel_grayscale(&color, &mut Mat::default(), &mut Mat::default())
            .is_err()
    );
    let gray = make(20, 20, 1, 7)?;
    assert!(
        fs.sobel_color_i16(&gray, &mut Mat::default(), &mut Mat::default())
            .is_err()
    );
    Ok(())
}
