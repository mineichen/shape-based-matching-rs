//! Backend-contract tests for the `fearless_simd` backend: buffer reuse and
//! error reporting. They are not parity tests — the comparison against
//! `opencv::imgproc` lives in tests/filters_opencv_parity.rs.
//!
//! The backend is feature-gated, so this file compiles to nothing unless
//! `fearless-simd` is enabled.
#![cfg(feature = "fearless-simd")]

use graph_matching::backend::{Backend, FearlessSimdBackend};
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

#[test]
fn dst_mats_are_reused() -> TestResult {
    // The production calling pattern reuses the destination `Mat`s: the backend
    // must not reallocate when the shape already matches.
    let src = make(37, 41, 3, 99)?;
    let mut dst = Mat::default();
    FearlessSimdBackend::default().gaussian_blur_7x7(&src, &mut dst)?;
    let first = dst.data_bytes()?;
    let ptr = first.as_ptr();
    let first = first.to_vec();
    FearlessSimdBackend::default().gaussian_blur_7x7(&src, &mut dst)?;
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
    let mut fs = FearlessSimdBackend::default();
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
