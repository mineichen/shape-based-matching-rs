//! The `fearless_simd` filters must be **bit-identical** to the `pulp`
//! reference: same integer kernels, same rounding, so switching backends can
//! never change a detector result. Every image size and channel count that
//! exercises the SIMD interior, the SIMD tail and the border frames is
//! compared byte for byte.
//!
//! The comparison against `opencv::imgproc` itself lives in
//! tests/filters_parity.rs and only needs the `pulp` reference: since
//! `fearless_simd` == `pulp` byte for byte (asserted here) and `pulp` is within
//! one LSB of `imgproc` (asserted there), `fearless_simd` is within one LSB too.
//! That is why this file needs no `imgproc` at all.
//!
//! `dst_mats_are_reused` and `rejects_bad_input` are not parity tests but
//! backend-contract tests (buffer reuse, error reporting); they live here
//! because they cover the `fearless_simd` backend, which nothing else does.
//!
//! Both backends are feature-gated, so this file compiles to nothing unless
//! `fearless-simd` and `pulp` are enabled.
#![cfg(all(feature = "fearless-simd", feature = "pulp"))]

use graph_matching::backend::{Backend, FearlessSimdBackend, PulpBackend};
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

fn assert_bytes_eq(a: &Mat, b: &Mat, what: &str) -> TestResult {
    assert_eq!(
        (a.rows(), a.cols(), a.typ()),
        (b.rows(), b.cols(), b.typ()),
        "{what}: shape/type mismatch"
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
            "{what}: outputs differ (bit-identical required), {n_diff}/{} bytes differ, \
             first at byte {first}",
            da.len()
        );
    }
    Ok(())
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
            PulpBackend.gaussian_blur_7x7(&src, &mut want)?;
            FearlessSimdBackend::default().gaussian_blur_7x7(&src, &mut got)?;
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
fn sobel_gray_bit_identical_to_pulp() -> TestResult {
    for (rows, cols) in SIZES {
        let src = make(rows, cols, 1, 0x5555 + rows as u64 * 3 + cols as u64)?;
        let mut ax = Mat::default();
        let mut ay = Mat::default();
        PulpBackend.sobel_grayscale(&src, &mut ax, &mut ay)?;
        let mut bx = Mat::default();
        let mut by = Mat::default();
        FearlessSimdBackend::default().sobel_grayscale(&src, &mut bx, &mut by)?;
        assert_bytes_eq(&ax, &bx, &format!("sobel gray dx {rows}x{cols}"))?;
        assert_bytes_eq(&ay, &by, &format!("sobel gray dy {rows}x{cols}"))?;
    }
    Ok(())
}

#[test]
fn sobel_color_bit_identical_to_pulp() -> TestResult {
    for (rows, cols) in SIZES {
        let src = make(rows, cols, 3, 0x9999 + rows as u64 * 5 + cols as u64)?;
        // The reference blur is itself the `pulp` one, so this test needs no
        // third implementation.
        let mut blurred = Mat::default();
        PulpBackend.gaussian_blur_7x7(&src, &mut blurred)?;
        let mut ax = Mat::default();
        let mut ay = Mat::default();
        PulpBackend.sobel_color_i16(&blurred, &mut ax, &mut ay)?;
        let mut bx = Mat::default();
        let mut by = Mat::default();
        FearlessSimdBackend::default().sobel_color_i16(&blurred, &mut bx, &mut by)?;
        assert_bytes_eq(&ax, &bx, &format!("sobel color dx {rows}x{cols}"))?;
        assert_bytes_eq(&ay, &by, &format!("sobel color dy {rows}x{cols}"))?;
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
            PulpBackend.pyr_down(&src, &mut want)?;
            FearlessSimdBackend::default().pyr_down(&src, &mut got)?;
            assert_bytes_eq(
                &want,
                &got,
                &format!("pyr_down ch={channels} {rows}x{cols}"),
            )?;
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
