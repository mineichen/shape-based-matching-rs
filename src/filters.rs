//! Image filters on the matching hot path, one module per SIMD backend.
//!
//! Each backend is a drop-in replacement for the `opencv::imgproc` calls the
//! matcher used before. The module names are plain code spans, because only the
//! feature-enabled ones exist in a given build:
//!
//! - `pulp`: [`pulp`](https://docs.rs/pulp) portable SIMD with runtime dispatch
//!   (`u16`/`i16` lanes). Requires the `pulp` cargo feature.
//! - `fsimd`: [`fearless_simd`](https://docs.rs/fearless_simd) portable SIMD,
//!   bit-identical to `pulp` but considerably faster. Requires the
//!   `fearless-simd` cargo feature.
//!
//! Both read from a shared `&Mat` and write into a caller-provided `&mut Mat`
//! (reallocated only on size/type mismatch), so `Mat`-allocated image buffers
//! are reused with zero copies. Row access goes through the safe
//! `at_row`/`at_row_mut` helpers, so non-continuous (ROI) inputs work.
//!
//! Border handling is the same on both sides: the blur and Sobel use
//! `BORDER_REPLICATE`, while `pyr_down` uses `BORDER_REFLECT_101` (the default
//! of `imgproc::pyr_down_def` / C++ `cv::pyrDown`).
//!
//! The [`crate::backend`] wrappers expose the filters as `Backend`
//! implementations; the helpers below are the ones both backends share.

#[cfg(feature = "fearless-simd")]
pub mod fsimd;
#[cfg(feature = "pulp")]
pub mod pulp;

// Only the pure-Rust backends need these; with the `opencv` backend alone the
// helpers would be dead code.
#[cfg(any(feature = "pulp", feature = "fearless-simd"))]
use opencv::{
    core::{self, Mat, Scalar},
    prelude::*,
};

#[cfg(any(feature = "pulp", feature = "fearless-simd"))]
pub(crate) fn bad_arg(msg: impl Into<String>) -> opencv::Error {
    opencv::Error::new(core::StsBadArg, msg)
}

#[cfg(any(feature = "pulp", feature = "fearless-simd"))]
pub(crate) fn ensure_dst(dst: &mut Mat, rows: i32, cols: i32, typ: i32) -> opencv::Result<()> {
    if dst.rows() != rows || dst.cols() != cols || dst.typ() != typ {
        *dst = Mat::new_rows_cols_with_default(rows, cols, typ, Scalar::all(0.0))?;
    }
    Ok(())
}

#[cfg(any(feature = "pulp", feature = "fearless-simd"))]
pub(crate) fn check_src_8u(src: &Mat) -> opencv::Result<(i32, i32)> {
    let rows = src.rows();
    let cols = src.cols();
    if rows <= 0 || cols <= 0 {
        return Err(bad_arg(format!("empty source image: {rows}x{cols}")));
    }
    if src.depth() != core::CV_8U || (src.channels() != 1 && src.channels() != 3) {
        return Err(bad_arg(format!(
            "expected CV_8UC1/CV_8UC3 source, got typ={} (depth={}, channels={})",
            src.typ(),
            src.depth(),
            src.channels()
        )));
    }
    Ok((rows, cols))
}

/// `BORDER_REFLECT_101` index mapping (mirror without repeating the edge pixel).
///
/// This matches the *default* border of `imgproc::pyr_down_def` / C++
/// `cv::pyrDown`, which the pyramid downsampling has always used (unlike the
/// blur/Sobel calls, which pass `BORDER_REPLICATE` explicitly).
#[cfg(any(feature = "pulp", feature = "fearless-simd"))]
#[inline]
pub(crate) fn reflect101(i: i32, n: i32) -> usize {
    debug_assert!(n > 0);
    if n == 1 {
        return 0;
    }
    let mut i = i;
    while i < 0 || i >= n {
        if i < 0 {
            i = -i;
        } else {
            i = 2 * n - 2 - i;
        }
    }
    i as usize
}
