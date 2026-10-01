//! Pure-Rust replacements for the `opencv::imgproc` calls on the matching hot path.
//!
//! Integer arithmetic only: no floating-point anywhere in this module.
//! Every function reads from a shared `&Mat` and writes into a caller-provided
//! `&mut Mat` (reallocated only on size/type mismatch), so `Mat`-allocated image
//! buffers are reused with zero copies.
//!
//! Border handling mirrors the previous `imgproc` calls exactly: the blur and
//! Sobel use `BORDER_REPLICATE`, while `pyr_down` uses `BORDER_REFLECT_101`
//! (the default of `pyr_down_def` / C++ `cv::pyrDown`). Row access goes through
//! the safe `at_row`/`at_row_mut` helpers, so non-continuous (ROI) inputs work.
//!
//! Performance: SIMD interiors via [`pulp`] (`u16`/`i16` lanes, runtime dispatch,
//! single-threaded) plus scalar border handling. Integer sums are
//! order-independent, so SIMD and scalar paths produce bit-identical results.
//!
//! Fixed-point design (verified against the OpenCV 4.13 `objdump`
//! disassembly, which uses the same widths):
//! - Gaussian 7x7: Q8 kernel in `u16` lanes, two-rounding separable. The kernel
//!   is dyadic, so Q8 taps are exact (`[8, 28, 56, 72, 56, 28, 8]`, sum 256)
//!   and Q8 two-rounding is bit-identical to Q14 two-rounding: with dyadic
//!   taps, `K14 = 64 * K8` exactly, hence `acc14 = 64 * acc8` and
//!   `(64*a + 8192) >> 14 == (a + 128) >> 8`. Max accumulator `255 * 256 =
//!   65280 < 65536`, so `u16` never overflows.
//! - `pyrDown` 5x5: raw `[1, 4, 6, 4, 1]` sums in `u16` lanes with a SINGLE
//!   rounding at the very end, like OpenCV's `pyrDown_` (its horizontal pass
//!   truncates to integer, its vertical pass adds 128 and shifts by 8 once).
//!   Max accumulator `255 * 256 = 65280 < 65536`. Max horizontal sum
//!   `255 * 16 = 4080`.
//! - Sobel 3x3: `i16` lanes (results within ±4080).

// Stencil code indexes several buffers by pixel coordinate; the iterator form
// suggested by `needless_range_loop` would obscure the border geometry.
#![allow(clippy::needless_range_loop)]

use opencv::{
    core::{self, Mat, Scalar},
    prelude::*,
};
use pulp::{Simd, WithSimd};

pub mod gaussian;
pub mod pyr_down;
pub mod sobel;

pub use gaussian::gaussian_blur_7x7;
pub use pyr_down::pyr_down;
pub use sobel::{sobel_color_i16, sobel_grayscale};

pub(crate) fn bad_arg(msg: impl Into<String>) -> opencv::Error {
    opencv::Error::new(core::StsBadArg, msg)
}

pub(crate) fn ensure_dst(dst: &mut Mat, rows: i32, cols: i32, typ: i32) -> opencv::Result<()> {
    if dst.rows() != rows || dst.cols() != cols || dst.typ() != typ {
        *dst = Mat::new_rows_cols_with_default(rows, cols, typ, Scalar::all(0.0))?;
    }
    Ok(())
}

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

// ---------------------------------------------------------------------------
// SIMD plumbing (pulp, u16/i16 lanes, single-threaded)
// ---------------------------------------------------------------------------

/// N-tap MAC over N `u16` windows into raw `u16` accumulators (caller narrows).
///
/// All windows must share one length. Returns the number of output elements
/// covered by SIMD (a prefix); the caller finishes the tail with scalar code.
///
/// Valid when every partial sum fits in `u16` — the case for the 5x5 `pyrDown`
/// kernel (`[1, 4, 6, 4, 1]`, total weight 16/256: max accumulator
/// `255 * 16 = 4080`).
pub(crate) struct Mac16<'a, const N: usize> {
    pub(crate) wins: [&'a [u16]; N],
    pub(crate) k: [u16; N],
    pub(crate) out: &'a mut [u16],
}

impl<const N: usize> WithSimd for Mac16<'_, N> {
    type Output = usize;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> usize {
        debug_assert!(N > 0);
        let ks = self.k.map(|v| simd.splat_u16s(v));
        // All windows share one length, so chunking agrees across them.
        let heads = self.wins.map(|w| S::as_simd_u16s(w).0);
        let (oh, _) = S::as_mut_simd_u16s(self.out);
        debug_assert_eq!(oh.len(), heads[0].len());
        for (i, o) in oh.iter_mut().enumerate() {
            let mut acc = simd.mul_u16s(heads[0][i], ks[0]);
            for j in 1..N {
                acc = simd.add_u16s(acc, simd.mul_u16s(heads[j][i], ks[j]));
            }
            *o = acc;
        }
        oh.len() * S::U16_LANES
    }
}

/// Widen a `u8` row to `i16` (exact; auto-vectorized).
#[inline]
pub(crate) fn widen_row_i16(dst: &mut [i16], src: &[u8]) {
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        *d = v as i16;
    }
}

/// Widen a `u8` row to `u16` (exact; auto-vectorized).
#[inline]
pub(crate) fn widen_row_u16(dst: &mut [u16], src: &[u8]) {
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        *d = v as u16;
    }
}

/// Narrow raw `u16` accumulators with round-and-shift.
#[inline]
pub(crate) fn narrow_row_u16(dst: &mut [u8], acc: &[u16], round: u16, shift: u32) {
    for (d, &a) in dst.iter_mut().zip(acc.iter()) {
        *d = ((a + round) >> shift) as u8;
    }
}

/// Deinterleave one interleaved `Vec3b` row into planar `i16` ring `slot`.
#[inline]
pub(crate) fn deinterleave_row_into_i16(
    ring: &mut [i16],
    slot: usize,
    w: usize,
    row: &[core::Vec3b],
) {
    let base = slot * 3 * w;
    for x in 0..w {
        ring[base + x] = row[x][0] as i16;
        ring[base + w + x] = row[x][1] as i16;
        ring[base + 2 * w + x] = row[x][2] as i16;
    }
}

/// Deinterleave a `Vec3b` row into three widened `u16` planes.
#[inline]
pub(crate) fn deinterleave_widen_row_u16(
    p0: &mut [u16],
    p1: &mut [u16],
    p2: &mut [u16],
    row: &[core::Vec3b],
) {
    for (x, px) in row.iter().enumerate() {
        p0[x] = px[0] as u16;
        p1[x] = px[1] as u16;
        p2[x] = px[2] as u16;
    }
}

/// Deinterleave one flat `RGBRGB..` tmp row into planar `u16` ring `slot`.
#[inline]
pub(crate) fn deinterleave_row_into_u16(ring: &mut [u16], slot: usize, w: usize, row: &[u8]) {
    let base = slot * 3 * w;
    for x in 0..w {
        ring[base + x] = row[x * 3] as u16;
        ring[base + w + x] = row[x * 3 + 1] as u16;
        ring[base + 2 * w + x] = row[x * 3 + 2] as u16;
    }
}
