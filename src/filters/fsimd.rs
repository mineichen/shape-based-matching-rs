#![allow(clippy::needless_range_loop)]

//! `fearless_simd` implementation of the matching-hot-path filters.
//!
//! Bit-identical to the `pulp` backend in `filters::pulp` (same integer
//! kernels, same rounding), but written against portable SIMD with runtime
//! dispatch, so one code path covers AVX2/AVX-512, SSE, NEON, and WASM
//! `simd128`.
//!
//! Integer arithmetic only; the sole use of floating point is the exact
//! `i16 -> f32` widening of the grayscale Sobel result (every `i16` is
//! representable in `f32`).
//!
//! Four structural differences to the `pulp` backend make it considerably
//! faster:
//!
//! 1. **No deinterleaving.** All channels share the same tap offsets, just
//!    scaled by the channel count, so color images are filtered in place: a tap
//!    at pixel `x + j` is a byte offset of `j * channels` horizontally and
//!    `j * channels * cols` vertically. Grayscale is the special case
//!    `channels == 1`, so the color paths cost about the same per pixel as the
//!    grayscale ones instead of `channels * channels`.
//! 2. **Only the frame is scalar.** The `radius`-wide column frame and the
//!    `radius`-tall row frame use the clamped-tap fallback; every other row and
//!    column goes through the same SIMD kernel as the interior.
//! 3. **Scratch is reused across calls.** See [`Scratch`]; allocating and
//!    zeroing the whole-image intermediates per call costs more than a whole
//!    filter pass.
//! 4. **No discarded work.** `pyrDown` decimates inside the horizontal pass, so
//!    the vertical pass and the ring only ever hold columns the output keeps.

mod gaussian;
mod pyr_down;
mod sobel;

pub use gaussian::gaussian_blur_7x7;
pub use pyr_down::pyr_down;
pub use sobel::{sobel_color_i16, sobel_grayscale};

use fearless_simd::prelude::*;
use fearless_simd_macros::simd;
use opencv::{
    core::{self, Mat},
    prelude::*,
};

/// Reusable scratch buffers for the `fearless_simd` backend.
///
/// The intermediates are whole-image (`u8`/`u16`) plus a few row-sized ones, so
/// allocating and zeroing them per call costs more than the Gaussian's vertical
/// pass on a 640x480 image. The buffers only ever grow.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    tmp_u8: Vec<u8>,
    tmp_u16: Vec<u16>,
    ring_u16: Vec<u16>,
    ring_i16: Vec<i16>,
    acc_u16: Vec<u16>,
    acc_i16: Vec<i16>,
}

/// Grow `v` to at least `n` and hand out exactly `n` elements. Zeroing only
/// ever happens on growth, never per call.
#[inline]
fn grow<T: Clone + Default>(v: &mut Vec<T>, n: usize) -> &mut [T] {
    if v.len() < n {
        v.resize(n, T::default());
    }
    &mut v[..n]
}

/// Round-and-shift at the output store: divisor `16 * 16 = 256`.
const ROUND: u16 = 1 << 7;
const SHIFT: u32 = 8;

/// Flat byte view of a `Mat`: borrows the `Mat`'s memory when it is continuous,
/// materializes one row-major copy for ROIs.
enum Rows<'a> {
    Direct(&'a [u8]),
    Owned(Vec<u8>),
}

impl<'a> Rows<'a> {
    fn new(mat: &'a Mat, bytes: usize, channels: usize) -> opencv::Result<Self> {
        if mat.is_continuous() {
            let data = mat.data_bytes()?;
            debug_assert_eq!(data.len(), bytes);
            Ok(Self::Direct(data))
        } else {
            let cols = mat.cols() as usize;
            let mut owned = vec![0u8; bytes];
            for (y, row) in owned.chunks_exact_mut(cols * channels).enumerate() {
                match channels {
                    1 => row.copy_from_slice(mat.at_row::<u8>(y as i32)?),
                    _ => {
                        let src = mat.at_row::<core::Vec3b>(y as i32)?;
                        for (o, px) in row.as_chunks_mut::<3>().0.iter_mut().zip(src) {
                            o.copy_from_slice(&px.0);
                        }
                    }
                }
            }
            Ok(Self::Owned(owned))
        }
    }

    #[inline]
    fn data(&self) -> &[u8] {
        match self {
            Self::Direct(d) => d,
            Self::Owned(v) => v,
        }
    }
}

/// Mutable flat byte view of a destination `Mat`, staged for ROIs.
enum Out<'a> {
    Direct(&'a mut [u8]),
    Staged(Vec<u8>),
}

impl<'a> Out<'a> {
    fn new(mat: &'a mut Mat, bytes: usize) -> opencv::Result<Self> {
        if mat.is_continuous() {
            Ok(Self::Direct(mat.data_bytes_mut()?))
        } else {
            Ok(Self::Staged(vec![0u8; bytes]))
        }
    }

    #[inline]
    fn bytes(&mut self) -> &mut [u8] {
        match self {
            Self::Direct(b) => b,
            Self::Staged(v) => v,
        }
    }

    /// Consume the view, releasing the borrow of the `Mat` and returning the
    /// staging buffer if the destination was an ROI.
    fn take_staged(self) -> Option<Vec<u8>> {
        match self {
            Self::Staged(v) => Some(v),
            Self::Direct(_) => None,
        }
    }
}

/// Copy a staged buffer into `mat`, one `copy_from_slice` per row.
fn write_back(
    mat: &mut Mat,
    v: Vec<u8>,
    rows: usize,
    row_bytes: usize,
    channels: usize,
) -> opencv::Result<()> {
    for (y, row) in v.chunks_exact(row_bytes).take(rows).enumerate() {
        match channels {
            1 => mat.at_row_mut::<u8>(y as i32)?.copy_from_slice(row),
            _ => {
                let dst = mat.at_row_mut::<core::Vec3b>(y as i32)?;
                for (px, o) in dst.iter_mut().zip(row.as_chunks::<3>().0) {
                    px.0.copy_from_slice(o);
                }
            }
        }
    }
    Ok(())
}

/// Row `y` of a flat, `rb`-byte-per-row buffer.
#[inline(always)]
fn row(data: &[u8], rb: usize, y: usize) -> &[u8] {
    &data[y * rb..(y + 1) * rb]
}

/// Widen `u8` -> `u16` over the whole slice.
///
/// Every MAC kernel takes `u16`/`i16` inputs and the callers widen each source
/// row exactly once into a scratch buffer. Widening straight from `u8` inside
/// the MAC would need one `vpmovzxbw`-class shuffle *per tap* (7 for the 7-tap
/// Gaussian); all of those issue on the single shuffle port, which then dominates
/// the runtime.
pub(crate) fn widen_u8_to_u16<S: Simd>(simd: S, src: &[u8], dst: &mut [u16]) {
    let n = <S as Simd>::u16s::LEN;
    let b = <S as Simd>::u8s::LEN;
    debug_assert_eq!(b, 2 * n);
    assert_eq!(src.len(), dst.len());
    let body = src.len() / b * b;
    let mut d = dst[..body].chunks_exact_mut(b);
    for s in src[..body].chunks_exact(b) {
        let v = <S as Simd>::u8s::from_slice(simd, s);
        let (lo, hi) = v.widen();
        let (dl, dh) = d.next().expect("chunks_exact_mut").split_at_mut(n);
        lo.store_slice(dl);
        hi.store_slice(dh);
    }
    for (d, &s) in dst[body..].iter_mut().zip(&src[body..]) {
        *d = s as u16;
    }
}

/// Widen `u8` -> `i16` over the whole slice.
#[simd]
pub(crate) fn widen_u8_to_i16<S: Simd>(simd: S, src: &[u8], dst: &mut [i16]) {
    let n = <S as Simd>::i16s::LEN;
    let b = <S as Simd>::i8s::LEN;
    debug_assert_eq!(b, 2 * n);
    assert_eq!(src.len(), dst.len());
    let body = src.len() / b * b;
    let mut d = dst[..body].chunks_exact_mut(b);
    for s in src[..body].chunks_exact(b) {
        let v = <S as Simd>::u8s::from_slice(simd, s);
        let (lo, hi) = v.widen();
        // `u16` and `i16` vectors are bit-identical, so widen once and reuse.
        let lo16: <S as Simd>::i16s = Bytes::from_bytes(lo.to_bytes());
        let hi16: <S as Simd>::i16s = Bytes::from_bytes(hi.to_bytes());
        let (dl, dh) = d.next().expect("chunks_exact_mut").split_at_mut(n);
        lo16.store_slice(dl);
        hi16.store_slice(dh);
    }
    for (d, &s) in dst[body..].iter_mut().zip(&src[body..]) {
        *d = s as i16;
    }
}

/// Read the tap windows of an `N`-tap filter out of one widened row.
///
/// `row` is the whole widened row. Window `j` starts at `start` plus the tap
/// offset `(j - radius) * tap_stride`, so `start` must already account for the
/// left border (`radius * tap_stride`). All windows have length `len`.
#[inline(always)]
pub(crate) fn windows<const N: usize>(
    row: &[u16],
    start: usize,
    tap_stride: usize,
    radius: usize,
    len: usize,
) -> [&[u16]; N] {
    std::array::from_fn(|j| {
        let off = start + j * tap_stride - radius * tap_stride;
        &row[off..off + len]
    })
}
