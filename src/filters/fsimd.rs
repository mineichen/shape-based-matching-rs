#![allow(clippy::needless_range_loop)]

//! `fearless_simd` implementation of the matching-hot-path filters: portable
//! SIMD with runtime dispatch, so one code path covers AVX2/AVX-512, SSE, NEON
//! and WASM `simd128`.
//!
//! Every filter is **bit-identical** to the `opencv::imgproc` call it replaces,
//! down to the rounding: `tests/filters_opencv_parity.rs` compares the output
//! bytes of every filter against `imgproc` for nine sizes and both channel
//! counts. That constraint is what decides the vertical pass of the Gaussian
//! blur, which is the hard one: OpenCV keeps 8 fractional bits after the
//! horizontal pass and rounds only once, at the end, so the vertical accumulator
//! needs 24 bits.
//!
//! What makes it fast:
//!
//! 1. **No deinterleaving.** All channels share the same tap offsets, just
//!    scaled by the channel count, so color images are filtered in place: a tap
//!    at pixel `x + j` is a byte offset of `j * channels` horizontally and
//!    `j * channels * cols` vertically. Grayscale is the special case
//!    `channels == 1`, so the color paths cost about the same per pixel as the
//!    grayscale ones instead of `channels * channels`.
//! 2. **Only the frame is scalar.** The `radius`-wide column frame uses the
//!    clamped-tap fallback; every other column goes through the same SIMD kernel
//!    as the interior. The Gaussian's row borders need no fallback at all: a
//!    replicated row is a duplicated ring row, so the vertical SIMD kernel
//!    handles them like any other row.
//! 3. **Scratch is reused across calls.** See [`Scratch`]; allocating and
//!    zeroing the whole-image intermediates per call costs more than a whole
//!    filter pass.
//! 4. **No discarded work.** `pyrDown` decimates inside the horizontal pass, so
//!    the vertical pass and the ring only ever hold columns the output keeps.
//! 5. **Fused row ring.** The Gaussian keeps only the seven horizontal rows the
//!    vertical pass is about to read, so its intermediate never round-trips
//!    through memory.
//!
//! Floating point: the Gaussian's vertical pass accumulates in `f32`, which is
//! exact here because every value is a dyadic rational with at most 24
//! significant bits (see the `gaussian` module docs), and the grayscale Sobel
//! widens `i16` to `f32`, which is exact for every `i16`. Everything else is
//! integer.
//!
//! # SIMD strategies tried for the Gaussian vertical pass
//!
//! The accumulator needs 24 bits, so `u16` lanes cannot hold it. What follows is
//! what was measured on this box (`ns` per output byte, 1920-wide rows, min of
//! five runs; `opencv_st` is 0.52 ns/byte):
//!
//! 1. **Round the horizontal pass to `u8`, accumulate in `u16` (the old code).**
//!    Cheapest possible vertical pass (0.07 ns/byte) but it rounds twice, which
//!    is off by one on ~1% of all pixels.
//! 2. **`u16` intermediate + `u32` accumulator with `vpmulld`.** Exact, but four
//!    `vpmulld` per vector are throughput-bound: 0.25 ns/byte, 3.5x the whole
//!    horizontal pass.
//! 3. **Two `u16` accumulators (high and low byte of the intermediate).** Exact,
//!    but `255 * 256 = 65280` already fills a `u16` for a *single* tap times its
//!    weight, so the low half costs a second 7-tap pass over the same data.
//! 4. **Shifts instead of multiplies** (`8 = 1<<3`, `28 = 1<<5 - 1<<2`, ...).
//!    Same op count as the multiplies and every shift is port 0, so no gain.
//! 5. **`f32` accumulator with `f32` ring, weights pre-scaled by `1/65536`, bias
//!    `0.5`, `trunc` at the end.** Four FMAs per vector instead of four
//!    multiply-add pairs, exact because all values are dyadic with <= 24
//!    significant bits: **0.10 ns/byte**, 2.4x faster than (2) — this is what
//!    ships.
//! 6. **Read the `u8` source directly in the horizontal MAC** instead of
//!    materializing a widened row: halves the loads (16 -> 7 bytes per output
//!    byte) but needs 14 live `u16` vectors and spills, measured slower.
//! 7. **Two or four output rows per vertical iteration** to share ring loads:
//!    the narrowing chain produces one contiguous store, so several rows need a
//!    staging copy, and the saved load traffic did not pay for it.
//! 8. **`f32` ring from a `u16` ring converted per tap.** Avoids doubling the
//!    ring traffic, but the per-tap `u16 -> u32 -> f32` conversions cost more
//!    than the traffic they save.

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
/// The Gaussian keeps a seven-row ring of its `f32` intermediate and one
/// widened source row; `pyrDown` keeps a five-row `u16` ring; the Sobels keep
/// three widened rows and two accumulator rows. Allocating and zeroing those per
/// call costs more than a whole filter pass on a 640x480 image, so the buffers
/// only ever grow.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    tmp_u16: Vec<u16>,
    ring_f32: Vec<f32>,
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
pub(crate) fn windows<const N: usize, T>(
    row: &[T],
    start: usize,
    tap_stride: usize,
    radius: usize,
    len: usize,
) -> [&[T]; N] {
    std::array::from_fn(|j| {
        let off = start + j * tap_stride - radius * tap_stride;
        &row[off..off + len]
    })
}
