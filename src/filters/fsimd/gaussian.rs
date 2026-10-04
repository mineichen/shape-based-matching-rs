//! 7x7 Gaussian blur, `BORDER_REPLICATE`, bit-identical to `imgproc::gaussian_blur`.
//!
//! # Fixed-point schedule
//!
//! OpenCV's `gaussianBlur` on `CV_8U` goes through `GaussianBlurFixedPoint`
//! (`smooth.dispatch.cpp`), which does **not** round after the horizontal pass:
//!
//! ```text
//! row  h[y][x] = sum_j K[j] * src[y][x + j - 3]        // integer, = value * 256
//! out[y][x]    = (sum_j K[j] * h[y + j - 3][x] + 32768) >> 16
//! ```
//!
//! The horizontal pass keeps 8 fractional bits and the vertical pass adds the
//! second 8, so there is exactly **one** rounding for the whole 2D filter and
//! the result is the correctly rounded 2D convolution of the *float* kernel.
//! Rounding the intermediate to `u8` instead double-rounds and is off by one on
//! about 1% of all pixels; `tests/filters_opencv_parity.rs` compares byte for
//! byte, so that is not optional.
//!
//! # Why `f32` in the vertical pass
//!
//! The vertical accumulator needs 24 bits (`255 * 256 * 256 = 16711680`), so it
//! cannot live in `u16` lanes, and it does not fit in two `u16` accumulators
//! either: `255 * 256 = 65280` already fills a `u16` for a *single* tap times its
//! weight, so the low half would cost a second 7-tap pass over the same data.
//!
//! In `u32` lanes the MAC is four `vpmulld`s per vector, and that is what this
//! used to be — measured at 0.25 ns/byte against the horizontal pass' 0.07. In
//! `f32` lanes it is four fused multiply-adds, which are twice as fast here, and
//! the arithmetic stays **exact**: every value is a dyadic rational with at most
//! 24 significant bits and magnitude below 256 (the Q8 intermediate is an
//! integer `<= 65280`, the weights are `K[j] / 65536`, the rounding bias is
//! `0.5`), so no product, sum or FMA ever rounds. `trunc` of the biased sum is
//! therefore exactly `(sum + 32768) >> 16`, on every backend and with or
//! without a fused multiply-add.

use fearless_simd::prelude::*;
use fearless_simd::{Level, dispatch};
use fearless_simd_macros::simd;
use opencv::{core::Mat, prelude::*};

use super::{Out, Rows, Scratch, grow, row, widen_u8_to_u16, windows, write_back};
use crate::filters::{check_src_8u, ensure_dst};

/// 7x7 Gaussian kernel for `sigma = 0` (auto) in Q8.
///
/// `getGaussianKernel(7, 0)` returns OpenCV's hard-coded small-Gaussian table
/// entry `[0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]`
/// (`small_gaussian_tab[3]`), which the fixed-point path scales by `1 << 8`.
/// The taps are dyadic, so those Q8 integers are the exact kernel.
const GAUSS_K: [u16; 7] = [8, 28, 56, 72, 56, 28, 8];
const GAUSS_R: usize = 3;

/// `GAUSS_K / 65536`, the weights the vertical pass multiplies with.
///
/// Each tap is a dyadic rational — `8 = 2^13`, `28 = 7 * 2^14`, `56 = 7 * 2^13`,
/// `72 = 9 * 2^13` — so the division by `2^16` is exact in `f32` and every partial
/// sum stays a dyadic rational below 256 with at most 24 significant bits. No
/// product, sum or fused multiply-add ever rounds.
const GAUSS_F: [f32; 7] = [
    8.0 / 65536.0,
    28.0 / 65536.0,
    56.0 / 65536.0,
    72.0 / 65536.0,
    56.0 / 65536.0,
    28.0 / 65536.0,
    8.0 / 65536.0,
];

// ---------------------------------------------------------------------------
// 7x7 Gaussian blur, `BORDER_REPLICATE`
// ---------------------------------------------------------------------------

/// 7x7 Gaussian blur (`sigma = 0` auto) on `CV_8UC1`/`CV_8UC3`.
///
/// The horizontal pass and the vertical pass are fused through a seven-row ring
/// of the Q8 intermediate, so the intermediate never leaves the ring: the
/// source is read once and the result written once, instead of round-tripping a
/// whole-image `u8` buffer through memory the way a row-then-column split would.
pub fn gaussian_blur_7x7(s: &mut Scratch, src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
    let (rows_i, cols_i) = check_src_8u(src)?;
    ensure_dst(dst, rows_i, cols_i, src.typ())?;
    let level = Level::new();
    let ch = src.channels() as usize;
    let rows = rows_i as usize;
    let rb = cols_i as usize * ch;
    let src = Rows::new(src, rows * rb, ch)?;
    let staged = {
        let mut out = Out::new(dst, rows * rb)?;
        let Scratch {
            ring_f32, acc_u16, ..
        } = s;
        // Output row `y` reads ring rows `y - 3 ..= y + 3`, so the ring holds the
        // horizontal pass of `radius + 1` new rows per output row.
        let nring = 2 * GAUSS_R + 1;
        let ring = grow(ring_f32, nring * rb);
        let wide = grow(acc_u16, rb);
        let lo = (GAUSS_R * ch).min(rb);
        let hi = rb.saturating_sub(GAUSS_R * ch);
        let dst_all = out.bytes();
        let mut loaded = 0usize;
        for y in 0..rows {
            let hi_row = (y + GAUSS_R).min(rows - 1);
            while loaded <= hi_row {
                let r = loaded;
                let base = r % nring * rb;
                let line = row(src.data(), rb, r);
                // The horizontal pass is the only place the source is widened,
                // and it widens each row exactly once: the MAC then reads the
                // `u16` row seven times instead of widening every tap.
                dispatch!(level, simd => widen_u8_to_u16(simd, line, wide));
                if hi > lo {
                    let w = windows::<7, _>(wide, lo, ch, GAUSS_R, hi - lo);
                    let t = &mut ring[base + lo..base + hi];
                    dispatch!(level, simd => mac_u16_to_f32(simd, w, &GAUSS_K, t));
                }
                // The `radius`-wide column frame: clamped taps, one pixel at a
                // time. Everything between `lo` and `hi` is the SIMD interior.
                let cols = rb / ch;
                for x in 0..lo {
                    ring[base + x] = gauss_h_tap(line, x, cols, ch);
                }
                for x in hi.max(lo)..rb {
                    ring[base + x] = gauss_h_tap(line, x, cols, ch);
                }
                loaded += 1;
            }

            // `BORDER_REPLICATE` up/down is just a duplicated ring row, because
            // the horizontal pass is linear — so the column frame is the only
            // part of this pass that needs the scalar fallback.
            let d = &mut dst_all[y * rb..(y + 1) * rb];
            // The seven taps of the column frame, resolved once per output row
            // instead of once per frame pixel: with `3 * channels` frame lanes on
            // each side that is the difference between a handful of address
            // computations and `2 * 3 * channels * 7` of them.
            let frame: [&[f32]; 7] = std::array::from_fn(|j| {
                let r = (y as i32 + j as i32 - GAUSS_R as i32).clamp(0, rows_i - 1) as usize;
                &ring[r % nring * rb..(r % nring + 1) * rb]
            });
            if hi > lo {
                let w: [&[f32]; 7] = std::array::from_fn(|j| &frame[j][lo..hi]);
                dispatch!(level, simd => gauss_v7(simd, w, &mut d[lo..hi]));
            }
            for x in 0..lo {
                d[x] = gauss_v_tap(&frame, x);
            }
            for x in hi.max(lo)..rb {
                d[x] = gauss_v_tap(&frame, x);
            }
        }
        out.take_staged()
    };
    match staged {
        Some(v) => write_back(dst, v, rows, rb, ch),
        None => Ok(()),
    }
}

/// Scalar horizontal 7-tap with `BORDER_REPLICATE`, unrounded (Q8 exact).
#[inline]
fn gauss_h_tap(s: &[u8], x: usize, cols: usize, ch: usize) -> f32 {
    let px = (x / ch) as i32;
    let c = x % ch;
    let cols = cols as i32;
    let mut acc = 0u32;
    for (j, &k) in GAUSS_K.iter().enumerate() {
        let xi = (px + j as i32 - GAUSS_R as i32).clamp(0, cols - 1) as usize * ch + c;
        acc += k as u32 * s[xi] as u32;
    }
    acc as f32
}

/// Scalar vertical 7-tap over the already-clamped ring rows, with the single
/// rounding of the whole filter.
#[inline]
fn gauss_v_tap(rows: &[&[f32]; 7], x: usize) -> u8 {
    let mut acc = 0.0f32;
    for (j, &k) in GAUSS_F.iter().enumerate() {
        acc += k * rows[j][x];
    }
    (acc + 0.5) as u8
}

/// N-tap `u16` MAC with no rounding, widened into the `f32` ring.
///
/// The `u16` sum is the exact Q8 intermediate (at most `255 * sum(k) = 65280`, so
/// it always fits) and widening it to `f32` is lossless.
#[simd]
fn mac_u16_to_f32<S: Simd>(simd: S, rows: [&[u16]; 7], k: &[u16; 7], out: &mut [f32]) -> usize {
    assert_eq!(rows[0].len(), out.len());
    for r in &rows[1..] {
        assert_eq!(r.len(), out.len());
    }
    let n = <S as Simd>::u16s::LEN;
    let m = <S as Simd>::f32s::LEN;
    debug_assert_eq!(2 * m, n);
    let len = out.len();
    let body = len / n * n;
    for i in (0..body).step_by(n) {
        // Constant-indexed taps and coefficients, so the seven multiplies stay
        // `vpmullw` with the weight broadcasts hoisted out of the loop.
        let tap = |j: usize| {
            <S as Simd>::u16s::from_slice(simd, &rows[j][i..i + n])
                * <S as Simd>::u16s::splat(simd, k[j])
        };
        let acc = tap(0) + tap(1) + tap(2) + tap(3) + tap(4) + tap(5) + tap(6);
        let (lo, hi) = acc.widen();
        let (dlo, dhi) = out[i..i + n].split_at_mut(m);
        <S as Simd>::f32s::float_from(lo).store_slice(dlo);
        <S as Simd>::f32s::float_from(hi).store_slice(dhi);
    }
    let mut i = body;
    while i < len {
        let mut acc = 0u16;
        for j in 0..7 {
            acc = acc.wrapping_add(rows[j][i] * k[j]);
        }
        out[i] = acc as f32;
        i += 1;
    }
    len
}

/// Vertical pass of the 7x7 Gaussian: seven Q8 rows in, one `u8` row out.
///
/// `out[x] = (sum_j K[j] * rows[j][x] + 32768) >> 16`, computed as
/// `trunc(sum_j (K[j] / 65536) * rows[j][x] + 0.5)` — see the module docs for
/// why that is exact.
///
/// The taps are symmetric, so the MAC folds the pairs `(K0, K6)`, `(K1, K5)`
/// and `(K2, K4)` into one add each and the four weights ride along in four
/// fused multiply-adds.
#[simd]
fn gauss_v7<S: Simd>(simd: S, rows: [&[f32]; 7], out: &mut [u8]) {
    assert_eq!(rows[0].len(), out.len());
    for r in &rows[1..] {
        assert_eq!(r.len(), out.len());
    }
    let n = <S as Simd>::f32s::LEN;
    let m = 2 * n;
    let len = out.len();
    // Two groups of `m` output bytes, so one narrowing chain per iteration
    // covers `2 * m` bytes — a full `u8` store.
    let step = 2 * m;
    let body = len / step * step;
    let ks = [
        <S as Simd>::f32s::splat(simd, GAUSS_F[0]),
        <S as Simd>::f32s::splat(simd, GAUSS_F[1]),
        <S as Simd>::f32s::splat(simd, GAUSS_F[2]),
        <S as Simd>::f32s::splat(simd, GAUSS_F[3]),
    ];
    let bias = <S as Simd>::f32s::splat(simd, 0.5);

    for i in (0..body).step_by(step) {
        let (a0, a1) = gauss_v_group(simd, rows, i, ks, bias);
        let (b0, b1) = gauss_v_group(simd, rows, i + m, ks, bias);
        let lo = <S as Simd>::u32s::truncate_from(a0).narrow(<S as Simd>::u32s::truncate_from(a1));
        let hi = <S as Simd>::u32s::truncate_from(b0).narrow(<S as Simd>::u32s::truncate_from(b1));
        lo.narrow(hi).store_slice(&mut out[i..i + step]);
    }

    let mut i = body;
    while i < len {
        let mut acc = 0.0f32;
        for j in 0..7 {
            acc += GAUSS_F[j] * rows[j][i];
        }
        out[i] = (acc + 0.5) as u8;
        i += 1;
    }
}

/// One group of [`gauss_v7`]: taps `rows[*][i..i + 2n]` MAC'd into the low and the
/// high vector of the group, which together cover `2n` output lanes.
#[inline(always)]
fn gauss_v_group<S: Simd>(
    simd: S,
    rows: [&[f32]; 7],
    i: usize,
    ks: [<S as Simd>::f32s; 4],
    bias: <S as Simd>::f32s,
) -> (<S as Simd>::f32s, <S as Simd>::f32s) {
    let n = <S as Simd>::f32s::LEN;
    let tap = |j: usize| {
        (
            <S as Simd>::f32s::from_slice(simd, &rows[j][i..i + n]),
            <S as Simd>::f32s::from_slice(simd, &rows[j][i + n..i + 2 * n]),
        )
    };
    let t0 = tap(0);
    let t1 = tap(1);
    let t2 = tap(2);
    let t3 = tap(3);
    let t4 = tap(4);
    let t5 = tap(5);
    let t6 = tap(6);
    // The pairs `(K0, K6)`, `(K1, K5)`, `(K2, K4)` fold into one add each, and the
    // bias is added *after* the MAC: inside the innermost `mul_add` it would be
    // scaled by the outer weights and stop being a rounding bias.
    let lo = (t0.0 + t6.0).mul_add(
        ks[0],
        (t1.0 + t5.0).mul_add(ks[1], (t2.0 + t4.0).mul_add(ks[2], t3.0 * ks[3])),
    ) + bias;
    let hi = (t0.1 + t6.1).mul_add(
        ks[0],
        (t1.1 + t5.1).mul_add(ks[1], (t2.1 + t4.1).mul_add(ks[2], t3.1 * ks[3])),
    ) + bias;
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use fearless_simd::{Level, dispatch};

    use super::{GAUSS_F, GAUSS_K, gauss_v7, mac_u16_to_f32};

    /// Every length the callers can produce, including the tails: the SIMD
    /// horizontal MAC must equal the scalar 7-tap definition lane for lane, and
    /// the SIMD vertical pass must equal the scalar one.
    #[test]
    fn kernels_match_scalar() {
        let level = Level::new();
        for len in [1usize, 7, 8, 13, 16, 17, 31, 32, 33, 48, 59, 64, 100, 333] {
            // The horizontal MAC is fed by `widen_u8_to_u16`, so the lanes are
            // `u8` source values: with the largest weight at 72 the taps peak at
            // `255 * 72 = 18360` and the Q8 sum at `255 * 256 = 65280`, both
            // inside `u16`. The reference below uses checked arithmetic on
            // purpose — it panics if that contract is ever violated.
            let u16row: Vec<u16> = (0..len * 7).map(|i| ((i * 7 + 3) % 256) as u16).collect();
            let rows: [&[u16]; 7] = std::array::from_fn(|j| &u16row[j * len..(j + 1) * len]);
            let mut got = vec![0.0f32; len];
            dispatch!(level, simd => mac_u16_to_f32(simd, rows, &GAUSS_K, &mut got));
            let want: Vec<f32> = (0..len)
                .map(|i| {
                    let mut acc = 0u16;
                    for j in 0..7 {
                        acc += rows[j][i] * GAUSS_K[j];
                    }
                    acc as f32
                })
                .collect();
            assert_eq!(got, want, "horizontal, len={len}");

            // Vertical: seven rows of exact Q8 values, so the scalar and the SIMD
            // pass must agree on `(sum + 32768) >> 16`.
            let ring: Vec<f32> = (0..len * 7).map(|i| ((i * 11 + 7) % 65281) as f32).collect();
            let vr: [&[f32]; 7] = std::array::from_fn(|j| &ring[j * len..(j + 1) * len]);
            let mut vout = vec![0u8; len];
            dispatch!(level, simd => gauss_v7(simd, vr, &mut vout));
            for i in 0..len {
                let mut acc = 0.0f32;
                for j in 0..7 {
                    acc += GAUSS_F[j] * vr[j][i];
                }
                assert_eq!(vout[i], (acc + 0.5) as u8, "vertical, len={len} i={i}");
            }
        }
    }
}