//! 7x7 Gaussian blur, `BORDER_REPLICATE`.

use fearless_simd::prelude::*;
use fearless_simd::{Level, dispatch};
use fearless_simd_macros::simd;
use opencv::{core::Mat, prelude::*};

use super::{Out, ROUND, Rows, SHIFT, Scratch, grow, row, widen_u8_to_u16, windows, write_back};
use crate::filters::{check_src_8u, ensure_dst};

/// 7x7 Gaussian kernel for `sigma = 0` (auto) in Q8.
///
/// `[0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]`, exactly
/// representable with 8 fractional bits and summing to `1 << 8`. The taps are
/// dyadic, so the Q8 two-rounding is bit-identical to Q14 two-rounding.
const GAUSS_K: [u16; 7] = [8, 28, 56, 72, 56, 28, 8];
const GAUSS_R: usize = 3;

// ---------------------------------------------------------------------------
// 7x7 Gaussian blur, `BORDER_REPLICATE`
// ---------------------------------------------------------------------------

/// 7x7 Gaussian blur (`sigma = 0` auto) on `CV_8UC1`/`CV_8UC3`.
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

        // Separable: horizontal into a `u8` tmp with one rounding, then
        // vertical.
        let Scratch {
            tmp_u8,
            ring_u16,
            acc_u16,
            ..
        } = s;
        let tmp = grow(tmp_u8, rows * rb);
        let wide = grow(acc_u16, rb);
        let nring = 2 * GAUSS_R + 1;
        let ring = grow(ring_u16, nring * rb);
        let lo = (GAUSS_R * ch).min(rb);
        let hi = rb.saturating_sub(GAUSS_R * ch);
        for y in 0..rows {
            let row = row(src.data(), rb, y);
            let t = &mut tmp[y * rb..(y + 1) * rb];
            dispatch!(level, simd => widen_u8_to_u16(simd, row, wide));
            if hi > lo {
                let w = windows::<7>(wide, lo, ch, GAUSS_R, hi - lo);
                dispatch!(level, simd => {
                    mac_round_pack_u8::<7, _>(simd, w, &GAUSS_K, ROUND, SHIFT, &mut t[lo..hi]);
                });
            }
            for x in 0..lo {
                t[x] = gauss_h_tap(row, x, rb, ch);
            }
            for x in hi.max(lo)..rb {
                t[x] = gauss_h_tap(row, x, rb, ch);
            }
        }

        // Vertical pass over `tmp` with a sliding window of widened rows. The
        // seven source rows stay in L1, so the MAC reads them from cache.
        let mut loaded = 0usize;
        let dst_all = out.bytes();
        for y in 0..rows {
            let hi_row = (y + GAUSS_R).min(rows - 1);
            while loaded <= hi_row {
                let r = loaded;
                let slice = &mut ring[r % nring * rb..(r % nring + 1) * rb];
                let row = row(tmp, rb, r);
                dispatch!(level, simd => widen_u8_to_u16(simd, row, slice));
                loaded += 1;
            }
            let d = &mut dst_all[y * rb..(y + 1) * rb];
            if y < GAUSS_R || y + GAUSS_R >= rows || hi <= lo {
                for (x, o) in d.iter_mut().enumerate() {
                    *o = gauss_v_tap(tmp, rb, y, x, rows);
                }
                continue;
            }
            let w: [&[u16]; 7] = std::array::from_fn(|j| {
                let r = y + j - GAUSS_R;
                &ring[r % nring * rb + lo..r % nring * rb + hi]
            });
            dispatch!(level, simd => {
                mac_round_pack_u8::<7, _>(simd, w, &GAUSS_K, ROUND, SHIFT, &mut d[lo..hi]);
            });
            for x in 0..lo {
                d[x] = gauss_v_tap(tmp, rb, y, x, rows);
            }
            for x in hi..rb {
                d[x] = gauss_v_tap(tmp, rb, y, x, rows);
            }
        }
        out.take_staged()
    };
    match staged {
        Some(v) => write_back(dst, v, rows, rb, ch),
        None => Ok(()),
    }
}

/// Scalar horizontal 7-tap with `BORDER_REPLICATE`, any byte position.
#[inline]
fn gauss_h_tap(s: &[u8], x: usize, rb: usize, ch: usize) -> u8 {
    let cols = rb / ch;
    let px = (x / ch) as i32;
    let c = x % ch;
    let mut acc = 0u16;
    for (j, &k) in GAUSS_K.iter().enumerate() {
        let xi = (px + j as i32 - GAUSS_R as i32).clamp(0, cols as i32 - 1) as usize * ch + c;
        acc += k * s[xi] as u16;
    }
    ((acc + ROUND) >> SHIFT) as u8
}

/// Scalar vertical 7-tap over flat `tmp` with `BORDER_REPLICATE`.
#[inline]
fn gauss_v_tap(tmp: &[u8], rb: usize, y: usize, x: usize, rows: usize) -> u8 {
    let mut acc = 0u16;
    for (j, &k) in GAUSS_K.iter().enumerate() {
        let r = (y as i32 + j as i32 - GAUSS_R as i32).clamp(0, rows as i32 - 1) as usize;
        acc += k * tmp[r * rb + x] as u16;
    }
    ((acc + ROUND) >> SHIFT) as u8
}

///
/// Each row must have exactly the same length as the output, which is what
/// `windows()` and the row slices in the callers produce.
#[inline(always)]
fn assert_same_len<const N: usize>(rows: [&[u16]; N], out: &[u8]) {
    assert_eq!(rows[0].len(), out.len());
    for r in &rows[1..] {
        assert_eq!(r.len(), out.len());
    }
}

/// N-tap `u16` MAC, `+ round`, `>> shift`, narrowed straight to `u8`.
///
/// Every row of `rows` is one tap, all of length `out.len()`.
///
/// Returns the number of output bytes written.
#[simd]
pub(crate) fn mac_round_pack_u8<const N: usize, S: Simd>(
    simd: S,
    rows: [&[u16]; N],
    k: &[u16],
    round: u16,
    shift: u32,
    out: &mut [u8],
) -> usize {
    assert_same_len(rows, out);
    let n = <S as Simd>::u16s::LEN;
    let pair = 2 * n;
    let len = out.len();
    let body = len / pair * pair;
    let rv = <S as Simd>::u16s::splat(simd, round);

    for i in (0..body).step_by(pair) {
        let mut lo = rv;
        let mut hi = rv;
        for j in 0..N {
            let kj = <S as Simd>::u16s::splat(simd, k[j]);
            let r = &rows[j][i..i + pair];
            let (rl, rh) = r.split_at(n);
            lo += <S as Simd>::u16s::from_slice(simd, rl) * kj;
            hi += <S as Simd>::u16s::from_slice(simd, rh) * kj;
        }
        (lo >> shift)
            .narrow(hi >> shift)
            .store_slice(&mut out[i..i + pair]);
    }

    let mut i = body;
    while i < len {
        let mut acc = 0u16;
        for j in 0..N {
            acc += rows[j][i] * k[j];
        }
        out[i] = ((acc + round) >> shift) as u8;
        i += 1;
    }
    len
}
