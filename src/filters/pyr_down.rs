// Stencil code indexes several buffers by pixel coordinate; the iterator form
// suggested by `needless_range_loop` would obscure the border geometry.
#![allow(clippy::needless_range_loop)]

use opencv::{
    core::{self, Mat},
    prelude::*,
};
use pulp::Arch;

use super::{
    Mac16, check_src_8u, deinterleave_widen_row_u16, ensure_dst, reflect101, widen_row_u16,
};

/// 5x5 `pyrDown` kernel as `u16` lanes (for [`Mac16`]); the hot path keeps raw
/// integer sums and rounds once at the very end, like OpenCV's `pyrDown_`.
const PYR_K_U16: [u16; 5] = [1, 4, 6, 4, 1];
const PYR_R: usize = 2;

/// `pyrDown` (5x5 Gaussian + 2x decimate) with `BORDER_REFLECT_101`.
///
/// Handles `CV_8UC1` and `CV_8UC3` (masks are single-channel). Output size is
/// `(cols + 1) / 2 x (rows + 1) / 2`, like `imgproc::pyr_down_def`.
pub fn pyr_down(src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
    let (rows, cols) = check_src_8u(src)?;
    let (orows, ocols) = ((rows + 1) / 2, (cols + 1) / 2);
    ensure_dst(dst, orows, ocols, src.typ())?;
    if src.channels() == 1 {
        pyr_down_gray(src, dst, rows, cols, orows)
    } else {
        pyr_down_color(src, dst, rows, cols, orows)
    }
}

/// Scalar 5-tap with per-tap `BORDER_REFLECT_101`, RAW sum (no rounding);
/// valid for every `x`. Max sum `255 * 16 = 4080`.
#[inline]
fn pyr_raw_tap_u8(s: &[u8], x: i32, cols: i32) -> u16 {
    let mut acc = 0u16;
    for (j, &k) in PYR_K_U16.iter().enumerate() {
        acc += k * s[reflect101(x + j as i32 - PYR_R as i32, cols)] as u16;
    }
    acc
}

/// Branch-free RAW 5-tap; caller guarantees `PYR_R <= x <= cols - 1 - PYR_R`.
#[inline]
fn pyr_raw_tap_interior_u8(s: &[u8], x: usize) -> u16 {
    PYR_K_U16[0] * s[x - 2] as u16
        + PYR_K_U16[1] * s[x - 1] as u16
        + PYR_K_U16[2] * s[x] as u16
        + PYR_K_U16[3] * s[x + 1] as u16
        + PYR_K_U16[4] * s[x + 2] as u16
}

fn pyr_down_gray(src: &Mat, dst: &mut Mat, rows: i32, cols: i32, orows: i32) -> opencv::Result<()> {
    let arch = Arch::new();
    let w = cols as usize;
    // Single-rounding separable: horizontal RAW sums into u16 tmp (no
    // rounding), then vertical Mac with the single (+128>>8) rounding at the
    // store — mirroring OpenCV's `pyrDown_` fixed-point structure.
    let mut tmp = vec![0u16; rows as usize * w];
    let mut wide = vec![0u16; w];
    let mut acc = vec![0u16; w];
    let mid_end = w.saturating_sub(PYR_R);
    let lo = PYR_R.min(w);
    for y in 0..rows {
        let s: &[u8] = src.at_row(y)?;
        let t = &mut tmp[y as usize * w..(y as usize + 1) * w];
        for x in 0..lo {
            t[x] = pyr_raw_tap_u8(s, x as i32, cols);
        }
        if lo < mid_end {
            widen_row_u16(&mut wide, s);
            let n = mid_end - lo;
            let wins: [&[u16]; 5] =
                std::array::from_fn(|j| &wide[lo + j - PYR_R..lo + j - PYR_R + n]);
            let covered = arch.dispatch(Mac16 {
                wins,
                k: PYR_K_U16,
                out: &mut acc[lo..lo + n],
            });
            t[lo..lo + covered].copy_from_slice(&acc[lo..lo + covered]);
            for x in lo + covered..mid_end {
                t[x] = pyr_raw_tap_interior_u8(s, x);
            }
        }
        for x in mid_end.max(lo)..w {
            t[x] = pyr_raw_tap_u8(s, x as i32, cols);
        }
    }
    // Decimated vertical pass over u16 tmp (scalar: strided column access has
    // no portable SIMD gather) with the single rounding at the store.
    // NOTE: the divisor here is 16 (horizontal kernel sum) * 16 (vertical
    // kernel sum) = 256, i.e. a >>8 shift — not PYR_SHIFT.
    for oy in 0..orows as usize {
        let cy = oy as i32 * 2;
        let d: &mut [u8] = dst.at_row_mut(oy as i32)?;
        for (ox, out) in d.iter_mut().enumerate() {
            let cx = ox * 2;
            let mut acc = 0u16;
            for (j, &k) in PYR_K_U16.iter().enumerate() {
                let sy = reflect101(cy + j as i32 - PYR_R as i32, rows) * w + cx;
                acc += k * tmp[sy];
            }
            *out = ((acc + 128) >> 8) as u8;
        }
    }
    Ok(())
}

fn pyr_down_color(
    src: &Mat,
    dst: &mut Mat,
    rows: i32,
    cols: i32,
    orows: i32,
) -> opencv::Result<()> {
    let arch = Arch::new();
    let w = cols as usize;
    // Single-rounding separable with interleaved u16 tmp (see gray version).
    let mut tmp = vec![0u16; rows as usize * w * 3];
    let mut planes = vec![0u16; 3 * w];
    let mut acc = vec![0u16; w];
    let mid_end = w.saturating_sub(PYR_R);
    let lo = PYR_R.min(w);
    for y in 0..rows {
        let s: &[core::Vec3b] = src.at_row(y)?;
        let t = &mut tmp[y as usize * w * 3..(y as usize + 1) * w * 3];
        if lo < mid_end {
            let (p0, rest) = planes.split_at_mut(w);
            let (p1, p2) = rest.split_at_mut(w);
            deinterleave_widen_row_u16(p0, p1, p2, s);
            let n = mid_end - lo;
            for (c, p) in [&p0[..], &p1[..], &p2[..]].into_iter().enumerate() {
                let wins: [&[u16]; 5] =
                    std::array::from_fn(|j| &p[lo + j - PYR_R..lo + j - PYR_R + n]);
                let covered = arch.dispatch(Mac16 {
                    wins,
                    k: PYR_K_U16,
                    out: &mut acc[lo..lo + n],
                });
                for x in lo..lo + covered {
                    t[x * 3 + c] = acc[x];
                }
                for x in lo + covered..mid_end {
                    t[x * 3 + c] = pyr_raw_tap_color_interior(s, x, c);
                }
            }
        }
        for x in 0..lo {
            pyr_raw_pixel_color(s, t, x, cols);
        }
        for x in mid_end.max(lo)..w {
            pyr_raw_pixel_color(s, t, x, cols);
        }
    }
    // Decimated vertical pass over u16 tmp with the single rounding at store.
    let ocols = dst.cols() as usize;
    for oy in 0..orows as usize {
        let d: &mut [core::Vec3b] = dst.at_row_mut(oy as i32)?;
        let cy = oy as i32 * 2;
        for ox in 0..ocols {
            d[ox] = pyr_vpixel_color_u16(&tmp, w, cy, ox, rows);
        }
    }
    Ok(())
}

/// Scalar horizontal RAW color pixel (all 3 channels, no rounding),
/// `BORDER_REFLECT_101`.
#[inline]
fn pyr_raw_pixel_color(s: &[core::Vec3b], t: &mut [u16], x: usize, cols: i32) {
    for c in 0..3 {
        let mut acc = 0u16;
        for (j, &k) in PYR_K_U16.iter().enumerate() {
            let xi = reflect101(x as i32 + j as i32 - PYR_R as i32, cols);
            acc += k * s[xi][c] as u16;
        }
        t[x * 3 + c] = acc;
    }
}

/// Branch-free horizontal RAW color tap for one channel (no rounding).
#[inline]
fn pyr_raw_tap_color_interior(s: &[core::Vec3b], x: usize, c: usize) -> u16 {
    PYR_K_U16[0] * s[x - 2][c] as u16
        + PYR_K_U16[1] * s[x - 1][c] as u16
        + PYR_K_U16[2] * s[x][c] as u16
        + PYR_K_U16[3] * s[x + 1][c] as u16
        + PYR_K_U16[4] * s[x + 2][c] as u16
}

/// Scalar decimated vertical color pixel over u16 tmp (all 3 channels) with
/// the single rounding at the store (divisor 16 * 16 = 256),
/// `BORDER_REFLECT_101`.
#[inline]
fn pyr_vpixel_color_u16(tmp: &[u16], w: usize, cy: i32, ox: usize, rows: i32) -> core::Vec3b {
    let cx = ox * 2;
    let mut px = core::Vec3b::from([0, 0, 0]);
    for c in 0..3 {
        let mut acc = 0u16;
        for (j, &k) in PYR_K_U16.iter().enumerate() {
            let sy = reflect101(cy + j as i32 - PYR_R as i32, rows) * w * 3 + cx * 3 + c;
            acc += k * tmp[sy];
        }
        px[c] = ((acc + 128) >> 8) as u8;
    }
    px
}
