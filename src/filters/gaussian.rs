// Stencil code indexes several buffers by pixel coordinate; the iterator form
// suggested by `needless_range_loop` would obscure the border geometry.
#![allow(clippy::needless_range_loop)]

use opencv::{
    core::{self, Mat},
    prelude::*,
};
use pulp::Arch;

use super::{
    Mac16, check_src_8u, deinterleave_row_into_u16, deinterleave_widen_row_u16,
    ensure_dst, narrow_row_u16, widen_row_u16,
};

/// 7x7 Gaussian kernel for `sigma = 0` (auto) in Q8.
///
/// Probed from `imgproc::get_gaussian_kernel(7, 0.0, CV_64F)`:
/// `[0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]`,
/// exactly representable with 8 fractional bits, summing to `1 << 8`.
const GAUSS_K_Q8: [u16; 7] = [8, 28, 56, 72, 56, 28, 8];
const GAUSS_SHIFT: u32 = 8;
const GAUSS_ROUND: u16 = 1 << (8 - 1);
const GAUSS_R: usize = 3;

/// 7x7 Gaussian blur (`sigma = 0` auto), `BORDER_REPLICATE`.
///
/// Handles `CV_8UC1` and `CV_8UC3`. Replaces
/// `imgproc::gaussian_blur(src, dst, 7x7, 0, 0, BORDER_REPLICATE)`.
pub fn gaussian_blur_7x7(src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
    let (rows, cols) = check_src_8u(src)?;
    ensure_dst(dst, rows, cols, src.typ())?;
    if src.channels() == 1 {
        blur_7x7_gray(src, dst, rows, cols)
    } else {
        blur_7x7_color(src, dst, rows, cols)
    }
}

/// Scalar 7-tap with per-tap `BORDER_REPLICATE`; valid for every `x`.
#[inline]
fn gauss_tap_u8(s: &[u8], x: i32, cols: i32) -> u16 {
    let mut acc = 0u16;
    for (j, &k) in GAUSS_K_Q8.iter().enumerate() {
        acc += k * s[(x + j as i32 - GAUSS_R as i32).clamp(0, cols - 1) as usize] as u16;
    }
    acc
}

/// Branch-free 7-tap; caller guarantees `GAUSS_R <= x <= cols - 1 - GAUSS_R`.
#[inline]
fn gauss_tap_interior_u8(s: &[u8], x: usize) -> u16 {
    GAUSS_K_Q8[0] * s[x - 3] as u16
        + GAUSS_K_Q8[1] * s[x - 2] as u16
        + GAUSS_K_Q8[2] * s[x - 1] as u16
        + GAUSS_K_Q8[3] * s[x] as u16
        + GAUSS_K_Q8[4] * s[x + 1] as u16
        + GAUSS_K_Q8[5] * s[x + 2] as u16
        + GAUSS_K_Q8[6] * s[x + 3] as u16
}

fn blur_7x7_gray(src: &Mat, dst: &mut Mat, rows: i32, cols: i32) -> opencv::Result<()> {
    let arch = Arch::new();
    let w = cols as usize;
    // Separable: horizontal into tmp (one rounding), then vertical into dst.
    let mut tmp = vec![0u8; rows as usize * w];
    let mut wide = vec![0u16; w];
    let mut acc = vec![0u16; w];
    let mid_end = w.saturating_sub(GAUSS_R);
    let lo = GAUSS_R.min(w);
    for y in 0..rows {
        let s: &[u8] = src.at_row(y)?;
        let t = &mut tmp[y as usize * w..(y as usize + 1) * w];
        for x in 0..lo {
            t[x] = ((gauss_tap_u8(s, x as i32, cols) + GAUSS_ROUND) >> GAUSS_SHIFT) as u8;
        }
        if lo < mid_end {
            widen_row_u16(&mut wide, s);
            let n = mid_end - lo;
            let wins: [&[u16]; 7] =
                std::array::from_fn(|j| &wide[lo + j - GAUSS_R..lo + j - GAUSS_R + n]);
            let covered = arch.dispatch(Mac16 {
                wins,
                k: GAUSS_K_Q8,
                out: &mut acc[lo..lo + n],
            });
            narrow_row_u16(
                &mut t[lo..lo + covered],
                &acc[lo..lo + covered],
                GAUSS_ROUND,
                GAUSS_SHIFT,
            );
            for x in lo + covered..mid_end {
                t[x] = ((gauss_tap_interior_u8(s, x) + GAUSS_ROUND) >> GAUSS_SHIFT) as u8;
            }
        }
        for x in mid_end.max(lo)..w {
            t[x] = ((gauss_tap_u8(s, x as i32, cols) + GAUSS_ROUND) >> GAUSS_SHIFT) as u8;
        }
    }
    // Vertical pass over tmp with a 7-row widened ring buffer.
    const WIN: usize = 7;
    let mut ring = vec![0u16; WIN * w];
    let init = (GAUSS_R as i32).min(rows - 1);
    for r in 0..=init {
        widen_row_u16(
            &mut ring[r as usize % WIN * w..(r as usize % WIN + 1) * w],
            &tmp[r as usize * w..(r as usize + 1) * w],
        );
    }
    let mut loaded = init;
    for y in 0..rows as usize {
        let hi = (y as i32 + GAUSS_R as i32).min(rows - 1);
        while loaded < hi {
            loaded += 1;
            widen_row_u16(
                &mut ring[loaded as usize % WIN * w..(loaded as usize % WIN + 1) * w],
                &tmp[loaded as usize * w..(loaded as usize + 1) * w],
            );
        }
        let d: &mut [u8] = dst.at_row_mut(y as i32)?;
        // Border rows need clamped taps: whole row goes scalar.
        if y < GAUSS_R || y as i32 >= rows - GAUSS_R as i32 {
            for x in 0..w {
                d[x] = gauss_vtap_u8(&tmp, w, y as i32, x, rows);
            }
            continue;
        }
        for x in 0..lo {
            d[x] = gauss_vtap_u8(&tmp, w, y as i32, x, rows);
        }
        if lo < mid_end {
            let n = mid_end - lo;
            let wins: [&[u16]; 7] = std::array::from_fn(|j| {
                let r = (y as i32 + j as i32 - GAUSS_R as i32).clamp(0, rows - 1) as usize;
                &ring[r % WIN * w + lo..r % WIN * w + lo + n]
            });
            let covered = arch.dispatch(Mac16 {
                wins,
                k: GAUSS_K_Q8,
                out: &mut acc[lo..lo + n],
            });
            narrow_row_u16(
                &mut d[lo..lo + covered],
                &acc[lo..lo + covered],
                GAUSS_ROUND,
                GAUSS_SHIFT,
            );
            for x in lo + covered..mid_end {
                d[x] = gauss_vtap_interior_u8(&tmp, w, y, x);
            }
        }
        for x in mid_end.max(lo)..w {
            d[x] = gauss_vtap_u8(&tmp, w, y as i32, x, rows);
        }
    }
    Ok(())
}

/// Scalar vertical 7-tap over flat `tmp` with `BORDER_REPLICATE`; any `y`.
#[inline]
fn gauss_vtap_u8(tmp: &[u8], w: usize, y: i32, x: usize, rows: i32) -> u8 {
    let mut acc = 0u16;
    for (j, &k) in GAUSS_K_Q8.iter().enumerate() {
        let sy = (y + j as i32 - GAUSS_R as i32).clamp(0, rows - 1) as usize * w + x;
        acc += k * tmp[sy] as u16;
    }
    ((acc + GAUSS_ROUND) >> GAUSS_SHIFT) as u8
}

/// Branch-free vertical 7-tap; caller guarantees in-range rows.
#[inline]
fn gauss_vtap_interior_u8(tmp: &[u8], w: usize, y: usize, x: usize) -> u8 {
    let mut acc = 0u16;
    for (j, &k) in GAUSS_K_Q8.iter().enumerate() {
        acc += k * tmp[(y + j - GAUSS_R) * w + x] as u16;
    }
    ((acc + GAUSS_ROUND) >> GAUSS_SHIFT) as u8
}

fn blur_7x7_color(src: &Mat, dst: &mut Mat, rows: i32, cols: i32) -> opencv::Result<()> {
    let arch = Arch::new();
    let w = cols as usize;
    let mut tmp = vec![0u8; rows as usize * w * 3];
    let mut planes = vec![0u16; 3 * w];
    let mut acc = vec![0u16; w];
    let mid_end = w.saturating_sub(GAUSS_R);
    let lo = GAUSS_R.min(w);
    for y in 0..rows {
        let s: &[core::Vec3b] = src.at_row(y)?;
        let t = &mut tmp[y as usize * w * 3..(y as usize + 1) * w * 3];
        if lo < mid_end {
            let (p0, rest) = planes.split_at_mut(w);
            let (p1, p2) = rest.split_at_mut(w);
            deinterleave_widen_row_u16(p0, p1, p2, s);
            let n = mid_end - lo;
            for (c, p) in [&p0[..], &p1[..], &p2[..]].into_iter().enumerate() {
                let wins: [&[u16]; 7] =
                    std::array::from_fn(|j| &p[lo + j - GAUSS_R..lo + j - GAUSS_R + n]);
                let covered = arch.dispatch(Mac16 {
                    wins,
                    k: GAUSS_K_Q8,
                    out: &mut acc[lo..lo + n],
                });
                for x in lo..lo + covered {
                    t[x * 3 + c] = ((acc[x] + GAUSS_ROUND) >> GAUSS_SHIFT) as u8;
                }
                for x in lo + covered..mid_end {
                    t[x * 3 + c] = gauss_tap_color_interior(s, x, c);
                }
            }
        }
        // Borders (also covers the whole row when too narrow for an interior).
        for x in 0..lo {
            gauss_pixel_color(s, t, x, cols);
        }
        for x in mid_end.max(lo)..w {
            gauss_pixel_color(s, t, x, cols);
        }
    }
    // Vertical pass with a 7-row planar ring buffer.
    const WIN: usize = 7;
    let mut ring = vec![0u16; WIN * 3 * w];
    let init = (GAUSS_R as i32).min(rows - 1);
    for r in 0..=init {
        deinterleave_row_into_u16(
            &mut ring,
            r as usize % WIN,
            w,
            &tmp[r as usize * w * 3..(r as usize + 1) * w * 3],
        );
    }
    let mut loaded = init;
    for y in 0..rows as usize {
        let hi = (y as i32 + GAUSS_R as i32).min(rows - 1);
        while loaded < hi {
            loaded += 1;
            deinterleave_row_into_u16(
                &mut ring,
                loaded as usize % WIN,
                w,
                &tmp[loaded as usize * w * 3..(loaded as usize + 1) * w * 3],
            );
        }
        let d: &mut [core::Vec3b] = dst.at_row_mut(y as i32)?;
        // Border rows need clamped taps: whole row goes scalar.
        if y < GAUSS_R || y as i32 >= rows - GAUSS_R as i32 {
            for x in 0..w {
                d[x] = gauss_vpixel_color(&tmp, w, y as i32, x, rows);
            }
            continue;
        }
        if lo < mid_end {
            let n = mid_end - lo;
            for c in 0..3 {
                let wins: [&[u16]; 7] = std::array::from_fn(|j| {
                    let r = (y as i32 + j as i32 - GAUSS_R as i32).clamp(0, rows - 1) as usize;
                    let base = r % WIN * 3 * w + c * w;
                    &ring[base + lo..base + lo + n]
                });
                let covered = arch.dispatch(Mac16 {
                    wins,
                    k: GAUSS_K_Q8,
                    out: &mut acc[lo..lo + n],
                });
                for x in lo..lo + covered {
                    d[x][c] = ((acc[x] + GAUSS_ROUND) >> GAUSS_SHIFT) as u8;
                }
                for x in lo + covered..mid_end {
                    d[x][c] = gauss_vtap_color_interior(&tmp, w, y, x, c, rows);
                }
            }
        }
        for x in 0..lo {
            d[x] = gauss_vpixel_color(&tmp, w, y as i32, x, rows);
        }
        for x in mid_end.max(lo)..w {
            d[x] = gauss_vpixel_color(&tmp, w, y as i32, x, rows);
        }
    }
    Ok(())
}

/// Scalar horizontal color pixel (all 3 channels), `BORDER_REPLICATE`.
#[inline]
fn gauss_pixel_color(s: &[core::Vec3b], t: &mut [u8], x: usize, cols: i32) {
    for c in 0..3 {
        let mut acc = 0u16;
        for (j, &k) in GAUSS_K_Q8.iter().enumerate() {
            let xi = (x as i32 + j as i32 - GAUSS_R as i32).clamp(0, cols - 1) as usize;
            acc += k * s[xi][c] as u16;
        }
        t[x * 3 + c] = ((acc + GAUSS_ROUND) >> GAUSS_SHIFT) as u8;
    }
}

/// Branch-free horizontal color tap for one channel.
#[inline]
fn gauss_tap_color_interior(s: &[core::Vec3b], x: usize, c: usize) -> u8 {
    let acc = GAUSS_K_Q8[0] * s[x - 3][c] as u16
        + GAUSS_K_Q8[1] * s[x - 2][c] as u16
        + GAUSS_K_Q8[2] * s[x - 1][c] as u16
        + GAUSS_K_Q8[3] * s[x][c] as u16
        + GAUSS_K_Q8[4] * s[x + 1][c] as u16
        + GAUSS_K_Q8[5] * s[x + 2][c] as u16
        + GAUSS_K_Q8[6] * s[x + 3][c] as u16;
    ((acc + GAUSS_ROUND) >> GAUSS_SHIFT) as u8
}

/// Scalar vertical color tap over flat interleaved `tmp`, `BORDER_REPLICATE`.
#[inline]
fn gauss_vtap_color_interior(tmp: &[u8], w: usize, y: usize, x: usize, c: usize, rows: i32) -> u8 {
    let _ = rows;
    let mut acc = 0u16;
    for (j, &k) in GAUSS_K_Q8.iter().enumerate() {
        acc += k * tmp[(y + j - GAUSS_R) * w * 3 + x * 3 + c] as u16;
    }
    ((acc + GAUSS_ROUND) >> GAUSS_SHIFT) as u8
}

/// Scalar vertical color pixel (all 3 channels) with `BORDER_REPLICATE`.
#[inline]
fn gauss_vpixel_color(tmp: &[u8], w: usize, y: i32, x: usize, rows: i32) -> core::Vec3b {
    let mut px = core::Vec3b::from([0, 0, 0]);
    for c in 0..3 {
        let mut acc = 0u16;
        for (j, &k) in GAUSS_K_Q8.iter().enumerate() {
            let yi =
                (y + j as i32 - GAUSS_R as i32).clamp(0, rows - 1) as usize * w * 3 + x * 3 + c;
            acc += k * tmp[yi] as u16;
        }
        px[c] = ((acc + GAUSS_ROUND) >> GAUSS_SHIFT) as u8;
    }
    px
}
