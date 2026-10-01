// Stencil code indexes several buffers by pixel coordinate; the iterator form
// suggested by `needless_range_loop` would obscure the border geometry.
#![allow(clippy::needless_range_loop)]

use opencv::{
    core::{self, Mat},
    prelude::*,
};
use pulp::{Arch, Simd, WithSimd};

use super::{bad_arg, check_src_8u, deinterleave_row_into_i16, ensure_dst, widen_row_i16};

/// 3x3 Sobel MAC over widened rows into raw `i16` accumulators.
///
/// `t`/`m`/`b` are top/mid/bottom rows; each holds 3 windows at taps
/// `x - 1`, `x`, `x + 1`. All intermediate values fit in `i16`
/// (kernel taps are -2..2, inputs 0..255, so results are within ±4080),
/// which doubles SIMD lanes and halves widen traffic vs `i32`.
/// Sign convention matches OpenCV: `gx = right - left`, `gy = bottom - top`.
struct Sobel3<'a> {
    t: [&'a [i16]; 3],
    m: [&'a [i16]; 3],
    b: [&'a [i16]; 3],
    ox: &'a mut [i16],
    oy: &'a mut [i16],
}

impl WithSimd for Sobel3<'_> {
    type Output = usize;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> usize {
        let two = simd.splat_i16s(2);
        let t0 = S::as_simd_i16s(self.t[0]).0;
        let t1 = S::as_simd_i16s(self.t[1]).0;
        let t2 = S::as_simd_i16s(self.t[2]).0;
        // Center column of the mid row has weight 0; skipped.
        let m0 = S::as_simd_i16s(self.m[0]).0;
        let m2 = S::as_simd_i16s(self.m[2]).0;
        let b0 = S::as_simd_i16s(self.b[0]).0;
        let b1 = S::as_simd_i16s(self.b[1]).0;
        let b2 = S::as_simd_i16s(self.b[2]).0;
        let (ohx, _) = S::as_mut_simd_i16s(self.ox);
        let (ohy, _) = S::as_mut_simd_i16s(self.oy);
        debug_assert_eq!(ohx.len(), t0.len());
        debug_assert_eq!(ohy.len(), t0.len());
        for i in 0..ohx.len() {
            // gx = (t2 + 2*m2 + b2) - (t0 + 2*m0 + b0)
            let right = simd.add_i16s(simd.add_i16s(t2[i], simd.mul_i16s(m2[i], two)), b2[i]);
            let left = simd.add_i16s(simd.add_i16s(t0[i], simd.mul_i16s(m0[i], two)), b0[i]);
            ohx[i] = simd.sub_i16s(right, left);
            // gy = (b0 + 2*b1 + b2) - (t0 + 2*t1 + t2)
            let bottom = simd.add_i16s(simd.add_i16s(b0[i], simd.mul_i16s(b1[i], two)), b2[i]);
            let top = simd.add_i16s(simd.add_i16s(t0[i], simd.mul_i16s(t1[i], two)), t2[i]);
            ohy[i] = simd.sub_i16s(bottom, top);
        }
        ohx.len() * S::I16_LANES
    }
}

/// 3x3 Sobel on single-channel `u8` input with `CV_32F` outputs, `BORDER_REPLICATE`.
///
/// Replaces `imgproc::sobel(smoothed, dx/dy, CV_32F, 1/0, 0/1, ksize=3,
/// BORDER_REPLICATE)`. The kernel taps are tiny integers and the result fits in
/// `i16`, so the exact integer sum is computed first and only cast to `f32`.
/// Sign convention matches OpenCV: `gx = right - left`, `gy = bottom - top`.
pub fn sobel_grayscale(src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()> {
    let (rows, cols) = check_src_8u(src)?;
    if src.channels() != 1 {
        return Err(bad_arg("sobel_grayscale needs a single-channel image"));
    }
    ensure_dst(dx, rows, cols, core::CV_32F)?;
    ensure_dst(dy, rows, cols, core::CV_32F)?;
    let arch = Arch::new();
    let (rows_usize, cols_usize) = (rows as usize, cols as usize);
    let w = cols_usize;
    // 3-row ring of widened rows: each source row is widened exactly once.
    let mut ring = vec![0i16; 3 * w];
    let mut accx = vec![0i16; w];
    let mut accy = vec![0i16; w];
    let init = 1i32.min(rows - 1);
    for r in 0..=init {
        let row: &[u8] = src.at_row(r)?;
        widen_row_i16(&mut ring[r as usize % 3 * w..(r as usize % 3 + 1) * w], row);
    }
    let mut loaded = init;
    for y in 0..rows_usize {
        let hi = (y as i32 + 1).min(rows - 1);
        while loaded < hi {
            loaded += 1;
            let row: &[u8] = src.at_row(loaded)?;
            widen_row_i16(
                &mut ring[loaded as usize % 3 * w..(loaded as usize % 3 + 1) * w],
                row,
            );
        }
        let border_row = y == 0 || y + 1 >= rows_usize;
        let r0: &[u8] = src.at_row(y.saturating_sub(1) as i32)?;
        let r1: &[u8] = src.at_row(y as i32)?;
        let r2: &[u8] = src.at_row((y + 1).min(rows_usize - 1) as i32)?;
        let ox: &mut [f32] = dx.at_row_mut(y as i32)?;
        let oy: &mut [f32] = dy.at_row_mut(y as i32)?;
        if !border_row && w > 2 {
            let n = w - 2;
            let slot = |r: usize| &ring[r % 3 * w..r % 3 * w + w];
            let (s0, s1, s2) = (slot(y - 1), slot(y), slot(y + 1));
            // Windows at taps x-1, x, x+1 over the interior [1, w-1).
            let t = [&s0[..n], &s0[1..1 + n], &s0[2..2 + n]];
            let m = [&s1[..n], &s1[1..1 + n], &s1[2..2 + n]];
            let b = [&s2[..n], &s2[1..1 + n], &s2[2..2 + n]];
            let covered = arch.dispatch(Sobel3 {
                t,
                m,
                b,
                ox: &mut accx[1..1 + n],
                oy: &mut accy[1..1 + n],
            });
            for x in 1..1 + covered {
                ox[x] = accx[x] as f32;
                oy[x] = accy[x] as f32;
            }
            for x in 1 + covered..w - 1 {
                ox[x] = sobel_gx_u8(r0, r1, r2, x, cols) as f32;
                oy[x] = sobel_gy_u8(r0, r2, x, cols) as f32;
            }
        } else {
            for x in 0..w {
                ox[x] = sobel_gx_u8(r0, r1, r2, x, cols) as f32;
                oy[x] = sobel_gy_u8(r0, r2, x, cols) as f32;
            }
            continue;
        }
        ox[0] = sobel_gx_u8(r0, r1, r2, 0, cols) as f32;
        oy[0] = sobel_gy_u8(r0, r2, 0, cols) as f32;
        if w > 1 {
            let x = w - 1;
            ox[x] = sobel_gx_u8(r0, r1, r2, x, cols) as f32;
            oy[x] = sobel_gy_u8(r0, r2, x, cols) as f32;
        }
    }
    Ok(())
}

/// Scalar 3x3 Sobel with per-tap `BORDER_REPLICATE`; valid for every `x`.
#[inline]
fn sobel_gx_u8(r0: &[u8], r1: &[u8], r2: &[u8], x: usize, cols: i32) -> i32 {
    let xm = (x as i32 - 1).clamp(0, cols - 1) as usize;
    let xp = (x as i32 + 1).clamp(0, cols - 1) as usize;
    (r0[xp] as i32 + 2 * r1[xp] as i32 + r2[xp] as i32)
        - (r0[xm] as i32 + 2 * r1[xm] as i32 + r2[xm] as i32)
}

#[inline]
fn sobel_gy_u8(r0: &[u8], r2: &[u8], x: usize, cols: i32) -> i32 {
    let xm = (x as i32 - 1).clamp(0, cols - 1) as usize;
    let xp = (x as i32 + 1).clamp(0, cols - 1) as usize;
    (r2[xm] as i32 + 2 * r2[x] as i32 + r2[xp] as i32)
        - (r0[xm] as i32 + 2 * r0[x] as i32 + r0[xp] as i32)
}

/// 3x3 Sobel on 3-channel `u8` input with `CV_16SC3` outputs, `BORDER_REPLICATE`.
///
/// Replaces `imgproc::sobel(smoothed, dx3/dy3, CV_16S, ...)`.
pub fn sobel_color_i16(src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()> {
    let (rows, cols) = check_src_8u(src)?;
    if src.channels() != 3 {
        return Err(bad_arg("sobel_color_i16 needs a 3-channel image"));
    }
    ensure_dst(dx, rows, cols, core::CV_16SC3)?;
    ensure_dst(dy, rows, cols, core::CV_16SC3)?;
    let arch = Arch::new();
    let (rows_usize, cols_usize) = (rows as usize, cols as usize);
    let w = cols_usize;
    // Planar ring: 3 rows x 3 channels of widened rows; each source row is
    // deinterleaved exactly once.
    let mut ring = vec![0i16; 9 * w];
    let mut accx = vec![0i16; w];
    let mut accy = vec![0i16; w];
    let init = 1i32.min(rows - 1);
    for r in 0..=init {
        let row: &[core::Vec3b] = src.at_row(r)?;
        deinterleave_row_into_i16(&mut ring, r as usize % 3, w, row);
    }
    let mut loaded = init;
    for y in 0..rows_usize {
        let hi = (y as i32 + 1).min(rows - 1);
        while loaded < hi {
            loaded += 1;
            let row: &[core::Vec3b] = src.at_row(loaded)?;
            deinterleave_row_into_i16(&mut ring, loaded as usize % 3, w, row);
        }
        let border_row = y == 0 || y + 1 >= rows_usize;
        let r0: &[core::Vec3b] = src.at_row(y.saturating_sub(1) as i32)?;
        let r1: &[core::Vec3b] = src.at_row(y as i32)?;
        let r2: &[core::Vec3b] = src.at_row((y + 1).min(rows_usize - 1) as i32)?;
        let ox: &mut [core::Vec3s] = dx.at_row_mut(y as i32)?;
        let oy: &mut [core::Vec3s] = dy.at_row_mut(y as i32)?;
        if !border_row && w > 2 {
            let n = w - 2;
            // Plane (row r, channel c) lives at ring[(r % 3) * 3 * w + c * w..].
            let plane =
                |r: usize, c: usize| &ring[r % 3 * 3 * w + c * w..r % 3 * 3 * w + c * w + w];
            for c in 0..3 {
                let t = [
                    &plane(y - 1, c)[..n],
                    &plane(y - 1, c)[1..1 + n],
                    &plane(y - 1, c)[2..2 + n],
                ];
                let m = [
                    &plane(y, c)[..n],
                    &plane(y, c)[1..1 + n],
                    &plane(y, c)[2..2 + n],
                ];
                let b = [
                    &plane(y + 1, c)[..n],
                    &plane(y + 1, c)[1..1 + n],
                    &plane(y + 1, c)[2..2 + n],
                ];
                let covered = arch.dispatch(Sobel3 {
                    t,
                    m,
                    b,
                    ox: &mut accx[1..1 + n],
                    oy: &mut accy[1..1 + n],
                });
                for x in 1..1 + covered {
                    ox[x][c] = accx[x];
                    oy[x][c] = accy[x];
                }
                for x in 1 + covered..w - 1 {
                    let (gx, gy) = sobel_color_pixel(r0, r1, r2, x, cols, c);
                    ox[x][c] = gx.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    oy[x][c] = gy.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                }
            }
        }
        // Borders (also the whole row when too narrow / on border rows).
        for x in 0..1.min(w) {
            let (gx, gy) = sobel_color_row(r0, r1, r2, x, cols);
            ox[x] = gx;
            oy[x] = gy;
        }
        if !border_row && w > 2 {
            // Interior already done; only the last column remains.
        } else {
            for x in 1.min(w)..w {
                let (gx, gy) = sobel_color_row(r0, r1, r2, x, cols);
                ox[x] = gx;
                oy[x] = gy;
            }
            continue;
        }
        if w > 1 {
            let x = w - 1;
            let (gx, gy) = sobel_color_row(r0, r1, r2, x, cols);
            ox[x] = gx;
            oy[x] = gy;
        }
    }
    Ok(())
}

/// Scalar 3x3 Sobel tap for one channel with `BORDER_REPLICATE`.
#[inline]
fn sobel_color_pixel(
    r0: &[core::Vec3b],
    r1: &[core::Vec3b],
    r2: &[core::Vec3b],
    x: usize,
    cols: i32,
    c: usize,
) -> (i32, i32) {
    let xm = (x as i32 - 1).clamp(0, cols - 1) as usize;
    let xp = (x as i32 + 1).clamp(0, cols - 1) as usize;
    let gx = (r0[xp][c] as i32 + 2 * r1[xp][c] as i32 + r2[xp][c] as i32)
        - (r0[xm][c] as i32 + 2 * r1[xm][c] as i32 + r2[xm][c] as i32);
    let gy = (r2[xm][c] as i32 + 2 * r2[x][c] as i32 + r2[xp][c] as i32)
        - (r0[xm][c] as i32 + 2 * r0[x][c] as i32 + r0[xp][c] as i32);
    (gx, gy)
}

/// Scalar 3x3 Sobel pixel (all 3 channels) with `BORDER_REPLICATE`.
#[inline]
fn sobel_color_row(
    r0: &[core::Vec3b],
    r1: &[core::Vec3b],
    r2: &[core::Vec3b],
    x: usize,
    cols: i32,
) -> (core::Vec3s, core::Vec3s) {
    let mut gx = core::Vec3s::from([0, 0, 0]);
    let mut gy = core::Vec3s::from([0, 0, 0]);
    for c in 0..3 {
        let (a, b) = sobel_color_pixel(r0, r1, r2, x, cols, c);
        gx[c] = a.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        gy[c] = b.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    (gx, gy)
}
