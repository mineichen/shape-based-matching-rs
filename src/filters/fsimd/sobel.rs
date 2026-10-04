//! 3x3 Sobel, `BORDER_REPLICATE`.

use fearless_simd::prelude::*;
use fearless_simd::{Level, dispatch};
use fearless_simd_macros::simd;
use opencv::{
    core::{self, Mat},
    prelude::*,
};

use super::{Out, Rows, Scratch, grow, row, widen_u8_to_i16, write_back};
use crate::filters::{bad_arg, check_src_8u, ensure_dst};

// ---------------------------------------------------------------------------
// 3x3 Sobel, `BORDER_REPLICATE`
// ---------------------------------------------------------------------------

/// The three halo-carrying widened rows `y - 1`, `y`, `y + 1`.
type SobelRows<'a> = [&'a [i16]; 3];

/// Sliding three-row window of widened `i16` source rows.
///
/// One [`SobelWindow::load`] per output row keeps every source row widened
/// exactly once; the interior kernel then runs straight into the output rows.
struct SobelWindow<'a> {
    ring: &'a mut [i16],
    rb: usize,
    loaded: usize,
    /// First byte of the column interior.
    lo: usize,
    /// One past the last byte of the column interior.
    hi: usize,
}

impl<'a> SobelWindow<'a> {
    fn new(ring: &'a mut [i16], rb: usize, ch: usize) -> Self {
        Self {
            ring,
            rb,
            loaded: 0,
            lo: ch.min(rb),
            hi: rb.saturating_sub(ch),
        }
    }

    /// Widen whatever rows row `y` needs. Returns `true` when the row has a
    /// usable interior.
    fn load(&mut self, level: &Level, src: &[u8], y: usize, rows: usize) -> bool {
        let hi_row = (y + 1).min(rows - 1);
        while self.loaded <= hi_row {
            let r = self.loaded;
            let rb = self.rb;
            let slice = &mut self.ring[r % 3 * rb..(r % 3 + 1) * rb];
            let row = row(src, rb, r);
            dispatch!(*level, simd => widen_u8_to_i16(simd, row, slice));
            self.loaded += 1;
        }
        y > 0 && y + 1 < rows && self.hi > self.lo
    }

    /// The three widened rows `y - 1`, `y`, `y + 1`, each carrying the tap halo.
    #[inline]
    fn taps(&self, ch: usize, y: usize) -> SobelRows<'_> {
        let rb = self.rb;
        let slot = |r: usize| &self.ring[r % 3 * rb..(r % 3 + 1) * rb];
        [
            sobel_row(slot(y - 1), self.lo, self.hi, ch),
            sobel_row(slot(y), self.lo, self.hi, ch),
            sobel_row(slot(y + 1), self.lo, self.hi, ch),
        ]
    }
}

/// 3x3 Sobel on `CV_8UC1` with `CV_32F` `dx`/`dy`.
pub fn sobel_grayscale(
    s: &mut Scratch,
    src: &Mat,
    dx: &mut Mat,
    dy: &mut Mat,
) -> opencv::Result<()> {
    let (rows_i, cols_i) = check_src_8u(src)?;
    if src.channels() != 1 {
        return Err(bad_arg("sobel_grayscale needs a single-channel image"));
    }
    ensure_dst(dx, rows_i, cols_i, core::CV_32F)?;
    ensure_dst(dy, rows_i, cols_i, core::CV_32F)?;
    let level = Level::new();
    let ch = 1usize;
    let rows = rows_i as usize;
    let rb = cols_i as usize;
    let src = Rows::new(src, rows * rb, ch)?;
    let Scratch { ring_i16, .. } = s;
    let mut win = SobelWindow::new(grow(ring_i16, 3 * rb), rb, ch);
    for y in 0..rows {
        let interior = win.load(&level, src.data(), y, rows);
        let r0 = row(src.data(), rb, y.saturating_sub(1));
        let r1 = row(src.data(), rb, y);
        let r2 = row(src.data(), rb, (y + 1).min(rows - 1));
        let ox: &mut [f32] = dx.at_row_mut(y as i32)?;
        let oy: &mut [f32] = dy.at_row_mut(y as i32)?;
        let (lo, hi) = (win.lo, win.hi);
        if interior {
            let [t, m, b] = win.taps(ch, y);
            dispatch!(level, simd => {
                sobel3_f32(simd, t, m, b, ch, &mut ox[lo..hi], &mut oy[lo..hi]);
            });
            // Only the `ch`-wide column frame needs the scalar fallback. Walking
            // the whole row and skipping the interior instead costs more than the
            // SIMD interior itself (a compare and a branch per output pixel).
            for x in 0..lo {
                let (gx, gy) = sobel_scalar(r0, r1, r2, x, rb, ch);
                ox[x] = gx;
                oy[x] = gy;
            }
            for x in hi..rb {
                let (gx, gy) = sobel_scalar(r0, r1, r2, x, rb, ch);
                ox[x] = gx;
                oy[x] = gy;
            }
        } else {
            // First and last row: no vertical neighbours, all scalar.
            for x in 0..rb {
                let (gx, gy) = sobel_scalar(r0, r1, r2, x, rb, ch);
                ox[x] = gx;
                oy[x] = gy;
            }
        }
    }
    Ok(())
}

/// 3x3 Sobel on `CV_8UC3` with `CV_16SC3` `dx`/`dy`.
pub fn sobel_color_i16(
    s: &mut Scratch,
    src: &Mat,
    dx: &mut Mat,
    dy: &mut Mat,
) -> opencv::Result<()> {
    let (rows_i, cols_i) = check_src_8u(src)?;
    if src.channels() != 3 {
        return Err(bad_arg("sobel_color_i16 needs a 3-channel image"));
    }
    ensure_dst(dx, rows_i, cols_i, core::CV_16SC3)?;
    ensure_dst(dy, rows_i, cols_i, core::CV_16SC3)?;
    let level = Level::new();
    let ch = 3usize;
    let rows = rows_i as usize;
    let rb = cols_i as usize * ch;
    let ob = rb * 2;
    let src = Rows::new(src, rows * rb, ch)?;
    // `CV_16SC3` has no `at_row_mut::<i16>` (the `opencv` crate matches the
    // whole `Mat` type), so the color path writes its rows through bytes and
    // runs the kernel into small `i16` staging rows. LLVM turns
    // `i16::to_le_bytes` stores into plain 16-bit stores, so the final copy
    // vectorizes.
    let staged = {
        let mut dxo = Out::new(dx, rows * ob)?;
        let mut dyo = Out::new(dy, rows * ob)?;
        let Scratch {
            ring_i16, acc_i16, ..
        } = s;
        let (accx, accy) = grow(acc_i16, 2 * rb).split_at_mut(rb);
        let ring = grow(ring_i16, 3 * rb);
        let mut win = SobelWindow::new(ring, rb, ch);
        for y in 0..rows {
            let interior = win.load(&level, src.data(), y, rows);
            let r0 = row(src.data(), rb, y.saturating_sub(1));
            let r1 = row(src.data(), rb, y);
            let r2 = row(src.data(), rb, (y + 1).min(rows - 1));
            if interior {
                let [t, m, b] = win.taps(ch, y);
                let (lo, hi) = (win.lo, win.hi);
                dispatch!(level, simd => {
                    sobel3_i16(simd, t, m, b, ch, &mut accx[lo..hi], &mut accy[lo..hi]);
                });
            }
            let xs = dxo.bytes();
            let ys = dyo.bytes();
            let ox = &mut xs[y * ob..(y + 1) * ob];
            let oy = &mut ys[y * ob..(y + 1) * ob];
            let put = |x: usize, ox: &mut [u8], oy: &mut [u8]| {
                let (gx, gy) = sobel_scalar(r0, r1, r2, x, rb, ch);
                let gx = gx.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                let gy = gy.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                ox[x * 2..x * 2 + 2].copy_from_slice(&gx.to_le_bytes());
                oy[x * 2..x * 2 + 2].copy_from_slice(&gy.to_le_bytes());
            };
            if interior {
                let (lo, hi) = (win.lo, win.hi);
                store_i16_le(&mut ox[lo * 2..hi * 2], &accx[lo..hi]);
                store_i16_le(&mut oy[lo * 2..hi * 2], &accy[lo..hi]);
                // Only the `ch`-wide column frame is scalar; see `sobel_grayscale`.
                for x in 0..lo {
                    put(x, ox, oy);
                }
                for x in hi..rb {
                    put(x, ox, oy);
                }
            } else {
                for x in 0..rb {
                    put(x, ox, oy);
                }
            }
        }
        (dxo.take_staged(), dyo.take_staged())
    };
    if let Some(v) = staged.0 {
        write_back(dx, v, rows, ob, ch)?;
    }
    if let Some(v) = staged.1 {
        write_back(dy, v, rows, ob, ch)?;
    }
    Ok(())
}

/// Store `i16` values as little-endian byte pairs. Auto-vectorized.
#[inline]
fn store_i16_le(dst: &mut [u8], src: &[i16]) {
    debug_assert_eq!(dst.len(), src.len() * 2);
    for (d, &v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
        d.copy_from_slice(&v.to_le_bytes());
    }
}

/// Scalar Sobel for byte `x` of one channel, `BORDER_REPLICATE`.
#[inline]
fn sobel_scalar(r0: &[u8], r1: &[u8], r2: &[u8], x: usize, rb: usize, ch: usize) -> (f32, f32) {
    let cols = rb / ch;
    let px = (x / ch) as i32;
    let c = x % ch;
    let xm = (px - 1).clamp(0, cols as i32 - 1) as usize * ch + c;
    let xp = (px + 1).clamp(0, cols as i32 - 1) as usize * ch + c;
    let gx = (r0[xp] as i32 + 2 * r1[xp] as i32 + r2[xp] as i32)
        - (r0[xm] as i32 + 2 * r1[xm] as i32 + r2[xm] as i32);
    let gy = (r2[xm] as i32 + 2 * r2[x] as i32 + r2[xp] as i32)
        - (r0[xm] as i32 + 2 * r0[x] as i32 + r0[xp] as i32);
    (gx as f32, gy as f32)
}

// ---------------------------------------------------------------------------

///
/// `t[i]`, `t[i + ch]`, `t[i + ch * 2]` are the left/centre/right taps of output
/// element `i`, so a row must be `out.len() + 2 * ch` long. Passing the halo
/// in (instead of three pre-sliced tap windows) is what lets the kernel drop
/// the eight per-iteration bounds checks of the old shape.
#[inline(always)]
fn assert_sobel_shape(t: &[i16], m: &[i16], b: &[i16], len: usize, ch: usize) {
    assert_eq!(t.len(), len + 2 * ch);
    assert_eq!(m.len(), len + 2 * ch);
    assert_eq!(b.len(), len + 2 * ch);
}

/// Scalar Sobel tail for the remaining `len % n` elements.
///
/// Row `j` already carries the halo, so element `i` reads columns
/// `(j - 1) * ch` away from output element `i`.
#[inline(always)]
fn sobel_tail(t: &[i16], m: &[i16], b: &[i16], ch: usize, i: usize) -> (i16, i16) {
    let (t0, t1, t2) = (t[i], t[i + ch], t[i + 2 * ch]);
    let (m0, m2) = (m[i], m[i + 2 * ch]);
    let (b0, b1, b2) = (b[i], b[i + ch], b[i + 2 * ch]);
    let gx = (t2 as i32 + 2 * m2 as i32 + b2 as i32) - (t0 as i32 + 2 * m0 as i32 + b0 as i32);
    let gy = (b0 as i32 + 2 * b1 as i32 + b2 as i32) - (t0 as i32 + 2 * t1 as i32 + t2 as i32);
    (gx as i16, gy as i16)
}

/// 3x3 Sobel straight into `f32` output rows (OpenCV's `CV_32F` result).
///
/// The `i16 -> f32` conversion is exact and fused, so no `i16` accumulator array
/// is written to memory.
#[simd]
pub(crate) fn sobel3_f32<S: Simd>(
    simd: S,
    t: &[i16],
    m: &[i16],
    b: &[i16],
    ch: usize,
    ox: &mut [f32],
    oy: &mut [f32],
) -> usize {
    assert_eq!(ox.len(), oy.len());
    assert_sobel_shape(t, m, b, ox.len(), ch);
    let n = <S as Simd>::i16s::LEN;
    let f = <S as Simd>::f32s::LEN;
    debug_assert_eq!(n, 2 * f);
    let len = ox.len();
    let body = len / n * n;

    for i in (0..body).step_by(n) {
        let l = |r: &[i16], k: usize| <S as Simd>::i16s::from_slice(simd, &r[i + k..i + k + n]);
        let (t0, t1, t2) = (l(t, 0), l(t, ch), l(t, 2 * ch));
        let (m0, m2) = (l(m, 0), l(m, 2 * ch));
        let (b0, b1, b2) = (l(b, 0), l(b, ch), l(b, 2 * ch));
        // gx = (t2 + 2*m2 + b2) - (t0 + 2*m0 + b0), written as three
        // differences so the weights cost one shift instead of two multiplies.
        let gx = (t2 - t0) + ((m2 - m0) << 1u32) + (b2 - b0);
        // gy = (b0 + 2*b1 + b2) - (t0 + 2*t1 + t2)
        let gy = (b0 - t0) + ((b1 - t1) << 1u32) + (b2 - t2);
        let (gxa, gxb) = gx.widen();
        let (gya, gyb) = gy.widen();
        let (xo0, xo1) = ox[i..i + n].split_at_mut(f);
        let (yo0, yo1) = oy[i..i + n].split_at_mut(f);
        let xa: <S as Simd>::f32s = SimdCvtFloat::float_from(gxa);
        let xb: <S as Simd>::f32s = SimdCvtFloat::float_from(gxb);
        let ya: <S as Simd>::f32s = SimdCvtFloat::float_from(gya);
        let yb: <S as Simd>::f32s = SimdCvtFloat::float_from(gyb);
        xa.store_slice(xo0);
        xb.store_slice(xo1);
        ya.store_slice(yo0);
        yb.store_slice(yo1);
    }
    for i in body..len {
        let (gx, gy) = sobel_tail(t, m, b, ch, i);
        ox[i] = gx as f32;
        oy[i] = gy as f32;
    }
    len
}

/// 3x3 Sobel into `i16` output rows (`CV_16S` result). Exact copy.
#[simd]
pub(crate) fn sobel3_i16<S: Simd>(
    simd: S,
    t: &[i16],
    m: &[i16],
    b: &[i16],
    ch: usize,
    ox: &mut [i16],
    oy: &mut [i16],
) -> usize {
    assert_eq!(ox.len(), oy.len());
    assert_sobel_shape(t, m, b, ox.len(), ch);
    let n = <S as Simd>::i16s::LEN;
    let len = ox.len();
    let body = len / n * n;

    for i in (0..body).step_by(n) {
        let l = |r: &[i16], k: usize| <S as Simd>::i16s::from_slice(simd, &r[i + k..i + k + n]);
        let (t0, t1, t2) = (l(t, 0), l(t, ch), l(t, 2 * ch));
        let (m0, m2) = (l(m, 0), l(m, 2 * ch));
        let (b0, b1, b2) = (l(b, 0), l(b, ch), l(b, 2 * ch));
        let gx = (t2 - t0) + ((m2 - m0) << 1u32) + (b2 - b0);
        let gy = (b0 - t0) + ((b1 - t1) << 1u32) + (b2 - t2);
        gx.store_slice(&mut ox[i..i + n]);
        gy.store_slice(&mut oy[i..i + n]);
    }
    for i in body..len {
        let (gx, gy) = sobel_tail(t, m, b, ch, i);
        ox[i] = gx;
        oy[i] = gy;
    }
    len
}

/// Halo window for the 3-tap Sobel: the `len + 2 * tap_stride` elements of `row`
/// starting `tap_stride` before the first output, so that element `i` of the
/// window has its left tap at `i`, its centre tap at `i + tap_stride` and its
/// right tap at `i + 2 * tap_stride`.
#[inline(always)]
pub(crate) fn sobel_row(row: &[i16], lo: usize, hi: usize, tap_stride: usize) -> &[i16] {
    &row[lo - tap_stride..hi + tap_stride]
}
