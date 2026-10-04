use fearless_simd::prelude::*;
use fearless_simd::{Level, dispatch};
use fearless_simd_macros::simd;
use opencv::{core::Mat, prelude::*};

use super::{Out, ROUND, Rows, SHIFT, Scratch, grow, row, widen_u8_to_u16, windows, write_back};
use crate::filters::{check_src_8u, ensure_dst, reflect101};

/// 5x5 `pyrDown` kernel as `u16` taps; raw integer sums with a single rounding
/// at the very end, exactly like OpenCV's `pyrDown_`.
const PYR_K: [u16; 5] = [1, 4, 6, 4, 1];
const PYR_R: usize = 2;

// ---------------------------------------------------------------------------
// pyrDown: 5x5 Gaussian + 2x decimate, `BORDER_REFLECT_101`
// ---------------------------------------------------------------------------

/// `pyrDown` on `CV_8UC1`/`CV_8UC3`; output is `((rows+1)/2) x ((cols+1)/2)`.
pub fn pyr_down(s: &mut Scratch, src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
    let (rows_i, cols_i) = check_src_8u(src)?;
    let orows = (rows_i + 1) / 2;
    let ocols = (cols_i + 1) / 2;
    ensure_dst(dst, orows, ocols, src.typ())?;
    let level = Level::new();
    let ch = src.channels() as usize;
    let rows = rows_i as usize;
    let rb = cols_i as usize * ch;
    let ob = ocols as usize * ch;
    let src = Rows::new(src, rows * rb, ch)?;
    let staged = {
        let mut out = Out::new(dst, orows as usize * ob)?;

        // Horizontal pass: RAW `u16` sums over the full row. Deferring the
        // single rounding to the vertical pass is what makes this
        // bit-compatible with OpenCV's `pyrDown_` (whose horizontal pass also
        // truncates to integer).
        let Scratch {
            tmp_u16,
            ring_u16,
            acc_u16,
            ..
        } = s;
        // Output row `oy` only ever reads input rows `2 * oy - 2 .. 2 * oy + 2`,
        // so the horizontal results live in a five-row ring instead of a
        // whole-image buffer: on a 3MP image that is ~6MB less of write and
        // ~6MB less of read traffic per call, and the ring is reused for every
        // output row.
        let nring = 2 * PYR_R + 1;
        let fused = ch == 1;
        // Fused gray: the horizontal pass runs on the even and odd input columns
        // separately, so it computes only the columns the output keeps and
        // writes the decimated row straight into the ring. Color needs *every*
        // column horizontally (output pixel `ox`, channel `c` reads input bytes
        // `6 * ox + c + 3 * t`), so its ring stays full width and the vertical
        // pass decimates - a fused color variant measured slower, because the
        // byte gather costs more than the halved vertical MAC saves.
        let ring = grow(tmp_u16, nring * if fused { ob } else { rb });
        // Even/odd halves of the widened row (fused) or the widened row itself.
        let half = rb / 2 + PYR_R + 2;
        let scratch_a = grow(ring_u16, if fused { half } else { rb });
        // Odd halves (fused) or the vertical `u16` accumulator (full width).
        let scratch_b = grow(acc_u16, if fused { half } else { rb });
        let lo = (PYR_R * ch).min(rb);
        let hi = rb.saturating_sub(PYR_R * ch);
        // Outputs whose five horizontal taps all land inside the row are the ones
        // the SIMD path can take; the two or three at each end need the reflected
        // column taps and are done scalar.
        let p_start = (PYR_R * ch).div_ceil(2).min(ob);
        let mut p_end = p_start;
        while p_end < ob && pyr_src_index(p_end, ch) + PYR_R * ch < rb {
            p_end += 1;
        }
        let dst_all = out.bytes();
        let mut loaded = 0usize;
        for oy in 0..orows as usize {
            // Rows `0..=loaded` are in the ring; pull in whatever this output row
            // still needs. Reflected taps always map onto the last few loaded
            // rows, so the five-row ring is enough for every row.
            let need = (oy * 2 + PYR_R).min(rows - 1);
            while loaded <= need {
                let y = loaded;
                let slot = (y % nring) * if fused { ob } else { rb };
                let row = row(src.data(), rb, y);
                let t = &mut ring[slot..slot + if fused { ob } else { rb }];
                if fused {
                    // RAW `u16` sums, decimated columns only. Deferring the single
                    // rounding to the vertical pass is what makes this
                    // bit-compatible with OpenCV's `pyrDown_` (whose horizontal
                    // pass also truncates to integer).
                    dispatch!(level, simd => {
                        widen_deinterleave(simd, row, scratch_a, scratch_b)
                    });
                    let e = &*scratch_a;
                    let o = &*scratch_b;
                    let n = p_end.saturating_sub(1);
                    if n > p_start {
                        // Taps of output `p`: `E[p-1]`, `O[p-1]`, `E[p]`, `O[p]`,
                        // `E[p+1]`, which is the symmetric 5-tap MAC over two
                        // contiguous arrays - no gather, no discarded column.
                        let w: [&[u16]; 5] = [
                            &e[p_start - 1..p_start - 1 + n],
                            &o[p_start - 1..p_start - 1 + n],
                            &e[p_start..p_start + n],
                            &o[p_start..p_start + n],
                            &e[p_start + 1..p_start + 1 + n],
                        ];
                        dispatch!(level, simd => {
                            mac_raw_u16::<5, _>(simd, w, &PYR_K, &mut t[p_start..p_end])
                        });
                    }
                } else {
                    // Color: the horizontal sums are needed at *every* column
                    // (output pixel `ox`, channel `c` reads input bytes
                    // `6 * ox + c + 3 * t`), so the horizontal pass cannot skip
                    // anything, and the ring keeps the full row width.
                    dispatch!(level, simd => widen_u8_to_u16(simd, row, scratch_a));
                    if hi > lo {
                        let w = windows::<5>(scratch_a, lo, ch, PYR_R, hi - lo);
                        dispatch!(level, simd => {
                            mac_raw_u16::<5, _>(simd, w, &PYR_K, &mut t[lo..hi])
                        });
                    }
                    for x in 0..lo {
                        t[x] = pyr_h_tap(row, x, rb, ch);
                    }
                    for x in hi.max(lo)..rb {
                        t[x] = pyr_h_tap(row, x, rb, ch);
                    }
                }
                if fused {
                    // Reflected column taps: one output at each end.
                    for p in 0..p_start {
                        t[p] = pyr_h_tap(row, pyr_src_index(p, ch), rb, ch);
                    }
                    for p in p_end..ob {
                        t[p] = pyr_h_tap(row, pyr_src_index(p, ch), rb, ch);
                    }
                }
                loaded += 1;
            }

            // Fused gray: the vertical pass reads the decimated columns, so it is
            // a 5-tap MAC with the single rounding fused into the `u8` store.
            // Color: full-width MAC into `scratch_b`, then the column
            // decimation on the way out.
            let cy = (oy * 2) as i32;
            let d = &mut dst_all[oy * ob..(oy + 1) * ob];
            if fused {
                let w: [&[u16]; 5] = std::array::from_fn(|j| {
                    let r = (reflect101(cy + j as i32 - PYR_R as i32, rows_i) % nring) * ob;
                    &ring[r..r + ob]
                });
                dispatch!(level, simd => {
                    pyr_v_narrow::<5, _>(simd, w, &PYR_K, ROUND, SHIFT, d)
                });
            } else {
                // Color: full-width MAC, single rounding and the column
                // decimation, fused - see `pyr_v_decimate`.
                let w: [&[u16]; 5] = std::array::from_fn(|j| {
                    let r = (reflect101(cy + j as i32 - PYR_R as i32, rows_i) % nring) * rb;
                    &ring[r..r + rb]
                });
                dispatch!(level, simd => {
                    pyr_v_decimate::<5, _>(simd, w, &PYR_K, ROUND, SHIFT, d, ch)
                });
            }
        }
        out.take_staged()
    };
    match staged {
        Some(v) => write_back(dst, v, orows as usize, ob, ch),
        None => Ok(()),
    }
}

/// Color `pyr_down` vertical pass: full-width 5-tap MAC, the single rounding,
/// the column decimation and the `u8` store, all in one kernel.
///
/// The horizontal sums of every column are needed by color (output pixel `ox`,
/// channel `c` reads input bytes `6 * ox + c + 3 * t`), so the ring stays full
/// width and this pass has to throw half the columns away. Doing that on
/// registers - round, shift, `narrow` to bytes, one byte-level gather - is
/// cheaper than a separate decimation pass over a `u16` accumulator array.
///
/// The gather works exactly like [`decimate_u16`]: `narrow` gives the byte
/// planes of 32 accumulator values, `swizzle_dyn_precise` gathers the ones the
/// 16 outputs need, and the `u16` view of the result is stored directly. The
/// store writes 32 bytes but only fills the first 16; the next iteration
/// overwrites the upper half, and the tail is finished scalar.
///
/// Rows are `rb` long, `out` is `ob` long, and every row of `rows` has the same
/// length as the ring row.
#[simd]
pub(crate) fn pyr_v_decimate<const N: usize, S: Simd>(
    simd: S,
    rows: [&[u16]; N],
    k: &[u16],
    round: u16,
    shift: u32,
    out: &mut [u8],
    ch: usize,
) {
    for r in &rows[1..] {
        assert_eq!(r.len(), rows[0].len());
    }
    let n = <S as Simd>::u16s::LEN;
    let len = out.len();
    // One `n`-output group needs at most `2 * n` accumulator columns, and the
    // gather mask depends only on `p % ch`.
    let mut masks = [[0u8; 64]; 3];
    for ph in 0..ch {
        for q in 0..n {
            masks[ph][q] = (2 * q + ph - ((ph + q) % ch)) as u8;
        }
    }
    let kmid = <S as Simd>::u16s::splat(simd, k[N / 2]);
    let koff = <S as Simd>::u16s::splat(simd, k[1]);
    let rv = <S as Simd>::u16s::splat(simd, round);

    let mut q = 0usize;
    while q + 2 * n <= len && 2 * q + 2 * n <= rows[0].len() {
        let ph = q % ch;
        let w0 = 2 * q - ph;
        let lm = <S as Simd>::u8s::from_fn(simd, |j| masks[ph][j]);
        // Straight-line, one explicit load per tap: routing this through a closure
        // makes `+`/`*` resolve to out-of-line 128-bit kernel calls instead of
        // inlined `vpaddw`/`vpmullw`.
        let a0 = <S as Simd>::u16s::from_slice(simd, &rows[0][w0..w0 + n]);
        let a1 = <S as Simd>::u16s::from_slice(simd, &rows[1][w0..w0 + n]);
        let a2 = <S as Simd>::u16s::from_slice(simd, &rows[N / 2][w0..w0 + n]);
        let a3 = <S as Simd>::u16s::from_slice(simd, &rows[N - 2][w0..w0 + n]);
        let a4 = <S as Simd>::u16s::from_slice(simd, &rows[N - 1][w0..w0 + n]);
        let b0 = <S as Simd>::u16s::from_slice(simd, &rows[0][w0 + n..w0 + 2 * n]);
        let b1 = <S as Simd>::u16s::from_slice(simd, &rows[1][w0 + n..w0 + 2 * n]);
        let b2 = <S as Simd>::u16s::from_slice(simd, &rows[N / 2][w0 + n..w0 + 2 * n]);
        let b3 = <S as Simd>::u16s::from_slice(simd, &rows[N - 2][w0 + n..w0 + 2 * n]);
        let b4 = <S as Simd>::u16s::from_slice(simd, &rows[N - 1][w0 + n..w0 + 2 * n]);
        let lo = (a0 + a4) + ((a1 + a3) * koff) + a2 * kmid + rv;
        let hi = (b0 + b4) + ((b1 + b3) * koff) + b2 * kmid + rv;
        (lo >> shift)
            .narrow(hi >> shift)
            .swizzle_dyn_precise(lm)
            .store_slice(&mut out[q..q + 2 * n]);
        q += n;
    }
    while q < len {
        let col = 2 * q - (q % ch);
        let mut a = 0u16;
        for j in 0..N {
            a += rows[j][col] * k[j];
        }
        out[q] = ((a + round) >> shift) as u8;
        q += 1;
    }
}

/// Scalar RAW horizontal 5-tap with `BORDER_REFLECT_101`.
#[inline]
fn pyr_h_tap(s: &[u8], x: usize, rb: usize, ch: usize) -> u16 {
    let cols = rb / ch;
    let px = (x / ch) as i32;
    let c = x % ch;
    let mut acc = 0u16;
    for (j, &k) in PYR_K.iter().enumerate() {
        let xi = reflect101(px + j as i32 - PYR_R as i32, cols as i32) * ch + c;
        acc += k * s[xi] as u16;
    }
    acc
}

///
/// Every row of `rows` is one tap, all of length `out.len()`. The taps are
/// symmetric (`k[j] == k[N - 1 - j]`, as `[1, 4, 6, 4, 1]` is), so the MAC is
/// evaluated as `(k0 + kN1) + 4 * (k1 + kN2) + 6 * k2`: the two `4 *` weights
/// become one shift and the two `1 *` weights disappear, which is two
/// instructions cheaper per vector than five multiplies and four adds.
///
/// Returns the number of output elements written.
#[simd]
pub(crate) fn mac_raw_u16<const N: usize, S: Simd>(
    simd: S,
    rows: [&[u16]; N],
    k: &[u16],
    out: &mut [u16],
) -> usize {
    assert_eq!(rows[0].len(), out.len());
    for r in &rows[1..] {
        assert_eq!(r.len(), out.len());
    }
    let n = <S as Simd>::u16s::LEN;
    let len = out.len();
    let body = len / n * n;
    let kmid = <S as Simd>::u16s::splat(simd, k[N / 2]);
    let koff = <S as Simd>::u16s::splat(simd, k[1]);
    let v = |r: &[u16], i: usize| <S as Simd>::u16s::from_slice(simd, &r[i..i + n]);

    for i in (0..body).step_by(n) {
        // (k[0] + k[N-1]) + k[1] * (k[1-row] + k[N-2-row]) + k[N/2] * middle
        let edge = v(rows[0], i) + v(rows[N - 1], i);
        let mid = v(rows[1], i) + v(rows[N - 2], i);
        let acc = edge + (mid * koff) + v(rows[N / 2], i) * kmid;
        acc.store_slice(&mut out[i..i + n]);
    }
    let mut i = body;
    while i < len {
        let mut acc = 0u16;
        for j in 0..N {
            acc += rows[j][i] * k[j];
        }
        out[i] = acc;
        i += 1;
    }
    len
}

/// Column decimation on `u16` values: `dst[p] = src[2 * p - (p % ch)]`.
///
/// This is what lets the `pyrDown` vertical pass read a *pre-decimated* ring, so
/// it neither computes nor stores the columns the output throws away. Sixteen
/// outputs always come from one 32-byte window of `src` (output byte `p` never
/// reads past `2 * p + 1`), and a single byte-level `swizzle_dyn_precise`
/// gathers them, so the compaction costs about one instruction per output.
///
/// The index mask depends only on `p0 % ch`, so at most `ch` masks are built,
/// once per call.
#[simd]
/// Widen a `u8` row to `u16` **and** split it into its even and odd elements.
///
/// Output column `2 * c` of `pyrDown` only ever reads even input columns and
/// output column `2 * c + 1` only ever reads odd ones, so a horizontal pass that
/// works on the two halves separately computes exactly the columns the output
/// needs - half the multiplies of a full-width pass - and writes the decimated
/// row straight into the ring with no compaction step at all.
///
/// `dst_even[c] == src[2 * c]`, `dst_odd[c] == src[2 * c + 1]`. Both halves are
/// filled as far as the source row reaches (`(len + 1) / 2` and `len / 2`), the
/// caller pads the rest.
#[simd]
pub(crate) fn widen_deinterleave<S: Simd>(
    simd: S,
    src: &[u8],
    dst_even: &mut [u16],
    dst_odd: &mut [u16],
) {
    let n = <S as Simd>::u16s::LEN;
    let b = <S as Simd>::u8s::LEN;
    debug_assert_eq!(b, 2 * n);
    assert!(dst_even.len() >= dst_odd.len());
    let body = src.len().min(dst_even.len() * 2) / b * b;
    let mut ev = dst_even[..body / 2].chunks_exact_mut(n);
    let mut od = dst_odd[..body / 2].chunks_exact_mut(n);
    for s in src[..body].chunks_exact(b) {
        let (lo, hi) = <S as Simd>::u8s::from_slice(simd, s).widen();
        let (e, o) = lo.deinterleave(hi);
        e.store_slice(ev.next().expect("chunks_exact_mut"));
        o.store_slice(od.next().expect("chunks_exact_mut"));
    }
    let ne = src.len().div_ceil(2);
    let no = src.len() / 2;
    let (ev, od) = (dst_even.len(), dst_odd.len());
    for (c, d) in dst_even[body / 2..ne.min(ev)].iter_mut().enumerate() {
        *d = src[body + 2 * c] as u16;
    }
    for (c, d) in dst_odd[body / 2..no.min(od)].iter_mut().enumerate() {
        *d = src[body + 2 * c + 1] as u16;
    }
}

///
/// Every row of `rows` is one tap and `out.len()` is the number of decimated
/// columns, so this replaces both a full-width `u16` accumulator pass and the
/// separate decimation pass. As in [`mac_raw_u16`] the symmetric taps are folded
/// into `(k0 + kN1) + 4 * (k1 + kN2) + 6 * k2`, and the `round` add rides along
/// in the same accumulator instead of costing its own instruction.
///
/// Nothing can overflow: the largest possible sum is `255 * 16 * 16 = 65280`,
/// and `+ round` stays below `65536`.
#[simd]
pub(crate) fn pyr_v_narrow<const N: usize, S: Simd>(
    simd: S,
    rows: [&[u16]; N],
    k: &[u16],
    round: u16,
    shift: u32,
    out: &mut [u8],
) {
    assert_eq!(rows[0].len(), out.len());
    for r in &rows[1..] {
        assert_eq!(r.len(), out.len());
    }
    let n = <S as Simd>::u16s::LEN;
    let b = <S as Simd>::u8s::LEN;
    debug_assert_eq!(b, 2 * n);
    let len = out.len();
    let body = len / b * b;
    let kmid = <S as Simd>::u16s::splat(simd, k[N / 2]);
    let koff = <S as Simd>::u16s::splat(simd, k[1]);
    let rv = <S as Simd>::u16s::splat(simd, round);

    // Written as straight-line code with one explicit load per tap: routing this
    // through a closure made `+`/`*` resolve to out-of-line 128-bit kernel calls
    // instead of inlined `vpaddw`/`vpmullw`, which cost more than the whole
    // non-fused vertical pass.
    for i in (0..body).step_by(b) {
        let a0 = <S as Simd>::u16s::from_slice(simd, &rows[0][i..i + n]);
        let a1 = <S as Simd>::u16s::from_slice(simd, &rows[1][i..i + n]);
        let a2 = <S as Simd>::u16s::from_slice(simd, &rows[N / 2][i..i + n]);
        let a3 = <S as Simd>::u16s::from_slice(simd, &rows[N - 2][i..i + n]);
        let a4 = <S as Simd>::u16s::from_slice(simd, &rows[N - 1][i..i + n]);
        let b0 = <S as Simd>::u16s::from_slice(simd, &rows[0][i + n..i + 2 * n]);
        let b1 = <S as Simd>::u16s::from_slice(simd, &rows[1][i + n..i + 2 * n]);
        let b2 = <S as Simd>::u16s::from_slice(simd, &rows[N / 2][i + n..i + 2 * n]);
        let b3 = <S as Simd>::u16s::from_slice(simd, &rows[N - 2][i + n..i + 2 * n]);
        let b4 = <S as Simd>::u16s::from_slice(simd, &rows[N - 1][i + n..i + 2 * n]);
        let lo = (a0 + a4) + ((a1 + a3) * koff) + a2 * kmid + rv;
        let hi = (b0 + b4) + ((b1 + b3) * koff) + b2 * kmid + rv;
        (lo >> shift)
            .narrow(hi >> shift)
            .store_slice(&mut out[i..i + b]);
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
}

/// Source index of `pyr_down` output byte `p`: output pixel `p / ch` is taken
/// from input pixel `2 * (p / ch)`.
#[inline(always)]
pub(crate) fn pyr_src_index(p: usize, ch: usize) -> usize {
    2 * p - (p % ch)
}

#[cfg(test)]
mod tests {
    use fearless_simd::{Level, dispatch};

    use super::pyr_v_decimate;
    const K: [u16; 5] = [1, 4, 6, 4, 1];

    /// The fused color vertical pass must equal the scalar definition
    /// `out[p] = (sum_j K[j] * ring[j][2p - p % ch] + 128) >> 8` for every
    /// channel count and every width (including the gather tail).
    #[test]
    fn color_vertical_matches_scalar() {
        let level = Level::new();
        for ch in [1usize, 3] {
            for rb in [40usize, 64, 96, 130, 200] {
                let cols = rb / ch;
                let ring: Vec<u16> = (0..5 * rb).map(|i| ((i * 7 + 3) % 4081) as u16).collect();
                let ob = cols.div_ceil(2) * ch;
                let mut out = vec![0u8; ob];
                let rows: [&[u16]; 5] = std::array::from_fn(|j| &ring[j * rb..(j + 1) * rb]);
                dispatch!(level, simd => {
                    pyr_v_decimate::<5, _>(simd, rows, &K, 128, 8, &mut out, ch);
                });
                for p in 0..ob {
                    let col = 2 * p - (p % ch);
                    let mut a = 0u16;
                    for j in 0..5 {
                        a += K[j] * ring[j * rb + col];
                    }
                    assert_eq!(out[p], ((a + 128) >> 8) as u8, "ch={ch} rb={rb} p={p}");
                }
            }
        }
    }
}
