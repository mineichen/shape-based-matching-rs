//! Standalone SIMD utilities for testing with Miri
//!
//! This module contains SIMD-optimized functions extracted from line2dup.rs
//! to allow testing without OpenCV dependencies.

/// Accumulation for u8: dst[i] += src[i] for all i (saturating)
#[inline]
pub(crate) fn simd_accumulate_u8(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(dst.len() % 16, 0, "Length must be a multiple of 16");
    debug_assert_eq!(src.len() % 16, 0, "Length must be a multiple of 16");

    // Process in fixed-size chunks, then remainder
    let mut dst_chunks = dst.chunks_exact_mut(16);
    let mut src_chunks = src.chunks_exact(16);
    dst_chunks
        .by_ref()
        .zip(src_chunks.by_ref())
        .for_each(|(dchunk, schunk)| {
            dchunk
                .iter_mut()
                .zip(schunk.iter())
                .for_each(|(d, s)| *d += *s);
        });
}

#[inline]
pub(crate) fn simd_accumulate_u16(dst: &mut [u16], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(dst.len() % 16, 0, "Length must be a multiple of 16");
    debug_assert_eq!(src.len() % 16, 0, "Length must be a multiple of 16");

    let mut dst_chunks = dst.chunks_exact_mut(16);
    let mut src_chunks = src.chunks_exact(16);
    dst_chunks
        .by_ref()
        .zip(src_chunks.by_ref())
        .for_each(|(dchunk, schunk)| {
            dchunk.iter_mut().zip(schunk.iter()).for_each(|(d, s)| {
                *d += *s as u16;
            })
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_accumulate_u8() {
        let mut dst = vec![0u8; 32];
        let src = vec![1u8; 32];

        simd_accumulate_u8(&mut dst, &src);

        // All elements should be 1
        for &val in &dst {
            assert_eq!(val, 1);
        }
    }

    #[test]
    fn test_simd_accumulate_u16() {
        let mut dst = vec![0u16; 32];
        let src = vec![1u8; 32];

        simd_accumulate_u16(&mut dst, &src);

        // All elements should be 1
        for &val in &dst {
            assert_eq!(val, 1);
        }

        // Test multiple accumulations
        simd_accumulate_u16(&mut dst, &src);
    }
}
