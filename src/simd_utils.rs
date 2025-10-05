//! Standalone SIMD utilities for testing with Miri
//! 
//! This module contains SIMD-optimized functions extracted from line2dup.rs
//! to allow testing without OpenCV dependencies.

/// SIMD-optimized accumulation for u8: dst[i] += src[i] for all i
#[inline]
pub(crate) unsafe fn simd_accumulate_u8(dst: *mut u8, src: *const u8, len: usize) {
    let mut i = 0;

    // Process 16 bytes at a time using SIMD (if available)
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            use std::arch::x86_64::*;
            
            while i + 16 <= len {
                unsafe {
                    let src_v = _mm_loadu_si128(src.add(i) as *const __m128i);
                    let dst_v = _mm_loadu_si128(dst.add(i) as *const __m128i);
                    let result = _mm_adds_epu8(dst_v, src_v); // Saturating add
                    _mm_storeu_si128(dst.add(i) as *mut __m128i, result);
                }
                i += 16;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        
        while i + 16 <= len {
            unsafe {
                let src_v = vld1q_u8(src.add(i));
                let dst_v = vld1q_u8(dst.add(i));
                let result = vqaddq_u8(dst_v, src_v); // Saturating add
                vst1q_u8(dst.add(i) as *mut u8, result);
            }
            i += 16;
        }
    }

    // Scalar fallback for remaining elements
    while i < len {
        unsafe {
            *dst.add(i) = (*dst.add(i)).saturating_add(*src.add(i));
        }
        i += 1;
    }
}

/// SIMD-optimized accumulation for u16: dst[i] += src[i] (u8 converted to u16)
/// dst is u16 array, src is u8 array
#[inline]
pub(crate) unsafe fn simd_accumulate_u16(dst: *mut u16, src: *const u8, dst_offset: usize, src_offset: usize, len: usize) {
    unsafe {
        let dst = dst.add(dst_offset);
        let src = src.add(src_offset);
        let mut i = 0;

        // Process 16 bytes at a time using SIMD (if available)
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                use std::arch::x86_64::*;
                
                while i + 16 <= len {
                    // Load 16 u8 values from src
                    let src_v = _mm_loadu_si128(src.add(i) as *const __m128i);
                    
                    // Split into low and high 8 bytes, convert to u16
                    let zero = _mm_setzero_si128();
                    let src_lo = _mm_unpacklo_epi8(src_v, zero); // First 8 u8 -> 8 u16
                    let src_hi = _mm_unpackhi_epi8(src_v, zero); // Last 8 u8 -> 8 u16
                    
                    // Load corresponding u16 values from dst
                    let dst_lo = _mm_loadu_si128(dst.add(i) as *const __m128i);
                    let dst_hi = _mm_loadu_si128(dst.add(i + 8) as *const __m128i);
                    
                    // Saturating add
                    let result_lo = _mm_adds_epu16(dst_lo, src_lo);
                    let result_hi = _mm_adds_epu16(dst_hi, src_hi);
                    
                    // Store results
                    _mm_storeu_si128(dst.add(i) as *mut __m128i, result_lo);
                    _mm_storeu_si128(dst.add(i + 8) as *mut __m128i, result_hi);
                    
                    i += 16;
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            
            while i + 16 <= len {
                // Load 16 u8 values from src
                let src_v = vld1q_u8(src.add(i));
                
                // Convert to two u16 vectors (low and high 8 bytes)
                let src_lo = vmovl_u8(vget_low_u8(src_v));
                let src_hi = vmovl_u8(vget_high_u8(src_v));
                
                // Load corresponding u16 values from dst
                let dst_lo = vld1q_u16(dst.add(i));
                let dst_hi = vld1q_u16(dst.add(i + 8));
                
                // Saturating add
                let result_lo = vqaddq_u16(dst_lo, src_lo);
                let result_hi = vqaddq_u16(dst_hi, src_hi);
                
                // Store results
                vst1q_u16(dst.add(i) as *mut u16, result_lo);
                vst1q_u16(dst.add(i + 8) as *mut u16, result_hi);
                
                i += 16;
            }
        }

        // Scalar fallback for remaining elements
        while i < len {
            *dst.add(i) = (*dst.add(i)).saturating_add(*src.add(i) as u16);
            i += 1;
        }
    }
}

/// Scalar fallback implementation for u8 accumulation
#[inline]
pub(crate) fn scalar_accumulate_u8(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = d.saturating_add(*s);
    }
}

/// Scalar fallback implementation for u16 accumulation
#[inline]
pub(crate) fn scalar_accumulate_u16(dst: &mut [u16], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = d.saturating_add(*s as u16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_accumulate_u8() {
        let mut dst = vec![0u8; 32];
        let src = vec![1u8; 32];
        
        unsafe {
            simd_accumulate_u8(dst.as_mut_ptr(), src.as_ptr(), 32);
        }
        
        // All elements should be 1
        for &val in &dst {
            assert_eq!(val, 1);
        }
        
        // Test saturating behavior
        dst.fill(255);
        unsafe {
            simd_accumulate_u8(dst.as_mut_ptr(), src.as_ptr(), 32);
        }
        
        // All elements should still be 255 (saturated)
        for &val in &dst {
            assert_eq!(val, 255);
        }
    }

    #[test]
    fn test_simd_accumulate_u16() {
        let mut dst = vec![0u16; 32];
        let src = vec![1u8; 32];
        
        unsafe {
            simd_accumulate_u16(dst.as_mut_ptr(), src.as_ptr(), 0, 0, 32);
        }
        
        // All elements should be 1
        for &val in &dst {
            assert_eq!(val, 1);
        }
        
        // Test multiple accumulations
        unsafe {
            simd_accumulate_u16(dst.as_mut_ptr(), src.as_ptr(), 0, 0, 32);
        }
        
        // All elements should be 2
        for &val in &dst {
            assert_eq!(val, 2);
        }
        
        // Test saturating behavior
        dst.fill(65535);
        unsafe {
            simd_accumulate_u16(dst.as_mut_ptr(), src.as_ptr(), 0, 0, 32);
        }
        
        // All elements should still be 65535 (saturated)
        for &val in &dst {
            assert_eq!(val, 65535);
        }
    }

    #[test]
    fn test_scalar_vs_simd_u8() {
        let mut dst_simd = vec![0u8; 64];
        let mut dst_scalar = vec![0u8; 64];
        let src = vec![5u8; 64];
        
        // Test SIMD version
        unsafe {
            simd_accumulate_u8(dst_simd.as_mut_ptr(), src.as_ptr(), 64);
        }
        
        // Test scalar version
        scalar_accumulate_u8(&mut dst_scalar, &src);
        
        // Results should be identical
        assert_eq!(dst_simd, dst_scalar);
    }

    #[test]
    fn test_scalar_vs_simd_u16() {
        let mut dst_simd = vec![0u16; 64];
        let mut dst_scalar = vec![0u16; 64];
        let src = vec![3u8; 64];
        
        // Test SIMD version
        unsafe {
            simd_accumulate_u16(dst_simd.as_mut_ptr(), src.as_ptr(), 0, 0, 64);
        }
        
        // Test scalar version
        scalar_accumulate_u16(&mut dst_scalar, &src);
        
        // Results should be identical
        assert_eq!(dst_simd, dst_scalar);
    }

    #[test]
    fn test_edge_cases_u8() {
        // Test with small arrays
        let mut dst = vec![0u8; 3];
        let src = vec![100u8; 3];
        
        unsafe {
            simd_accumulate_u8(dst.as_mut_ptr(), src.as_ptr(), 3);
        }
        
        for &val in &dst {
            assert_eq!(val, 100);
        }
        
        // Test with single element
        let mut dst = vec![0u8; 1];
        let src = vec![200u8; 1];
        
        unsafe {
            simd_accumulate_u8(dst.as_mut_ptr(), src.as_ptr(), 1);
        }
        
        assert_eq!(dst[0], 200);
    }

    #[test]
    fn test_edge_cases_u16() {
        // Test with small arrays
        let mut dst = vec![0u16; 3];
        let src = vec![50u8; 3];
        
        unsafe {
            simd_accumulate_u16(dst.as_mut_ptr(), src.as_ptr(), 0, 0, 3);
        }
        
        for &val in &dst {
            assert_eq!(val, 50);
        }
        
        // Test with single element
        let mut dst = vec![0u16; 1];
        let src = vec![150u8; 1];
        
        unsafe {
            simd_accumulate_u16(dst.as_mut_ptr(), src.as_ptr(), 0, 0, 1);
        }
        
        assert_eq!(dst[0], 150);
    }

    #[test]
    fn test_large_arrays() {
        // Test with large arrays to stress test SIMD
        let size = 10000;
        let mut dst = vec![0u8; size];
        let src = vec![1u8; size];
        
        unsafe {
            simd_accumulate_u8(dst.as_mut_ptr(), src.as_ptr(), size);
        }
        
        // All elements should be 1
        for &val in &dst {
            assert_eq!(val, 1);
        }
    }

    #[test]
    fn test_memory_alignment() {
        // Test with unaligned memory to ensure SIMD handles it correctly
        let mut data = vec![0u8; 100];
        let src = vec![2u8; 100];
        
        // Use offset to create unaligned access
        unsafe {
            simd_accumulate_u8(data.as_mut_ptr().add(1), src.as_ptr().add(1), 99);
        }
        
        // Check that elements 1-99 were updated
        for i in 1..100 {
            assert_eq!(data[i], 2);
        }
        // Element 0 should remain 0
        assert_eq!(data[0], 0);
    }
}
