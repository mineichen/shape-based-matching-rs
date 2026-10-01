//! Pluggable filter backends for the matching hot path.
//!
//! [`Backend`] is the algorithm trait ("TAlgo"): the four image operations the
//! matcher needs. Implementations receive `&mut self` so they can reuse
//! internal scratch buffers across calls in the future.
//!
//! - [`Native`]: the pure-Rust SIMD implementation in [`crate::filters`]
//!   (default).
//! - [`OpenCv`]: delegates to `opencv::imgproc` — the previous behavior.
//!   Requires the `opencv` cargo feature.

use opencv::core::Mat;

use crate::filters;

/// The four image operations the matcher needs from its filter backend.
pub trait Backend {
    /// 7x7 Gaussian blur (`sigma = 0` auto), `BORDER_REPLICATE`, `CV_8UC1`/`CV_8UC3`.
    fn gaussian_blur_7x7(&mut self, src: &Mat, dst: &mut Mat) -> opencv::Result<()>;

    /// 3x3 Sobel on single-channel `u8` with `CV_32F` `dx`/`dy`, `BORDER_REPLICATE`.
    fn sobel_grayscale(&mut self, src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()>;

    /// 3x3 Sobel on 3-channel `u8` with `CV_16SC3` `dx`/`dy`, `BORDER_REPLICATE`.
    fn sobel_color_i16(&mut self, src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()>;

    /// 5x5 Gaussian + 2x decimate with `BORDER_REFLECT_101`, `CV_8UC1`/`CV_8UC3`.
    fn pyr_down(&mut self, src: &Mat, dst: &mut Mat) -> opencv::Result<()>;
}

/// Default backend: pure-Rust SIMD filters (see [`crate::filters`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct Native;

impl Backend for Native {
    #[inline]
    fn gaussian_blur_7x7(&mut self, src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
        filters::gaussian_blur_7x7(src, dst)
    }

    #[inline]
    fn sobel_grayscale(&mut self, src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()> {
        filters::sobel_grayscale(src, dx, dy)
    }

    #[inline]
    fn sobel_color_i16(&mut self, src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()> {
        filters::sobel_color_i16(src, dx, dy)
    }

    #[inline]
    fn pyr_down(&mut self, src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
        filters::pyr_down(src, dst)
    }
}

/// Backend delegating to `opencv::imgproc` (the pre-port implementation).
///
/// Requires the `opencv` cargo feature.
#[cfg(feature = "opencv")]
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCv;

#[cfg(feature = "opencv")]
mod opencv_impl {
    use super::{Backend, OpenCv};
    use opencv::{
        core::{self, Mat},
        imgproc,
    };

    impl Backend for OpenCv {
        fn gaussian_blur_7x7(&mut self, src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
            imgproc::gaussian_blur(
                src,
                dst,
                core::Size::new(7, 7),
                0.0,
                0.0,
                core::BORDER_REPLICATE,
                core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )
        }

        fn sobel_grayscale(&mut self, src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()> {
            imgproc::sobel(
                src,
                dx,
                core::CV_32F,
                1,
                0,
                3,
                1.0,
                0.0,
                core::BORDER_REPLICATE,
            )?;
            imgproc::sobel(
                src,
                dy,
                core::CV_32F,
                0,
                1,
                3,
                1.0,
                0.0,
                core::BORDER_REPLICATE,
            )
        }

        fn sobel_color_i16(&mut self, src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()> {
            imgproc::sobel(
                src,
                dx,
                core::CV_16S,
                1,
                0,
                3,
                1.0,
                0.0,
                core::BORDER_REPLICATE,
            )?;
            imgproc::sobel(
                src,
                dy,
                core::CV_16S,
                0,
                1,
                3,
                1.0,
                0.0,
                core::BORDER_REPLICATE,
            )
        }

        fn pyr_down(&mut self, src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
            imgproc::pyr_down_def(src, dst)
        }
    }
}
