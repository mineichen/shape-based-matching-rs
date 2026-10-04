//! Pluggable filter backends for the matching hot path.
//!
//! [`Backend`] is the algorithm trait ("TAlgo"): the four image operations the
//! matcher needs. Implementations receive `&mut self` so they can reuse
//! internal scratch buffers across calls in the future.
//!
//! One implementation per cargo feature, so the type names below are plain
//! code spans: only the enabled ones exist in a given build.
//!
//! - `FearlessSimdBackend`: the pure-Rust implementation built on
//!   [`fearless_simd`](https://docs.rs/fearless_simd) portable SIMD in
//!   `crate::filters::fsimd`, with runtime dispatch. Requires the
//!   `fearless-simd` feature.
//! - `OpenCvBackend`: delegates to `opencv::imgproc` — the previous behavior.
//!   Requires the `opencv` cargo feature.
//!
//! At least one backend feature must be enabled, otherwise the crate does not
//! compile: there is no fallback implementation behind the feature flags.
//! [`DefaultBackend`] names the one the matcher uses when none is requested:
//! `FearlessSimdBackend` if `fearless-simd` is on, else `OpenCvBackend` if
//! `opencv` is on.

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

#[cfg(not(any(feature = "fearless-simd", feature = "opencv")))]
compile_error!(
    "graph_matching needs at least one filter backend. Enable one of:\n\
     - `fearless-simd` (default): pure-Rust `fearless_simd` filters, the fastest backend\n\
     - `opencv`       : delegate the filters to `opencv::imgproc`\n\
     For example: `cargo build --features fearless-simd`."
);

/// Backend built on [`fearless_simd`] portable SIMD with runtime dispatch.
///
/// Faster than `opencv::imgproc` because it vectorizes the color paths without
/// deinterleaving: all channels share the same tap offsets, scaled by the channel
/// count.
///
/// Requires the `fearless-simd` cargo feature.
#[cfg(feature = "fearless-simd")]
#[derive(Debug, Clone, Default)]
pub struct FearlessSimdBackend {
    /// Reused across calls, so the whole-image intermediates are not
    /// reallocated and re-zeroed on every filter invocation.
    scratch: crate::filters::fsimd::Scratch,
}

#[cfg(feature = "fearless-simd")]
impl Backend for FearlessSimdBackend {
    #[inline]
    fn gaussian_blur_7x7(&mut self, src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
        filters::fsimd::gaussian_blur_7x7(&mut self.scratch, src, dst)
    }

    #[inline]
    fn sobel_grayscale(&mut self, src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()> {
        filters::fsimd::sobel_grayscale(&mut self.scratch, src, dx, dy)
    }

    #[inline]
    fn sobel_color_i16(&mut self, src: &Mat, dx: &mut Mat, dy: &mut Mat) -> opencv::Result<()> {
        filters::fsimd::sobel_color_i16(&mut self.scratch, src, dx, dy)
    }

    #[inline]
    fn pyr_down(&mut self, src: &Mat, dst: &mut Mat) -> opencv::Result<()> {
        filters::fsimd::pyr_down(&mut self.scratch, src, dst)
    }
}

/// Backend delegating to `opencv::imgproc` (the pre-port implementation).
///
/// Requires the `opencv` cargo feature.
#[cfg(feature = "opencv")]
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCvBackend;

#[cfg(feature = "opencv")]
mod opencv_impl {
    use super::{Backend, OpenCvBackend};
    use opencv::{
        core::{self, Mat},
        imgproc,
    };

    impl Backend for OpenCvBackend {
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

/// The backend [`crate::Detector`] and [`crate::DetectorBuilder`] use when no
/// other one is requested: `FearlessSimdBackend` if the `fearless-simd`
/// feature is enabled, else `OpenCvBackend` if the `opencv` feature is.
#[cfg(feature = "fearless-simd")]
pub type DefaultBackend = FearlessSimdBackend;

#[cfg(all(not(feature = "fearless-simd"), feature = "opencv"))]
pub type DefaultBackend = OpenCvBackend;
