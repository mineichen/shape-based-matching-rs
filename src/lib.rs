pub mod backend;
pub mod filters;
mod image_buffer;
mod line2dup;
mod match_entry;
mod matches;
mod pyramid;
mod simd_utils;

#[cfg(feature = "fearless-simd")]
pub use backend::FearlessSimd;
#[cfg(feature = "opencv")]
pub use backend::OpenCv;
pub use backend::{Backend, Native};
pub use line2dup::{BuilderError, Detector, DetectorBuilder, Feature, TemplateConfigHandle};
pub use match_entry::Match;
pub use matches::Matches;

/// Marker unit for image coordinates in this crate.
///
/// Public convention: the top-left subpixel of the top-left pixel is
/// `(0, 0)`, and pixel `N`'s center is `N + 0.5`. Every float position
/// produced by this crate is a pixel center (i.e. `X.5`); caller-supplied
/// pivots (rotation/scale centers) are pixel indices ([`Point2i`]) — the
/// pivot is the center of that pixel. The matcher's internals keep
/// working on integer pixel indices and convert at the API boundary.
pub enum ImageSpace {}

/// Fixed-point image-space point: pixel `N`'s center is `N + 0.5`.
/// This is the public position type for [`Match::pos`] and
/// [`Match::center_point`]: until subpixel accuracy is implemented every
/// position is exactly `X.5`. Backed by `fixed::types::U20F12`
/// (unsigned, 12 fractional bits — exactly representable `X.5`).
pub type Point2Fixed = euclid::Point2D<fixed::types::U20F12, ImageSpace>;

/// Fixed-point image-space translation, the offset-counterpart of
/// [`PointFixed`].
pub type Vector2Fixed = euclid::Vector2D<fixed::types::U20F12, ImageSpace>;

/// Integer pixel-index point. Template pivots (rotation/scale centers)
/// take this type: the pivot is the center of the given pixel, i.e. the
/// public image coordinate of pixel `N` is `N as f32 + 0.5`. Features
/// and all matcher-internal positions are discrete pixel indices; the
/// internal float <-> pixel bridge is `v as i32`, never `v.round()`.
pub type Point2i = euclid::Point2D<i32, ImageSpace>;

/// Integer pixel-index translation ([`euclid::Vector2D<i32>`]); the
/// offset-counterpart of [`Point2i`].
pub type Vector2i = euclid::Vector2D<i32, ImageSpace>;

/// Pixel index -> public image coordinates: pixel `N`'s center is
/// `N + 0.5`. Counterpart to [`to_pixel_pt`].
///
/// Build [`PointFixed`] positions from [`Point2i`] with this helper
/// instead of `cast::<f32>() + 0.5`.
#[inline(always)]
fn from_pixel_pt(p: Point2i) -> Point2Fixed {
    debug_assert!(
        p.x >= 0 && p.y >= 0,
        "from_pixel_pt needs non-negative indices for U20F12, got {p:?}"
    );
    const HALF: fixed::types::U20F12 = fixed::types::U20F12::from_bits(1 << 11);
    Point2Fixed::new(
        fixed::types::U20F12::from_num(p.x as u32) + HALF,
        fixed::types::U20F12::from_num(p.y as u32) + HALF,
    )
}
