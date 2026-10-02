pub mod backend;
pub mod filters;
mod image_buffer;
mod line2dup;
mod match_entry;
mod matches;
mod pyramid;
mod simd_utils;

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

/// Sub-pixel-capable image-space point, backed by [`euclid::Point2D`]
/// (same layout as two `f32`, zero cost). Replaces `opencv::core::Point2f`
/// in this crate's public API so callers can use the library without
/// depending on opencv. Pixel `N`'s center is at `N + 0.5`; `(0, 0)` is
/// the top-left subpixel corner of the image.
pub type Point2f = euclid::Point2D<f32, ImageSpace>;

/// Image-space translation/offset, backed by [`euclid::Vector2D`] (same
/// layout as two `f32`, zero cost). Vectors have no absolute position: they
/// only translate points (`point + vector`), which keeps position-vs-offset
/// confusion out of the type system.
pub type Vector2f = euclid::Vector2D<f32, ImageSpace>;

/// Integer pixel-index point. Template pivots (rotation/scale centers)
/// take this type: the pivot is the center of the given pixel, i.e. the
/// public image coordinate of pixel `N` is `N as f32 + 0.5`. Features
/// and all matcher-internal positions are discrete pixel indices; the
/// internal float <-> pixel bridge is `v as i32`, never `v.round()`.
pub type Point2i = euclid::Point2D<i32, ImageSpace>;

/// Integer pixel-index translation ([`euclid::Vector2D<i32>`]); the
/// offset-counterpart of [`Point2i`].
pub type Vector2i = euclid::Vector2D<i32, ImageSpace>;
