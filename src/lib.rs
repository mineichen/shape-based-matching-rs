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
/// Every position produced by this crate is a pixel center: an integral
/// value denotes exactly that pixel, and `(0, 0)` is the center of the
/// top-left pixel. Half-pixel values are never emitted. Caller-supplied
/// positions (e.g. rotation/scale pivots) are used verbatim.
pub enum ImageSpace {}

/// Sub-pixel-capable image-space point, backed by [`euclid::Point2D`]
/// (same layout as two `f32`, zero cost). Replaces `opencv::core::Point2f`
/// in this crate's public API so callers can use the library without
/// depending on opencv. Integral values are pixel centers.
pub type Point2f = euclid::Point2D<f32, ImageSpace>;

/// Image-space translation/offset, backed by [`euclid::Vector2D`] (same
/// layout as two `f32`, zero cost). Vectors have no absolute position: they
/// only translate points (`point + vector`), which keeps position-vs-offset
/// confusion out of the type system.
pub type Vector2f = euclid::Vector2D<f32, ImageSpace>;

/// Integer pixel-index point ([`Point2f`] with `i32` components). Features
/// and all matcher-internal positions are discrete by definition; the
/// float <-> pixel bridge is `(v + 0.5) as i32`, never `v.round()`.
pub type Point2i = euclid::Point2D<i32, ImageSpace>;

/// Integer pixel-index translation ([`euclid::Vector2D<i32>`]); the
/// offset-counterpart of [`Point2i`].
pub type Vector2i = euclid::Vector2D<i32, ImageSpace>;
