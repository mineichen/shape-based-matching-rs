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
