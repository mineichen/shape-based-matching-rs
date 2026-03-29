mod image_buffer;
mod line2dup;
mod match_entry;
mod matches;
mod pyramid;
mod simd_utils;

pub use line2dup::{BuilderError, Detector, DetectorBuilder, Feature, TemplateConfigHandle};
pub use match_entry::Match;
pub use matches::Matches;
