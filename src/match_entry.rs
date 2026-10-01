//! Match structures for shape-based matching
//!
//! This module contains the data structures and implementations for match results
//! from the line2dup shape-based matching algorithm.

use crate::pyramid::Template;
use crate::{Point2f, Point2i};
use euclid::vec2;

/// A match result with position, similarity score, and template information
#[derive(Debug, Clone)]
pub struct Match<'a> {
    /// Top-left match position: the center of the matched variant bbox's
    /// top-left pixel, always a full number (integral value = that pixel's
    /// center; `(0, 0)` is the center of the top-left pixel).
    pub pos: Point2f,
    /// Similarity score as percentage (0.0 to 100.0)
    pub similarity: f32,
    pub class_id: &'a str,
    templates: &'a [Vec<Template>],
    template_id: usize,
}

/// Internal match structure used during matching process.
///
/// Stays integer: the detection pyramid genuinely works on grid positions.
/// Convert to [`Match`] with `as f32` (+ `POSITION_OFFSET` as float), no
/// rounding change.
#[derive(Debug)]
pub(crate) struct MatchRaw<T> {
    pub pos: Point2i,
    pub raw_score: T,
}

impl<'a> Match<'a> {
    pub fn new(
        pos: Point2f,
        similarity: f32,
        class_id: &'a str,
        template_id: usize,
        templates: &'a [Vec<Template>],
    ) -> Self {
        assert!(!templates.is_empty(), "Match needs at least one template");
        Match {
            pos,
            similarity,
            class_id,
            template_id,
            templates,
        }
    }

    pub fn match_template(&self) -> &Template {
        &self.templates[self.template_id][0]
    }

    /// Center pixel of the matched bbox: `pos + (w / 2)`. Always a full
    /// number — every position this crate outputs is a pixel center; no
    /// half-pixel values are ever emitted.
    pub fn center_point(&self) -> Point2f {
        let templ = self.match_template();
        self.pos
            + vec2(
                (templ.width.get() / 2) as f32,
                (templ.height.get() / 2) as f32,
            )
    }

    pub fn ref_template(&self) -> &Template {
        &self.templates[0][0]
    }

    pub fn angle(&self) -> f32 {
        self.templates[self.template_id][0].rotation_angle
    }

    pub fn scale(&self) -> f32 {
        self.templates[self.template_id][0].scale_factor
    }
}

impl<'a> PartialEq for Match<'a> {
    fn eq(&self, other: &Self) -> bool {
        // Positions are integral f32, so direct float equality is exact
        // here; no epsilon is needed.
        self.pos == other.pos
            && self.similarity == other.similarity
            && self.class_id == other.class_id
    }
}

impl<'a> Eq for Match<'a> {}

impl<'a> PartialOrd for Match<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for Match<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort by similarity (ascending), so max() returns best match.
        // Position tiebreak uses total_cmp: positions are integral floats,
        // so this matches integer ordering.
        match self.similarity.total_cmp(&other.similarity) {
            std::cmp::Ordering::Equal => self.pos.y.total_cmp(&other.pos.y),
            ord => ord,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Match;
    use crate::Point2f;
    use crate::Vector2i;
    use crate::pyramid::Template;
    use std::num::NonZeroUsize;

    fn dummy_template() -> [Vec<Template>; 1] {
        [vec![Template {
            width: NonZeroUsize::MIN,
            height: NonZeroUsize::MIN,
            tl: Vector2i::zero(),
            pyramid_level: 0,
            features: vec![],
            rotation_angle: 0.0,
            scale_factor: 1.0,
        }]]
    }

    #[test]
    fn center_point_is_a_full_number_pixel_center() {
        // center_point() == pos + floor(w/2) — integral for every width,
        // so it always points at an actual pixel center.
        for x in [0, 1, 60, 221, 400] {
            for w in [1usize, 2, 3, 40, 80, 91, 137] {
                let t = [vec![Template {
                    width: w.try_into().unwrap(),
                    height: w.try_into().unwrap(),
                    tl: Vector2i::zero(),
                    pyramid_level: 0,
                    features: vec![],
                    rotation_angle: 0.0,
                    scale_factor: 1.0,
                }]];
                let m = Match::new(Point2f::new(x as f32, x as f32), 0.9, "t", 0, &t);
                let c = m.center_point();
                assert_eq!(c.x, x as f32 + (w / 2) as f32);
                assert_eq!(c.y, x as f32 + (w / 2) as f32);
                assert_eq!(c.x as i32, x + (w as i32) / 2);
                assert_eq!(c.y as i32, x + (w as i32) / 2);
            }
        }
    }

    #[test]
    fn test_match_ordering() {
        let t = dummy_template();
        let m1 = Match::new(Point2f::zero(), 0.9, "test", 0, &t);
        let m2 = Match::new(Point2f::zero(), 0.8, "test", 0, &t);
        assert!(m1 > m2); // Ascending Ord: higher similarity is greater
        assert_eq!(std::cmp::max(&m1, &m2), &m1); // max() returns best
    }
}
