//! Match structures for shape-based matching
//!
//! This module contains the data structures and implementations for match results
//! from the line2dup shape-based matching algorithm.

use crate::pyramid::Template;
use crate::{Point2Fixed, Point2i};

/// A match result with position, similarity score, and template information
#[derive(Debug, Clone)]
pub struct Match<'a> {
    /// Top-left match position: the center of the matched bbox's top-left
    /// pixel in public image coordinates, where pixel `N`'s center is
    /// `N + 0.5` and `(0, 0)` is the top-left subpixel of the image.
    /// Until subpixel accuracy is implemented, its always `X.5`;  corner is at `pos - 0.5`.
    /// Backed by `fixed::types::U20F12` — build with [`crate::from_pixel_pt`].
    pub pos: Point2Fixed,
    /// Similarity score as percentage (0.0 to 100.0)
    pub similarity: f32,
    pub class_id: &'a str,
    templates: &'a [Vec<Template>],
    template_id: usize,
}

/// Internal match structure used during matching process.
///
/// Stays integer: the detection pyramid genuinely works on grid positions.
/// Convert to [`Match`] with [`crate::from_pixel_pt`] (after adding
/// `POSITION_OFFSET` as integers), no rounding change.
#[derive(Debug)]
pub(crate) struct MatchRaw<T> {
    pub pos: Point2i,
    pub raw_score: T,
}

impl<'a> Match<'a> {
    pub fn new(
        pos: Point2Fixed,
        similarity: f32,
        class_id: &'a str,
        template_id: usize,
        templates: &'a [Vec<Template>],
    ) -> Self {
        assert!(!templates.is_empty(), "Match needs at least one template");
        debug_assert_eq!(
            pos.map(|x| x.frac()),
            Point2Fixed::splat(fixed::types::U20F12::from_num(0.5)),
            "Match pos must currently be a pixel center (X.5), if we have no subpixel accuracy, got {pos:?}"
        );
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

    /// Center of the matched bbox: `pos + floor((w, h) / 2)`. Always
    /// `X.5` — every position this crate outputs is a pixel center.
    /// Odd widths coincide with the geometric center; even widths land
    /// on the upper-middle pixel's center.
    pub fn center_point(&self) -> Point2Fixed {
        let templ = self.match_template();
        self.pos
            + crate::Vector2Fixed::new(
                fixed::types::U20F12::from_num((templ.width.get() / 2) as u32),
                fixed::types::U20F12::from_num((templ.height.get() / 2) as u32),
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
        // Positions are exactly representable fixed-point (X.5), so
        // direct equality is exact; no epsilon is needed.
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
        // Position tiebreak uses fixed-point Ord, which matches integer
        // ordering since all positions are X.5.
        match self.similarity.total_cmp(&other.similarity) {
            std::cmp::Ordering::Equal => self.pos.y.cmp(&other.pos.y),
            ord => ord,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Match;
    use crate::Vector2i;
    use crate::pyramid::Template;
    use crate::{Point2i, from_pixel_pt};
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
    fn center_point_is_a_pixel_center() {
        // center_point() == pos + floor(w/2): X.5 in -> X.5 out for both
        // parities — every position this crate outputs is a pixel center.
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
                let point = Point2i::splat(x);
                let m = Match::new(from_pixel_pt(point), 0.9, "t", 0, &t);
                let c = m.center_point();
                let expected = from_pixel_pt(point + Vector2i::splat((w / 2) as i32));
                assert_eq!(c, expected);
                assert_eq!(c.x.frac(), fixed::types::U20F12::from_num(0.5));
            }
        }
    }

    #[test]
    fn test_match_ordering() {
        let t = dummy_template();
        let m1 = Match::new(from_pixel_pt(Point2i::zero()), 0.9, "test", 0, &t);
        let m2 = Match::new(from_pixel_pt(Point2i::zero()), 0.8, "test", 0, &t);
        assert!(m1 > m2); // Ascending Ord: higher similarity is greater
        assert_eq!(std::cmp::max(&m1, &m2), &m1); // max() returns best
    }
}
