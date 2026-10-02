use crate::{Point2Fixed, Vector2Fixed, match_entry::Match};

#[cfg(feature = "visualize")]
mod visualize;

/// A thin wrapper around a vector of matches.
///
/// This type exists so we can add convenience methods (e.g. debug visualization)
/// without changing the underlying data structure.
#[derive(Debug, Clone, Default)]
pub struct Matches<'a>(Vec<Match<'a>>);

impl<'a> Matches<'a> {
    pub fn new(matches: Vec<Match<'a>>) -> Self {
        Self(matches)
    }

    pub fn into_vec(self) -> Vec<Match<'a>> {
        self.0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Match<'a>> {
        self.0.iter()
    }

    /// Filters matches in-place: keeps only the best match within `min_distance` (center-to-center).
    ///
    /// Sorts matches from best-to-worst
    /// This does not allocate; it compacts `self` by swapping kept elements to the front and truncating.
    pub fn filter_min_center_distance(&mut self, min_distance: f32) {
        self.0.sort_unstable_by(|a, b| b.cmp(a)); // Descending: best match first

        let min_distance2 = min_distance * min_distance;

        let len = self.0.len();
        let mut keep_len: usize = 1;

        'next: for i in 1..len {
            // Same centers as `Match::center_point` — shared so the two
            // can never diverge. Centers are `U20F12` (unsigned): subtract in
            // `f32` — `Point2Fixed - Point2Fixed` would panic on negative
            // components (unsigned underflow).
            let c = self.0[i].center_point();

            for j in 0..keep_len {
                let existing_c = self.0[j].center_point();
                let d = Vector2Fixed::new(c.x.dist(existing_c.x), c.y.dist(existing_c.y));
                let distance2 = d.dot(d);

                if distance2 < min_distance2 {
                    continue 'next;
                }
            }

            self.0.swap(i, keep_len);
            keep_len += 1;
        }

        self.0.truncate(keep_len);
    }
}

impl<'a> std::ops::Deref for Matches<'a> {
    type Target = Vec<Match<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> std::ops::DerefMut for Matches<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> From<Vec<Match<'a>>> for Matches<'a> {
    fn from(value: Vec<Match<'a>>) -> Self {
        Self(value)
    }
}

impl<'a> From<Matches<'a>> for Vec<Match<'a>> {
    fn from(value: Matches<'a>) -> Self {
        value.0
    }
}

impl<'a> IntoIterator for Matches<'a> {
    type Item = Match<'a>;
    type IntoIter = std::vec::IntoIter<Match<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Matches<'a> {
    type Item = &'a Match<'a>;
    type IntoIter = std::slice::Iter<'a, Match<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut Matches<'a> {
    type Item = &'a mut Match<'a>;
    type IntoIter = std::slice::IterMut<'a, Match<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point2i;
    use crate::Vector2i;
    use crate::from_pixel_pt;
    use crate::match_entry::Match;
    use crate::pyramid::Template;

    fn dummy_templates() -> [Vec<Template>; 1] {
        [vec![Template {
            width: 10.try_into().unwrap(),
            height: 10.try_into().unwrap(),
            tl: Vector2i::zero(),
            pyramid_level: 0,
            features: vec![],
            rotation_angle: 0.0,
            scale_factor: 1.0,
        }]]
    }

    #[test]
    fn test_filter_keeps_best_match() {
        let t = dummy_templates();
        // All matches at the same position: only best should survive distance filter
        let matches = Matches::new(vec![
            Match::new(from_pixel_pt(Point2i::zero()), 0.3, "test", 0, &t),
            Match::new(from_pixel_pt(Point2i::zero()), 0.9, "test", 0, &t),
            Match::new(from_pixel_pt(Point2i::zero()), 0.5, "test", 0, &t),
        ]);

        let mut filtered = matches;
        filtered.filter_min_center_distance(100.0);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].similarity, 0.9);
    }

    #[test]
    fn test_filter_respects_order() {
        let t = dummy_templates();
        let matches = Matches::new(vec![
            Match::new(from_pixel_pt(Point2i::splat(0)), 0.9, "test", 0, &t),
            Match::new(from_pixel_pt(Point2i::splat(400)), 0.5, "test", 0, &t),
            Match::new(from_pixel_pt(Point2i::splat(200)), 0.7, "test", 0, &t),
        ]);

        let mut filtered = matches;
        filtered.filter_min_center_distance(1.0);

        // Best first, worst last
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].similarity, 0.9);
        assert_eq!(filtered[1].similarity, 0.7);
        assert_eq!(filtered[2].similarity, 0.5);
    }

    #[test]
    fn filter_handles_worse_match_up_left_of_best_without_underflow() {
        // Regression: `center_point()` is `U20F12` (unsigned), so
        // `Point - Point` panics on negative components (unsigned underflow
        // in `fixed`). The worse match sits up-left of the best one, forcing
        // `worse - best` negative — the old `match_c - existing_c` overflowed
        // here instead of returning a large distance.
        let t = dummy_templates();
        let mut matches = Matches::new(vec![
            Match::new(from_pixel_pt(Point2i::splat(0)), 0.5, "test", 0, &t),
            Match::new(from_pixel_pt(Point2i::splat(100)), 0.9, "test", 0, &t),
        ]);
        matches.filter_min_center_distance(1.0);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].similarity, 0.9);
        assert_eq!(matches[1].similarity, 0.5);
    }
}
