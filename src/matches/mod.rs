use crate::match_entry::Match;

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
    /// Assumes matches are sorted best-to-worst (e.g. by similarity descending).
    /// This does not allocate; it compacts `self` by swapping kept elements to the front and truncating.
    pub fn filter_min_center_distance(&mut self, min_distance: f32) {
        self.0.sort_unstable();

        let min_distance2 = min_distance * min_distance;

        let len = self.0.len();
        let mut keep_len: usize = 1;

        'next: for i in 1..len {
            let (match_cx, match_cy) = {
                let match_item = &self.0[i];
                let match_templ = match_item.match_template();
                (
                    match_item.x as f32 + match_templ.width as f32 / 2.0,
                    match_item.y as f32 + match_templ.height as f32 / 2.0,
                )
            };

            for j in 0..keep_len {
                let (existing_cx, existing_cy) = {
                    let existing = &self.0[j];
                    let existing_templ = existing.match_template();
                    (
                        existing.x as f32 + existing_templ.width as f32 / 2.0,
                        existing.y as f32 + existing_templ.height as f32 / 2.0,
                    )
                };

                let dx = match_cx - existing_cx;
                let dy = match_cy - existing_cy;
                let distance2 = dx * dx + dy * dy;

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
