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
