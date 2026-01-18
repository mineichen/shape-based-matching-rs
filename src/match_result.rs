//! Match result structures for shape-based matching
//!
//! This module contains the data structures and implementations for match results
//! from the line2dup shape-based matching algorithm.

use opencv::core::Point;

use crate::pyramid::Template;

/// A thin wrapper around a vector of matches.
///
/// This type exists so we can add convenience methods (e.g. debug visualization)
/// without changing the underlying data structure.
#[derive(Debug, Clone, Default)]
pub struct Matches<'a>(Vec<Match<'a>>);

/// A match result with position, similarity score, and template information
#[derive(Debug, Clone)]
pub struct Match<'a> {
    pub x: i32,
    pub y: i32,
    /// Similarity score as percentage (0.0 to 100.0)
    pub similarity: f32,
    pub class_id: &'a str,
    templates: &'a [Vec<Template>],
    template_id: usize,
}

/// Internal match structure used during matching process
#[derive(Debug)]
pub(crate) struct RawMatch<T> {
    pub x: i32,
    pub y: i32,
    pub raw_score: T,
}

impl<'a> Match<'a> {
    pub fn new(
        x: i32,
        y: i32,
        similarity: f32,
        class_id: &'a str,
        template_id: usize,
        templates: &'a [Vec<Template>],
    ) -> Self {
        assert!(!templates.is_empty(), "Match needs at least one template");
        Match {
            x,
            y,
            similarity,
            class_id,
            template_id,
            templates,
        }
    }

    pub fn match_template(&self) -> &Template {
        &self.templates[self.template_id][0]
    }

    pub fn center_point(&self) -> Point {
        let templ = self.match_template();
        Point::new(self.x + templ.width / 2, self.y + templ.height / 2)
    }

    pub fn ref_template(&self) -> &Template {
        &self.templates[0][0]
    }

    pub fn angle(&self) -> f32 {
        self.templates[self.template_id][0].features[0].theta
    }
}

impl<'a> Matches<'a> {
    pub fn new(matches: Vec<Match<'a>>) -> Self {
        Self(matches)
    }

    pub fn into_vec(self) -> Vec<Match<'a>> {
        self.0
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

impl<'a> PartialEq for Match<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.y == other.y
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
        // Sort by similarity (descending), then by template_id
        match other.similarity.partial_cmp(&self.similarity) {
            Some(std::cmp::Ordering::Equal) => self.y.cmp(&other.y),
            Some(ord) => ord,
            None => std::cmp::Ordering::Equal,
        }
    }
}
