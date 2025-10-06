//! Match result structures for shape-based matching
//!
//! This module contains the data structures and implementations for match results
//! from the line2dup shape-based matching algorithm.

use crate::pyramid::Template;

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

    pub fn ref_template(&self) -> &Template {
        &self.templates[0][0]
    }

    pub fn angle(&self) -> f32 {
        self.templates[self.template_id][0].features[0].theta
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
