//! Line2Dup shape-based matching module
//!
//! This module implements a shape-based matching algorithm using gradient orientation features.
//! It's a Rust port of the C++ line2Dup implementation, adapted for OpenCV 4.x.

use opencv::{
    core::{self, Mat, Point2f, Scalar},
    imgproc,
    prelude::*,
};
use std::collections::HashMap;

use crate::pyramid::{ColorGradientPyramid, Template};

/// Handle to a template that can be used to add rotated/scaled variants

pub struct TemplateHandle<'a> {
    class_id: &'a str,
    template_id: usize,
    detector: &'a mut Detector,
}

impl<'a> TemplateHandle<'a> {
    fn new(class_id: &'a str, template_id: usize, detector: &'a mut Detector) -> Self {
        Self {
            class_id,
            template_id,
            detector,
        }
    }

    /// Get the template ID
    pub fn template_id(&self) -> usize {
        self.template_id
    }

    /// Add a rotated version of this template
    ///
    /// # Arguments
    /// * `theta` - Rotation angle in degrees (positive = counter-clockwise)
    /// * `center` - Center of rotation in absolute image coordinates
    pub fn add_rotated(
        &mut self,
        theta: f32,
        center: Point2f,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.detector
            .add_template_rotate_internal(self.class_id, self.template_id, theta, center)
    }
}

// ============================================================================
// PUBLIC API - Data Structures
// ============================================================================

/// A feature point with position, label (quantized orientation), and angle
#[derive(Debug, Clone)]
pub struct Feature {
    pub x: i32,
    pub y: i32,
    pub label: i32,
    pub theta: f32,
}

impl Feature {
    pub fn new(x: i32, y: i32, label: i32) -> Self {
        Feature {
            x,
            y,
            label,
            theta: 0.0,
        }
    }
}

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

// ============================================================================
// PUBLIC API - Main Detector
// ============================================================================

/// Main detector for shape-based matching
pub struct Detector {
    weak_threshold: f32,
    num_features: usize,
    strong_threshold: f32,
    t_at_level: Vec<i32>,
    pyramid_levels: i32,
    class_templates: HashMap<String, TemplatePyramidsLevels>,
}

#[derive(Default)]
struct TemplatePyramidsLevels(Vec<Vec<Template>>);

impl TemplatePyramidsLevels {
    pub fn insert(&mut self, templates: Vec<Template>) -> usize {
        self.0.push(templates);
        self.0.len() - 1
    }
}

impl Detector {
    /// Create a new detector with default parameters
    pub fn new() -> Self {
        Detector::with_params(63, vec![4, 8], 30.0, 60.0)
    }

    /// Create a new detector with custom parameters
    ///
    /// # Arguments
    /// * `num_features` - Number of features to extract per template
    /// * `t` - Spread values at each pyramid level (e.g., [4, 8])
    /// * `weak_threshold` - Weak gradient threshold
    /// * `strong_threshold` - Strong gradient threshold
    pub fn with_params(
        num_features: usize,
        t: Vec<i32>,
        weak_threshold: f32,
        strong_threshold: f32,
    ) -> Self {
        Detector {
            weak_threshold,
            num_features,
            strong_threshold,
            t_at_level: t.clone(),
            pyramid_levels: t.len() as i32,
            class_templates: HashMap::new(),
        }
    }

    /// Add a template to the detector
    ///
    /// # Arguments
    /// * `source` - Source image containing the object
    /// * `class_id` - Identifier for this template class
    /// * `object_mask` - Mask indicating the object region (optional)
    ///
    /// # Returns
    /// Template handle that can be used to add rotated/scaled variants
    pub fn add_template<'a>(
        &'a mut self,
        source: &Mat,
        class_id: &'a str,
        object_mask: Option<&Mat>,
    ) -> Result<TemplateHandle<'a>, Box<dyn std::error::Error>> {
        let mask = match object_mask {
            Some(m) => m.clone(),
            None => Mat::new_rows_cols_with_default(
                source.rows(),
                source.cols(),
                core::CV_8UC1,
                Scalar::all(255.0),
            )?,
        };

        // Create pyramid and extract templates
        let mut pyramid = ColorGradientPyramid::new(
            source,
            &mask,
            self.weak_threshold,
            self.num_features,
            self.strong_threshold,
        )?;

        let mut template_pyramid = Vec::new();

        for level in 0..self.pyramid_levels {
            let mut templ = Template {
                pyramid_level: level,
                ..Default::default()
            };

            if pyramid.extract_template(&mut templ)? {
                template_pyramid.push(templ);
            }

            if level < self.pyramid_levels - 1 {
                pyramid.pyr_down()?;
            }
        }

        // Crop templates to minimal bounding box
        crop_templates(&mut template_pyramid);

        // Store in class_templates
        let templates = self
            .class_templates
            .entry(class_id.to_string())
            .or_default();
        let template_id = templates.insert(template_pyramid);
        Ok(TemplateHandle::new(class_id, template_id, self))
    }

    /// Internal method to add a rotated version of an existing template
    fn add_template_rotate_internal(
        &mut self,
        class_id: &str,
        template_id: usize,
        theta: f32,
        center: Point2f,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get the base template pyramid
        let base_pyramid = self
            .class_templates
            .get(class_id)
            .and_then(|templates| templates.0.get(template_id))
            .ok_or("Base template not found")?
            .clone();

        let mut rotated_pyramid = Vec::new();
        let mut pyramid_center = center;

        for base_templ in &base_pyramid {
            let mut new_templ = Template {
                pyramid_level: base_templ.pyramid_level,
                ..Default::default()
            };

            // Scale center for pyramid level (cumulative division like C++)
            if base_templ.pyramid_level > 0 {
                pyramid_center.x /= 2.0;
                pyramid_center.y /= 2.0;
            }

            // Rotation angle in radians (negative because C++ uses CW rotation)
            let theta_rad = -theta.to_radians();
            let cos_theta = theta_rad.cos();
            let sin_theta = theta_rad.sin();

            for feat in &base_templ.features {
                // Convert feature position to absolute coordinates (add tl_x, tl_y)
                let abs_x = (feat.x + base_templ.tl_x) as f32;
                let abs_y = (feat.y + base_templ.tl_y) as f32;

                // Rotate point around center in absolute coordinates
                let dx = abs_x - pyramid_center.x;
                let dy = abs_y - pyramid_center.y;

                let new_x = (cos_theta * dx - sin_theta * dy + pyramid_center.x + 0.5) as i32;
                let new_y = (sin_theta * dx + cos_theta * dy + pyramid_center.y + 0.5) as i32;

                // Rotate the orientation angle (feat.theta is in degrees like C++)
                let mut new_theta_deg = feat.theta - theta;
                while new_theta_deg > 360.0 {
                    new_theta_deg -= 360.0;
                }
                while new_theta_deg < 0.0 {
                    new_theta_deg += 360.0;
                }

                // Quantize to label (C++ uses 16 bins then masks to 8)
                let new_label = ((new_theta_deg * 16.0 / 360.0 + 0.5) as i32) & 7;

                let mut new_feat = Feature::new(new_x, new_y, new_label);
                new_feat.theta = new_theta_deg; // Keep in degrees
                new_templ.features.push(new_feat);
            }

            rotated_pyramid.push(new_templ);
        }

        // Crop templates to recalculate bounding box (like C++ does)
        crop_templates(&mut rotated_pyramid);

        // Store rotated pyramid
        let templates = self
            .class_templates
            .entry(class_id.to_string())
            .or_default();

        templates.insert(rotated_pyramid);
        Ok(())
    }

    /// Match templates against a source image
    /// Automatically chooses u8 or u16 accumulator based on template features
    ///
    /// # Arguments
    /// * `source` - Image to search in
    /// * `threshold` - Similarity threshold in percentage (0.0 to 100.0)
    /// * `class_ids` - Optional list of class IDs to search for (empty = all)
    /// * `masks` - Optional mask for the source image
    ///
    /// # Returns
    /// Vector of matches (unsorted - caller can sort as needed)
    fn match_templates_generic<'a, T: SimilarityAccumulator + 'static>(
        &'a self,
        source: &Mat,
        threshold: f32,
        class_ids: Option<&[&'a str]>,
        masks: Option<&Mat>,
    ) -> Result<Vec<Match<'a>>, Box<dyn std::error::Error>> {
        let time = std::time::Instant::now();
        let mask = match masks {
            Some(m) => m.clone(),
            None => Mat::default(),
        };

        // Determine which classes to search
        let search_classes: Vec<&str> = match class_ids {
            Some(ids) if !ids.is_empty() => ids.to_vec(),
            _ => self.class_templates.keys().map(|s| s.as_str()).collect(),
        };

        // Build linear memories for ALL pyramid levels (like C++)
        let mut pyramid = ColorGradientPyramid::new(
            source,
            &mask,
            self.weak_threshold,
            self.num_features,
            self.strong_threshold,
        )?;
        println!("Time taken to build pyramid: {:?}", time.elapsed());

        // Pre-compute linear memories for each pyramid level
        let mut linear_memory_pyramid: Vec<Vec<Mat>> = Vec::new();
        let mut pyramid_sizes: Vec<(i32, i32)> = Vec::new();

        for level in 0..self.pyramid_levels {
            let t = self.t_at_level[level as usize];
            let mut quantized = Mat::default();
            pyramid.quantize(&mut quantized)?;

            pyramid_sizes.push((quantized.cols(), quantized.rows()));

            let spread_quantized = spread_quantized_image(&quantized, t)?;
            let linear_memories = compute_and_linearize_response_maps(&spread_quantized, t)?;
            linear_memory_pyramid.push(linear_memories);

            // Downsample for next level (except last)
            if level < self.pyramid_levels - 1 {
                pyramid.pyr_down()?;
            }
        }
        println!(
            "Time taken to build linear memory pyramid: {:?}",
            time.elapsed()
        );
        // Match all templates using coarse-to-fine pyramid refinement (like C++)
        let mut matches = Vec::new();
        for class_id in &search_classes {
            if let Some(template_pyramids) = self.class_templates.get(*class_id) {
                for (template_id, template_pyramid) in template_pyramids.0.iter().enumerate() {
                    if template_pyramid.is_empty() {
                        continue;
                    }

                    matches.extend(self.match_template_pyramid::<T>(
                        &linear_memory_pyramid,
                        &pyramid_sizes,
                        template_pyramid,
                        threshold,
                        class_id,
                        template_id,
                    ));
                }
            }
        }
        println!("Time taken to match_templates: {:?}", time.elapsed());

        Ok(matches)
    }

    /// Match templates against a source image
    /// Automatically chooses u8 or u16 accumulator based on template features
    ///
    /// # Arguments
    /// * `source` - Image to search in
    /// * `threshold` - Similarity threshold in percentage (0.0 to 100.0)
    /// * `class_ids` - Optional list of class IDs to search for (empty = all)
    /// * `masks` - Optional mask for the source image
    ///
    /// # Returns
    /// Iterator of matches (unsorted - caller can collect and sort as needed)
    pub fn match_templates<'a>(
        &'a self,
        source: &Mat,
        threshold: f32,
        class_ids: Option<&[&'a str]>,
        masks: Option<&Mat>,
    ) -> Result<impl Iterator<Item = Match<'a>> + 'a, Box<dyn std::error::Error>> {
        // Determine which classes to search (for accumulator type check)
        let search_classes: Vec<&str> = match class_ids {
            Some(ids) if !ids.is_empty() => ids.to_vec(),
            _ => self.class_templates.keys().map(|s| s.as_str()).collect(),
        };

        // Check if any template has 64+ features to decide accumulator type
        let use_u16 = search_classes.iter().any(|class_id| {
            if let Some(template_pyramids) = self.class_templates.get(*class_id) {
                template_pyramids.0.iter().any(|pyramid| {
                    pyramid
                        .first()
                        .is_some_and(|templ| templ.features.len() >= 64)
                })
            } else {
                false
            }
        });

        if use_u16 {
            let matches = self.match_templates_generic::<u16>(
                source,
                threshold,
                Some(&search_classes),
                masks,
            )?;
            Ok(MatchIterator::U16(matches.into_iter()))
        } else {
            let matches = self.match_templates_generic::<u8>(
                source,
                threshold,
                Some(&search_classes),
                masks,
            )?;
            Ok(MatchIterator::U8(matches.into_iter()))
        }
    }

    /// Get number of templates for a class
    pub fn num_templates(&self, class_id: &str) -> usize {
        self.class_templates
            .get(class_id)
            .map(|v| v.0.len())
            .unwrap_or(0)
    }

    /// Get all class IDs
    pub fn class_ids(&self) -> Vec<String> {
        self.class_templates.keys().cloned().collect()
    }

    // Match a template pyramid using coarse-to-fine refinement (like C++)
    fn match_template_pyramid<'a, T: SimilarityAccumulator + 'static>(
        &'a self,
        linear_memory_pyramid: &[Vec<Mat>],
        pyramid_sizes: &[(i32, i32)],
        template_pyramid: &[Template],
        threshold: f32,
        class_id: &'a str,
        template_id: usize,
    ) -> Vec<Match<'a>> {
        // Start at the coarsest pyramid level (last in array)
        let lowest_level = (self.pyramid_levels - 1) as usize;
        let lowest_t = self.t_at_level[lowest_level];
        let (src_cols, src_rows) = pyramid_sizes[lowest_level];

        // Get template at coarsest level
        let coarse_template = &template_pyramid[lowest_level];

        // Match at coarcompute_similarity_at_positionsest level to get initial candidates
        let mut candidates: Vec<Match> = self
            .match_template_with_linear_memory::<T>(
                &linear_memory_pyramid[lowest_level],
                coarse_template,
                threshold,
                class_id,
                template_id,
                src_cols,
                src_rows,
                lowest_t,
            )
            .collect();

        // Refine candidates by marching up the pyramid (from coarse to fine)
        for level in (0..lowest_level).rev() {
            let t = self.t_at_level[level];
            let (src_cols, src_rows) = pyramid_sizes[level];
            let template = &template_pyramid[level];
            let border = 8 * t;
            let _offset = t / 2 + (t % 2 - 1);

            let max_x = src_cols - template.width - border;
            let max_y = src_rows - template.height - border;

            let mut refined_candidates = Vec::new();

            for candidate in &candidates {
                // Scale up position from previous level (2x)
                let x = candidate.x * 2 + 1;
                let y = candidate.y * 2 + 1;

                // Require 8 (reduced) rows/cols to the up/left
                if x < border || y < border || x > max_x || y > max_y {
                    continue;
                }

                // Search in a 5x5 window around the scaled position
                let mut best_match = candidate.clone();
                best_match.similarity = 0.0;

                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let search_x = x + dx * t;
                        let search_y = y + dy * t;

                        if search_x < border
                            || search_y < border
                            || search_x > max_x
                            || search_y > max_y
                        {
                            continue;
                        }

                        // Compute similarity at this position
                        let similarity = self.compute_similarity_at_position::<T>(
                            &linear_memory_pyramid[level],
                            template,
                            search_x,
                            search_y,
                            src_cols,
                            src_rows,
                            t,
                        );

                        if similarity > best_match.similarity {
                            best_match.x = search_x;
                            best_match.y = search_y;
                            best_match.similarity = similarity;
                        }
                    }
                }

                // Keep refined match if it still passes threshold
                if best_match.similarity >= threshold {
                    refined_candidates.push(best_match);
                }
            }

            candidates = refined_candidates;
        }

        candidates
    }

    // Compute similarity at a specific position
    fn compute_similarity_at_position<T: SimilarityAccumulator + 'static>(
        &self,
        linear_memories: &[Mat],
        templ: &Template,
        x: i32,
        y: i32,
        src_cols: i32,
        src_rows: i32,
        t: i32,
    ) -> f32 {
        let w = src_cols / t;
        let mut score: T = T::default();

        for feat in &templ.features {
            let label = feat.label as usize;
            if label >= linear_memories.len() {
                continue;
            }

            let feat_x = feat.x + x;
            let feat_y = feat.y + y;

            // Check bounds
            if feat_x < 0 || feat_x >= src_cols || feat_y < 0 || feat_y >= src_rows {
                continue;
            }

            // Access the correct linear memory from the TxT grid
            let grid_x = feat_x % t;
            let grid_y = feat_y % t;
            let grid_index = grid_y * t + grid_x;

            let memory_grid = &linear_memories[label];
            if grid_index >= memory_grid.rows() {
                continue;
            }

            // Feature position in decimated coordinates
            let fx = feat_x / t;
            let fy = feat_y / t;
            let lm_index = (fy * w + fx) as usize;

            if lm_index >= memory_grid.cols() as usize {
                continue;
            }

            unsafe {
                let linear_memory_ptr = memory_grid.ptr(grid_index).unwrap();
                let response = T::from_u8(*linear_memory_ptr.add(lm_index));
                score = score + response;
            }
        }

        // Convert to percentage
        (score.into() * 100.0) / (4.0 * templ.features.len() as f32)
    }

    // Match a single template using pre-computed linear memories
    #[allow(clippy::too_many_arguments)]
    fn match_template_with_linear_memory<'a, T: SimilarityAccumulator + 'static>(
        &'a self,
        linear_memories: &[Mat],
        templ: &Template,
        threshold: f32,
        class_id: &'a str,
        template_id: usize,
        src_cols: i32,
        src_rows: i32,
        t: i32,
    ) -> impl Iterator<Item = Match<'a>> {
        // Extract matches from similarity map
        let w = src_cols / t;
        let h = src_rows / t;

        // Calculate offset to center the match position (like C++ version)
        let offset = t / 2 + (t % 2 - 1);

        let similarity_map =
            compute_similarity_map::<T>(linear_memories, templ, src_cols, src_rows, t);
        let templ_len = templ.features.len();

        // Pre-calculate threshold in terms of raw_score to avoid repeated calculations
        let raw_threshold = T::from_f32((threshold * 4.0 * templ_len as f32) / 100.0);

        (0..h)
            .flat_map(move |y| (0..w).map(move |x| (y, x)))
            .filter_map(move |(y, x)| {
                let raw_score = *similarity_map.at_2d::<T>(y, x).unwrap();

                (raw_score >= raw_threshold).then(|| {
                    let similarity = (raw_score.into() * 100.0) / (4.0 * templ_len as f32);
                    Match::new(
                        x * t + offset,
                        y * t + offset,
                        similarity,
                        class_id,
                        template_id,
                        &self.class_templates.get(class_id).unwrap().0,
                    )
                })
            })
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator wrapper to handle both u8 and u16 accumulator types
pub enum MatchIterator<'a> {
    U8(std::vec::IntoIter<Match<'a>>),
    U16(std::vec::IntoIter<Match<'a>>),
}

impl<'a> Iterator for MatchIterator<'a> {
    type Item = Match<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            MatchIterator::U8(iter) => iter.next(),
            MatchIterator::U16(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            MatchIterator::U8(iter) => iter.size_hint(),
            MatchIterator::U16(iter) => iter.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for MatchIterator<'a> {
    fn len(&self) -> usize {
        match self {
            MatchIterator::U8(iter) => iter.len(),
            MatchIterator::U16(iter) => iter.len(),
        }
    }
}

// ============================================================================
// ColorGradientPyramid - Gradient computation and feature extraction
// ============================================================================

// ============================================================================
// ShapeInfoProducer - Generate rotated/scaled templates
// ============================================================================

/// Information about a transformed shape (rotation and scale)
#[derive(Debug, Clone)]
pub struct ShapeInfo {
    pub angle: f32,
    pub scale: f32,
}

impl ShapeInfo {
    pub fn new(angle: f32, scale: f32) -> Self {
        ShapeInfo { angle, scale }
    }
}

/// Producer for generating multiple shape variations
pub struct ShapeInfoProducer {
    pub src: Mat,
    pub mask: Mat,
    pub angle_range: Vec<f32>,
    pub scale_range: Vec<f32>,
    pub angle_step: f32,
    pub scale_step: f32,
    pub infos: Vec<ShapeInfo>,
}

impl ShapeInfoProducer {
    /// Create a new shape info producer
    pub fn new(src: Mat, mask: Option<Mat>) -> Result<Self, Box<dyn std::error::Error>> {
        let mask = match mask {
            Some(m) => m,
            None => Mat::new_rows_cols_with_default(
                src.rows(),
                src.cols(),
                core::CV_8UC1,
                Scalar::all(255.0),
            )?,
        };

        Ok(ShapeInfoProducer {
            src,
            mask,
            angle_range: vec![0.0],
            scale_range: vec![1.0],
            angle_step: 15.0,
            scale_step: 0.5,
            infos: Vec::new(),
        })
    }

    /// Set angle range (start, end) in degrees
    pub fn set_angle_range(&mut self, start: f32, end: f32) {
        self.angle_range = vec![start, end];
    }

    /// Set scale range (start, end)
    pub fn set_scale_range(&mut self, start: f32, end: f32) {
        self.scale_range = vec![start, end];
    }

    /// Generate all shape info combinations
    pub fn produce_infos(&mut self) {
        self.infos.clear();

        let angles: Vec<f32> = if self.angle_range.len() == 2 {
            let mut angles = Vec::new();
            let mut angle = self.angle_range[0];
            while angle <= self.angle_range[1] + 0.0001 {
                angles.push(angle);
                angle += self.angle_step;
            }
            angles
        } else {
            vec![self.angle_range[0]]
        };

        let scales: Vec<f32> = if self.scale_range.len() == 2 {
            let mut scales = Vec::new();
            let mut scale = self.scale_range[0];
            while scale <= self.scale_range[1] + 0.0001 {
                scales.push(scale);
                scale += self.scale_step;
            }
            scales
        } else {
            vec![self.scale_range[0]]
        };

        for scale in &scales {
            for angle in &angles {
                self.infos.push(ShapeInfo::new(*angle, *scale));
            }
        }
    }

    /// Transform source image according to shape info
    pub fn transform_src(&self, info: &ShapeInfo) -> Result<Mat, Box<dyn std::error::Error>> {
        transform_image(&self.src, info.angle, info.scale)
    }

    /// Transform mask according to shape info
    pub fn transform_mask(&self, info: &ShapeInfo) -> Result<Mat, Box<dyn std::error::Error>> {
        let transformed = transform_image(&self.mask, info.angle, info.scale)?;
        let mut result = Mat::default();
        core::compare(&transformed, &Scalar::all(0.0), &mut result, core::CMP_GT)?;
        Ok(result)
    }
}
/// Crop templates to minimal bounding box
fn crop_templates(templates: &mut [Template]) {
    if templates.is_empty() {
        return;
    }

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    // Find bounding box
    for templ in templates.iter() {
        for feat in &templ.features {
            let x = feat.x << templ.pyramid_level;
            let y = feat.y << templ.pyramid_level;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    // Make even
    if min_x % 2 == 1 {
        min_x -= 1;
    }
    if min_y % 2 == 1 {
        min_y -= 1;
    }

    // Update templates
    for templ in templates.iter_mut() {
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y >> templ.pyramid_level;

        for feat in &mut templ.features {
            feat.x -= templ.tl_x;
            feat.y -= templ.tl_y;
        }
    }
}

/// Transform (rotate and scale) an image
fn transform_image(src: &Mat, angle: f32, scale: f32) -> Result<Mat, Box<dyn std::error::Error>> {
    let mut dst = Mat::default();
    let center = Point2f::new(src.cols() as f32 / 2.0, src.rows() as f32 / 2.0);
    let rot_mat = imgproc::get_rotation_matrix_2d(center, angle as f64, scale as f64)?;
    imgproc::warp_affine(
        src,
        &mut dst,
        &rot_mat,
        src.size()?,
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    Ok(dst)
}

// ============================================================================
// Optimized Matching Helpers - Linear Memory and Response Maps
// ============================================================================

/// Spread quantized orientations to neighboring pixels (like C++ spread function)
/// For each offset (r,c) in TxT grid, OR source values into destination
fn spread_quantized_image(quantized: &Mat, t: i32) -> Result<Mat, Box<dyn std::error::Error>> {
    let mut spread = Mat::new_rows_cols_with_default(
        quantized.rows(),
        quantized.cols(),
        core::CV_8UC1,
        Scalar::all(0.0),
    )?;

    // Fill in spread gradient image (section 2.3 of paper)
    // For each offset (r, c) in TxT grid, OR source[r:,c:] into destination
    // Use raw pointer access for true in-place OR without copy overhead
    for r in 0..t {
        for c in 0..t {
            let height = quantized.rows() - r;
            let width = quantized.cols() - c;

            unsafe {
                for y in 0..height {
                    let src_ptr = quantized.ptr((y + r) as i32)?.add(c as usize);
                    let dst_ptr = spread.ptr_mut(y as i32)?;

                    // Perform bitwise OR for the entire row at once
                    for x in 0..width as usize {
                        *dst_ptr.add(x) |= *src_ptr.add(x);
                    }
                }
            }
        }
    }

    Ok(spread)
}

/// Similarity lookup table (from C++ implementation)
/// Maps bit patterns to similarity scores:
/// - Single orientation: 4 points
/// - Adjacent orientations: 3 points  
/// - No match: 0 points
const LUT3: u8 = 3;
const SIMILARITY_LUT: [u8; 256] = [
    0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3,
    LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3,
    4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3,
    LUT3, LUT3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3,
    LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0,
    LUT3, 0, LUT3, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, 4,
    LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0,
    LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, LUT3, LUT3,
    LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4,
];

/// Compute response maps and linearize them in a single pass (combined optimization)
/// Directly produces linearized memories without creating intermediate full-resolution response maps
/// Returns Vec<Mat> where each Mat has T×T rows and (w×h) cols
fn compute_and_linearize_response_maps(
    spread_quantized: &Mat,
    t: i32,
) -> Result<Vec<Mat>, Box<dyn std::error::Error>> {
    // Align with C++ precondition: dimensions divisible by 16 and by t
    if spread_quantized.rows() % t != 0 || spread_quantized.cols() % t != 0 {
        return Err("Image width and height must be divisible by T".into());
    }

    let rows = spread_quantized.rows();
    let cols = spread_quantized.cols();
    let mem_width = cols / t;
    let mem_height = rows / t;
    let mem_size = mem_width * mem_height;

    let mut linear_memories = Vec::new();

    // Process each orientation
    for ori in 0..8 {
        let lut_offset = 32 * ori;

        // Create Mat with T×T rows, where each row is a linear memory
        let mut linearized =
            Mat::new_rows_cols_with_default(t * t, mem_size, core::CV_8UC1, Scalar::all(0.0))?;

        // Use raw pointer access for maximum performance
        let mut grid_index = 0;
        for r_start in 0..t {
            for c_start in 0..t {
                unsafe {
                    let linear_row_ptr = linearized.ptr_mut(grid_index)?;
                    let mut mem_idx = 0;

                    for r in (r_start..rows).step_by(t as usize) {
                        let src_row_ptr = spread_quantized.ptr(r)?;
                        for c in (c_start..cols).step_by(t as usize) {
                            let val = *src_row_ptr.add(c as usize);

                            // Split into LSB4 and MSB4
                            let lsb4 = (val & 15) as usize;
                            let msb4 = ((val & 240) >> 4) as usize;

                            // Use LUT to compute response (like C++)
                            let response_val = SIMILARITY_LUT[lut_offset + lsb4]
                                .max(SIMILARITY_LUT[lut_offset + 16 + msb4]);

                            // Store directly in linearized memory
                            *linear_row_ptr.add(mem_idx) = response_val;
                            mem_idx += 1;
                        }
                    }
                }
                grid_index += 1;
            }
        }

        linear_memories.push(linearized);
    }

    Ok(linear_memories)
}

/// Trait for similarity accumulation with different data types
pub(crate) trait SimilarityAccumulator:
    opencv::core::DataType + Into<f32> + Eq + Ord + Default + std::ops::Add<Output = Self>
{
    const CV_TYPE: i32;

    fn from_f32(f: f32) -> Self;

    fn from_u8(v: u8) -> Self;

    fn accumulate_row(similarity_slice: &mut [Self], linear_memory_slice: &[u8]);
}

impl SimilarityAccumulator for u8 {
    const CV_TYPE: i32 = core::CV_8UC1;

    fn accumulate_row(similarity_slice: &mut [Self], linear_memory_slice: &[u8]) {
        // Use safe slice-based accumulation
        crate::simd_utils::simd_accumulate_u8(similarity_slice, linear_memory_slice);
    }

    fn from_f32(f: f32) -> Self {
        f.round() as Self
    }

    fn from_u8(v: u8) -> Self {
        v
    }
}

impl SimilarityAccumulator for u16 {
    const CV_TYPE: i32 = core::CV_16UC1;

    fn accumulate_row(similarity_slice: &mut [Self], linear_memory_slice: &[u8]) {
        // Use safe slice-based accumulation
        crate::simd_utils::simd_accumulate_u16(similarity_slice, linear_memory_slice);
    }

    fn from_f32(f: f32) -> Self {
        f.round() as Self
    }

    fn from_u8(v: u8) -> Self {
        v as u16
    }
}

/// Generic similarity map computation
fn compute_similarity_map<T: SimilarityAccumulator + 'static>(
    linear_memories: &[Mat],
    templ: &Template,
    src_cols: i32,
    src_rows: i32,
    t: i32,
) -> Mat {
    let w = src_cols / t;
    let h = src_rows / t;

    let mut similarity_map =
        Mat::new_rows_cols_with_default(h, w, T::CV_TYPE, Scalar::all(0.0)).unwrap();

    // Decimated template dimensions
    let wf = (templ.width - 1) / t + 1;
    let hf = (templ.height - 1) / t + 1;

    // Get raw pointer for SIMD access
    let similarity_ptr = similarity_map.data_mut();

    // For each feature, add its contribution to the similarity map
    for feat in &templ.features {
        let label = feat.label as usize;
        if label >= linear_memories.len() {
            continue;
        }

        // Discard feature if out of bounds
        if feat.x < 0 || feat.x >= src_cols || feat.y < 0 || feat.y >= src_rows {
            continue;
        }

        // Access the correct linear memory from the TxT grid (stored as Mat rows)
        let grid_x = feat.x % t;
        let grid_y = feat.y % t;
        let grid_index = grid_y * t + grid_x;

        let memory_grid = &linear_memories[label];
        if grid_index >= memory_grid.rows() {
            continue;
        }

        // Feature position in decimated coordinates
        let fx = feat.x / t;
        let fy = feat.y / t;
        let lm_index = (fy * w + fx) as usize;

        if lm_index >= memory_grid.cols() as usize {
            continue;
        }

        let linear_memory_ptr = memory_grid.ptr(grid_index).unwrap();
        // Add this feature's response to all valid template positions using safe element access
        for y in 0..(h - hf + 1) {
            let base_lm_idx = lm_index + (y * w) as usize;

            if base_lm_idx >= memory_grid.cols() as usize {
                continue;
            }

            let base_lm_idx = lm_index + (y * w) as usize;
            let base_sim_idx = (y * w) as usize;
            let span_x = (w - wf + 1) as usize;

            // Process full 16-byte blocks via simd_accumulate_u8, then handle tail
            // SAFETY: Construct slices for the aligned region
            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut((similarity_ptr as *mut T).add(base_sim_idx), span_x)
            };
            let src_slice =
                unsafe { std::slice::from_raw_parts(linear_memory_ptr.add(base_lm_idx), span_x) };

            T::accumulate_row(dst_slice, src_slice);
        }
    }

    similarity_map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_creation() {
        let feat = Feature::new(10, 20, 3);
        assert_eq!(feat.x, 10);
        assert_eq!(feat.y, 20);
        assert_eq!(feat.label, 3);
    }

    #[test]
    fn test_match_ordering() {
        let template = [vec![Template {
            width: 0,
            height: 0,
            tl_x: 0,
            tl_y: 0,
            pyramid_level: 0,
            features: vec![],
        }]];
        let m1 = Match::new(0, 0, 0.9, "test", 0, &template);
        let m2 = Match::new(0, 0, 0.8, "test", 1, &template);
        assert!(m1 < m2); // Higher similarity comes first
    }
}
