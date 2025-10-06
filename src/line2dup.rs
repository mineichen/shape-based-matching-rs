//! Line2Dup shape-based matching module
//!
//! This module implements a shape-based matching algorithm using gradient orientation features.
//! It's a Rust port of the C++ line2Dup implementation, adapted for OpenCV 4.x.

use opencv::{
    core::{self, Mat, Point2f, Scalar},
    prelude::*,
};
use std::collections::HashMap;

use crate::pyramid::{ColorGradientPyramid, Template};

/// Handle to a template that can be used to add rotated/scaled variants

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

pub(crate) struct RawMatch {
    pub x: i32,
    pub y: i32,
    pub similarity: f32,
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
    strong_threshold: f32,
    pyramid_levels: Vec<u8>,
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
    /// Create a new builder; call `build()` to get a `Detector`.
    pub fn builder() -> DetectorBuilder {
        DetectorBuilder::default()
    }

    // add_template moved to DetectorBuilder; Detector is read-only post-build

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
        let mut pyramid =
            ColorGradientPyramid::new(source, &mask, self.weak_threshold, self.strong_threshold)?;
        println!("Time taken to build pyramid: {:?}", time.elapsed());

        // Pre-compute linear memories for each pyramid level
        let mut linear_memory_pyramid: Vec<Vec<Mat>> = Vec::new();
        let mut pyramid_sizes: Vec<(i32, i32)> = Vec::new();

        for level in 0..self.pyramid_levels.len() {
            let t = self.pyramid_levels[level as usize] as i32;
            let mut quantized = Mat::default();
            pyramid.quantize(&mut quantized)?;

            pyramid_sizes.push((quantized.cols(), quantized.rows()));

            let spread_quantized = spread_quantized_image(&quantized, t)?;
            let linear_memories = compute_and_linearize_response_maps(&spread_quantized, t)?;
            linear_memory_pyramid.push(linear_memories);

            // Downsample for next level (except last)
            if level < self.pyramid_levels.len() - 1 {
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

                    matches.extend(
                        self.match_template_pyramid::<T>(
                            &linear_memory_pyramid,
                            &pyramid_sizes,
                            template_pyramid,
                            threshold,
                        )
                        .into_iter()
                        .map(move |raw_match| Match {
                            x: raw_match.x,
                            y: raw_match.y,
                            similarity: raw_match.similarity,
                            class_id: class_id,
                            templates: template_pyramids.0.as_slice(),
                            template_id,
                        }),
                    )
                }
            }
        }

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
    ) -> Result<Vec<Match<'a>>, Box<dyn std::error::Error>> {
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
            Ok(self.match_templates_generic::<u16>(
                source,
                threshold,
                Some(&search_classes),
                masks,
            )?)
        } else {
            Ok(self.match_templates_generic::<u8>(
                source,
                threshold,
                Some(&search_classes),
                masks,
            )?)
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
    ) -> Vec<RawMatch> {
        // Start at the coarsest pyramid level (last in array)
        let lowest_level = (self.pyramid_levels.len() - 1) as usize;
        let lowest_t = self.pyramid_levels[lowest_level] as i32;
        let (src_cols, src_rows) = pyramid_sizes[lowest_level];

        // Get template at coarsest level
        let coarse_template = &template_pyramid[lowest_level];

        // Match at coarcompute_similarity_at_positionsest level to get initial candidates
        let mut candidates: Vec<_> = self
            .match_template_with_linear_memory::<T>(
                &linear_memory_pyramid[lowest_level],
                coarse_template,
                threshold,
                src_cols,
                src_rows,
                lowest_t,
            )
            .collect();

        // Refine candidates by marching up the pyramid (from coarse to fine)
        for level in (0..lowest_level).rev() {
            let t = self.pyramid_levels[level] as i32;
            let (src_cols, src_rows) = pyramid_sizes[level];
            let template = &template_pyramid[level];
            let border = 8 * t;
            let _offset = t / 2 + (t % 2 - 1);

            let max_x = src_cols - template.width - border;
            let max_y = src_rows - template.height - border;

            let mut refined_candidates = Vec::with_capacity(candidates.len());

            for candidate in candidates.into_iter() {
                // Scale up position from previous level (2x)
                let x = candidate.x * 2 + 1;
                let y = candidate.y * 2 + 1;

                // Require 8 (reduced) rows/cols to the up/left
                if x < border || y < border || x > max_x || y > max_y {
                    continue;
                }

                // Search in a 5x5 window around the scaled position
                let mut best_match = candidate;
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
        src_cols: i32,
        src_rows: i32,
        t: i32,
    ) -> impl Iterator<Item = RawMatch> {
        // Extract matches from similarity map
        let w = src_cols / t;
        let h = src_rows / t;

        // Calculate offset to center the match position (like C++ version)
        let offset = t / 2 + (t % 2 - 1);

        let similarity_map =
            compute_similarity_map::<T>(linear_memories, templ, src_cols, src_rows, t);
        let similarity_map_slice = unsafe {
            std::slice::from_raw_parts(
                similarity_map.ptr(0).unwrap() as *const T,
                h as usize * w as usize,
            )
        };
        let templ_len = templ.features.len();

        // Pre-calculate threshold in terms of raw_score to avoid repeated calculations
        let raw_threshold = T::from_f32((threshold * 4.0 * templ_len as f32) / 100.0);
        (0..h)
            .flat_map(move |y| (0..w).map(move |x| (y, x)))
            .zip(similarity_map_slice.iter())
            .filter_map(move |((y, x), &raw_score)| {
                let _safety_dont_drop_similarity_map = &similarity_map;
                // static CTR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                // let c = CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // println!("ctr: {c}",);
                (raw_score >= raw_threshold).then(|| {
                    let similarity = (raw_score.into() * 100.0) / (4.0 * templ_len as f32);
                    RawMatch {
                        x: x * t + offset,
                        y: y * t + offset,
                        similarity,
                    }
                })
            })
    }
}

/// Builder for `Detector` configuration. Call `build()` to create the `Detector`.
pub struct DetectorBuilder {
    num_features: usize,
    pyramid_levels: Vec<u8>,
    weak_threshold: f32,
    strong_threshold: f32,
    pending_templates: Vec<PendingTemplate>,
}

impl Default for DetectorBuilder {
    fn default() -> Self {
        DetectorBuilder {
            num_features: 63,
            pyramid_levels: vec![4, 8],
            weak_threshold: 30.0,
            strong_threshold: 60.0,
            pending_templates: Vec::new(),
        }
    }
}

impl DetectorBuilder {
    pub fn num_features(mut self, num_features: usize) -> Self {
        self.num_features = num_features;
        self
    }

    pub fn pyramid_levels(mut self, pyramid_levels: Vec<u8>) -> Self {
        self.pyramid_levels = pyramid_levels;
        self
    }

    pub fn weak_threshold(mut self, weak_threshold: f32) -> Self {
        self.weak_threshold = weak_threshold;
        self
    }

    pub fn strong_threshold(mut self, strong_threshold: f32) -> Self {
        self.strong_threshold = strong_threshold;
        self
    }

    /// Add a template and configure it via a closure, allowing chaining
    /// Note: No zero angle template is added implicitly - all rotations must be explicitly specified
    pub fn with_template<F>(mut self, class_id: &str, mask: &Mat, f: F) -> Self
    where
        F: FnOnce(TemplateConfigHandle),
    {
        let idx = self.pending_templates.len();
        self.pending_templates.push(PendingTemplate {
            source: mask.clone(), // Use mask as source
            mask: mask.clone(),
            class_id: class_id.to_string(),
            rotations: Vec::new(),
        });
        let cfg = TemplateConfigHandle {
            builder: &mut self,
            idx,
        };
        f(cfg);
        self
    }

    // add_template removed - use with_template instead

    /// Queue a rotated variant for a previously queued template.
    pub fn add_rotated(&mut self, handle: &TemplateBuildHandle, theta: f32, center: Point2f) {
        if let Some(p) = self.pending_templates.get_mut(handle.idx) {
            p.rotations.push((theta, center));
        }
    }

    pub fn build(self) -> Detector {
        let mut detector = Detector {
            weak_threshold: self.weak_threshold,
            strong_threshold: self.strong_threshold,
            pyramid_levels: self.pyramid_levels,
            class_templates: HashMap::new(),
        };

        for p in self.pending_templates {
            // Inline previous add_template logic
            let mut pyramid = ColorGradientPyramid::new(
                &p.source,
                &p.mask,
                detector.weak_threshold,
                detector.strong_threshold,
            )
            .expect("pyramid creation failed");

            let mut template_pyramid = Vec::new();
            for level in 0..detector.pyramid_levels.len() as u8 {
                let mut templ = Template {
                    pyramid_level: level,
                    ..Default::default()
                };
                if pyramid
                    .extract_template(&mut templ, self.num_features)
                    .expect("extract_template failed")
                {
                    template_pyramid.push(templ);
                }
                if level < detector.pyramid_levels.len() as u8 - 1 {
                    pyramid.pyr_down().expect("pyr_down failed");
                }
            }
            crop_templates(&mut template_pyramid);

            let templates = detector
                .class_templates
                .entry(p.class_id.clone())
                .or_default();
            let template_id = templates.insert(template_pyramid);

            for (theta, center) in p.rotations {
                // Avoid duplicating base template (0°)
                if theta.rem_euclid(360.0) == 0.0 {
                    continue;
                }
                let _ =
                    detector.add_template_rotate_internal(&p.class_id, template_id, theta, center);
            }
        }

        detector
    }
}

#[derive(Clone, Copy)]
pub struct TemplateBuildHandle {
    idx: usize,
}

struct PendingTemplate {
    source: Mat,
    mask: Mat,
    class_id: String,
    rotations: Vec<(f32, Point2f)>,
}

/// Narrow configuration handle exposed to with_template callback
pub struct TemplateConfigHandle<'a> {
    builder: &'a mut DetectorBuilder,
    idx: usize,
}

impl<'a> TemplateConfigHandle<'a> {
    pub fn add_rotated(&mut self, theta: f32, center: Point2f) {
        self.builder.pending_templates[self.idx]
            .rotations
            .push((theta, center));
    }

    pub fn add_rotated_range<T: Into<f32>>(
        &mut self,
        theta_range: impl Iterator<Item = T>,
        center: Point2f,
    ) {
        for theta in theta_range {
            self.add_rotated(theta.into(), center);
        }
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
