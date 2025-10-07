//! Line2Dup shape-based matching module
//!
//! This module implements a shape-based matching algorithm using gradient orientation features.
//! It's a Rust port of the C++ line2Dup implementation, adapted for OpenCV 4.x.

use opencv::{
    core::{self, Mat, Point2f, Scalar},
    prelude::*,
};
use std::collections::HashMap;

use crate::image_buffer::ImageBuffer;
use crate::match_result::{Match, RawMatch};
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

// ============================================================================
// PUBLIC API - Main Detector
// ============================================================================

/// Main detector for shape-based matching
pub struct Detector {
    weak_threshold: f32,
    strong_threshold: f32,
    /// T-shift values for each pyramid level (log2 of T).
    /// For example, T=4 -> shift=2, T=8 -> shift=3.
    /// This allows efficient bit shift operations: `1 << t_shift` gives T value.
    t_shifts: Vec<u8>,
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
        #[cfg(feature = "profile")]
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
        #[cfg(feature = "profile")]
        println!("- Time taken to build pyramid: {:?}", time.elapsed());

        // Pre-compute linear memories for each pyramid level
        let mut linear_memory_pyramid: Vec<[ImageBuffer; 8]> = Vec::new();
        let mut pyramid_sizes: Vec<(i32, i32)> = Vec::new();

        for level in 0..self.t_shifts.len() {
            let t_shift = self.t_shifts[level as usize];
            let t = 1i32 << t_shift; // Compute T from shift: T = 2^t_shift
            let mut quantized = Mat::default();
            pyramid.quantize(&mut quantized)?;

            pyramid_sizes.push((quantized.cols(), quantized.rows()));

            let spread_quantized = spread_quantized_image(&quantized, t)?;
            let linear_memories = compute_and_linearize_response_maps(&spread_quantized, t_shift);
            linear_memory_pyramid.push(linear_memories);

            // Downsample for next level (except last)
            if level < self.t_shifts.len() - 1 {
                pyramid.pyr_down()?;
            }
        }
        #[cfg(feature = "profile")]
        println!(
            "- Time taken to build linear memory pyramid: {:?}",
            time.elapsed()
        );
        // Match all templates using coarse-to-fine pyramid refinement (like C++)
        let mut matches = Vec::new();
        let mut candidates_buffer: Vec<RawMatch<T>> = Vec::new();

        for class_id in &search_classes {
            if let Some(template_pyramids) = self.class_templates.get(*class_id) {
                #[cfg(feature = "profile")]
                println!(
                    "- Processing class '{class_id}' with {} templates: {:?}",
                    template_pyramids.0.len(),
                    time.elapsed()
                );
                for (template_id, template_pyramid) in template_pyramids.0.iter().enumerate() {
                    #[cfg(feature = "profile")]
                    let subtime = std::time::Instant::now();
                    if template_pyramid.is_empty() {
                        continue;
                    }

                    let templ_len = template_pyramid[0].features.len() as f32;
                    let similarity_multiplier = 100.0 / (4.0 * templ_len);
                    let raw_threshold = T::from_f32((threshold * 4.0 * templ_len) / 100.0);

                    // Reuse candidates buffer
                    self.match_template_pyramid::<T>(
                        &linear_memory_pyramid,
                        &pyramid_sizes,
                        template_pyramid,
                        raw_threshold,
                        &mut candidates_buffer,
                    );

                    // Convert candidates to matches
                    matches.extend(candidates_buffer.iter().map(|raw_match| {
                        let similarity = raw_match.raw_score.into() * similarity_multiplier;
                        Match::new(
                            raw_match.x,
                            raw_match.y,
                            similarity,
                            class_id,
                            template_id,
                            template_pyramids.0.as_slice(),
                        )
                    }));

                    #[cfg(feature = "profile")]
                    println!(
                        "-- Time taken to match template {template_id}: {:?}",
                        subtime.elapsed(),
                    );
                }
            }
        }

        #[cfg(feature = "profile")]
        println!("- Time taken after all classes: {:?}", time.elapsed());
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
        linear_memory_pyramid: &[[ImageBuffer; 8]],
        pyramid_sizes: &[(i32, i32)],
        template_pyramid: &[Template],
        raw_threshold: T,
        candidates: &mut Vec<RawMatch<T>>,
    ) {
        // Start at the coarsest pyramid level (last in array)
        let lowest_level = (self.t_shifts.len() - 1) as usize;
        let lowest_t_shift = self.t_shifts[lowest_level];
        let (src_cols, src_rows) = pyramid_sizes[lowest_level];

        // Get template at coarsest level
        #[cfg(feature = "profile")]
        let time = std::time::Instant::now();
        // Match at coarsest level to get initial candidates (reuse buffer)
        candidates.clear();
        candidates.extend(self.match_template_with_linear_memory::<T>(
            &linear_memory_pyramid[lowest_level],
            &template_pyramid[lowest_level],
            raw_threshold,
            src_cols,
            src_rows,
            lowest_t_shift,
        ));

        // #[cfg(feature = "profile")]
        // println!(
        //     "--- Found {} candidates at level {lowest_level} for refinemnt: {:?}",
        //     candidates.len(),
        //     time.elapsed()
        // );
        // #[cfg(feature = "profile")]
        // for candidate in &candidates {
        //     println!("---- Candidate: {:?}", candidate);
        // }

        // Refine candidates by marching up the pyramid (from coarse to fine)
        for level in (0..lowest_level).rev() {
            let t_shift = self.t_shifts[level];
            let t = 1i32 << t_shift; // T = 2^t_shift
            let (src_cols, src_rows) = pyramid_sizes[level];
            let template = &template_pyramid[level];
            let border = 8 * t;
            let _offset = t / 2 + (t % 2 - 1);

            let max_x = src_cols - template.width - border;
            let max_y = src_rows - template.height - border;

            candidates.retain_mut(|candidate| {
                // Scale up position from previous level (2x)
                let x = candidate.x * 2 + 1;
                let y = candidate.y * 2 + 1;

                // Require 8 (reduced) rows/cols to the up/left
                if x < border || y < border || x > max_x || y > max_y {
                    return false;
                }

                // Search in a 5x5 window around the scaled position
                candidate.raw_score = T::default();

                const NEIGHBOURHOOD: i32 = 2;

                for dy in -NEIGHBOURHOOD..=NEIGHBOURHOOD {
                    for dx in -NEIGHBOURHOOD..=NEIGHBOURHOOD {
                        let search_x = x + dx * t;
                        let search_y = y + dy * t;

                        if search_x < border
                            || search_y < border
                            || search_x > max_x
                            || search_y > max_y
                        {
                            continue;
                        }

                        // Compute raw score at this position
                        let raw_score = self.compute_similarity_at_position::<T>(
                            &linear_memory_pyramid[level],
                            template,
                            search_x,
                            search_y,
                            src_cols,
                            src_rows,
                            t_shift,
                        );

                        if raw_score > candidate.raw_score {
                            candidate.x = search_x;
                            candidate.y = search_y;
                            candidate.raw_score = raw_score;
                        }
                    }
                }

                // Keep refined match if it still passes threshold
                candidate.raw_score >= raw_threshold
            });

            // #[cfg(feature = "profile")]
            // println!(
            //     "--- Refining candidates at level {level}: {:?}",
            //     time.elapsed()
            // );
        }
    }

    #[inline(always)]
    fn compute_similarity_at_position<T: SimilarityAccumulator + 'static>(
        &self,
        linear_memories: &[ImageBuffer; 8],
        templ: &Template,
        x: i32,
        y: i32,
        src_cols: i32,
        src_rows: i32,
        t_shift: u8,
    ) -> T {
        // Precompute constants outside the hot loop
        let t = 1i32 << t_shift; // T = 2^t_shift
        let w = src_cols >> t_shift; // Efficient division by power of 2
        let t_mask = t - 1; // For efficient modulo with power of 2

        let mut score: T = T::default();

        // Pre-cache base pointers and strides for all 8 orientations
        // This eliminates repeated Mat access overhead in the hot loop
        let mut memory_base_ptrs: [*const u8; 8] = [std::ptr::null(); 8];
        let mut stride_cache: [usize; 8] = [0; 8];

        unsafe {
            for i in 0..8 {
                let mat = linear_memories.get_unchecked(i);
                memory_base_ptrs[i] = mat.as_ptr();
                stride_cache[i] = mat.cols() as usize;
            }
        }

        // Hot loop with cached pointers and strides
        for feat in &templ.features {
            let label = feat.label as usize;
            debug_assert!(label < linear_memories.len());

            let feat_x = feat.x + x;
            let feat_y = feat.y + y;

            // Check bounds
            debug_assert!(feat_x >= 0 && feat_x < src_cols && feat_y >= 0 && feat_y < src_rows);

            // Access the correct linear memory from the TxT grid using bit operations
            let grid_x = feat_x & t_mask; // Efficient modulo for power of 2
            let grid_y = feat_y & t_mask;
            let grid_index = (grid_y * t + grid_x) as usize;

            // Feature position in decimated coordinates using bit shift
            let fx = feat_x >> t_shift;
            let fy = feat_y >> t_shift;
            let lm_index = (fy * w + fx) as usize;

            unsafe {
                // Use cached pointer and stride instead of repeated Mat access
                let base_ptr = memory_base_ptrs[label];
                let stride = stride_cache[label];
                let row_ptr = base_ptr.add(grid_index * stride);
                let response = T::from_u8(*row_ptr.add(lm_index));
                score = score + response;
            }
        }

        // Return raw score
        score
    }

    // Match a single template using pre-computed linear memories
    #[allow(clippy::too_many_arguments)]
    fn match_template_with_linear_memory<'a, T: SimilarityAccumulator + 'static>(
        &'a self,
        linear_memories: &[ImageBuffer; 8],
        templ: &Template,
        raw_threshold: T,
        src_cols: i32,
        src_rows: i32,
        t_shift: u8,
    ) -> impl Iterator<Item = RawMatch<T>> {
        // Compute T using bit shift (T = 2^t_shift)
        let t = 1i32 << t_shift;
        // Extract matches from similarity map using efficient bit operations
        let w = src_cols >> t_shift; // Efficient division by power of 2
        let h = src_rows >> t_shift;

        // Calculate offset to center the match position (like C++ version)
        let offset = t / 2 + (t % 2 - 1);

        let similarity_map =
            compute_similarity_map::<T>(linear_memories, templ, src_cols, src_rows, t_shift);
        let similarity_map_slice = unsafe {
            std::slice::from_raw_parts(
                similarity_map.ptr(0).unwrap() as *const T,
                h as usize * w as usize,
            )
        };

        (0..h)
            .flat_map(move |y| (0..w).map(move |x| (y, x)))
            .zip(similarity_map_slice.iter())
            .filter_map(move |((y, x), &raw_score)| {
                let _safety_dont_drop_similarity_map = &similarity_map;
                // static CTR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                // let c = CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // println!("ctr: {c}",);
                (raw_score >= raw_threshold).then(|| RawMatch {
                    x: x * t + offset,
                    y: y * t + offset,
                    raw_score,
                })
            })
    }
}

/// Builder for `Detector` configuration. Call `build()` to create the `Detector`.
pub struct DetectorBuilder {
    num_features: usize,
    t_shifts: Vec<u8>,
    weak_threshold: f32,
    strong_threshold: f32,
    pending_templates: Vec<PendingTemplate>,
}

impl Default for DetectorBuilder {
    fn default() -> Self {
        DetectorBuilder {
            num_features: 63,
            t_shifts: vec![2, 3], // T=4 (2^2), T=8 (2^3)
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

    /// Set pyramid T values (they will be converted to shifts internally).
    /// T values must be powers of 2 (e.g., 4, 8, 16).
    pub fn pyramid_levels(mut self, t_values: Vec<u8>) -> Self {
        // Convert T values to shifts, validating they're powers of 2
        self.t_shifts = t_values
            .into_iter()
            .map(|t| {
                assert!(
                    t > 0 && (t & (t - 1)) == 0,
                    "Pyramid level T value must be a power of 2, got {}",
                    t
                );
                t.trailing_zeros() as u8
            })
            .collect();
        self
    }

    /// Set pyramid T-shift values directly (log2 of T values).
    /// For example, for T=4 use shift=2, for T=8 use shift=3.
    pub fn pyramid_t_shifts(mut self, t_shifts: Vec<u8>) -> Self {
        self.t_shifts = t_shifts;
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
            t_shifts: self.t_shifts,
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
            for level in 0..detector.t_shifts.len() as u8 {
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
                if level < detector.t_shifts.len() as u8 - 1 {
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

    let rows = quantized.rows();
    let cols = quantized.cols();

    // Fill in spread gradient image (section 2.3 of paper)
    // For each offset (r, c) in TxT grid, OR source[r:,c:] into destination
    // Optimized: use u64 loads/stores for 8x throughput
    for r in 0..t {
        let height = rows - r;
        for c in 0..t {
            let width = cols - c;

            unsafe {
                for y in 0..height {
                    let src_ptr = quantized.ptr((y + r) as i32)?.add(c as usize);
                    let dst_ptr = spread.ptr_mut(y as i32)?;

                    let mut x = 0usize;
                    let width_usize = width as usize;

                    // Process 8 bytes at a time using u64
                    while x + 8 <= width_usize {
                        let src_u64 = (src_ptr.add(x) as *const u64).read_unaligned();
                        let dst_u64 = (dst_ptr.add(x) as *const u64).read_unaligned();
                        (dst_ptr.add(x) as *mut u64).write_unaligned(dst_u64 | src_u64);
                        x += 8;
                    }

                    // Handle remaining bytes
                    while x < width_usize {
                        *dst_ptr.add(x) |= *src_ptr.add(x);
                        x += 1;
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
fn compute_and_linearize_response_maps(spread_quantized: &Mat, t_shift: u8) -> [ImageBuffer; 8] {
    // Align with C++ precondition: dimensions divisible by 16 and by t
    let mask = (t_shift - 1) as i32;
    assert!(
        spread_quantized.rows() & mask == 0 && spread_quantized.cols() & mask == 0,
        "Image width and height must be divisible by T"
    );

    let t = 1i32 << t_shift;
    let rows = spread_quantized.rows();
    let cols = spread_quantized.cols();
    let mem_width = cols >> t_shift;
    let mem_height = rows >> t_shift;
    let mem_size = mem_width * mem_height;

    // Process each orientation
    std::array::from_fn(|ori| {
        let lut_offset = 32 * ori;

        // Create buffer with T×T rows and mem_size cols
        let mut linearized = ImageBuffer::new_zeroed(t * t, mem_size);

        // Use raw pointer access for maximum performance
        let mut grid_index = 0;
        for r_start in 0..t {
            for c_start in 0..t {
                unsafe {
                    let linear_row_ptr = linearized
                        .as_mut_ptr()
                        .add((grid_index * mem_size) as usize);
                    let mut mem_idx = 0;

                    for r in (r_start..rows).step_by(t as usize) {
                        let src_row_ptr = spread_quantized.ptr(r).unwrap();
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
        linearized
    })
}

/// Trait for similarity accumulation with different data types
pub(crate) trait SimilarityAccumulator:
    opencv::core::DataType
    + Into<f32>
    + Eq
    + Ord
    + Default
    + std::ops::Add<Output = Self>
    + std::fmt::Debug
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
    linear_memories: &[ImageBuffer; 8],
    templ: &Template,
    src_cols: i32,
    src_rows: i32,
    t_shift: u8,
) -> Mat {
    // Compute T and dimensions using bit shifts (T = 2^t_shift)
    let t = 1i32 << t_shift;
    let w = src_cols >> t_shift; // Efficient division by power of 2
    let h = src_rows >> t_shift;
    let t_mask = t - 1; // For efficient modulo with power of 2

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
        debug_assert!(label < linear_memories.len());

        // Discard feature if out of bounds
        debug_assert!(feat.x >= 0 && feat.x < src_cols && feat.y >= 0 && feat.y < src_rows);

        // Access the correct linear memory from the TxT grid using bit operations
        let grid_x = feat.x & t_mask; // Efficient modulo for power of 2
        let grid_y = feat.y & t_mask;
        let grid_index = grid_y * t + grid_x;

        // Safety: label is < 8 and linear_memories is [ReadOnlyBuffer; 8]
        let memory_grid = unsafe { linear_memories.get_unchecked(label) };
        debug_assert!(grid_index < memory_grid.rows());

        // Feature position in decimated coordinates using bit shift
        let fx = feat.x >> t_shift; // Efficient division by power of 2
        let fy = feat.y >> t_shift;
        let lm_index = (fy * w + fx) as usize;

        if lm_index >= memory_grid.cols() as usize {
            continue;
        }

        let linear_memory_ptr = unsafe {
            memory_grid
                .as_ptr()
                .add((grid_index * memory_grid.cols()) as usize)
        };
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
