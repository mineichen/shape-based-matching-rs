//! Line2Dup shape-based matching module
//!
//! This module implements a shape-based matching algorithm using gradient orientation features.
//! It's a Rust port of the C++ line2Dup implementation, adapted for OpenCV 4.x.

use opencv::{
    core::{self, Mat, Point2f, Scalar, Size},
    imgproc,
    prelude::*,
};
use std::collections::HashMap;

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

/// A template representing a shape at a specific pyramid level
#[derive(Debug, Clone, Default)]
pub struct Template {
    pub width: i32,
    pub height: i32,
    pub tl_x: i32,
    pub tl_y: i32,
    pub pyramid_level: i32,
    pub features: Vec<Feature>,
}

/// A match result with position, similarity score, and template information
#[derive(Debug, Clone)]
pub struct Match {
    pub x: i32,
    pub y: i32,
    /// Similarity score as percentage (0.0 to 100.0)
    pub similarity: f32,
    pub class_id: String,
    pub template_id: usize,
}

impl Match {
    pub fn new(x: i32, y: i32, similarity: f32, class_id: String, template_id: usize) -> Self {
        Match {
            x,
            y,
            similarity,
            class_id,
            template_id,
        }
    }
}

impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.y == other.y
            && self.similarity == other.similarity
            && self.class_id == other.class_id
    }
}

impl Eq for Match {}

impl PartialOrd for Match {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Match {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort by similarity (descending), then by template_id
        match other.similarity.partial_cmp(&self.similarity) {
            Some(std::cmp::Ordering::Equal) => self.template_id.cmp(&other.template_id),
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
    pub class_templates: HashMap<String, Vec<Vec<Template>>>,
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
    /// Template ID on success
    pub fn add_template(
        &mut self,
        source: &Mat,
        class_id: &str,
        object_mask: Option<&Mat>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
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
        templates.push(template_pyramid);

        Ok(templates.len() - 1)
    }

    /// Add a rotated version of an existing template
    ///
    /// # Arguments
    /// * `class_id` - Class identifier
    /// * `zero_id` - ID of the base (0-degree) template
    /// * `theta` - Rotation angle in degrees (positive = counter-clockwise)
    /// * `center` - Center of rotation in absolute image coordinates
    pub fn add_template_rotate(
        &mut self,
        class_id: &str,
        zero_id: usize,
        theta: f32,
        center: Point2f,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        // Get the base template pyramid
        let base_pyramid = self
            .class_templates
            .get(class_id)
            .and_then(|templates| templates.get(zero_id))
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
        templates.push(rotated_pyramid);

        Ok(templates.len() - 1)
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
    /// Vector of matches sorted by similarity (as percentage 0-100)
    fn match_templates_generic<T: SimilarityAccumulator + 'static>(
        &self,
        source: &Mat,
        threshold: f32,
        class_ids: Option<Vec<String>>,
        masks: Option<&Mat>,
    ) -> Result<Vec<Match>, Box<dyn std::error::Error>> {
        let time = std::time::Instant::now();
        let mask = match masks {
            Some(m) => m.clone(),
            None => Mat::default(),
        };

        let mut matches = Vec::new();

        // Determine which classes to search
        let search_classes: Vec<String> = match class_ids {
            Some(ids) if !ids.is_empty() => ids,
            _ => self.class_templates.keys().cloned().collect(),
        };

        let (source_cols, source_rows) = (source.cols(), source.rows());
        // Build linear memories (quantized gradient maps) for matching
        let pyramid = ColorGradientPyramid::new(
            source,
            &mask,
            self.weak_threshold,
            self.num_features,
            self.strong_threshold,
        )?;
        println!("Time taken to build pyramid: {:?}", time.elapsed());

        // Pre-compute linear memories once for all templates (major optimization!)
        let t = self.t_at_level[0]; // Use first pyramid level
        let mut quantized = Mat::default();
        pyramid.quantize(&mut quantized)?;
        println!("Time taken to quantize: {:?}", time.elapsed());
        let spread_quantized = spread_quantized_image(&quantized, t)?;
        println!("Time taken to spread quantized: {:?}", time.elapsed());
        let linear_memories = compute_and_linearize_response_maps(&spread_quantized, t)?;
        println!(
            "Time taken to compute and linearize response maps: {:?}",
            time.elapsed()
        );

        // // Match all templates sequentially using pre-computed linear memories
        for class_id in search_classes {
            if let Some(template_pyramids) = self.class_templates.get(&class_id) {
                for (template_id, template_pyramid) in template_pyramids.iter().enumerate() {
                    if let Some(templ) = template_pyramid.first() {
                        matches.extend(self.match_template_with_linear_memory::<T>(
                            &linear_memories,
                            templ,
                            threshold,
                            &class_id,
                            template_id,
                            source_cols,
                            source_rows,
                            t,
                        ));
                    }
                }
            }
        }
        println!("Time taken to match_templates: {:?}", time.elapsed());
        // Sort matches by similarity
        matches.sort_unstable();

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
    /// Vector of matches sorted by similarity (as percentage 0-100)
    pub fn match_templates(
        &self,
        source: &Mat,
        threshold: f32,
        class_ids: Option<Vec<String>>,
        masks: Option<&Mat>,
    ) -> Result<Vec<Match>, Box<dyn std::error::Error>> {
        // Determine which classes to search
        let search_classes: Vec<String> = match &class_ids {
            Some(ids) if !ids.is_empty() => ids.clone(),
            _ => self.class_templates.keys().cloned().collect(),
        };

        // Check if any template has 64+ features to decide accumulator type
        let use_u16 = search_classes.iter().any(|class_id| {
            if let Some(template_pyramids) = self.class_templates.get(class_id) {
                template_pyramids.iter().any(|pyramid| {
                    pyramid
                        .first()
                        .is_some_and(|templ| templ.features.len() >= 64)
                })
            } else {
                false
            }
        });

        if use_u16 {
            self.match_templates_generic::<u16>(source, threshold, class_ids, masks)
        } else {
            self.match_templates_generic::<u8>(source, threshold, class_ids, masks)
        }
    }

    /// Get number of templates for a class
    pub fn num_templates(&self, class_id: &str) -> usize {
        self.class_templates
            .get(class_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get all class IDs
    pub fn class_ids(&self) -> Vec<String> {
        self.class_templates.keys().cloned().collect()
    }

    // Match a single template using pre-computed linear memories
    #[allow(clippy::too_many_arguments)]
    fn match_template_with_linear_memory<T: SimilarityAccumulator + 'static>(
        &self,
        linear_memories: &[Mat],
        templ: &Template,
        threshold: f32,
        class_id: &str,
        template_id: usize,
        src_cols: i32,
        src_rows: i32,
        t: i32,
    ) -> impl Iterator<Item = Match> {
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
                        class_id.to_string(),
                        template_id,
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

// ============================================================================
// ColorGradientPyramid - Gradient computation and feature extraction
// ============================================================================

pub struct ColorGradientPyramid {
    pub src: Mat,
    pub mask: Mat,
    pub pyramid_level: i32,
    pub angle: Mat,     // quantized 8-direction bitmask (CV_8U)
    pub angle_ori: Mat, // original orientation in degrees (CV_32F)
    pub magnitude: Mat,
    pub weak_threshold: f32,
    pub num_features: usize,
    pub strong_threshold: f32,
}

impl ColorGradientPyramid {
    pub fn new(
        src: &Mat,
        mask: &Mat,
        weak_threshold: f32,
        num_features: usize,
        strong_threshold: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut pyramid = ColorGradientPyramid {
            src: src.clone(),
            mask: if mask.empty() {
                Mat::new_rows_cols_with_default(
                    src.rows(),
                    src.cols(),
                    core::CV_8UC1,
                    Scalar::all(255.0),
                )?
            } else {
                mask.clone()
            },
            pyramid_level: 0,
            angle: Mat::default(),
            angle_ori: Mat::default(),
            magnitude: Mat::default(),
            weak_threshold,
            num_features,
            strong_threshold,
        };

        pyramid.update()?;
        Ok(pyramid)
    }

    /// Compute gradients and quantized orientations (C++-aligned)
    pub fn update(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Smooth input
        let mut smoothed = Mat::default();
        imgproc::gaussian_blur(
            &self.src,
            &mut smoothed,
            Size::new(7, 7),
            0.0,
            0.0,
            core::BORDER_REPLICATE,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Derivatives and angle/magnitude
        self.magnitude = Mat::default();
        self.angle_ori = Mat::default();

        if self.src.channels() == 1 {
            let mut dx = Mat::default();
            let mut dy = Mat::default();
            imgproc::sobel(
                &smoothed,
                &mut dx,
                core::CV_32F,
                1,
                0,
                3,
                1.0,
                0.0,
                core::BORDER_REPLICATE,
            )?;
            imgproc::sobel(
                &smoothed,
                &mut dy,
                core::CV_32F,
                0,
                1,
                3,
                1.0,
                0.0,
                core::BORDER_REPLICATE,
            )?;
            // magnitude as squared magnitude to match C++
            let mut dx2 = Mat::default();
            let mut dy2 = Mat::default();
            core::multiply(&dx, &dx, &mut dx2, 1.0, -1)?;
            core::multiply(&dy, &dy, &mut dy2, 1.0, -1)?;
            core::add(&dx2, &dy2, &mut self.magnitude, &core::no_array(), -1)?;
            // angle in degrees
            core::phase(&dx, &dy, &mut self.angle_ori, true)?;
        } else {
            // color path: compute per-channel int16 Sobel, then pick channel by max magnitude
            let mut dx3 = Mat::default();
            let mut dy3 = Mat::default();
            imgproc::sobel(
                &smoothed,
                &mut dx3,
                core::CV_16S,
                1,
                0,
                3,
                1.0,
                0.0,
                core::BORDER_REPLICATE,
            )?;
            imgproc::sobel(
                &smoothed,
                &mut dy3,
                core::CV_16S,
                0,
                1,
                3,
                1.0,
                0.0,
                core::BORDER_REPLICATE,
            )?;

            let size = smoothed.size()?;
            let mut dx = Mat::new_size_with_default(size, core::CV_32F, Scalar::all(0.0))?;
            let mut dy = Mat::new_size_with_default(size, core::CV_32F, Scalar::all(0.0))?;
            self.magnitude = Mat::new_size_with_default(size, core::CV_32F, Scalar::all(0.0))?;

            // Iterate rows and choose channel with largest squared magnitude
            for r in 0..size.height {
                for c in 0..size.width {
                    let vx = *dx3.at_2d::<core::Vec3s>(r, c)?;
                    let vy = *dy3.at_2d::<core::Vec3s>(r, c)?;
                    let x0 = vx[0] as i32;
                    let y0 = vy[0] as i32;
                    let x1 = vx[1] as i32;
                    let y1 = vy[1] as i32;
                    let x2 = vx[2] as i32;
                    let y2 = vy[2] as i32;
                    let m0 = x0 * x0 + y0 * y0;
                    let m1 = x1 * x1 + y1 * y1;
                    let m2 = x2 * x2 + y2 * y2;
                    let (xb, yb, mb) = if m0 >= m1 && m0 >= m2 {
                        (x0, y0, m0)
                    } else if m1 >= m0 && m1 >= m2 {
                        (x1, y1, m1)
                    } else {
                        (x2, y2, m2)
                    };
                    *dx.at_2d_mut::<f32>(r, c)? = xb as f32;
                    *dy.at_2d_mut::<f32>(r, c)? = yb as f32;
                    *self.magnitude.at_2d_mut::<f32>(r, c)? = mb as f32;
                }
            }

            core::phase(&dx, &dy, &mut self.angle_ori, true)?;
        }

        // Hysteresis-like quantization similar to C++ hysteresisGradient
        // Step 1: raw quantization to 16 bins (0..360)
        let mut quant_unfiltered = Mat::default();
        self.angle_ori
            .convert_to(&mut quant_unfiltered, core::CV_8U, 16.0 / 360.0, 0.0)?;

        // zero borders
        if quant_unfiltered.rows() > 0 {
            let cols = quant_unfiltered.cols();
            unsafe {
                let p0 = quant_unfiltered.ptr_mut(0)?;
                std::ptr::write_bytes(p0, 0u8, cols as usize);
                let p1 = quant_unfiltered.ptr_mut(quant_unfiltered.rows() - 1)?;
                std::ptr::write_bytes(p1, 0u8, cols as usize);
            }
            for r in 0..quant_unfiltered.rows() {
                *quant_unfiltered.at_2d_mut::<u8>(r, 0)? = 0;
                *quant_unfiltered.at_2d_mut::<u8>(r, cols - 1)? = 0;
            }
        }

        // Mask to 8 bins (keep lower 3 bits)
        for r in 1..(quant_unfiltered.rows() - 1) {
            for c in 1..(quant_unfiltered.cols() - 1) {
                let v = *quant_unfiltered.at_2d::<u8>(r, c)? & 7;
                *quant_unfiltered.at_2d_mut::<u8>(r, c)? = v;
            }
        }

        // Hysteresis filter using magnitude threshold and 3x3 majority
        self.angle = Mat::new_rows_cols_with_default(
            self.angle_ori.rows(),
            self.angle_ori.cols(),
            core::CV_8UC1,
            Scalar::all(0.0),
        )?;
        for r in 1..(self.angle_ori.rows() - 1) {
            for c in 1..(self.angle_ori.cols() - 1) {
                let mag = *self.magnitude.at_2d::<f32>(r, c)?;
                if mag > self.weak_threshold * self.weak_threshold {
                    let mut hist = [0i32; 8];
                    // 3x3 patch histogram
                    for pr in -1..=1 {
                        for pc in -1..=1 {
                            let v = *quant_unfiltered.at_2d::<u8>(r + pr, c + pc)? as usize;
                            hist[v] += 1;
                        }
                    }
                    // find max vote
                    let mut max_votes = 0;
                    let mut index = 0;
                    for (i, &h) in hist.iter().enumerate() {
                        if h > max_votes {
                            max_votes = h;
                            index = i;
                        }
                    }
                    if max_votes >= 5 {
                        *self.angle.at_2d_mut::<u8>(r, c)? = 1u8 << index;
                    }
                }
            }
        }

        Ok(())
    }

    /// Quantize gradients: copy precomputed bitmask `angle` with mask (C++ behavior)
    pub fn quantize(&self, dst: &mut Mat) -> Result<(), Box<dyn std::error::Error>> {
        *dst = Mat::new_rows_cols_with_default(
            self.angle.rows(),
            self.angle.cols(),
            core::CV_8UC1,
            Scalar::all(0.0),
        )?;
        self.angle.copy_to_masked(dst, &self.mask)?;
        Ok(())
    }

    /// Extract a template from the current pyramid level
    pub fn extract_template(
        &self,
        templ: &mut Template,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Erode mask once to avoid border features (like C++)
        let mut local_mask = Mat::default();
        if !self.mask.empty() {
            imgproc::erode(
                &self.mask,
                &mut local_mask,
                &Mat::default(),
                core::Point::new(-1, -1),
                1,
                core::BORDER_REPLICATE,
                imgproc::morphology_default_border_value()?,
            )?;
        }

        let no_mask = local_mask.empty();
        let threshold_sq = self.strong_threshold * self.strong_threshold;
        let nms_kernel = 5i32;
        let k = nms_kernel / 2;

        let mut magnitude_valid = Mat::new_rows_cols_with_default(
            self.magnitude.rows(),
            self.magnitude.cols(),
            core::CV_8UC1,
            Scalar::all(255.0),
        )?;

        let mut candidates: Vec<Candidate> = Vec::new();

        for r in k..(self.magnitude.rows() - k) {
            for c in k..(self.magnitude.cols() - k) {
                let mask_ok = no_mask || *local_mask.at_2d::<u8>(r, c)? > 0;
                if !mask_ok {
                    continue;
                }

                let mut score = 0.0f32;
                if *magnitude_valid.at_2d::<u8>(r, c)? > 0 {
                    score = *self.magnitude.at_2d::<f32>(r, c)?;
                    let mut is_max = true;
                    'outer: for dr in -k..=k {
                        for dc in -k..=k {
                            if dr == 0 && dc == 0 {
                                continue;
                            }
                            if score < *self.magnitude.at_2d::<f32>(r + dr, c + dc)? {
                                score = 0.0;
                                is_max = false;
                                break 'outer;
                            }
                        }
                    }
                    if is_max {
                        for dr in -k..=k {
                            for dc in -k..=k {
                                if dr == 0 && dc == 0 {
                                    continue;
                                }
                                *magnitude_valid.at_2d_mut::<u8>(r + dr, c + dc)? = 0;
                            }
                        }
                    }
                }

                // require strong magnitude and a quantized angle bit present
                if score > threshold_sq {
                    let ang = *self.angle.at_2d::<u8>(r, c)?;
                    if ang > 0 {
                        // convert angle bitmask to label index as in C++ getLabel
                        let label = bit_to_label(ang);
                        let mut feat = Feature::new(c, r, label);
                        feat.theta = *self.angle_ori.at_2d::<f32>(r, c)?;
                        candidates.push(Candidate { f: feat, score });
                    }
                }
            }
        }

        if candidates.len() <= 4 {
            return Ok(false);
        }

        // Sort high score first
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        // Select scattered features (use fixed distance heuristic close to C++)
        templ.features = select_scattered_features(
            &candidates,
            self.num_features,
            candidates.len() as f32 / self.num_features as f32 + 1.0,
        );

        // Set meta
        templ.width = -1;
        templ.height = -1;
        templ.pyramid_level = self.pyramid_level;

        Ok(!templ.features.is_empty())
    }

    /// Downsample the pyramid
    pub fn pyr_down(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut src_down = Mat::default();
        let mut mask_down = Mat::default();

        imgproc::pyr_down(
            &self.src,
            &mut src_down,
            Size::default(),
            core::BORDER_DEFAULT,
        )?;
        imgproc::pyr_down(
            &self.mask,
            &mut mask_down,
            Size::default(),
            core::BORDER_DEFAULT,
        )?;

        self.src = src_down;
        self.mask = mask_down;
        self.pyramid_level += 1;

        self.update()?;

        Ok(())
    }
}

struct Candidate {
    f: Feature,
    score: f32,
}

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

fn bit_to_label(bitmask: u8) -> i32 {
    match bitmask {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        64 => 6,
        128 => 7,
        _ => 0,
    }
}

/// Select scattered features from candidates
fn select_scattered_features(
    candidates: &[Candidate],
    num_features: usize,
    distance: f32,
) -> Vec<Feature> {
    let mut features: Vec<Feature> = Vec::new();
    let distance_sq = distance * distance;

    for candidate in candidates {
        if features.len() >= num_features {
            break;
        }

        // Check distance to all existing features
        let mut keep = true;
        for feat in &features {
            let dx = (candidate.f.x - feat.x) as f32;
            let dy = (candidate.f.y - feat.y) as f32;
            if dx * dx + dy * dy < distance_sq {
                keep = false;
                break;
            }
        }

        if keep {
            features.push(candidate.f.clone());
        }
    }

    features
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
    if spread_quantized.cols() % 16 != 0 || spread_quantized.rows() % 16 != 0 {
        return Err(format!(
            "Image width and height must each be divisible by 16. Got {}x{}",
            spread_quantized.cols(),
            spread_quantized.rows()
        )
        .into());
    }
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

        // Outer two loops iterate over top-left T×T starting pixels
        let mut grid_index = 0;
        for r_start in 0..t {
            for c_start in 0..t {
                // Inner two loops: compute response and linearize in one pass
                let mut mem_idx = 0;
                for r in (r_start..rows).step_by(t as usize) {
                    for c in (c_start..cols).step_by(t as usize) {
                        let val = *spread_quantized.at_2d::<u8>(r, c)?;

                        // Split into LSB4 and MSB4
                        let lsb4 = (val & 15) as usize;
                        let msb4 = ((val & 240) >> 4) as usize;

                        // Use LUT to compute response (like C++)
                        let response_val = SIMILARITY_LUT[lut_offset + lsb4]
                            .max(SIMILARITY_LUT[lut_offset + 16 + msb4]);

                        // Store directly in linearized memory
                        *linearized.at_2d_mut::<u8>(grid_index, mem_idx)? = response_val;
                        mem_idx += 1;
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
    opencv::core::DataType + Into<f32> + Eq + Ord
{
    const CV_TYPE: i32;

    fn from_f32(f: f32) -> Self;

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
        let m1 = Match::new(0, 0, 0.9, "test".to_string(), 0);
        let m2 = Match::new(0, 0, 0.8, "test".to_string(), 1);
        assert!(m1 < m2); // Higher similarity comes first
    }
}
