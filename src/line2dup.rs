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

use crate::simd_utils::{simd_accumulate_u16, simd_accumulate_u8};

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
#[derive(Debug, Clone)]
pub struct Template {
    pub width: i32,
    pub height: i32,
    pub tl_x: i32,
    pub tl_y: i32,
    pub pyramid_level: i32,
    pub features: Vec<Feature>,
}

impl Template {
    pub fn new() -> Self {
        Template {
            width: 0,
            height: 0,
            tl_x: 0,
            tl_y: 0,
            pyramid_level: 0,
            features: Vec::new(),
        }
    }
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
        Detector::with_params(128, vec![4, 8], 30.0, 60.0)
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
            let mut templ = Template::new();
            templ.pyramid_level = level;

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
            .or_insert_with(Vec::new);
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
            let mut new_templ = Template::new();
            new_templ.pyramid_level = base_templ.pyramid_level;

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
            .or_insert_with(Vec::new);
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
    pub fn match_templates_generic<T: SimilarityAccumulator + 'static>(
        &self,
        source: &Mat,
        threshold: f32,
        class_ids: Option<Vec<String>>,
        masks: Option<&Mat>,
    ) -> Result<Vec<Match>, Box<dyn std::error::Error>> {
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

        // Build linear memories (quantized gradient maps) for matching
        let pyramid = ColorGradientPyramid::new(
            source,
            &mask,
            self.weak_threshold,
            self.num_features,
            self.strong_threshold,
        )?;

        // Pre-compute linear memories once for all templates (major optimization!)
        let t = self.t_at_level[0]; // Use first pyramid level
        let mut quantized = Mat::default();
        pyramid.quantize(&mut quantized)?;
        let spread_quantized = spread_quantized_image(&quantized, t)?;
        let response_maps = compute_response_maps(&spread_quantized)?;
        let linear_memories = linearize_response_maps(&response_maps, t)?;

        // Match all templates sequentially using pre-computed linear memories
        for class_id in search_classes {
            if let Some(template_pyramids) = self.class_templates.get(&class_id) {
                for (template_id, template_pyramid) in template_pyramids.iter().enumerate() {
                    if let Some(templ) = template_pyramid.first() {
                        let template_matches = self.match_template_with_linear_memory::<T>(
                                &linear_memories,
                                templ,
                                threshold,
                                &class_id,
                                template_id,
                                source.cols(),
                                source.rows(),
                                t,
                        )?;
                        matches.extend(template_matches);
                    }
                }
            }
        }

        // Sort matches by similarity
        matches.sort();

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
                    pyramid.first().map_or(false, |templ| templ.features.len() >= 64)
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
    ) -> Result<Vec<Match>, Box<dyn std::error::Error>> {
        // Extract matches from similarity map
        let mut matches = Vec::new();
        let w = src_cols / t;
        let h = src_rows / t;
        
        // Calculate offset to center the match position (like C++ version)
        let offset = t / 2 + (t % 2 - 1);

        let similarity_map = compute_similarity_map::<T>(linear_memories, templ, src_cols, src_rows, t)?;
        
        for y in 0..h {
            for x in 0..w {
                let raw_score = match std::any::TypeId::of::<T>() {
                    id if id == std::any::TypeId::of::<u8>() => {
                        *similarity_map.at_2d::<u8>(y, x)? as f32
                    }
                    id if id == std::any::TypeId::of::<u16>() => {
                        *similarity_map.at_2d::<u16>(y, x)? as f32
                    }
                    _ => panic!("Unsupported accumulator type"),
                };
                let similarity = (raw_score * 100.0) / (4.0 * templ.features.len() as f32);

                if similarity >= threshold {
                    matches.push(Match::new(
                        x * t + offset,
                        y * t + offset,
                        similarity,
                        class_id.to_string(),
                        template_id,
                    ));
                }
            }
        }

        Ok(matches)
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
    pub angle: Mat,
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
            magnitude: Mat::default(),
            weak_threshold,
            num_features,
            strong_threshold,
        };

        pyramid.update()?;
        Ok(pyramid)
    }

    /// Compute gradients (angle and magnitude)
    pub fn update(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Convert to grayscale if needed
        let gray = if self.src.channels() == 3 {
            let mut gray_img = Mat::default();
            imgproc::cvt_color(&self.src, &mut gray_img, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
            gray_img
        } else {
            self.src.clone()
        };

        // Compute gradients using Sobel
        let mut grad_x = Mat::default();
        let mut grad_y = Mat::default();

        imgproc::sobel(
            &gray,
            &mut grad_x,
            core::CV_32F,
            1,
            0,
            3,
            1.0,
            0.0,
            core::BORDER_DEFAULT,
        )?;
        imgproc::sobel(
            &gray,
            &mut grad_y,
            core::CV_32F,
            0,
            1,
            3,
            1.0,
            0.0,
            core::BORDER_DEFAULT,
        )?;

        // Compute magnitude and angle
        self.magnitude = Mat::default();
        self.angle = Mat::default();

        core::cart_to_polar(&grad_x, &grad_y, &mut self.magnitude, &mut self.angle, true)?;

        Ok(())
    }

    /// Quantize gradients to 8 orientations
    pub fn quantize(&self, dst: &mut Mat) -> Result<(), Box<dyn std::error::Error>> {
        *dst = Mat::new_rows_cols_with_default(
            self.angle.rows(),
            self.angle.cols(),
            core::CV_8UC1,
            Scalar::all(0.0),
        )?;

        for y in 0..self.angle.rows() {
            for x in 0..self.angle.cols() {
                let mag = *self.magnitude.at_2d::<f32>(y, x)?;
                if mag > self.weak_threshold {
                    let angle_val = *self.angle.at_2d::<f32>(y, x)?;
                    let label = quantize_angle(angle_val.to_radians());
                    *dst.at_2d_mut::<u8>(y, x)? = 1 << label;
                }
            }
        }

        Ok(())
    }

    /// Extract a template from the current pyramid level
    pub fn extract_template(&self, templ: &mut Template) -> Result<bool, Box<dyn std::error::Error>> {
        // Find candidate features
        let mut candidates = Vec::new();

        for y in 0..self.magnitude.rows() {
            for x in 0..self.magnitude.cols() {
                let mag = *self.magnitude.at_2d::<f32>(y, x)?;
                let mask_val = *self.mask.at_2d::<u8>(y, x)?;

                if mag > self.strong_threshold && mask_val > 0 {
                    let angle_val = *self.angle.at_2d::<f32>(y, x)?;
                    let label = quantize_angle(angle_val.to_radians());

                    let mut feat = Feature::new(x, y, label);
                    feat.theta = angle_val; // Store angle in degrees (like C++)
                    
                    candidates.push(Candidate {
                        f: feat,
                        score: mag,
                    });
                }
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        // Sort by score (magnitude)
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Select scattered features
        templ.features = select_scattered_features(&candidates, self.num_features, 8.0);

        Ok(!templ.features.is_empty())
    }

    /// Downsample the pyramid
    pub fn pyr_down(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut src_down = Mat::default();
        let mut mask_down = Mat::default();

        imgproc::pyr_down(&self.src, &mut src_down, Size::default(), core::BORDER_DEFAULT)?;
        imgproc::pyr_down(&self.mask, &mut mask_down, Size::default(), core::BORDER_DEFAULT)?;

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

// ============================================================================
// Helper Functions
// ============================================================================

/// Quantize angle to one of 8 orientations (0-7)
fn quantize_angle(angle_rad: f32) -> i32 {
    let mut angle = angle_rad;
    while angle < 0.0 {
        angle += 2.0 * std::f32::consts::PI;
    }
    while angle >= 2.0 * std::f32::consts::PI {
        angle -= 2.0 * std::f32::consts::PI;
    }

    let bin = ((angle * 8.0 / (2.0 * std::f32::consts::PI)) + 0.5) as i32;
    bin % 8
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
fn transform_image(
    src: &Mat,
    angle: f32,
    scale: f32,
) -> Result<Mat, Box<dyn std::error::Error>> {
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
    for r in 0..t {
        for c in 0..t {
            // OR values from source starting at (r,c) into destination starting at (0,0)
            let height = quantized.rows() - r;
            let width = quantized.cols() - c;
            
            for y in 0..height {
                for x in 0..width {
                    let src_val = *quantized.at_2d::<u8>(y + r, x + c)?;
                    let dst_val = *spread.at_2d::<u8>(y, x)?;
                    *spread.at_2d_mut::<u8>(y, x)? = dst_val | src_val;
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
    0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4,
    0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3,
    0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4,
    0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3,
    0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3,
    0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4,
    0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3,
    0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4,
];

/// Compute response maps using similarity LUT (like C++)
/// Maps bit patterns to scores: 4 for exact match, 3 for adjacent, 0 for no match
fn compute_response_maps(spread_quantized: &Mat) -> Result<Vec<Mat>, Box<dyn std::error::Error>> {
    let mut response_maps = Vec::new();

    for ori in 0..8 {
        let mut response = Mat::new_rows_cols_with_default(
            spread_quantized.rows(),
            spread_quantized.cols(),
            core::CV_8UC1,
            Scalar::all(0.0),
        )?;

        let lut_offset = 32 * ori;

        for y in 0..spread_quantized.rows() {
            for x in 0..spread_quantized.cols() {
                let val = *spread_quantized.at_2d::<u8>(y, x)?;
                
                // Split into LSB4 and MSB4
                let lsb4 = (val & 15) as usize;
                let msb4 = ((val & 240) >> 4) as usize;
                
                // Use LUT to compute response (like C++)
                let response_val = SIMILARITY_LUT[lut_offset + lsb4].max(SIMILARITY_LUT[lut_offset + 16 + msb4]);
                *response.at_2d_mut::<u8>(y, x)? = response_val;
            }
        }

        response_maps.push(response);
    }

    Ok(response_maps)
}

/// Linearize response maps for fast memory access
/// Creates a TxT grid of linear memories for each orientation (like C++)
/// Returns Vec<Mat> where each Mat has T×T rows and (w×h) cols
fn linearize_response_maps(
    response_maps: &[Mat],
    t: i32
) -> Result<Vec<Mat>, Box<dyn std::error::Error>> {
    let mut linear_memories = Vec::new();

    for response_map in response_maps {
        let rows = response_map.rows();
        let cols = response_map.cols();
        let mem_width = cols / t;
        let mem_height = rows / t;

        // Create Mat with T×T rows, where each row is a linear memory
        let mut linearized = Mat::new_rows_cols_with_default(
            t * t,
            mem_width * mem_height,
            core::CV_8UC1,
            Scalar::all(0.0),
        )?;

        // Outer two loops iterate over top-left T×T starting pixels
        let mut index = 0;
        for r_start in 0..t {
            for c_start in 0..t {
                // Inner two loops copy every T-th pixel into the linear memory
                let mut mem_idx = 0;
                for r in (r_start..rows).step_by(t as usize) {
                    for c in (c_start..cols).step_by(t as usize) {
                        let val = *response_map.at_2d::<u8>(r, c)?;
                        *linearized.at_2d_mut::<u8>(index, mem_idx)? = val;
                        mem_idx += 1;
                    }
                }
                index += 1;
            }
        }

        linear_memories.push(linearized);
    }

    Ok(linear_memories)
}

/// Trait for similarity accumulation with different data types
trait SimilarityAccumulator: opencv::core::DataType {
    type Acc: Copy;
    type LinearPtr: Copy;
    const CV_TYPE: i32;
    
    fn get_ptr(data: *mut u8) -> *mut Self::Acc;
    fn get_linear_ptr(data: *const u8) -> Self::LinearPtr;
    
    fn accumulate_row(
        similarity_ptr: *mut Self::Acc,
        linear_memory_ptr: Self::LinearPtr,
        y: i32,
        w: i32,
        wf: i32,
        lm_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

impl SimilarityAccumulator for u8 {
    type Acc = u8;
    type LinearPtr = *const u8;
    const CV_TYPE: i32 = core::CV_8UC1;
    
    fn get_ptr(data: *mut u8) -> *mut Self::Acc {
        data
    }
    
    fn get_linear_ptr(data: *const u8) -> Self::LinearPtr {
        data
    }
    
    fn accumulate_row(
        similarity_ptr: *mut Self::Acc,
        linear_memory_ptr: Self::LinearPtr,
        y: i32,
        w: i32,
        wf: i32,
        lm_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let base_lm_idx = lm_index + (y * w) as usize;
        let base_sim_idx = (y * w) as usize;
        let span_x = (w - wf + 1) as usize;
        
        unsafe {
            simd_accumulate_u8(
                similarity_ptr.add(base_sim_idx),
                linear_memory_ptr.add(base_lm_idx),
                span_x,
            );
        }
        Ok(())
    }
}

impl SimilarityAccumulator for u16 {
    type Acc = u16;
    type LinearPtr = *const u8;
    const CV_TYPE: i32 = core::CV_16UC1;
    
    fn get_ptr(data: *mut u8) -> *mut Self::Acc {
        data as *mut u16
    }
    
    fn get_linear_ptr(data: *const u8) -> Self::LinearPtr {
        data
    }
    
    fn accumulate_row(
        similarity_ptr: *mut Self::Acc,
        linear_memory_ptr: Self::LinearPtr,
        y: i32,
        w: i32,
        wf: i32,
        lm_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let base_lm_idx = lm_index + (y * w) as usize;
        let base_sim_idx = (y * w) as usize;
        let span_x = (w - wf + 1) as usize;
        
        unsafe {
            simd_accumulate_u16(
                similarity_ptr,
                linear_memory_ptr,
                base_sim_idx,
                base_lm_idx,
                span_x,
            );
        }
        Ok(())
    }
}

/// Generic similarity map computation
fn compute_similarity_map<T: SimilarityAccumulator>(
    linear_memories: &[Mat],
    templ: &Template,
    src_cols: i32,
    src_rows: i32,
    t: i32,
) -> Result<Mat, Box<dyn std::error::Error>> {
    let w = src_cols / t;
    let h = src_rows / t;

    let mut similarity_map = Mat::new_rows_cols_with_default(
        h,
        w,
        T::CV_TYPE,
        Scalar::all(0.0),
    )?;

    // Decimated template dimensions
    let wf = (templ.width - 1) / t + 1;
    let hf = (templ.height - 1) / t + 1;

    // Get raw pointer for SIMD access
    let similarity_ptr = T::get_ptr(similarity_map.data_mut());

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

        // Get pointer to the linear memory row
        let linear_memory_ptr = memory_grid.ptr(grid_index)?;
        let linear_ptr = T::get_linear_ptr(linear_memory_ptr);

        // Add this feature's response to all valid template positions
        for y in 0..(h - hf + 1) {
            let base_lm_idx = lm_index + (y * w) as usize;
            
            if base_lm_idx >= memory_grid.cols() as usize {
                continue;
            }

            T::accumulate_row(
                similarity_ptr,
                linear_ptr,
                y,
                w,
                wf,
                lm_index,
            )?;
        }
    }

    Ok(similarity_map)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_angle() {
        assert_eq!(quantize_angle(0.0), 0);
        assert_eq!(quantize_angle(std::f32::consts::PI / 4.0), 1);
        assert_eq!(quantize_angle(std::f32::consts::PI / 2.0), 2);
    }

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
