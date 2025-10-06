use crate::line2dup::Feature;
use opencv::{
    core::{self, Mat, Scalar, Size},
    imgproc,
    prelude::*,
};

pub struct ColorGradientPyramid {
    pub src: Mat,
    pub mask: Mat,
    pub pyramid_level: u8,
    pub angle: Mat,     // quantized 8-direction bitmask (CV_8U)
    pub angle_ori: Mat, // original orientation in degrees (CV_32F)
    pub magnitude: Mat,
    pub weak_threshold: f32,
    pub strong_threshold: f32,
}

impl ColorGradientPyramid {
    pub fn new(
        src: &Mat,
        mask: &Mat,
        weak_threshold: f32,
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
            // Use raw pointer access for better performance
            for r in 0..size.height {
                unsafe {
                    let vx_row = dx3.ptr(r)? as *const core::Vec3s;
                    let vy_row = dy3.ptr(r)? as *const core::Vec3s;
                    let dx_row = dx.ptr_mut(r)? as *mut f32;
                    let dy_row = dy.ptr_mut(r)? as *mut f32;
                    let mag_row = self.magnitude.ptr_mut(r)? as *mut f32;

                    for c in 0..size.width {
                        let vx = *vx_row.add(c as usize);
                        let vy = *vy_row.add(c as usize);
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
                        *dx_row.add(c as usize) = xb as f32;
                        *dy_row.add(c as usize) = yb as f32;
                        *mag_row.add(c as usize) = mb as f32;
                    }
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

                // Zero left and right borders
                for r in 0..quant_unfiltered.rows() {
                    let row_ptr = quant_unfiltered.ptr_mut(r)? as *mut u8;
                    *row_ptr = 0;
                    *row_ptr.add((cols - 1) as usize) = 0;
                }
            }
        }

        // Mask to 8 bins (keep lower 3 bits) using raw pointer access
        unsafe {
            for r in 1..(quant_unfiltered.rows() - 1) {
                let row_ptr = quant_unfiltered.ptr_mut(r)? as *mut u8;
                for c in 1..(quant_unfiltered.cols() - 1) {
                    *row_ptr.add(c as usize) &= 7;
                }
            }
        }

        // Hysteresis filter using magnitude threshold and 3x3 majority
        self.angle = Mat::new_rows_cols_with_default(
            self.angle_ori.rows(),
            self.angle_ori.cols(),
            core::CV_8UC1,
            Scalar::all(0.0),
        )?;
        let threshold_sq = self.weak_threshold * self.weak_threshold;

        // Use raw pointer access for performance
        for r in 1..(self.angle_ori.rows() - 1) {
            unsafe {
                let mag_row = self.magnitude.ptr(r)? as *const f32;
                let angle_row = self.angle.ptr_mut(r)? as *mut u8;

                for c in 1..(self.angle_ori.cols() - 1) {
                    let mag = *mag_row.add(c as usize);
                    if mag > threshold_sq {
                        let mut hist = [0i32; 8];
                        // 3x3 patch histogram
                        for pr in -1..=1 {
                            let quant_row = quant_unfiltered.ptr((r + pr) as i32)? as *const u8;
                            for pc in -1..=1 {
                                let v = *quant_row.add((c + pc) as usize) as usize;
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
                            *angle_row.add(c as usize) = 1u8 << index;
                        }
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
        num_features: usize,
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

        // Use raw pointer access for performance
        for r in k..(self.magnitude.rows() - k) {
            unsafe {
                let mask_row = if no_mask {
                    std::ptr::null()
                } else {
                    local_mask.ptr(r)?
                };
                let mag_valid_row = magnitude_valid.ptr(r)? as *const u8;
                let mag_row = self.magnitude.ptr(r)? as *const f32;
                let angle_row = self.angle.ptr(r)? as *const u8;
                let angle_ori_row = self.angle_ori.ptr(r)? as *const f32;

                for c in k..(self.magnitude.cols() - k) {
                    let mask_ok = no_mask || *mask_row.add(c as usize) > 0;
                    if !mask_ok {
                        continue;
                    }

                    let mut score = 0.0f32;
                    if *mag_valid_row.add(c as usize) > 0 {
                        score = *mag_row.add(c as usize);
                        let mut is_max = true;
                        'outer: for dr in -k..=k {
                            let mag_neighbor_row =
                                self.magnitude.ptr((r + dr) as i32)? as *const f32;
                            for dc in -k..=k {
                                if dr == 0 && dc == 0 {
                                    continue;
                                }
                                if score < *mag_neighbor_row.add((c + dc) as usize) {
                                    score = 0.0;
                                    is_max = false;
                                    break 'outer;
                                }
                            }
                        }
                        if is_max {
                            for dr in -k..=k {
                                let mag_valid_neighbor_row =
                                    magnitude_valid.ptr_mut((r + dr) as i32)? as *mut u8;
                                for dc in -k..=k {
                                    if dr == 0 && dc == 0 {
                                        continue;
                                    }
                                    *mag_valid_neighbor_row.add((c + dc) as usize) = 0;
                                }
                            }
                        }
                    }

                    // require strong magnitude and a quantized angle bit present
                    if score > threshold_sq {
                        let ang = *angle_row.add(c as usize);
                        if ang > 0 {
                            // convert angle bitmask to label index as in C++ getLabel
                            let label = bit_to_label(ang);
                            let mut feat = Feature::new(c, r, label);
                            feat.theta = *angle_ori_row.add(c as usize);
                            candidates.push(Candidate { f: feat, score });
                        }
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
            num_features,
            candidates.len() as f32 / num_features as f32 + 1.0,
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

        imgproc::pyr_down_def(&self.src, &mut src_down)?;
        imgproc::pyr_down_def(&self.mask, &mut mask_down)?;

        self.src = src_down;
        self.mask = mask_down;
        self.pyramid_level += 1;

        self.update()?;

        Ok(())
    }
}

/// A template representing a shape at a specific pyramid level
#[derive(Debug, Clone, Default)]
pub struct Template {
    pub width: i32,
    pub height: i32,
    pub tl_x: i32,
    pub tl_y: i32,
    pub pyramid_level: u8,
    pub features: Vec<Feature>,
}

struct Candidate {
    f: Feature,
    score: f32,
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
