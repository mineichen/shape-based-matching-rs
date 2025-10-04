use opencv::{
    core::{self, Point, Size},
    imgproc,
    prelude::*,
};
// ximgproc is optional; we'll attempt to use it for thinning if available
#[allow(unused_imports)]
use opencv::ximgproc;

#[derive(Debug, Clone)]
pub struct EdgeDetectionParams {
    pub bilateral_d: i32,
    pub bilateral_sigma_color: f64,
    pub bilateral_sigma_space: f64,
    pub adaptive_thresh_max: f64,
    pub adaptive_thresh_block_size: i32,
    pub adaptive_thresh_c: f64,
    pub canny_low: f64,
    pub canny_high: f64,
    pub canny_aperture: i32,
    pub gaussian_kernel_size: i32,
    pub gaussian_sigma: f64,
    pub pre_skeleton_morph_pixels: i32,
}

impl Default for EdgeDetectionParams {
    fn default() -> Self {
        Self {
            bilateral_d: 9,
            bilateral_sigma_color: 75.0,
            bilateral_sigma_space: 75.0,
            adaptive_thresh_max: 255.0,
            adaptive_thresh_block_size: 11,
            adaptive_thresh_c: 2.0,
            canny_low: 30.0,
            canny_high: 100.0,
            canny_aperture: 3,
            gaussian_kernel_size: 3,
            gaussian_sigma: 0.5,
            pre_skeleton_morph_pixels: 15,
        }
    }
}

/// Ultra-sensitive edge detection parameters for capturing all inner and outer circles
impl EdgeDetectionParams {
    pub fn ultra_sensitive() -> Self {
        Self {
            bilateral_d: 9,
            bilateral_sigma_color: 75.0,
            bilateral_sigma_space: 75.0,
            adaptive_thresh_max: 255.0,
            adaptive_thresh_block_size: 11,
            adaptive_thresh_c: 2.0,
            canny_low: 8.0,
            canny_high: 20.0,
            canny_aperture: 3,
            gaussian_kernel_size: 1,
            gaussian_sigma: 0.0,
            pre_skeleton_morph_pixels: 0,
        }
    }
}

#[derive(Clone)]
pub struct EdgeImage {
    pub mat: core::Mat,
    pub params: EdgeDetectionParams,
}

impl EdgeImage {
    pub fn raw(&self) -> &core::Mat { &self.mat }
    pub fn into_raw(self) -> core::Mat { self.mat }
    fn morph_only(&self) -> Result<core::Mat, Box<dyn std::error::Error>> {
        let n = self.params.pre_skeleton_morph_pixels;
        let after_dilate = apply_dilation_configurable(&self.mat, n)?;
        let after_erode = apply_erosion_configurable(&after_dilate, n)?;
        Ok(after_erode)
    }
    // Returns skeleton from the non-morphed base image
    pub fn not_morphed(&self) -> Result<core::Mat, Box<dyn std::error::Error>> {
        apply_thinning_skeleton(&self.mat)
    }
    // Returns skeleton after applying morph (dilate/erode) first
    pub fn morphed(&self) -> Result<core::Mat, Box<dyn std::error::Error>> {
        let morphed = self.morph_only()?;
        apply_thinning_skeleton(&morphed)
    }
}

pub fn preprocess_for_edge_detection(
    image: &core::Mat,
    params: &EdgeDetectionParams,
) -> Result<core::Mat, Box<dyn std::error::Error>> {
    // Convert to grayscale if needed
    let mut gray = core::Mat::default();
    if image.channels() > 1 {
        imgproc::cvt_color(
            image, 
            &mut gray, 
            imgproc::COLOR_BGR2GRAY, 
            0, 
            core::AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
    } else {
        gray = image.clone();
    }

    // Apply bilateral filter to reduce noise while preserving edges
    let mut filtered = core::Mat::default();
    imgproc::bilateral_filter(
        &gray,
        &mut filtered,
        params.bilateral_d,
        params.bilateral_sigma_color,
        params.bilateral_sigma_space,
        core::BORDER_DEFAULT,
    )?;

    Ok(filtered)
}

pub fn apply_adaptive_threshold(
    image: &core::Mat,
    params: &EdgeDetectionParams,
) -> Result<core::Mat, Box<dyn std::error::Error>> {
    let mut thresh = core::Mat::default();
    imgproc::adaptive_threshold(
        image,
        &mut thresh,
        params.adaptive_thresh_max,
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        imgproc::THRESH_BINARY,
        params.adaptive_thresh_block_size,
        params.adaptive_thresh_c,
    )?;
    Ok(thresh)
}


fn apply_gaussian_blur(
    image: &core::Mat,
    params: &EdgeDetectionParams,
) -> Result<core::Mat, Box<dyn std::error::Error>> {
    let mut blurred = core::Mat::default();
    imgproc::gaussian_blur(
        image,
        &mut blurred,
        Size::new(params.gaussian_kernel_size, params.gaussian_kernel_size),
        params.gaussian_sigma,
        params.gaussian_sigma,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(blurred)
}


fn apply_multi_level_canny(
    image: &core::Mat,
    _params: &EdgeDetectionParams,
) -> Result<core::Mat, Box<dyn std::error::Error>> {
    // Apply different sensitivity levels
    let mut edges_low = core::Mat::default();
    imgproc::canny(image, &mut edges_low, 20.0, 60.0, 3, false)?;
    
    let mut edges_high = core::Mat::default();
    imgproc::canny(image, &mut edges_high, 50.0, 150.0, 3, false)?;
    
    // Combine different sensitivity levels
    let mut multi_level = core::Mat::default();
    core::bitwise_or(&edges_low, &edges_high, &mut multi_level, &core::Mat::default())?;
    
    Ok(multi_level)
}

pub fn apply_ultra_sensitive_edge_detection(image: &core::Mat) -> Result<core::Mat, Box<dyn std::error::Error>> {
    let params = EdgeDetectionParams::ultra_sensitive();
    let filtered = preprocess_for_edge_detection(image, &params)?;
    
    // Apply minimal blur to preserve all details
    let mut blurred_ultra_light = core::Mat::default();
    imgproc::gaussian_blur(
        &filtered,
        &mut blurred_ultra_light,
        Size::new(1, 1),
        0.0, 0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Apply multiple levels of Canny with extremely sensitive parameters
    let mut edges_ultra_sensitive = core::Mat::default();
    imgproc::canny(&blurred_ultra_light, &mut edges_ultra_sensitive, 8.0, 20.0, 3, false)?;
    
    let mut edges_sensitive = core::Mat::default();
    imgproc::canny(&blurred_ultra_light, &mut edges_sensitive, 15.0, 45.0, 3, false)?;
    
    let mut edges_standard = core::Mat::default();
    imgproc::canny(&blurred_ultra_light, &mut edges_standard, 30.0, 90.0, 3, false)?;
    
    // Combine all sensitivity levels
    let mut temp1 = core::Mat::default();
    core::bitwise_or(&edges_ultra_sensitive, &edges_sensitive, &mut temp1, &core::Mat::default())?;
    
    let mut all_canny = core::Mat::default();
    core::bitwise_or(&temp1, &edges_standard, &mut all_canny, &core::Mat::default())?;

    // Apply adaptive thresholding for additional edge enhancement
    let mut thresh = core::Mat::default();
    imgproc::adaptive_threshold(
        &filtered,
        &mut thresh,
        255.0,
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        imgproc::THRESH_BINARY,
        11, 2.0,
    )?;

    // Apply morphological gradient to enhance edges
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(3, 3),
        Point::new(-1, -1),
    )?;
    
    let mut gradient = core::Mat::default();
    imgproc::morphology_ex(
        &all_canny,
        &mut gradient,
        imgproc::MORPH_GRADIENT,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_DEFAULT,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
    )?;
    
    // Combine all edge detection methods
    let mut combined_edges = core::Mat::default();
    core::bitwise_or(&all_canny, &gradient, &mut combined_edges, &core::Mat::default())?;

    // Apply minimal morphological closing to connect nearby edges and fill gaps
    let close_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(1, 1),  // Minimal kernel size to reduce artificial connections
        Point::new(-1, -1),
    )?;
    
    let mut closed = core::Mat::default();
    imgproc::morphology_ex(
        &combined_edges,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &close_kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_DEFAULT,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
    )?;

    Ok(closed)
}



pub fn detect_edges_comprehensive(
    image: &core::Mat,
    params: &EdgeDetectionParams,
) -> Result<EdgeImage, Box<dyn std::error::Error>> {
    // Preprocess the image
    let filtered = preprocess_for_edge_detection(image, params)?;
    
    // Apply Gaussian blur
    let blurred = apply_gaussian_blur(&filtered, params)?;
    
    // Apply multi-level Canny
    let multi_level = apply_multi_level_canny(&blurred, params)?;
    
    // Use multi-level Canny directly (edges=255, background=0)
    let combined_edges = multi_level.clone();

    Ok(EdgeImage { mat: combined_edges, params: params.clone() })
}

/// Applies distance transform to find skeleton
pub fn apply_distance_transform(image: &core::Mat) -> Result<core::Mat, Box<dyn std::error::Error>> {
    let mut dist_transform = core::Mat::default();
    imgproc::distance_transform(
        image,
        &mut dist_transform,
        imgproc::DIST_L2,
        imgproc::DIST_MASK_5,
        core::CV_32F,
    )?;
    Ok(dist_transform)
}

/// Normalizes distance transform and converts to binary skeleton
pub fn create_skeleton_from_distance_transform(dist_transform: &core::Mat) -> Result<core::Mat, Box<dyn std::error::Error>> {
    // Normalize distance transform
    let mut normalized = core::Mat::default();
    core::normalize(
        dist_transform, 
        &mut normalized, 
        0.0, 
        255.0, 
        core::NORM_MINMAX, 
        core::CV_8U, 
        &core::Mat::default()
    )?;

    // Threshold to get skeleton lines
    let mut skeleton = core::Mat::default();
    imgproc::threshold(
        &normalized,
        &mut skeleton,
        0.0, 
        255.0, 
        imgproc::THRESH_BINARY,
    )?;
    
    Ok(skeleton)
}





/// Applies dilation with 5-pixel kernel
pub fn apply_dilation_5px(image: &core::Mat) -> Result<core::Mat, Box<dyn std::error::Error>> {
    let dilate_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;
    
    let mut dilated = core::Mat::default();
    imgproc::morphology_ex(
        image,
        &mut dilated,
        imgproc::MORPH_DILATE,
        &dilate_kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_DEFAULT,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
    )?;
    
    Ok(dilated)
}

/// Applies dilation with configurable pixel kernel
pub fn apply_dilation_configurable(image: &core::Mat, pixels: i32) -> Result<core::Mat, Box<dyn std::error::Error>> {
    if pixels <= 0 { return Ok(image.clone()); }
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(pixels, pixels),
        Point::new(-1, -1),
    )?;
    let mut out = core::Mat::default();
    imgproc::dilate(image, &mut out, &kernel, Point::new(-1, -1), 1, core::BORDER_DEFAULT, core::Scalar::new(0.0,0.0,0.0,0.0))?;
    Ok(out)
}

/// Applies erosion with configurable pixel kernel
pub fn apply_erosion_configurable(image: &core::Mat, pixels: i32) -> Result<core::Mat, Box<dyn std::error::Error>> {
    if pixels <= 0 { return Ok(image.clone()); }
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(pixels, pixels),
        Point::new(-1, -1),
    )?;
    let mut out = core::Mat::default();
    imgproc::erode(image, &mut out, &kernel, Point::new(-1, -1), 1, core::BORDER_DEFAULT, core::Scalar::new(0.0,0.0,0.0,0.0))?;
    Ok(out)
}

/// Applies thinning (skeletonization) if available, falls back to simple erosion if not
pub fn apply_thinning_skeleton(image: &core::Mat) -> Result<core::Mat, Box<dyn std::error::Error>> {
    // Ensure input is binary 8-bit
    let mut binary = core::Mat::default();
    imgproc::threshold(image, &mut binary, 0.0, 255.0, imgproc::THRESH_OTSU | imgproc::THRESH_BINARY)?;
    
    // Try ximgproc thinning (Zhang-Suen); if unavailable at runtime, fall back
    let mut thinned = core::Mat::default();
    if ximgproc::thinning(&binary, &mut thinned, ximgproc::THINNING_ZHANGSUEN).is_ok() {
        return Ok(thinned);
    }
    
    // Fallback: one-pass morphological erosion (minimal) to slightly thin lines
    let erode_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_CROSS,
        Size::new(3, 3),
        Point::new(-1, -1),
    )?;
    imgproc::erode(&binary, &mut thinned, & erode_kernel, Point::new(-1, -1), 1, core::BORDER_DEFAULT, core::Scalar::new(0.0,0.0,0.0,0.0))?;
    Ok(thinned)
}
