use opencv::{
    core::{self, Point},
    prelude::*,
};

use crate::edge_detection::{EdgeDetectionParams, detect_edges_comprehensive};

// Public API - Data Structures
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    pub points: Vec<Point>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LineSegment {
    pub start: Point,
    pub end: Point,
}

#[derive(Debug, Clone)]
pub struct CurveDetectionParams {
    pub min_curve_length: f64,
    pub min_curve_points: usize,
}

impl Default for CurveDetectionParams {
    fn default() -> Self {
        Self {
            min_curve_length: 20.0,
            min_curve_points: 5,
        }
    }
}

// Public API - Main Functions
/// Detects curves from a pre-computed edge image
pub fn detect_curves_from_edges(edge_image: &core::Mat) -> Result<Vec<Curve>, Box<dyn std::error::Error>> {
    let curve_params = CurveDetectionParams::default();
    find_curves_by_tracing(edge_image, &curve_params)
}

/// Detects curves from a pre-computed edge image with custom parameters
pub fn detect_curves_from_edges_with_params(edge_image: &core::Mat, params: &CurveDetectionParams) -> Result<Vec<Curve>, Box<dyn std::error::Error>> {
    find_curves_by_tracing(edge_image, params)
}

/// Detects curves using comprehensive edge detection (convenience function)
pub fn detect_curves(image: &core::Mat) -> Result<Vec<Curve>, Box<dyn std::error::Error>> {
    let params = EdgeDetectionParams::default();
    let edge_image = detect_edges_comprehensive(image, &params)?;
    let skeleton = edge_image.morphed()?;
    detect_curves_from_edges(&skeleton)
}

/// Detects curves using skeleton-based approach (convenience function)
pub fn detect_curves_skeleton(image: &core::Mat) -> Result<Vec<Curve>, Box<dyn std::error::Error>> {
    let params = EdgeDetectionParams::default();
    let filtered = crate::edge_detection::preprocess_for_edge_detection(image, &params)?;
    let thresh = crate::edge_detection::apply_adaptive_threshold(&filtered, &params)?;
    
    // Apply distance transform to find skeleton
    let dist_transform = crate::edge_detection::apply_distance_transform(&thresh)?;
    let skeleton = crate::edge_detection::create_skeleton_from_distance_transform(&dist_transform)?;
    
    detect_curves_from_edges(&skeleton)
}

/// Detects curves using ultra-sensitive edge detection (convenience function)
pub fn detect_curves_ultra_sensitive(image: &core::Mat) -> Result<Vec<Curve>, Box<dyn std::error::Error>> {
    let closed = crate::edge_detection::apply_ultra_sensitive_edge_detection(image)?;
    let dist_transform = crate::edge_detection::apply_distance_transform(&closed)?;
    let skeleton = crate::edge_detection::create_skeleton_from_distance_transform(&dist_transform)?;
    
    detect_curves_from_edges(&skeleton)
}

/// Extracts line segments from curves by finding straight portions
pub fn extract_line_segments_from_curves(curves: &[Curve]) -> Vec<LineSegment> {
    let mut line_segments = Vec::new();
    
    for curve in curves {
        if curve.points.len() < 2 {
            continue;
        }
        
        // Find straight line segments in the curve
        let mut i = 0;
        while i < curve.points.len() - 1 {
            let start_point = curve.points[i];
            let mut end_point = curve.points[i + 1];
            
            // Try to extend the line segment as long as it remains straight
            let mut j = i + 2;
            while j < curve.points.len() {
                let next_point = curve.points[j];
                
                // Check if the next point maintains the straight line
                if is_point_on_line(&start_point, &end_point, &next_point) {
                    end_point = next_point;
                    j += 1;
                } else {
                    break;
                }
            }
            
            // Only add line segments that are long enough
            let length = ((end_point.x - start_point.x) as f64).powi(2) + 
                        ((end_point.y - start_point.y) as f64).powi(2);
            if length.sqrt() >= 10.0 {
                line_segments.push(LineSegment { 
                    start: start_point, 
                    end: end_point 
                });
            }
            
            i = j;
        }
    }
    
    line_segments
}

/// Detects line segments by first detecting curves and then extracting straight portions
pub fn detect_line_segments(image: &core::Mat) -> Result<Vec<LineSegment>, Box<dyn std::error::Error>> {
    let curves = detect_curves(image)?;
    Ok(extract_line_segments_from_curves(&curves))
}

// Private Helper Functions
/// Traces a curve by following connected edge pixels from a starting point
fn trace_curve_from_point(
    edge_image: &core::Mat,
    start_point: Point,
    visited: &mut Vec<Vec<bool>>,
    width: i32,
    height: i32,
) -> Result<Vec<Point>, Box<dyn std::error::Error>> {
    let mut curve_points = Vec::new();
    let mut current_point = start_point;
    
    // 8-connected neighborhood offsets
    let offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ];
    
    // Trace the curve by following connected pixels sequentially
    loop {
        if current_point.x < 0 || current_point.x >= width || 
           current_point.y < 0 || current_point.y >= height {
            break;
        }
        
        if visited[current_point.y as usize][current_point.x as usize] {
            break;
        }
        
        // Check if this pixel is an edge pixel
        let pixel_value = edge_image.at_2d::<u8>(current_point.y, current_point.x)?;
        if *pixel_value == 0 {
            break;
        }
        
        // Mark as visited and add to curve
        visited[current_point.y as usize][current_point.x as usize] = true;
        curve_points.push(current_point);
        
        // Find the next connected edge pixel
        let mut next_point_found = false;
        for (dx, dy) in offsets.iter() {
            let new_x = current_point.x + dx;
            let new_y = current_point.y + dy;
            
            if new_x >= 0 && new_x < width && new_y >= 0 && new_y < height {
                if !visited[new_y as usize][new_x as usize] {
                    // Check if this neighbor is an edge pixel
                    let neighbor_value = edge_image.at_2d::<u8>(new_y, new_x)?;
                    if *neighbor_value != 0 {
                        current_point = Point::new(new_x, new_y);
                        next_point_found = true;
                        break;
                    }
                }
            }
        }
        
        if !next_point_found {
            break;
        }
    }
    
    Ok(curve_points)
}

/// Finds all curves by tracing connected edge pixels
fn find_curves_by_tracing(edge_image: &core::Mat, params: &CurveDetectionParams) -> Result<Vec<Curve>, Box<dyn std::error::Error>> {
    let width = edge_image.cols();
    let height = edge_image.rows();
    let edge_src: &core::Mat = edge_image;
    
    // Create visited matrix
    let mut visited = vec![vec![false; width as usize]; height as usize];
    let mut curves = Vec::new();
    
    // Scan the image for unvisited edge pixels
    for y in 0..height {
        for x in 0..width {
            if visited[y as usize][x as usize] {
                continue;
            }
            
            // Check if this pixel is an edge pixel
            let pixel_value = edge_src.at_2d::<u8>(y, x)?;
            if *pixel_value == 0 {
                continue;
            }
            
            // Trace curve from this point
            let curve_points = trace_curve_from_point(
                edge_src,
                Point::new(x, y),
                &mut visited,
                width,
                height,
            )?;
            
            // Apply filtering
            if curve_points.len() >= params.min_curve_points {
                
                // Calculate curve length
                let mut total_length = 0.0;
                for i in 0..curve_points.len() - 1 {
                    let dx = (curve_points[i + 1].x - curve_points[i].x) as f64;
                    let dy = (curve_points[i + 1].y - curve_points[i].y) as f64;
                    total_length += (dx * dx + dy * dy).sqrt();
                }
                
                // Only keep curves that are long enough
                if total_length >= params.min_curve_length {
                    curves.push(Curve { points: curve_points });
                }
            }
        }
    }
    
    Ok(curves)
}

/// Checks if a point lies approximately on a line defined by two other points
fn is_point_on_line(p1: &Point, p2: &Point, p3: &Point) -> bool {
    const TOLERANCE: f64 = 3.0; // pixels
    
    // Calculate the distance from p3 to the line defined by p1 and p2
    let a = (p2.y - p1.y) as f64;
    let b = (p1.x - p2.x) as f64;
    let c = (p2.x * p1.y - p1.x * p2.y) as f64;
    
    let distance = (a * p3.x as f64 + b * p3.y as f64 + c).abs() / 
                   (a * a + b * b).sqrt();
    
    distance <= TOLERANCE
}