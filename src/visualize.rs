use opencv::{
    core::{self, Mat, Point, Rect, Scalar},
    imgcodecs, imgproc,
    prelude::*,
};

use crate::match_result::{Match, Matches};

impl<'a> Matches<'a> {
    /// Create a debug visualization of these matches on the source image.
    pub fn debug_visual(&self, input: Mat, template_region: Option<Rect>) -> Result<Mat, opencv::Error> {
        debug_visual(input, self.as_slice(), template_region)
    }
}

/// Create a debug visualization of matches on the source image.
/// Returns encoded PNG image bytes.
///
/// # Arguments
/// * `source` - Source image to draw matches on
/// * `matches` - Vector of matches to visualize
/// * `template_region` - Optional region where the original template was extracted (drawn in blue)
///
/// # Returns
/// Encoded PNG image bytes
pub fn debug_visual(
    input: Mat,
    matches: &[Match],
    template_region: Option<Rect>,
) -> Result<Mat, opencv::Error> {
    // Convert grayscale to BGR if needed
    let mut result = if input.channels() == 1 {
        let mut result = Mat::default();
        imgproc::cvt_color_def(&input, &mut result, imgproc::COLOR_GRAY2BGR)?;
        result
    } else {
        input
    };

    // Draw original template region in blue if provided
    if let Some(region) = template_region {
        let blue = Scalar::new(255.0, 0.0, 0.0, 0.0);
        imgproc::rectangle(&mut result, region, blue, 3, imgproc::LINE_8, 0)?;
        imgproc::put_text(
            &mut result,
            "Original Template",
            Point::new(region.x, region.y - 10),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            blue,
            2,
            imgproc::LINE_8,
            false,
        )?;
    }

    const BORDER_PADDING: i32 = 4;

    for match_item in matches {
        // Get template dimensions
        let templ = match_item.match_template();

        // Draw rectangle around match
        let color = Scalar::new(0.0, 100.0, 0.0, 0.0); // Dark green
        imgproc::rectangle(
            &mut result,
            Rect::new(
                match_item.x - BORDER_PADDING,
                match_item.y - BORDER_PADDING,
                templ.width + 2 * BORDER_PADDING,
                templ.height + 2 * BORDER_PADDING,
            ),
            color,
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Draw center point
        let center_point = match_item.center_point();
        imgproc::circle(
            &mut result,
            center_point,
            5,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?; // Red filled circle
        imgproc::circle(
            &mut result,
            center_point,
            10,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?; // Red outline

        // Draw features
        for feat in &templ.features {
            imgproc::circle(
                &mut result,
                Point::new(feat.x + match_item.x, feat.y + match_item.y),
                2,
                color,
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }

        // Draw similarity and angle text
        let angle = match_item.angle();
        let label = format!("{}% @{}deg", match_item.similarity.round() as i32, angle);
        imgproc::put_text(
            &mut result,
            &label,
            Point::new(match_item.x + 5, match_item.y + 20),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(0.0, 180.0, 0.0, 0.0), // Brighter green for text readability
            2,
            imgproc::LINE_8,
            false,
        )?;
    }

    // Encode to PNG bytes
    Ok(result)
}
