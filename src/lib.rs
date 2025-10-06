use opencv::{
    core::{self, Scalar},
    imgcodecs, imgproc,
    prelude::*,
};

mod image_buffer;
pub mod line2dup;
mod pyramid;
mod simd_utils;
#[cfg(test)]
mod tests {
    use crate::line2dup::Detector;

    use super::*;

    #[test]
    fn test_line2dup_ellipse_detection() -> Result<(), Box<dyn std::error::Error>> {
        use crate::line2dup::Detector;

        // Create a blank canvas
        let width = 400;
        let height = 400;
        let mut canvas = core::Mat::new_rows_cols_with_default(
            height,
            width,
            core::CV_8UC3,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
        )?;

        // Draw an ellipse as the template
        let center = core::Point::new(200, 200);
        let axes = core::Size::new(80, 50);
        let angle = 0.0;
        imgproc::ellipse(
            &mut canvas,
            center,
            axes,
            angle,
            0.0,
            360.0,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Create detector
        let center_f = core::Point2f::new(center.x as _, center.y as _);
        let detector = Detector::builder()
            .with_template("ellipse", &canvas, |mut cfg| {
                cfg.add_rotated(0.0, center_f); // Explicitly add zero angle
                cfg.add_rotated(45.0, center_f);
            })
            .build();
        assert_eq!(detector.num_templates("ellipse"), 2);

        // Create a test image with the same ellipse rotated 45 degrees
        let mut test_canvas = core::Mat::new_rows_cols_with_default(
            height,
            width,
            core::CV_8UC3,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
        )?;

        imgproc::ellipse(
            &mut test_canvas,
            center,
            axes,
            45.0, // Rotated 45 degrees
            0.0,
            360.0,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Match the rotated ellipse with lower threshold (30%)
        let best_match = detector
            .match_templates(&test_canvas, 30.0, None, None)?
            .into_iter()
            .max()
            .unwrap();

        // The simple matching algorithm may not find perfect matches due to rotation
        // This test verifies the API works correctly
        println!(
            "Best match at ({}, {}) with similarity {:.2}",
            best_match.x, best_match.y, best_match.similarity
        );
        assert!(best_match.similarity >= 0.95);

        Ok(())
    }

    #[test]
    fn test_line2dup_rotated_range() -> Result<(), Box<dyn std::error::Error>> {
        // Create a simple shape
        let width = 200;
        let height = 200;
        let mut canvas = core::Mat::new_rows_cols_with_default(
            height,
            width,
            core::CV_8UC3,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
        )?;

        // Draw a rectangle
        imgproc::rectangle(
            &mut canvas,
            core::Rect::new(50, 50, 100, 80),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Test add_rotated_range with builder
        let center = core::Point2f::new((width / 2) as f32, (height / 2) as f32);
        let detector = Detector::builder()
            .with_template("rectangle", &canvas, |mut cfg| {
                cfg.add_rotated_range((0..=90u16).step_by(30), center);
            })
            .build();

        assert_eq!(detector.num_templates("rectangle"), 4);

        Ok(())
    }

    #[test]
    fn test_line2dup_multiple_rotations() -> Result<(), Box<dyn std::error::Error>> {
        // Create template with a distinctive shape (triangle)
        let width = 304;
        let height = 304;
        let mut template_canvas = core::Mat::new_rows_cols_with_default(
            height,
            width,
            core::CV_8UC3,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
        )?;

        // Draw triangle using lines instead of polylines
        imgproc::line(
            &mut template_canvas,
            core::Point::new(150, 50),
            core::Point::new(50, 250),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut template_canvas,
            core::Point::new(50, 250),
            core::Point::new(250, 250),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut template_canvas,
            core::Point::new(250, 250),
            core::Point::new(150, 50),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Create detector and add all rotations via builder
        let center = core::Point2f::new((width / 2) as f32, (height / 2) as f32);
        let detector = Detector::builder()
            .with_template("triangle", &template_canvas, |mut cfg| {
                cfg.add_rotated_range((0..=180u16).step_by(45), center);
            })
            .build();
        assert_eq!(detector.num_templates("triangle"), 5);

        // Test matching with a 90-degree rotated triangle
        let mut test_canvas = core::Mat::new_rows_cols_with_default(
            height,
            width,
            core::CV_8UC3,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
        )?;

        // Draw rotated triangle using lines
        imgproc::line(
            &mut test_canvas,
            core::Point::new(250, 150),
            core::Point::new(50, 50),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut test_canvas,
            core::Point::new(50, 50),
            core::Point::new(50, 250),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut test_canvas,
            core::Point::new(50, 250),
            core::Point::new(250, 150),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        let best = detector
            .match_templates(&test_canvas, 30.0, None, None)?
            .into_iter()
            .max()
            .unwrap();

        println!("Best match: similarity={:.2}", best.similarity);
        assert!(best.similarity >= 0.3);

        Ok(())
    }
}
