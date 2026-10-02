use opencv::{
    core::{self as cv, Mat, Point, Scalar},
    imgproc,
    prelude::*,
};

use super::Matches;
use crate::{Point2Fixed, Point2i};

impl<'a> Matches<'a> {
    /// Debug visualization.
    ///
    /// Rasterization stays entirely on integer pixels: fixed positions are
    /// quantized once via `to_pixel_pt`, and (integral) match offsets are
    /// added as integers afterwards — never folded into the position before
    /// quantization. No `LINE_AA`, no float drawing.
    pub fn debug_visual(
        &self,
        input: Mat,
        template_region: Option<cv::Rect>,
    ) -> Result<Mat, opencv::Error> {
        let matches = self.0.as_slice();
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
            // Quantize the match position once; all integer draw arithmetic
            // below adds these pixel indices.
            let mpx = to_pixel_pt(match_item.pos);

            // Draw rectangle around match
            let color = Scalar::new(0.0, 100.0, 0.0, 0.0); // Dark green
            imgproc::rectangle(
                &mut result,
                cv::Rect::new(
                    mpx.x - BORDER_PADDING,
                    mpx.y - BORDER_PADDING,
                    templ.width.get() as i32 + 2 * BORDER_PADDING,
                    templ.height.get() as i32 + 2 * BORDER_PADDING,
                ),
                color,
                2,
                imgproc::LINE_8,
                0,
            )?;

            // Draw center point
            let center_px = to_pixel_pt(match_item.center_point());
            imgproc::circle(
                &mut result,
                Point::new(center_px.x, center_px.y),
                5,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?; // Red filled circle
            imgproc::circle(
                &mut result,
                Point::new(center_px.x, center_px.y),
                10,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?; // Red outline

            // Draw features. Both are integer pixel indices: add the match
            // offset as a vector (offset, not position).
            for feat in &templ.features {
                let px = feat.pos + mpx.to_vector();
                imgproc::circle(
                    &mut result,
                    Point::new(px.x, px.y),
                    2,
                    color,
                    -1,
                    imgproc::LINE_8,
                    0,
                )?;
            }

            // Draw similarity and angle text
            let label = format!(
                "Score: {}% @{}deg, scale: {}",
                (match_item.similarity * 100.0).round() as i32,
                match_item.angle(),
                match_item.scale()
            );
            imgproc::put_text(
                &mut result,
                &label,
                Point::new(mpx.x + 5, mpx.y + 20),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar::new(0.0, 180.0, 0.0, 0.0), // Brighter green for text readability
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        Ok(result)
    }
}

#[inline(always)]
fn to_pixel_pt(p: Point2Fixed) -> Point2i {
    Point2i::new(to_pixel(p.x), to_pixel(p.y))
}

#[inline(always)]
fn to_pixel(v: fixed::types::U20F12) -> i32 {
    v.to_num()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_pixel_pt;

    #[test]
    fn to_pixel_matches_legacy_truncation() {
        // to_pixel MUST stay truncation (drop fractional bits). All
        // positions this crate outputs are `integer + 0.5`, so pixel N's
        // center (N.5) quantizes to index N. The byte-exact
        // debug_visual hash tests in tests/debug_visual_hash.rs enforce
        // this end-to-end.
        use fixed::types::U20F12;
        assert_eq!(to_pixel(U20F12::from_num(0.5)), 0); // center of pixel 0
        assert_eq!(to_pixel(U20F12::from_num(1.5)), 1);
        assert_eq!(to_pixel(U20F12::from_num(99.5)), 99);
        // Fractional parts are dropped (no rounding):
        assert_eq!(to_pixel(U20F12::from_num(10.7)), 10);
        // to_pixel_pt quantizes component-wise, offset-safe.
        let p = to_pixel_pt(from_pixel_pt(Point2i::new(11, 22)));
        assert_eq!((p.x, p.y), (11, 22));
    }

    #[test]
    fn from_pixel_pt_is_center() {
        use fixed::types::U20F12;
        // Pixel N's center is N + 0.5, exactly representable in U20F12.
        let p = from_pixel_pt(Point2i::new(0, 1));
        assert_eq!(p.x, U20F12::from_num(0.5));
        assert_eq!(p.y, U20F12::from_num(1.5));
        // Roundtrip: from -> to is identity on non-negative indices.
        for (x, y) in [(0, 0), (11, 22), (99, 199), (400, 300)] {
            let idx = Point2i::new(x, y);
            assert_eq!(to_pixel_pt(from_pixel_pt(idx)), idx);
        }
    }
}
