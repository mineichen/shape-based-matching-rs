use opencv::{
    core::{self as cv, Mat, Point, Scalar},
    imgproc,
    prelude::*,
};

use super::Matches;

/// Float -> pixel index for public image coordinates, where pixel `N`'s
/// center is `N + 0.5`: plain truncation, `v as i32`. Keep exactly this
/// arithmetic. Do NOT substitute `v.round()`: it differs for negative
/// half-way values and within half an ulp of integer boundaries (f32).
///
/// This is bit-equivalent to the legacy `(v - 0.5 + 0.5) as i32`: all
/// positions this crate outputs are `integer + 0.5`, so truncation drops
/// exactly the offset. It only does real work on genuinely fractional
/// floats.
#[inline(always)]
fn to_pixel(v: f32) -> i32 {
    v as i32
}

/// Component-wise [`to_pixel`] for points. Quantize FIRST, then add
/// integral offsets as integers — never fold offsets into the float before
/// quantization (boundary rounding can differ).
#[inline(always)]
fn to_pixel_pt(p: crate::Point2f) -> crate::Point2i {
    crate::Point2i::new(to_pixel(p.x), to_pixel(p.y))
}

impl<'a> Matches<'a> {
    /// Debug visualization.
    ///
    /// Rasterization stays entirely on integer pixels: float positions are
    /// quantized once via `to_pixel_pt`, and (integral) match offsets are
    /// added as integers afterwards — never folded into the float before
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

#[cfg(test)]
mod tests {
    use super::{to_pixel, to_pixel_pt};
    use crate::Point2f;

    #[test]
    fn to_pixel_matches_legacy_truncation() {
        // to_pixel MUST stay `v as i32` (truncation toward zero). All
        // positions this crate outputs are `integer + 0.5`, so pixel N's
        // center (N.5) quantizes to index N — bit-equivalent to the legacy
        // `(v_old + 0.5) as i32` for v_old = v - 0.5. The byte-exact
        // debug_visual hash tests in tests/debug_visual_hash.rs enforce
        // this end-to-end.
        assert_eq!(to_pixel(0.5), 0); // center of pixel 0
        assert_eq!(to_pixel(1.5), 1);
        assert_eq!(to_pixel(99.5), 99);
        // Fractional parts are dropped (no rounding):
        assert_eq!(to_pixel(10.7), 10);
        assert_eq!(to_pixel(-1.5), -1); // NOT -2: truncation toward zero
        // round() would move these pixels:
        assert_ne!(to_pixel(-1.5), -1.5f32.round() as i32);
        assert_ne!(to_pixel(-2.5), -2.5f32.round() as i32);
        // to_pixel_pt quantizes component-wise, offset-safe. Note the
        // negative value: -2.7 as i32 truncates TOWARD zero -> -2.
        let p = to_pixel_pt(Point2f::new(11.2, -2.7));
        assert_eq!((p.x, p.y), (11, -2));
    }
}
