use graph_matching::Detector;
use opencv::{
    core::{self, Point, Point2f, Rect, Scalar},
    imgcodecs, imgproc,
    prelude::*,
};
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create debug_images directory
    fs::create_dir_all("debug_images")?;
    // Get image path from environment or use default
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let image_file = format!("{}/Downloads/Bilder/10.tif", home);
    println!("Loading image: {}", image_file);
    let test = imgcodecs::imread(&image_file, imgcodecs::IMREAD_COLOR)?;

    if test.empty() {
        eprintln!("Failed to load image");
        return Err("Image not found".into());
    }

    println!("Original image: {}x{}", test.cols(), test.rows());

    // Crop image to dimensions divisible by 128 (16 * max pyramid level)
    // Pyramid levels are {4, 8}, so we need divisibility by 16 * 8 = 128
    let crop_width = test.cols() - (test.cols() % 128);
    let crop_height = test.rows() - (test.rows() % 128);
    let test_cropped =
        core::Mat::roi(&test, Rect::new(0, 0, crop_width, crop_height))?.try_clone()?;

    println!(
        "Cropped image: {}x{}",
        test_cropped.cols(),
        test_cropped.rows()
    );

    // Extract template from fixed region (dimensions must be multiples of 16)
    let template_region = Rect::new(2650, 200, 592, 592);
    let img = core::Mat::roi(&test_cropped, template_region)?.try_clone()?;

    println!("Template: {}x{}", img.cols(), img.rows());

    let class_id = "template";
    let center = Point2f::new(img.cols() as f32 / 2.0, img.rows() as f32 / 2.0);
    // Use builder to add templates before building detector
    let time = std::time::Instant::now();
    let detector = Detector::builder()
        .num_features(192)
        .pyramid_levels(vec![4, 8])
        .weak_threshold(25.0)
        .strong_threshold(50.0)
        .with_template(class_id, &img, |mut cfg| {
            // Add rotated versions (every 1 degree from -180 to 180)
            cfg.add_rotated_range(0..360u16, center);
        })
        .build();
    println!(
        "Rotated templates queued and detector built, time: {:?}",
        time.elapsed()
    );
    // Match with threshold 50% (like C++ example)
    let mut matches = detector.match_templates(&test_cropped, 60.0, None, None)?;
    println!(
        "Found {} raw match(es), time: {:?}",
        matches.len(),
        time.elapsed()
    );
    matches.sort_unstable();
    println!("Sorted match(es), time: {:?}", time.elapsed());

    // Filter matches: keep only best match within min_distance (center-to-center)
    let min_distance = 50.0f32;
    let mut filtered_matches: Vec<graph_matching::Match> = Vec::new();

    for match_item in &matches {
        // Get template dimensions for this match to calculate center
        let match_templ = match_item.match_template();
        let match_cx = match_item.x as f32 + match_templ.width as f32 / 2.0;
        let match_cy = match_item.y as f32 + match_templ.height as f32 / 2.0;

        let mut keep = true;

        // Check if this match is too close to a better match already in filtered list
        for existing in &filtered_matches {
            let existing_templ = existing.match_template();
            let existing_cx = existing.x as f32 + existing_templ.width as f32 / 2.0;
            let existing_cy = existing.y as f32 + existing_templ.height as f32 / 2.0;

            let dx = match_cx - existing_cx;
            let dy = match_cy - existing_cy;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance < min_distance {
                // Too close - keep the one with higher similarity
                if match_item.similarity <= existing.similarity {
                    keep = false;
                    break;
                }
                // This match is better, will remove the existing one later
            }
        }

        if keep {
            // Remove any worse matches that are too close to this one
            filtered_matches.retain(|existing: &graph_matching::Match| {
                let existing_templ = existing.match_template();
                let existing_cx = existing.x as f32 + existing_templ.width as f32 / 2.0;
                let existing_cy = existing.y as f32 + existing_templ.height as f32 / 2.0;

                let dx = match_cx - existing_cx;
                let dy = match_cy - existing_cy;
                let distance = (dx * dx + dy * dy).sqrt();
                !(distance < min_distance && match_item.similarity > existing.similarity)
            });
            filtered_matches.push(match_item.clone());
        }
    }

    #[cfg(feature = "profile")]
    println!(
        "After filtering: {} match(es) (min_distance={}px)",
        filtered_matches.len(),
        min_distance
    );

    // Draw matches on image
    let mut result = test_cropped.clone();

    // Draw original template region in blue
    let blue = Scalar::new(255.0, 0.0, 0.0, 0.0);
    imgproc::rectangle(&mut result, template_region, blue, 3, imgproc::LINE_8, 0)?;
    imgproc::put_text(
        &mut result,
        "Original Template",
        Point::new(template_region.x, template_region.y - 10),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        blue,
        2,
        imgproc::LINE_8,
        false,
    )?;

    for (i, match_item) in filtered_matches.iter().enumerate() {
        // Get template dimensions
        let templ = match_item.match_template();
        let w = templ.width;
        let h = templ.height;

        // Draw rectangle around match
        let color = Scalar::new(0.0, 255.0, 0.0, 0.0); // Green
        imgproc::rectangle(
            &mut result,
            Rect::new(match_item.x, match_item.y, w, h),
            color,
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Draw center point
        let center_point = Point::new(match_item.x + w / 2, match_item.y + h / 2);
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

        // Calculate angle from template_id (0 = 0°, 1 = -180°, 2 = -179°, ...)
        let angle = match_item.angle();

        // Draw similarity and angle text (similarity is already in percentage)
        let label = format!("{}% @{}deg", match_item.similarity.round() as i32, angle);
        imgproc::put_text(
            &mut result,
            &label,
            Point::new(match_item.x + 5, match_item.y + 20),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            imgproc::LINE_8,
            false,
        )?;

        println!(
            "  Match {}: angle={}° similarity={} pos=({},{}) center=({},{})",
            i,
            angle,
            match_item.similarity,
            match_item.x,
            match_item.y,
            center_point.x,
            center_point.y
        );
    }

    // Save result
    let output_file = "debug_images/match_result.png";
    imgcodecs::imwrite(output_file, &result, &core::Vector::new())?;
    println!("Result saved to {}", output_file);

    Ok(())
}
