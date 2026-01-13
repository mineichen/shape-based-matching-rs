use graph_matching::{Detector, debug_visual};
use opencv::{
    core::{self, Point, Point2f, Rect},
    imgcodecs,
    prelude::*,
};
use std::{env, num::NonZeroU8, path::PathBuf};
use std::{fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = PathBuf::from(env::var("HOME").unwrap_or_else(|_| ".".to_string()));

    let image_file = home.join("Downloads/Bilder/10.tif");
    let template_region = Rect::new(2650, 200, 592, 592);
    process_image(
        image_file,
        template_region,
        vec![
            const { NonZeroU8::new(4).unwrap() },
            const { NonZeroU8::new(8).unwrap() },
        ],
        60.0,
    )?;

    println!("--------------------------------");

    let image_file = home.join("Downloads/blech_twin/1.png");
    let template_region = Rect::new(530, 420, 120, 120);
    process_image(
        image_file,
        template_region,
        vec![
            const { NonZeroU8::new(2).unwrap() },
            const { NonZeroU8::new(4).unwrap() },
        ],
        40.0,
    )?;

    Ok(())
}
fn process_image(
    image_file: PathBuf,
    template_region: Rect,
    levels: Vec<NonZeroU8>,
    threshold: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut iter = std::env::args().skip(1).fuse();
    let num_features = iter
        .next()
        .map(|x| x.parse::<usize>().expect("Invalid number of features"))
        .unwrap_or(192);
    let debug_image_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("debug_images");
    // Create debug_images directory
    fs::create_dir_all(&debug_image_dir)?;
    // Get image path from environment or use default

    println!("Loading image: {:?}", image_file);
    let test = imgcodecs::imread(&image_file.to_str().unwrap(), imgcodecs::IMREAD_GRAYSCALE)?;

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

    let img = core::Mat::roi(&test_cropped, template_region)?.try_clone()?;

    println!("Template: {}x{}", img.cols(), img.rows());

    let class_id = "template";
    let center = Point2f::new(img.cols() as f32 / 2.0, img.rows() as f32 / 2.0);
    // Use builder to add templates before building detector
    let time = std::time::Instant::now();
    let detector = Detector::builder()
        .num_features(num_features)
        .pyramid_levels(levels)?
        .weak_threshold(20.0)
        .strong_threshold(40.0)
        .with_template(class_id, &img, |mut cfg| {
            // Add rotated versions (every 1 degree from -180 to 180)

            //cfg.add_rotated_range(0..360u16, center);
            cfg.add_rotated_range((0..720).map(|x| x as f32 / 2.0), center);
            //cfg.add_rotated_range((0..90u16).map(|x| x as f32 * 4.0), center);
        })
        .build();
    println!(
        "Rotated templates queued and detector built, time: {:?}",
        time.elapsed()
    );
    let time_before_match = std::time::Instant::now();
    let mut matches = detector.match_templates(&test_cropped, threshold, None, None)?;
    println!(
        "Found {} raw match(es), time: {:?}",
        matches.len(),
        time_before_match.elapsed()
    );
    matches.sort_unstable();
    println!("Sorted match(es), time: {:?}", time_before_match.elapsed());

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

    // Print match information
    for (i, match_item) in filtered_matches.iter().enumerate() {
        let center_point = match_item.center_point();
        let angle = match_item.angle();
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

    // Generate debug visualization
    let image_bytes = debug_visual(&test_cropped, &filtered_matches, Some(template_region))?;

    // Save result
    let output_file = debug_image_dir.join(format!(
        "match_result_{}.png",
        image_file.file_stem().unwrap().to_str().unwrap()
    ));
    std::fs::write(&output_file, image_bytes)?;
    println!("Result saved to {:?}", output_file);

    Ok(())
}
