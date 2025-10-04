use graph_matching::process_image;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example usage of the image processing function
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let input_path = format!("{}/Downloads/Bilder/10.tif", home);
    let output_filename = "processed_with_lines.png";

    println!("Processing image: {}", input_path);

    match process_image(&input_path, output_filename) {
        Ok(_) => println!("Image processing completed successfully!"),
        Err(e) => {
            eprintln!("Error processing image: {}", e);
            eprintln!("Make sure the input image exists and is a valid image file.");
        }
    }

    Ok(())
}
