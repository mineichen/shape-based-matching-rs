//! Side-by-side benches: pure-Rust `graph_matching::filters` vs `opencv::imgproc`.
//!
//! Destination `Mat`s are preallocated outside the timed loop, so this measures
//! steady-state filter cost with buffer reuse (the production calling pattern).
//!
//! Each group measures three variants:
//! - `rust`: the new implementation (single-threaded).
//! - `opencv`: previous implementation with default threading (real-world old).
//! - `opencv_st`: previous implementation forced to 1 thread — the fair
//!   per-thread comparison for the single-threaded Rust code.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use graph_matching::{Point2i, filters};
use opencv::{
    core::{self, Mat, Scalar, Size},
    imgproc,
    prelude::*,
};

const ROWS: i32 = 480;
const COLS: i32 = 640;

fn make_input(typ: i32, channels: i32, seed: u64) -> Mat {
    let mut m = Mat::new_rows_cols_with_default(ROWS, COLS, typ, Scalar::all(0.0)).unwrap();
    let mut s = seed.max(1);
    for r in 0..ROWS {
        let row_ptr = m.ptr_mut(r).unwrap();
        unsafe {
            for c in 0..(COLS * channels) as usize {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *row_ptr.add(c) = (s >> 33) as u8;
            }
        }
    }
    m
}

/// Run `f` with OpenCV forced to 1 thread, restoring the previous setting after.
fn single_threaded(f: impl FnOnce()) {
    let prev = core::get_num_threads().unwrap_or(0);
    core::set_num_threads(1).unwrap();
    f();
    core::set_num_threads(prev).unwrap();
}

fn bench_gaussian(c: &mut Criterion) {
    let gray = make_input(core::CV_8UC1, 1, 0x1234);
    let color = make_input(core::CV_8UC3, 3, 0x5678);
    let run_gray = |src: &Mat, dst: &mut Mat| {
        imgproc::gaussian_blur(
            src,
            dst,
            Size::new(7, 7),
            0.0,
            0.0,
            core::BORDER_REPLICATE,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap()
    };
    let run_color = |src: &Mat, dst: &mut Mat| {
        imgproc::gaussian_blur(
            src,
            dst,
            Size::new(7, 7),
            0.0,
            0.0,
            core::BORDER_REPLICATE,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap()
    };

    let mut group = c.benchmark_group("gaussian_7x7_gray_640x480");
    group.bench_function("rust", |b| {
        let mut dst = Mat::default();
        filters::gaussian_blur_7x7(&gray, &mut dst).unwrap();
        b.iter(|| filters::gaussian_blur_7x7(black_box(&gray), black_box(&mut dst)).unwrap());
    });
    group.bench_function("opencv", |b| {
        let mut dst = Mat::default();
        run_gray(&gray, &mut dst);
        b.iter(|| run_gray(black_box(&gray), black_box(&mut dst)));
    });
    group.bench_function("opencv_st", |b| {
        single_threaded(|| {
            let mut dst = Mat::default();
            run_gray(&gray, &mut dst);
            b.iter(|| run_gray(black_box(&gray), black_box(&mut dst)));
        });
    });
    group.finish();

    let mut group = c.benchmark_group("gaussian_7x7_color_640x480");
    group.bench_function("rust", |b| {
        let mut dst = Mat::default();
        filters::gaussian_blur_7x7(&color, &mut dst).unwrap();
        b.iter(|| filters::gaussian_blur_7x7(black_box(&color), black_box(&mut dst)).unwrap());
    });
    group.bench_function("opencv", |b| {
        let mut dst = Mat::default();
        run_color(&color, &mut dst);
        b.iter(|| run_color(black_box(&color), black_box(&mut dst)));
    });
    group.bench_function("opencv_st", |b| {
        single_threaded(|| {
            let mut dst = Mat::default();
            run_color(&color, &mut dst);
            b.iter(|| run_color(black_box(&color), black_box(&mut dst)));
        });
    });
    group.finish();
}

fn bench_sobel(c: &mut Criterion) {
    let gray = make_input(core::CV_8UC1, 1, 0x9ABC);
    let color = make_input(core::CV_8UC3, 3, 0xDEF0);
    let run_gray = |src: &Mat, dx: &mut Mat, dy: &mut Mat| {
        imgproc::sobel(
            src,
            dx,
            core::CV_32F,
            1,
            0,
            3,
            1.0,
            0.0,
            core::BORDER_REPLICATE,
        )
        .unwrap();
        imgproc::sobel(
            src,
            dy,
            core::CV_32F,
            0,
            1,
            3,
            1.0,
            0.0,
            core::BORDER_REPLICATE,
        )
        .unwrap();
    };
    let run_color = |src: &Mat, dx: &mut Mat, dy: &mut Mat| {
        imgproc::sobel(
            src,
            dx,
            core::CV_16S,
            1,
            0,
            3,
            1.0,
            0.0,
            core::BORDER_REPLICATE,
        )
        .unwrap();
        imgproc::sobel(
            src,
            dy,
            core::CV_16S,
            0,
            1,
            3,
            1.0,
            0.0,
            core::BORDER_REPLICATE,
        )
        .unwrap();
    };

    let mut group = c.benchmark_group("sobel_3x3_gray_640x480");
    group.bench_function("rust", |b| {
        let mut dx = Mat::default();
        let mut dy = Mat::default();
        filters::sobel_grayscale(&gray, &mut dx, &mut dy).unwrap();
        b.iter(|| {
            filters::sobel_grayscale(black_box(&gray), black_box(&mut dx), black_box(&mut dy))
                .unwrap()
        });
    });
    group.bench_function("opencv", |b| {
        let mut dx = Mat::default();
        let mut dy = Mat::default();
        run_gray(&gray, &mut dx, &mut dy);
        b.iter(|| run_gray(black_box(&gray), black_box(&mut dx), black_box(&mut dy)));
    });
    group.bench_function("opencv_st", |b| {
        single_threaded(|| {
            let mut dx = Mat::default();
            let mut dy = Mat::default();
            run_gray(&gray, &mut dx, &mut dy);
            b.iter(|| run_gray(black_box(&gray), black_box(&mut dx), black_box(&mut dy)));
        });
    });
    group.finish();

    let mut group = c.benchmark_group("sobel_3x3_color_640x480");
    group.bench_function("rust", |b| {
        let mut dx = Mat::default();
        let mut dy = Mat::default();
        filters::sobel_color_i16(&color, &mut dx, &mut dy).unwrap();
        b.iter(|| {
            filters::sobel_color_i16(black_box(&color), black_box(&mut dx), black_box(&mut dy))
                .unwrap()
        });
    });
    group.bench_function("opencv", |b| {
        let mut dx = Mat::default();
        let mut dy = Mat::default();
        run_color(&color, &mut dx, &mut dy);
        b.iter(|| run_color(black_box(&color), black_box(&mut dx), black_box(&mut dy)));
    });
    group.bench_function("opencv_st", |b| {
        single_threaded(|| {
            let mut dx = Mat::default();
            let mut dy = Mat::default();
            run_color(&color, &mut dx, &mut dy);
            b.iter(|| run_color(black_box(&color), black_box(&mut dx), black_box(&mut dy)));
        });
    });
    group.finish();
}

fn bench_pyr_down(c: &mut Criterion) {
    let gray = make_input(core::CV_8UC1, 1, 0x1357);
    let color = make_input(core::CV_8UC3, 3, 0x2468);
    let run = |src: &Mat, dst: &mut Mat| {
        imgproc::pyr_down_def(src, dst).unwrap();
    };

    let mut group = c.benchmark_group("pyr_down_gray_640x480");
    group.bench_function("rust", |b| {
        let mut dst = Mat::default();
        filters::pyr_down(&gray, &mut dst).unwrap();
        b.iter(|| filters::pyr_down(black_box(&gray), black_box(&mut dst)).unwrap());
    });
    group.bench_function("opencv", |b| {
        let mut dst = Mat::default();
        run(&gray, &mut dst);
        b.iter(|| run(black_box(&gray), black_box(&mut dst)));
    });
    group.bench_function("opencv_st", |b| {
        single_threaded(|| {
            let mut dst = Mat::default();
            run(&gray, &mut dst);
            b.iter(|| run(black_box(&gray), black_box(&mut dst)));
        });
    });
    group.finish();

    let mut group = c.benchmark_group("pyr_down_color_640x480");
    group.bench_function("rust", |b| {
        let mut dst = Mat::default();
        filters::pyr_down(&color, &mut dst).unwrap();
        b.iter(|| filters::pyr_down(black_box(&color), black_box(&mut dst)).unwrap());
    });
    group.bench_function("opencv", |b| {
        let mut dst = Mat::default();
        run(&color, &mut dst);
        b.iter(|| run(black_box(&color), black_box(&mut dst)));
    });
    group.bench_function("opencv_st", |b| {
        single_threaded(|| {
            let mut dst = Mat::default();
            run(&color, &mut dst);
            b.iter(|| run(black_box(&color), black_box(&mut dst)));
        });
    });
    group.finish();
}

/// End-to-end: detector build (pyramid + template extraction) and matching on a
/// synthetic 640x480 image. Absolute numbers for the new implementation; the
/// old-implementation total is estimated from the micro ratios above.
fn bench_end_to_end(c: &mut Criterion) {
    use graph_matching::Detector;

    const TEMPLATE_SIZE: i32 = 256;

    // Synthetic image: white background, black rectangle outline with margin
    // around it (edges must not touch the template border).
    let mut img =
        Mat::new_rows_cols_with_default(ROWS, COLS, core::CV_8UC1, Scalar::all(255.0)).unwrap();
    imgproc::rectangle(
        &mut img,
        core::Rect::new(232, 152, 176, 176),
        Scalar::all(0.0),
        3,
        imgproc::LINE_8,
        0,
    )
    .unwrap();
    let template_region = core::Rect::new(192, 112, TEMPLATE_SIZE, TEMPLATE_SIZE);
    let template = core::Mat::roi(&img, template_region)
        .unwrap()
        .try_clone()
        .unwrap();
    // Pivot is the center pixel of the template.
    let center = Point2i::new(TEMPLATE_SIZE / 2, TEMPLATE_SIZE / 2);

    let mut group = c.benchmark_group("end_to_end_640x480_gray");
    group.sample_size(10);
    group.bench_function("detector_build_1_template", |b| {
        b.iter(|| {
            Detector::builder()
                .num_features(63)
                .with_template("rect", black_box(&template), |mut cfg| {
                    cfg.add_rotated(0.0, *black_box(&center));
                })
                .build()
                .unwrap()
        });
    });
    let mut detector = Detector::builder()
        .num_features(63)
        .with_template("rect", &template, |mut cfg| {
            cfg.add_rotated(0.0, center);
        })
        .build()
        .unwrap();
    group.bench_function("match_templates_1_template", |b| {
        b.iter(|| {
            let matches = detector
                .match_templates(black_box(&img), 0.8, None)
                .unwrap();
            black_box(matches.len())
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_gaussian,
    bench_sobel,
    bench_pyr_down,
    bench_end_to_end
);
criterion_main!(benches);
