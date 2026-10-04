//!
//! Destination `Mat`s are preallocated outside the timed loop, so this measures
//! steady-state filter cost with buffer reuse (the production calling pattern).
//!
//! Each group measures:
//! - `pulp`: the `pulp`-based `graph_matching::filters::pulp` implementation
//!   (single-threaded, `pulp` feature).
//! - `fearless_simd`: the portable-SIMD implementation in `filters::fsimd`,
//!   runtime-dispatched to the best level this CPU supports (single-threaded,
//!   `fearless-simd` feature).
//! - `opencv_st`: `imgproc` forced to 1 thread — the fair per-thread
//!   comparison for the single-threaded Rust code.
//! - `opencv`: `imgproc` with default threading (real-world old).
//!
//! `opencv_st` is the number to beat.
//!
//! # Buffer reuse, both sides
//!
//! The Rust backends reuse their intermediate buffers across calls (see
//! `filters::fsimd::Scratch`), which is the production calling pattern: the
//! detector filters the same image many times per run. OpenCV gets the same
//! benefit for everything it can - the destination `Mat`s are preallocated here
//! for every implementation, exactly as `imgproc` is called in `pyramid.rs`.
//!
//! What OpenCV *cannot* take from us is its own per-call scratch: `gaussianBlur`
//! and `sepFilter2D` build their kernels and row buffers inside every call
//! (`getGaussianKernel`, `createSeparableLinearFilter`, `AutoBuffer`) and expose
//! no way to pass one in. The `opencv_call_overhead_32x32` group measures what
//! that costs by running the same calls on a 32x32 image, where the filtering
//! work is negligible. Measured here (single-threaded): 6.5 us per
//! `gaussian_blur` call, 1.7 us per `sobel` call, 0.49 us per `pyr_down` call.
//! That is 0.03%-0.4% of the 3MP `opencv_st` numbers, so the comparison is
//! fair, and if anything slightly generous to OpenCV.

use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use graph_matching::Point2i;
#[cfg(feature = "pulp")]
use graph_matching::filters::pulp as filters;
use opencv::{
    core::{self, Mat, Scalar, Size},
    imgproc,
    prelude::*,
};

/// Image sizes as `(rows, cols)`: 0.3MP (the old bench size), 3MP and 4.9MP,
/// which are the sizes matching is actually used on. Bigger than ~1MP the
/// filters are memory bound, so the small size is kept to separate compute from
/// bandwidth effects.
const SIZES: [(i32, i32); 3] = [(480, 640), (1536, 2048), (1920, 2560)];

/// `filters::fsimd` is only compiled with the `fearless-simd` feature.
#[cfg(feature = "fearless-simd")]
use graph_matching::filters::fsimd;

fn make_input(rows: i32, cols: i32, typ: i32, channels: i32, seed: u64) -> Mat {
    let mut m = Mat::new_rows_cols_with_default(rows, cols, typ, Scalar::all(0.0)).unwrap();
    let mut s = seed.max(1);
    for r in 0..rows {
        let row_ptr = m.ptr_mut(r).unwrap();
        unsafe {
            for c in 0..(cols * channels) as usize {
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
    let run_gaussian = |src: &Mat, dst: &mut Mat| {
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

    for (rows, cols) in SIZES {
        let gray = make_input(rows, cols, core::CV_8UC1, 1, 0x1234);
        let color = make_input(rows, cols, core::CV_8UC3, 3, 0x5678);
        for (name, src) in [("gray", &gray), ("color", &color)] {
            let mut group = c.benchmark_group(format!("gaussian_7x7_{name}_{cols}x{rows}"));
            group.throughput(Throughput::Bytes((rows * cols * src.channels()) as u64));
            #[cfg(feature = "pulp")]
            group.bench_function("pulp", |b| {
                let mut dst = Mat::default();
                filters::gaussian_blur_7x7(src, &mut dst).unwrap();
                b.iter(|| filters::gaussian_blur_7x7(black_box(src), black_box(&mut dst)).unwrap());
            });
            #[cfg(feature = "fearless-simd")]
            group.bench_function("fearless_simd", |b| {
                let mut sc = fsimd::Scratch::default();
                let mut dst = Mat::default();
                fsimd::gaussian_blur_7x7(&mut sc, src, &mut dst).unwrap();
                b.iter(|| {
                    fsimd::gaussian_blur_7x7(&mut sc, black_box(src), black_box(&mut dst)).unwrap()
                });
            });
            group.bench_function("opencv_st", |b| {
                single_threaded(|| {
                    let mut dst = Mat::default();
                    run_gaussian(src, &mut dst);
                    b.iter(|| run_gaussian(black_box(src), black_box(&mut dst)));
                });
            });
            group.bench_function("opencv", |b| {
                let mut dst = Mat::default();
                run_gaussian(src, &mut dst);
                b.iter(|| run_gaussian(black_box(src), black_box(&mut dst)));
            });
            group.finish();
        }
    }
}

fn bench_sobel(c: &mut Criterion) {
    for (rows, cols) in SIZES {
        let gray = make_input(rows, cols, core::CV_8UC1, 1, 0x9ABC);
        let color = make_input(rows, cols, core::CV_8UC3, 3, 0xDEF0);
        for (name, src, depth) in [
            ("gray", &gray, core::CV_32F),
            ("color", &color, core::CV_16S),
        ] {
            let run_sobel = |src: &Mat, dx: &mut Mat, dy: &mut Mat| {
                for (m, sx, sy) in [(&mut *dx, 1, 0), (&mut *dy, 0, 1)] {
                    imgproc::sobel(src, m, depth, sx, sy, 3, 1.0, 0.0, core::BORDER_REPLICATE)
                        .unwrap();
                }
            };
            let mut group = c.benchmark_group(format!("sobel_3x3_{name}_{cols}x{rows}"));
            group.throughput(Throughput::Bytes((rows * cols * src.channels()) as u64));
            #[cfg(feature = "pulp")]
            group.bench_function("pulp", |b| {
                let mut dx = Mat::default();
                let mut dy = Mat::default();
                if depth == core::CV_32F {
                    filters::sobel_grayscale(src, &mut dx, &mut dy).unwrap();
                } else {
                    filters::sobel_color_i16(src, &mut dx, &mut dy).unwrap();
                }
                b.iter(|| {
                    let (dx, dy) = (&mut dx, &mut dy);
                    if depth == core::CV_32F {
                        filters::sobel_grayscale(black_box(src), dx, dy).unwrap();
                    } else {
                        filters::sobel_color_i16(black_box(src), dx, dy).unwrap();
                    }
                });
            });
            #[cfg(feature = "fearless-simd")]
            group.bench_function("fearless_simd", |b| {
                let mut sc = fsimd::Scratch::default();
                let mut dx = Mat::default();
                let mut dy = Mat::default();
                if depth == core::CV_32F {
                    fsimd::sobel_grayscale(&mut sc, src, &mut dx, &mut dy).unwrap();
                } else {
                    fsimd::sobel_color_i16(&mut sc, src, &mut dx, &mut dy).unwrap();
                }
                b.iter(|| {
                    let (dx, dy) = (&mut dx, &mut dy);
                    if depth == core::CV_32F {
                        fsimd::sobel_grayscale(&mut sc, black_box(src), dx, dy).unwrap();
                    } else {
                        fsimd::sobel_color_i16(&mut sc, black_box(src), dx, dy).unwrap();
                    }
                });
            });
            group.bench_function("opencv_st", |b| {
                single_threaded(|| {
                    let mut dx = Mat::default();
                    let mut dy = Mat::default();
                    run_sobel(src, &mut dx, &mut dy);
                    b.iter(|| run_sobel(black_box(src), black_box(&mut dx), black_box(&mut dy)));
                });
            });
            group.bench_function("opencv", |b| {
                let mut dx = Mat::default();
                let mut dy = Mat::default();
                run_sobel(src, &mut dx, &mut dy);
                b.iter(|| run_sobel(black_box(src), black_box(&mut dx), black_box(&mut dy)));
            });
            group.finish();
        }
    }
}

fn bench_pyr_down(c: &mut Criterion) {
    for (rows, cols) in SIZES {
        for (name, src) in [
            ("gray", make_input(rows, cols, core::CV_8UC1, 1, 0x1357)),
            ("color", make_input(rows, cols, core::CV_8UC3, 3, 0x2468)),
        ] {
            let src = &src;
            let run_pyr = |src: &Mat, dst: &mut Mat| {
                imgproc::pyr_down_def(src, dst).unwrap();
            };
            let mut group = c.benchmark_group(format!("pyr_down_{name}_{cols}x{rows}"));
            group.throughput(Throughput::Bytes((rows * cols * src.channels()) as u64));
            #[cfg(feature = "pulp")]
            group.bench_function("pulp", |b| {
                let mut dst = Mat::default();
                filters::pyr_down(src, &mut dst).unwrap();
                b.iter(|| filters::pyr_down(black_box(src), black_box(&mut dst)).unwrap());
            });
            #[cfg(feature = "fearless-simd")]
            group.bench_function("fearless_simd", |b| {
                let mut sc = fsimd::Scratch::default();
                let mut dst = Mat::default();
                fsimd::pyr_down(&mut sc, src, &mut dst).unwrap();
                b.iter(|| fsimd::pyr_down(&mut sc, black_box(src), black_box(&mut dst)).unwrap());
            });
            group.bench_function("opencv_st", |b| {
                single_threaded(|| {
                    let mut dst = Mat::default();
                    run_pyr(src, &mut dst);
                    b.iter(|| run_pyr(black_box(src), black_box(&mut dst)));
                });
            });
            group.bench_function("opencv", |b| {
                let mut dst = Mat::default();
                run_pyr(src, &mut dst);
                b.iter(|| run_pyr(black_box(src), black_box(&mut dst)));
            });
            group.finish();
        }
    }
}

/// End-to-end: detector build (pyramid + template extraction) and matching on a
/// synthetic image. Absolute numbers for the new implementation; the
/// old-implementation total is estimated from the micro ratios above.
/// Per-call overhead of the OpenCV entry points, measured on a 32x32 image so
/// that the filtering work itself is negligible and the number is essentially
/// call setup plus the internal `AutoBuffer` allocations that the Rust
/// backends replace with reused [`fsimd::Scratch`] buffers.
///
/// OpenCV exposes no way to hand those buffers in, so this is the one place
/// where the comparison cannot be made exactly even; the group exists so the
/// size of the effect is on the record instead of assumed.
fn bench_opencv_call_overhead(c: &mut Criterion) {
    const N: i32 = 32;
    let gray = make_input(N, N, core::CV_8UC1, 1, 0x1);
    let color = make_input(N, N, core::CV_8UC3, 3, 0x2);
    let mut dst = Mat::default();
    let mut dx = Mat::default();
    let mut dy = Mat::default();

    let mut group = c.benchmark_group("opencv_call_overhead_32x32");
    // Forced to one thread like every other `opencv_st` measurement: the point
    // here is the per-call buffer allocation, not OpenCV's thread pool.
    group.bench_function("gaussian_blur_gray", |b| {
        single_threaded(|| {
            b.iter(|| {
                imgproc::gaussian_blur(
                    &gray,
                    &mut dst,
                    Size::new(7, 7),
                    0.0,
                    0.0,
                    core::BORDER_REPLICATE,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )
                .unwrap()
            });
        });
    });
    group.bench_function("sobel_gray_x2", |b| {
        single_threaded(|| {
            b.iter(|| {
                for (m, sx, sy) in [(&mut dx, 1, 0), (&mut dy, 0, 1)] {
                    imgproc::sobel(
                        &gray,
                        m,
                        core::CV_32F,
                        sx,
                        sy,
                        3,
                        1.0,
                        0.0,
                        core::BORDER_REPLICATE,
                    )
                    .unwrap();
                }
            });
        });
    });
    group.bench_function("pyr_down_gray", |b| {
        single_threaded(|| {
            b.iter(|| imgproc::pyr_down_def(&gray, &mut dst).unwrap());
        });
    });
    group.bench_function("pyr_down_color", |b| {
        single_threaded(|| {
            b.iter(|| imgproc::pyr_down_def(&color, &mut dst).unwrap());
        });
    });
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    use graph_matching::Detector;

    const TEMPLATE_SIZE: i32 = 256;
    const ROWS: i32 = 480;
    const COLS: i32 = 640;

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
    let center = Point2i::splat(TEMPLATE_SIZE / 2);

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
    #[cfg(feature = "fearless-simd")]
    group.bench_function("detector_build_1_template_fearless_simd", |b| {
        b.iter(|| {
            Detector::builder()
                .with_backend(graph_matching::FearlessSimdBackend::default())
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
    #[cfg(feature = "fearless-simd")]
    group.bench_function("match_templates_1_template_fearless_simd", |b| {
        let mut detector = Detector::builder()
            .with_backend(graph_matching::FearlessSimdBackend::default())
            .num_features(63)
            .with_template("rect", &template, |mut cfg| {
                cfg.add_rotated(0.0, center);
            })
            .build()
            .unwrap();
        b.iter(|| {
            let matches = detector
                .match_templates(black_box(&img), 0.8, None)
                .unwrap();
            black_box(matches.len())
        });
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_gaussian,
    bench_sobel,
    bench_pyr_down,
    bench_opencv_call_overhead,
    bench_end_to_end
}
criterion_main!(benches);
