//! Test helpers shared by the integration tests.
//!
//! This module is not a test target itself: Cargo only picks up the `.rs` files
//! directly inside `tests/`, so `tests/common/mod.rs` is compiled *into* every
//! test file that declares `mod common;`.
//!
//! # `backend_test!`
//!
//! A filter backend is an implementation detail, never of the result: every
//! enabled backend must turn the same input into the same output. So a
//! functional test must not silently run against "the default backend" only —
//! whichever features happen to be enabled would decide what is actually
//! covered, which is how backends end up untested. `backend_test!` runs the test
//! body with *every* enabled backend and asserts that all of them return the
//! very same value, so any divergence fails the test naming the backend that
//! diverged:
//!
//! ```ignore
//! mod common;
//! use common::backend_test;
//! use graph_matching::{Backend, Detector, Point2i};
//!
//! backend_test! {
//!     /// Every backend must find the rectangle at the same position.
//!     fn rect_match(backend: impl Backend) -> TestResult<(Point2Fixed, f32)> {
//!         let mut detector = Detector::builder()
//!             .with_backend(backend)
//!             .with_template("rect", &img, |mut cfg| {
//!                 cfg.add_rotated(0.0, Point2i::splat(100));
//!             })
//!             .build()?;
//!         let best = detector.match_templates(&img, 0.5, None)?.into_iter().max().unwrap();
//!         // Pins, checked per backend ...
//!         assert!(best.similarity > 0.99);
//!         // ... and everything that must be equal across backends is returned.
//!         Ok((best.pos, best.similarity))
//!     }
//! }
//! ```
//!
//! The body takes the backend under test by value, so it can be handed to
//! `DetectorBuilder::with_backend`, and returns `Result<Value, Error>`: the
//! `Value` needs `PartialEq + Debug` (it is what gets compared), an error in
//! any backend fails the test naming that backend.

/// Run a functional test with every enabled filter backend and compare the
/// returned values. See the module docs.
macro_rules! backend_test {
    (
        $(#[$meta:meta])*
        fn $name:ident($backend:ident: $backend_ty:ty $(,)?) -> $ret:ty $body:block
    ) => {
        $(#[$meta])*
        #[test]
        fn $name() {
            // Nested so the body can be a plain function; it still resolves
            // the imports of the file it was written in.
            fn run($backend: $backend_ty) -> $ret $body

            let mut results: ::std::vec::Vec<(&str, $ret)> = ::std::vec::Vec::new();
            #[cfg(feature = "fearless-simd")]
            results.push((
                "fearless_simd",
                run(::graph_matching::FearlessSimdBackend::default()),
            ));
            #[cfg(feature = "opencv")]
            results.push(("opencv", run(::graph_matching::OpenCvBackend)));

            // The first enabled backend is the reference every other one is
            // compared against.
            let mut results = results.into_iter();
            let (reference_name, reference) = results
                .next()
                .expect("no filter backend enabled, the crate requires at least one");
            let reference = reference
                .unwrap_or_else(|e| panic!("backend {reference_name} failed: {e:?}"));
            for (name, result) in results {
                let value = result.unwrap_or_else(|e| panic!("backend {name} failed: {e:?}"));
                assert_eq!(
                    value, reference,
                    "backend {name} diverged from {reference_name}"
                );
            }
        }
    };
}

pub(crate) use backend_test;
