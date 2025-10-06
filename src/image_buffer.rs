//! Internal module for 2D buffer with row/column layout

use std::ops::{Deref, DerefMut};

/// A 2D buffer of u8 data with row-major layout.
///
/// Memory layout: `data[row * cols + col]`
#[derive(Debug)]
pub(crate) struct ImageBuffer {
    data: Box<[u8]>,
    rows: i32,
    cols: i32,
}

impl ImageBuffer {
    pub(crate) fn new_zeroed(rows: i32, cols: i32) -> Self {
        let size = (rows * cols) as usize;
        let data = vec![0u8; size].into_boxed_slice();
        Self { data, rows, cols }
    }

    #[inline]
    pub(crate) fn rows(&self) -> i32 {
        self.rows
    }

    #[inline]
    pub(crate) fn cols(&self) -> i32 {
        self.cols
    }
}

impl Deref for ImageBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for ImageBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
