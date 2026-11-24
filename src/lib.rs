//! Per-recipient invisible watermark for artwork.
//!
//! Each copy handed out carries a 32-bit recipient id as a faint, smooth
//! brightness pattern. Reading it back needs the original image: the suspect
//! copy is aligned onto the original (undoing crop / resize / mirroring), the
//! original is subtracted, and what is left is correlated with the pattern.
//!
//! The library works on plain 8-bit RGB buffers and has no dependencies.

mod code;
mod detect;
mod embed;
mod pattern;
mod register;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use code::{decode_soft, encode_id, CODE_BITS};
pub use detect::{detect, Detection, Verdict};
pub use embed::{embed, EmbedOptions};
pub use register::{register, Transform};

/// Borrowed 8-bit RGB image, row-major, 3 bytes per pixel.
#[derive(Clone, Copy)]
pub struct Rgb<'a> {
    pub width: usize,
    pub height: usize,
    pub data: &'a [u8],
}

impl<'a> Rgb<'a> {
    pub fn new(width: usize, height: usize, data: &'a [u8]) -> Self {
        assert_eq!(data.len(), width * height * 3, "RGB buffer has the wrong length");
        Rgb { width, height, data }
    }
}
