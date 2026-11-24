//! The keyed layout that turns code bits into a spatial pattern.

use crate::code::CODE_BITS;

/// Cells per side. The grid is relative to the image, so it scales with it.
pub const GRID: usize = 48;

pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn key_seed(key: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Which code bit each cell carries, and with which sign.
pub struct Layout {
    pub owner: Vec<u8>,
    pub sign: Vec<f32>,
}

impl Layout {
    pub fn new(key: &str) -> Self {
        let n = GRID * GRID;
        let mut rng = SplitMix64(key_seed(key));
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        let owner = perm.iter().map(|p| (p % CODE_BITS) as u8).collect();
        let sign = (0..n).map(|_| if rng.next() & 1 == 0 { -1.0 } else { 1.0 }).collect();
        Layout { owner, sign }
    }

    /// Expected sign of every cell for a given codeword.
    pub fn cells(&self, bits: &[bool; CODE_BITS]) -> Vec<f32> {
        (0..GRID * GRID)
            .map(|i| if bits[self.owner[i] as usize] { self.sign[i] } else { -self.sign[i] })
            .collect()
    }
}

/// Cell index of a pixel coordinate along one axis.
#[inline]
pub fn cell_of(p: usize, len: usize) -> usize {
    (p * GRID / len).min(GRID - 1)
}
