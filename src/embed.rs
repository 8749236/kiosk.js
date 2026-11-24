use crate::code::encode_id;
use crate::pattern::{key_seed, Layout, SplitMix64, GRID};
use crate::Rgb;

#[derive(Clone, Debug)]
pub struct EmbedOptions {
    /// Secret that fixes the pattern layout. Detection must use the same key.
    pub key: String,
    /// Brightness shift, in 8-bit levels, applied in flat areas.
    pub strength: f32,
    /// Extra shift allowed where the artwork is busy (lines, texture), as a multiple of the default.
    pub texture_boost: f32,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        EmbedOptions { key: String::new(), strength: 1.0, texture_boost: 1.0 }
    }
}

/// Catmull-Rom taps for upsampling GRID cells to `len` pixels.
fn taps(len: usize) -> Vec<([usize; 4], [f32; 4])> {
    (0..len)
        .map(|p| {
            let u = (p as f32 + 0.5) * GRID as f32 / len as f32 - 0.5;
            let i = u.floor();
            let t = u - i;
            let w = [
                ((-0.5 * t + 1.0) * t - 0.5) * t,
                (1.5 * t - 2.5) * t * t + 1.0,
                ((-1.5 * t + 2.0) * t + 0.5) * t,
                (0.5 * t - 0.5) * t * t,
            ];
            let idx = [-1i64, 0, 1, 2].map(|d| (i as i64 + d).clamp(0, GRID as i64 - 1) as usize);
            (idx, w)
        })
        .collect()
}

/// Mean over a (2r+1) window along both axes, edges clamped.
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; w * h];
    let mut out = vec![0f32; w * h];
    let pass = |src: &[f32], dst: &mut [f32], len: usize, lines: usize, step: usize, line_step: usize| {
        for l in 0..lines {
            let at = |i: i64| src[l * line_step + (i.clamp(0, len as i64 - 1) as usize) * step];
            let mut acc: f32 = (-(r as i64)..=r as i64).map(at).sum();
            for i in 0..len {
                dst[l * line_step + i * step] = acc / (2 * r + 1) as f32;
                acc += at(i as i64 + r as i64 + 1) - at(i as i64 - r as i64);
            }
        }
    };
    pass(src, &mut tmp, w, h, 1, w);
    pass(&tmp, &mut out, h, w, w, 1);
    out
}

/// Return a copy of `img` carrying `id`.
pub fn embed(img: Rgb, id: u32, opt: &EmbedOptions) -> Vec<u8> {
    let (w, h) = (img.width, img.height);
    let layout = Layout::new(&opt.key);
    let cells = layout.cells(&encode_id(id));

    // Smooth field in [-1, 1]: interpolate the cell grid, columns first.
    let tx = taps(w);
    let ty = taps(h);
    let mut rows = vec![0f32; GRID * w];
    for gy in 0..GRID {
        for (x, (idx, wt)) in tx.iter().enumerate() {
            rows[gy * w + x] = (0..4).map(|k| wt[k] * cells[gy * GRID + idx[k]]).sum();
        }
    }

    // Local contrast decides how much extra can be hidden.
    let boost = if opt.texture_boost > 0.0 {
        let gray: Vec<f32> = img
            .data
            .chunks_exact(3)
            .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
            .collect();
        let sq: Vec<f32> = gray.iter().map(|g| g * g).collect();
        let m = box_blur(&gray, w, h, 4);
        let m2 = box_blur(&sq, w, h, 4);
        Some((m, m2))
    } else {
        None
    };

    let mut rng = SplitMix64(key_seed(&opt.key) ^ 0xD17E_5EED);
    let mut out = img.data.to_vec();
    for y in 0..h {
        let (iy, wy) = &ty[y];
        for x in 0..w {
            let f: f32 = (0..4).map(|k| wy[k] * rows[iy[k] * w + x]).sum::<f32>().clamp(-1.0, 1.0);
            let mut amp = opt.strength;
            if let Some((m, m2)) = &boost {
                let i = y * w + x;
                let sd = (m2[i] - m[i] * m[i]).max(0.0).sqrt();
                amp += (opt.texture_boost * sd / 8.0).min(4.0 * opt.texture_boost) * opt.strength;
            }
            // Dithered rounding: no contour lines where the field crosses a level.
            let d = (f * amp + rng.unit()).floor() as i32;
            if d != 0 {
                let p = &mut out[(y * w + x) * 3..(y * w + x) * 3 + 3];
                for c in p {
                    *c = (*c as i32 + d).clamp(0, 255) as u8;
                }
            }
        }
    }
    out
}
