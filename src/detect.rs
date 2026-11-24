use crate::code::{decode_soft, encode_id, CODE_BITS};
use crate::pattern::{cell_of, Layout, GRID};
use crate::register::{register, resize_plane, Transform};
use crate::Rgb;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The id was read and its checksum holds, or a listed id matches clearly.
    Match,
    /// Something is there but it is too faint to name anyone with confidence.
    Weak,
    /// No watermark found.
    None,
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub verdict: Verdict,
    /// Id read directly from the image (checksum verified), if any.
    pub id: Option<u32>,
    /// How many uncertain bits had to be flipped for the checksum to hold.
    pub repaired_bits: usize,
    /// Best id among the supplied candidates and its strength.
    pub best_candidate: Option<(u32, f32)>,
    /// Pattern strength for the reported id, in standard deviations above chance.
    pub strength: f32,
    /// Share of the original's area present in the suspect copy.
    pub coverage: f32,
    pub transform: Transform,
}

/// Residuals beyond this many levels cannot come from the watermark.
const OUTLIER: f32 = 12.0;

/// Look for a watermark in `suspect`, given the unmarked `original`.
/// `candidates` are ids known to have been issued; they are matched directly,
/// which works on copies too damaged to be read outright.
pub fn detect(original: Rgb, suspect: Rgb, key: &str, candidates: &[u32]) -> Detection {
    let t = register(original, suspect);
    let (w, h) = (original.width, original.height);
    let (sw, sh) = (suspect.width, suspect.height);

    // The original as it would look at the suspect's resolution. Comparing there, rather than
    // blowing the suspect back up, keeps detail lost to downscaling out of the residual.
    let (rw, rh) = (
        ((w as f32 * t.ax).round() as usize).clamp(1, w),
        ((h as f32 * t.ay).round() as usize).clamp(1, h),
    );
    let planes: Vec<Vec<f32>> = (0..3)
        .map(|c| {
            let plane: Vec<f32> = original.data.iter().skip(c).step_by(3).map(|&v| v as f32).collect();
            resize_plane(&plane, w, h, rw, rh)
        })
        .collect();
    let (rkx, rky) = (rw as f32 / w as f32, rh as f32 / h as f32);
    // For a suspect pixel: the matching original colour and the cell it falls in.
    let reference = |x: usize, y: usize| -> Option<([f32; 3], usize)> {
        let lx = if t.mirrored { sw as f32 - (x as f32 + 0.5) } else { x as f32 + 0.5 };
        let (ox, oy) = t.invert(lx, y as f32 + 0.5);
        if ox < 0.0 || oy < 0.0 || ox >= w as f32 || oy >= h as f32 {
            return None;
        }
        let fx = (ox * rkx - 0.5).clamp(0.0, (rw - 1) as f32);
        let fy = (oy * rky - 0.5).clamp(0.0, (rh - 1) as f32);
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(rw - 1), (y0 + 1).min(rh - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let mut out = [0f32; 3];
        for (c, o) in out.iter_mut().enumerate() {
            let p = &planes[c];
            let top = p[y0 * rw + x0] * (1.0 - tx) + p[y0 * rw + x1] * tx;
            let bot = p[y1 * rw + x0] * (1.0 - tx) + p[y1 * rw + x1] * tx;
            *o = top * (1.0 - ty) + bot * ty;
        }
        Some((out, cell_of(oy as usize, h) * GRID + cell_of(ox as usize, w)))
    };

    // Pass 1: learn the tone curve the copy went through (brightness, contrast, colour shifts).
    // Done twice: the second time ignores pixels far off the first estimate, so that stickers,
    // captions or other overpainting do not skew the curve.
    let mut lut = [[0f32; 256]; 3];
    let mut covered = 0usize;
    for round in 0..2 {
        let mut sum = [[0f64; 256]; 3];
        let mut cnt = [[0u32; 256]; 3];
        covered = 0;
        for y in 0..sh {
            for x in 0..sw {
                if let Some((r, _)) = reference(x, y) {
                    covered += 1;
                    let s = &suspect.data[(y * sw + x) * 3..];
                    for c in 0..3 {
                        let v = r[c].round().clamp(0.0, 255.0) as usize;
                        let shift = s[c] as f32 - r[c];
                        if round == 0 || (shift - lut[c][v]).abs() <= OUTLIER {
                            sum[c][v] += shift as f64;
                            cnt[c][v] += 1;
                        }
                    }
                }
            }
        }
        for c in 0..3 {
            for v in 0..256 {
                if cnt[c][v] > 0 {
                    lut[c][v] = (sum[c][v] / cnt[c][v] as f64) as f32;
                } else if round == 0 {
                    lut[c][v] = 0.0;
                }
            }
        }
    }

    // Pass 2: residual after removing the original, averaged per cell.
    let n = GRID * GRID;
    let mut acc = vec![0f64; n];
    let mut hits = vec![0u32; n];
    for y in 0..sh {
        for x in 0..sw {
            if let Some((r, cell)) = reference(x, y) {
                let s = &suspect.data[(y * sw + x) * 3..];
                let d = (0..3)
                    .map(|c| s[c] as f32 - r[c] - lut[c][r[c].round().clamp(0.0, 255.0) as usize])
                    .sum::<f32>()
                    / 3.0;
                // Far-off pixels are edits or misaligned edges, not watermark: leave them out.
                if d.abs() <= OUTLIER {
                    acc[cell] += d as f64;
                    hits[cell] += 1;
                }
            }
        }
    }
    // A cell counts if at least half of it is present in the suspect.
    let px_per_cell = (w * h) as f32 / n as f32 * t.ax * t.ay;
    let valid: Vec<bool> = (0..n).map(|i| hits[i] > 0 && hits[i] as f32 * 2.0 >= px_per_cell).collect();
    let mut x: Vec<f32> = (0..n).map(|i| if valid[i] { (acc[i] / hits[i] as f64) as f32 } else { 0.0 }).collect();
    let used = valid.iter().filter(|&&v| v).count().max(1);
    let mean = x.iter().sum::<f32>() / used as f32;
    for i in 0..n {
        if valid[i] {
            x[i] -= mean;
        }
    }
    let energy = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);

    let layout = Layout::new(key);
    let mut soft = [0f32; CODE_BITS];
    for i in 0..n {
        soft[layout.owner[i] as usize] += layout.sign[i] * x[i];
    }
    let strength_of = |id: u32| -> f32 {
        let cells = layout.cells(&encode_id(id));
        x.iter().zip(&cells).map(|(a, b)| a * b).sum::<f32>() / energy
    };

    let read = decode_soft(&soft);
    let best_candidate = candidates
        .iter()
        .map(|&id| (id, strength_of(id)))
        .max_by(|a, b| a.1.total_cmp(&b.1));

    // A fixed id scoring 5 sigma is not chance. An id read off the image is the best-fitting
    // one by construction, so there the checksum is the evidence and the bar is higher.
    let (verdict, strength) = match (read, best_candidate) {
        (Some((id, repaired)), _) => {
            let s = strength_of(id);
            let ok = repaired == 0 || s >= 7.0 || best_candidate.map_or(false, |c| c.0 == id);
            (if ok { Verdict::Match } else { Verdict::Weak }, s)
        }
        (None, Some((_, s))) if s >= 5.0 => (Verdict::Match, s),
        (None, Some((_, s))) if s >= 4.0 => (Verdict::Weak, s),
        (None, Some((_, s))) => (Verdict::None, s),
        (None, None) => (Verdict::None, 0.0),
    };

    Detection {
        verdict,
        id: read.map(|r| r.0),
        repaired_bits: read.map_or(0, |r| r.1),
        best_candidate,
        strength,
        coverage: (covered as f32 / (t.ax * t.ay) / (w * h) as f32).min(1.0),
        transform: t,
    }
}
