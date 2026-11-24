//! Align a suspect copy onto the original: uniform-ish scale, shift, optional mirror.
//!
//! Coarse step: slide a shrunken copy of the suspect over a shrunken original at
//! many scales. Fine step: hill-climb scale and position at higher resolution.

use crate::Rgb;

pub(crate) struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Gray {
    pub fn from_rgb(img: Rgb) -> Self {
        let data = img
            .data
            .chunks_exact(3)
            .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
            .collect();
        Gray { w: img.width, h: img.height, data }
    }

    fn mirrored(&self) -> Gray {
        let mut data = self.data.clone();
        for row in data.chunks_exact_mut(self.w) {
            row.reverse();
        }
        Gray { w: self.w, h: self.h, data }
    }

    fn resized(&self, nw: usize, nh: usize) -> Gray {
        let (nw, nh) = (nw.max(1), nh.max(1));
        Gray { w: nw, h: nh, data: resize_plane(&self.data, self.w, self.h, nw, nh) }
    }

    fn fit(&self, max_side: usize) -> Gray {
        let m = self.w.max(self.h);
        if m <= max_side {
            return self.resized(self.w, self.h);
        }
        let k = max_side as f64 / m as f64;
        self.resized((self.w as f64 * k).round() as usize, (self.h as f64 * k).round() as usize)
    }

    /// Bilinear sample at continuous coordinates (pixel centres at +0.5).
    #[inline]
    fn sample(&self, x: f32, y: f32) -> Option<f32> {
        let (fx, fy) = (x - 0.5, y - 0.5);
        if fx < 0.0 || fy < 0.0 || fx > (self.w - 1) as f32 || fy > (self.h - 1) as f32 {
            return None;
        }
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let d = &self.data;
        let top = d[y0 * self.w + x0] * (1.0 - tx) + d[y0 * self.w + x1] * tx;
        let bot = d[y1 * self.w + x0] * (1.0 - tx) + d[y1 * self.w + x1] * tx;
        Some(top * (1.0 - ty) + bot * ty)
    }
}

/// Area-averaging resize of one channel.
pub(crate) fn resize_plane(src: &[f32], w: usize, h: usize, nw: usize, nh: usize) -> Vec<f32> {
    if nw == w && nh == h {
        return src.to_vec();
    }
    let spans = |src: usize, dst: usize| -> Vec<Vec<(usize, f32)>> {
        let r = src as f64 / dst as f64;
        (0..dst)
            .map(|i| {
                let (a, b) = (i as f64 * r, ((i + 1) as f64 * r).min(src as f64));
                let mut v = Vec::new();
                let mut p = a.floor() as usize;
                while (p as f64) < b && p < src {
                    let cover = (b.min(p as f64 + 1.0) - a.max(p as f64)) / (b - a);
                    if cover > 0.0 {
                        v.push((p, cover as f32));
                    }
                    p += 1;
                }
                v
            })
            .collect()
    };
    let sx = spans(w, nw);
    let sy = spans(h, nh);
    let mut tmp = vec![0f32; nw * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for (x, s) in sx.iter().enumerate() {
            tmp[y * nw + x] = s.iter().map(|&(p, wgt)| row[p] * wgt).sum();
        }
    }
    let mut data = vec![0f32; nw * nh];
    for (y, s) in sy.iter().enumerate() {
        for x in 0..nw {
            data[y * nw + x] = s.iter().map(|&(p, wgt)| tmp[p * nw + x] * wgt).sum();
        }
    }
    data
}

/// Maps original coordinates to suspect coordinates:
/// `suspect = centre_of_suspect + scale * (original - centre)`, after mirroring
/// the suspect left-to-right when `mirrored` is set.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub cx: f32,
    pub cy: f32,
    pub ax: f32,
    pub ay: f32,
    pub mirrored: bool,
    /// Similarity of the two images once aligned, -1..1. Low means the alignment is a guess.
    pub score: f32,
    pub(crate) sw: f32,
    pub(crate) sh: f32,
}

impl Transform {
    /// Suspect coordinates back to original coordinates.
    #[inline]
    pub fn invert(&self, x: f32, y: f32) -> (f32, f32) {
        (self.cx + (x - self.sw / 2.0) / self.ax, self.cy + (y - self.sh / 2.0) / self.ay)
    }

    #[inline]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (self.sw / 2.0 + self.ax * (x - self.cx), self.sh / 2.0 + self.ay * (y - self.cy))
    }

    fn full_frame(ow: usize, oh: usize, sw: usize, sh: usize) -> Self {
        Transform {
            cx: ow as f32 / 2.0,
            cy: oh as f32 / 2.0,
            ax: sw as f32 / ow as f32,
            ay: sh as f32 / oh as f32,
            mirrored: false,
            score: 0.0,
            sw: sw as f32,
            sh: sh as f32,
        }
    }
}

/// Best zero-mean normalised correlation of `t` slid over `o`, with its position.
fn slide(o: &Gray, t: &Gray) -> (f32, usize, usize) {
    let n = (t.w * t.h) as f64;
    let mean = t.data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let tz: Vec<f32> = t.data.iter().map(|&v| v - mean as f32).collect();
    let tnorm = tz.iter().map(|&v| (v * v) as f64).sum::<f64>().sqrt();
    if tnorm < 1e-3 * n.sqrt() {
        return (0.0, 0, 0);
    }
    // Integral images of the original for windowed mean / variance.
    let iw = o.w + 1;
    let mut s1 = vec![0f64; iw * (o.h + 1)];
    let mut s2 = vec![0f64; iw * (o.h + 1)];
    for y in 0..o.h {
        let (mut r1, mut r2) = (0f64, 0f64);
        for x in 0..o.w {
            let v = o.data[y * o.w + x] as f64;
            r1 += v;
            r2 += v * v;
            s1[(y + 1) * iw + x + 1] = s1[y * iw + x + 1] + r1;
            s2[(y + 1) * iw + x + 1] = s2[y * iw + x + 1] + r2;
        }
    }
    let win = |s: &[f64], x: usize, y: usize| s[(y + t.h) * iw + x + t.w] - s[y * iw + x + t.w] - s[(y + t.h) * iw + x] + s[y * iw + x];
    let mut best = (0f32, 0usize, 0usize);
    for py in 0..=o.h - t.h {
        for px in 0..=o.w - t.w {
            let (a, b) = (win(&s1, px, py), win(&s2, px, py));
            let var = b - a * a / n;
            if var < 1e-6 * n {
                continue;
            }
            let mut dot = 0f32;
            for ty in 0..t.h {
                let orow = &o.data[(py + ty) * o.w + px..(py + ty) * o.w + px + t.w];
                let trow = &tz[ty * t.w..(ty + 1) * t.w];
                dot += orow.iter().zip(trow).map(|(a, b)| a * b).sum::<f32>();
            }
            let score = (dot as f64 / (tnorm * var.sqrt())) as f32;
            if score > best.0 {
                best = (score, px, py);
            }
        }
    }
    best
}

/// Correlation between the original and the suspect under `t`, on a grid of sample points.
/// Returns the score and how many points landed inside the suspect.
fn agreement(o: &Gray, okx: f32, oky: f32, l: &Gray, lkx: f32, lky: f32, t: &Transform, stride: usize) -> (f32, usize) {
    let (mut n, mut so, mut sl, mut soo, mut sll, mut sol) = (0usize, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mut y = stride / 2;
    while y < o.h {
        let mut x = stride / 2;
        while x < o.w {
            let (lx, ly) = t.apply((x as f32 + 0.5) / okx, (y as f32 + 0.5) / oky);
            if let Some(v) = l.sample(lx * lkx, ly * lky) {
                let (a, b) = (o.data[y * o.w + x] as f64, v as f64);
                n += 1;
                so += a;
                sl += b;
                soo += a * a;
                sll += b * b;
                sol += a * b;
            }
            x += stride;
        }
        y += stride;
    }
    if n < 16 {
        return (-1.0, n);
    }
    let nf = n as f64;
    let (va, vb) = (soo - so * so / nf, sll - sl * sl / nf);
    if va < 1e-9 || vb < 1e-9 {
        return (-1.0, n);
    }
    (((sol - so * sl / nf) / (va * vb).sqrt()) as f32, n)
}

/// Find how `suspect` sits inside `original`.
pub fn register(original: Rgb, suspect: Rgb) -> Transform {
    let og = Gray::from_rgb(original);
    let sg = Gray::from_rgb(suspect);
    let (ow, oh, sw, sh) = (og.w as f32, og.h as f32, sg.w as f32, sg.h as f32);
    let fallback = Transform::full_frame(og.w, og.h, sg.w, sg.h);

    // ---- coarse: exhaustive over scale and position at ~96 px
    let small = og.fit(96);
    let (kx, ky) = (small.w as f32 / ow, small.h as f32 / oh);
    let pre = sg.fit(192);
    let pre_m = pre.mirrored();
    let rmax = (ow / sw).min(oh / sh);
    let mut best: Option<(f32, Transform)> = None;
    let mut seen = std::collections::HashSet::new();
    let mut r = rmax;
    loop {
        let tw = ((sw * r * kx).round() as usize).min(small.w);
        let th = ((sh * r * ky).round() as usize).min(small.h);
        if tw < 10 || th < 10 || (tw * th) as f32 <= 0.02 * (small.w * small.h) as f32 {
            break;
        }
        if seen.insert((tw, th)) {
            for (mirrored, src) in [(false, &pre), (true, &pre_m)] {
                let t = src.resized(tw, th);
                let (score, px, py) = slide(&small, &t);
                if best.as_ref().map_or(true, |b| score > b.0) {
                    let (ax, ay) = (sw * kx / tw as f32, sh * ky / th as f32);
                    let tr = Transform {
                        cx: (px as f32 + tw as f32 / 2.0) / kx,
                        cy: (py as f32 + th as f32 / 2.0) / ky,
                        ax,
                        ay,
                        mirrored,
                        score,
                        sw,
                        sh,
                    };
                    best = Some((score, tr));
                }
            }
        }
        r *= 0.96;
    }
    let Some((coarse, mut t)) = best else { return fallback };
    if coarse < 0.3 {
        return fallback; // nothing to hold on to (e.g. a flat colour): assume a plain resize
    }
    let s_full = if t.mirrored { sg.mirrored() } else { sg };

    // ---- fine: hill-climb at two resolutions
    for (side, tie_axes, stop) in [(256usize, true, 0.25f32), (1024, false, 0.08)] {
        let o = og.fit(side);
        let (okx, oky) = (o.w as f32 / ow, o.h as f32 / oh);
        let shrink = (okx / t.ax).min(1.0);
        let l = s_full.resized((sw * shrink).round() as usize, (sh * shrink).round() as usize);
        let (lkx, lky) = (l.w as f32 / sw, l.h as f32 / sh);
        let stride = (((o.w * o.h) as f32 / 200_000.0).sqrt().ceil() as usize).max(1);
        let eval = |t: &Transform| agreement(&o, okx, oky, &l, lkx, lky, t, stride);
        let (mut cur, n0) = eval(&t);
        let min_n = n0 * 4 / 5;
        let mut step = 1.5f32; // in pixels of this level
        while step >= stop {
            let shift = step / okx;
            let zoom = 1.0 + 2.0 * step / o.w.max(o.h) as f32;
            let mut moves: Vec<Box<dyn Fn(&mut Transform)>> = vec![
                Box::new(move |t| t.cx += shift),
                Box::new(move |t| t.cx -= shift),
                Box::new(move |t| t.cy += shift),
                Box::new(move |t| t.cy -= shift),
            ];
            if tie_axes {
                moves.push(Box::new(move |t| { t.ax *= zoom; t.ay *= zoom }));
                moves.push(Box::new(move |t| { t.ax /= zoom; t.ay /= zoom }));
            } else {
                moves.push(Box::new(move |t| t.ax *= zoom));
                moves.push(Box::new(move |t| t.ax /= zoom));
                moves.push(Box::new(move |t| t.ay *= zoom));
                moves.push(Box::new(move |t| t.ay /= zoom));
            }
            let mut improved = true;
            let mut rounds = 0;
            while improved && rounds < 40 {
                improved = false;
                rounds += 1;
                for m in &moves {
                    let mut cand = t;
                    m(&mut cand);
                    let (s, n) = eval(&cand);
                    if n >= min_n && s > cur + 1e-6 {
                        t = cand;
                        cur = s;
                        improved = true;
                    }
                }
            }
            step *= 0.5;
        }
        t.score = cur;
    }
    t
}
