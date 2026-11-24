use tracemark::{detect, embed, EmbedOptions, Rgb, Verdict};

/// Procedural stand-in for an illustration: flat-coloured discs with dark outlines on a
/// gradient, placed irregularly so that every region looks different.
fn artwork(w: usize, h: usize) -> Vec<u8> {
    let mut seed = 0x1234_5678u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let discs: Vec<[f32; 6]> = (0..60)
        .map(|_| [rnd() * w as f32, rnd() * h as f32, 15.0 + rnd() * 70.0, 40.0 + rnd() * 200.0, 40.0 + rnd() * 200.0, 40.0 + rnd() * 200.0])
        .collect();
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let mut px = [250.0 - 60.0 * y as f32 / h as f32, 235.0, 215.0 + 30.0 * x as f32 / w as f32];
            for d in &discs {
                let r = ((x as f32 - d[0]).powi(2) + (y as f32 - d[1]).powi(2)).sqrt();
                if r < d[2] {
                    px = if r > d[2] - 3.0 { [25.0; 3] } else { [d[3], d[4], d[5]] };
                }
            }
            for c in 0..3 {
                out[(y * w + x) * 3 + c] = px[c] as u8;
            }
        }
    }
    out
}

/// Crop a region then shrink it by an integer factor, optionally mirrored.
fn damage(src: &[u8], w: usize, x0: usize, y0: usize, cw: usize, ch: usize, k: usize, mirror: bool) -> (usize, usize, Vec<u8>) {
    let (nw, nh) = (cw / k, ch / k);
    let mut out = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        for x in 0..nw {
            for c in 0..3 {
                let mut s = 0u32;
                for dy in 0..k {
                    for dx in 0..k {
                        s += src[((y0 + y * k + dy) * w + x0 + x * k + dx) * 3 + c] as u32;
                    }
                }
                let dx = if mirror { nw - 1 - x } else { x };
                out[(y * nw + dx) * 3 + c] = ((s + (k * k / 2) as u32) / (k * k) as u32) as u8;
            }
        }
    }
    (nw, nh, out)
}

const W: usize = 900;
const H: usize = 700;

#[test]
fn reads_untouched_copy() {
    let art = artwork(W, H);
    let marked = embed(Rgb::new(W, H, &art), 123_456, &EmbedOptions::default());
    let max = art.iter().zip(&marked).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    assert!(max <= 5, "pixels changed by up to {max}");
    let d = detect(Rgb::new(W, H, &art), Rgb::new(W, H, &marked), "", &[]);
    assert_eq!((d.verdict, d.id), (Verdict::Match, Some(123_456)));
}

#[test]
fn survives_crop_shrink_and_mirror() {
    let art = artwork(W, H);
    let marked = embed(Rgb::new(W, H, &art), 77, &EmbedOptions::default());
    for mirror in [false, true] {
        let (w, h, leak) = damage(&marked, W, 150, 100, 600, 480, 2, mirror);
        let d = detect(Rgb::new(W, H, &art), Rgb::new(w, h, &leak), "", &[]);
        assert_eq!((d.verdict, d.id, d.transform.mirrored), (Verdict::Match, Some(77), mirror));
    }
}

#[test]
fn unmarked_copy_names_nobody() {
    let art = artwork(W, H);
    let (w, h, copy) = damage(&art, W, 0, 0, W, H, 2, false);
    let ids: Vec<u32> = (0..500).collect();
    let d = detect(Rgb::new(W, H, &art), Rgb::new(w, h, &copy), "", &ids);
    assert_eq!(d.verdict, Verdict::None);
    assert_eq!(d.id, None);
}

#[test]
fn wrong_key_reads_nothing() {
    let art = artwork(W, H);
    let opt = EmbedOptions { key: "alice".into(), ..Default::default() };
    let marked = embed(Rgb::new(W, H, &art), 9, &opt);
    assert_eq!(detect(Rgb::new(W, H, &art), Rgb::new(W, H, &marked), "alice", &[]).id, Some(9));
    let d = detect(Rgb::new(W, H, &art), Rgb::new(W, H, &marked), "mallory", &[9]);
    assert_eq!(d.verdict, Verdict::None);
}
