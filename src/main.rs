use clap::{Parser, Subcommand};
use image::{ImageBuffer, ImageFormat, Rgba, RgbaImage};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tracemark::{detect, embed, EmbedOptions, Rgb, Verdict};

#[derive(Parser)]
#[command(version, about = "Per-recipient invisible watermark for artwork")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a copy of an image that carries a recipient id
    Embed {
        input: PathBuf,
        /// Output path. With several --id, include "{id}" in the name, e.g. out/page1_{id}.png
        output: String,
        /// Recipient id (0..4294967295). Repeat to produce one copy per recipient.
        #[arg(long, required = true)]
        id: Vec<u32>,
        /// Secret key; detection needs the same one
        #[arg(long, default_value = "")]
        key: String,
        /// Brightness change in flat areas, in 8-bit levels
        #[arg(long, default_value_t = 1.0)]
        strength: f32,
    },
    /// Find out whose copy a suspect image is, given your original
    Detect {
        original: PathBuf,
        suspect: PathBuf,
        #[arg(long, default_value = "")]
        key: String,
        /// Ids you have issued (repeatable, or comma separated). Helps on heavily damaged copies.
        #[arg(long, value_delimiter = ',')]
        id: Vec<u32>,
        /// Print machine-readable JSON
        #[arg(long)]
        json: bool,
    },
}

fn load(path: &Path) -> Result<RgbaImage, String> {
    image::open(path).map(|i| i.to_rgba8()).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

/// RGB over a white backdrop, which is what a flattened re-upload looks like.
fn flatten(img: &RgbaImage) -> Vec<u8> {
    img.pixels()
        .flat_map(|p| {
            let a = p[3] as u32;
            [0, 1, 2].map(|c| ((p[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8)
        })
        .collect()
}

fn save(path: &Path, w: u32, h: u32, rgb: &[u8], alpha: &RgbaImage) -> Result<(), String> {
    let err = |e: image::ImageError| format!("cannot write {}: {e}", path.display());
    let has_alpha = alpha.pixels().any(|p| p[3] != 255);
    let format = ImageFormat::from_path(path).map_err(err)?;
    if format == ImageFormat::Jpeg {
        let file = std::fs::File::create(path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 95);
        return image::RgbImage::from_raw(w, h, rgb.to_vec()).unwrap().write_with_encoder(enc).map_err(err);
    }
    if has_alpha {
        let out: RgbaImage = ImageBuffer::from_fn(w, h, |x, y| {
            let i = ((y * w + x) * 3) as usize;
            Rgba([rgb[i], rgb[i + 1], rgb[i + 2], alpha.get_pixel(x, y)[3]])
        });
        out.save(path).map_err(err)
    } else {
        image::RgbImage::from_raw(w, h, rgb.to_vec()).unwrap().save(path).map_err(err)
    }
}

fn run() -> Result<ExitCode, String> {
    match Cli::parse().cmd {
        Cmd::Embed { input, output, id, key, strength } => {
            if id.len() > 1 && !output.contains("{id}") {
                return Err("with several --id, the output name must contain {id}".into());
            }
            let img = load(&input)?;
            let (w, h) = img.dimensions();
            // Straight (un-premultiplied) colour channels: transparency is kept as it is.
            let rgb: Vec<u8> = img.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
            let opt = EmbedOptions { key, strength, ..Default::default() };
            for i in id {
                let out = embed(Rgb::new(w as usize, h as usize, &rgb), i, &opt);
                let path = PathBuf::from(output.replace("{id}", &i.to_string()));
                save(&path, w, h, &out, &img)?;
                println!("{} -> id {}", path.display(), i);
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Detect { original, suspect, key, id, json } => {
            let (o, s) = (load(&original)?, load(&suspect)?);
            let (of, sf) = (flatten(&o), flatten(&s));
            let d = detect(
                Rgb::new(o.width() as usize, o.height() as usize, &of),
                Rgb::new(s.width() as usize, s.height() as usize, &sf),
                &key,
                &id,
            );
            let verdict = match d.verdict {
                Verdict::Match => "match",
                Verdict::Weak => "weak",
                Verdict::None => "none",
            };
            let named = d.id.or(d.best_candidate.filter(|_| d.verdict != Verdict::None).map(|c| c.0));
            if json {
                let opt = |v: Option<u32>| v.map_or("null".to_string(), |v| v.to_string());
                println!(
                    "{{\"verdict\":\"{verdict}\",\"id\":{},\"read_id\":{},\"repaired_bits\":{},\"strength\":{:.2},\"coverage\":{:.3},\"alignment\":{:.3},\"mirrored\":{}}}",
                    opt(named), opt(d.id), d.repaired_bits, d.strength, d.coverage, d.transform.score, d.transform.mirrored
                );
            } else {
                match (d.verdict, named) {
                    (Verdict::Match, Some(i)) => println!("Match: id {i}"),
                    (Verdict::Weak, Some(i)) => println!("Weak trace of id {i} - not enough to be sure"),
                    _ => println!("No watermark found"),
                }
                println!(
                    "  strength {:.1} sigma, {:.0}% of the original present, alignment {:.2}{}",
                    d.strength,
                    d.coverage * 100.0,
                    d.transform.score,
                    if d.transform.mirrored { ", mirrored" } else { "" }
                );
                if d.repaired_bits > 0 {
                    println!("  {} uncertain bit(s) corrected", d.repaired_bits);
                }
            }
            Ok(if d.verdict == Verdict::Match { ExitCode::SUCCESS } else { ExitCode::from(2) })
        }
    }
}

fn main() -> ExitCode {
    run().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        ExitCode::FAILURE
    })
}
