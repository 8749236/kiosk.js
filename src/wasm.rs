//! Plain C-style exports for the browser build. No bindgen: JS passes pointers into
//! the module's memory. Images are RGBA, as handed out by a canvas.

use crate::{detect, embed, EmbedOptions, Rgb, Verdict};

#[no_mangle]
pub extern "C" fn tm_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` must come from `tm_alloc(len)`.
#[no_mangle]
pub unsafe extern "C" fn tm_free(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len.max(1)));
}

unsafe fn text<'a>(ptr: *const u8, len: usize) -> &'a str {
    std::str::from_utf8(std::slice::from_raw_parts(ptr, len)).unwrap_or("")
}

/// Watermark an RGBA buffer in place. Transparency is left alone.
///
/// # Safety
/// `rgba` must hold `w * h * 4` bytes and `key` must hold `key_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn tm_embed(rgba: *mut u8, w: usize, h: usize, id: u32, key: *const u8, key_len: usize, strength: f32) {
    let px = std::slice::from_raw_parts_mut(rgba, w * h * 4);
    let rgb: Vec<u8> = px.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let opt = EmbedOptions { key: text(key, key_len).to_string(), strength, ..Default::default() };
    let out = embed(Rgb::new(w, h, &rgb), id, &opt);
    for (p, o) in px.chunks_exact_mut(4).zip(out.chunks_exact(3)) {
        p[..3].copy_from_slice(o);
    }
}

/// RGB over white, which is what a flattened re-upload looks like.
unsafe fn flatten(rgba: *const u8, w: usize, h: usize) -> Vec<u8> {
    std::slice::from_raw_parts(rgba, w * h * 4)
        .chunks_exact(4)
        .flat_map(|p| {
            let a = p[3] as u32;
            [0, 1, 2].map(|c| ((p[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8)
        })
        .collect()
}

/// Writes 8 little-endian f64 to `out`:
/// verdict (0 none, 1 weak, 2 match), id (-1 if nobody can be named), repaired bits,
/// strength, coverage, alignment score, mirrored (0/1), read-directly (0/1).
///
/// # Safety
/// Buffers must have the stated sizes; `ids` holds `ids_len` u32; `out` holds 8 f64.
#[no_mangle]
pub unsafe extern "C" fn tm_detect(
    orig: *const u8, ow: usize, oh: usize,
    sus: *const u8, sw: usize, sh: usize,
    key: *const u8, key_len: usize,
    ids: *const u32, ids_len: usize,
    out: *mut f64,
) {
    let (o, s) = (flatten(orig, ow, oh), flatten(sus, sw, sh));
    let ids: Vec<u32> = (0..ids_len).map(|i| ids.add(i).read_unaligned()).collect();
    let d = detect(Rgb::new(ow, oh, &o), Rgb::new(sw, sh, &s), text(key, key_len), &ids);
    let named = d.id.or(d.best_candidate.filter(|_| d.verdict != Verdict::None).map(|c| c.0));
    let res = [
        match d.verdict { Verdict::None => 0.0, Verdict::Weak => 1.0, Verdict::Match => 2.0 },
        named.map_or(-1.0, |v| v as f64),
        d.repaired_bits as f64,
        d.strength as f64,
        d.coverage as f64,
        d.transform.score as f64,
        d.transform.mirrored as u8 as f64,
        d.id.is_some() as u8 as f64,
    ];
    for (i, v) in res.iter().enumerate() {
        out.add(i).write_unaligned(*v);
    }
}
