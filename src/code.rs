//! 32-bit id + CRC-16, so a misread is detected instead of naming the wrong person.

pub const CODE_BITS: usize = 48;

fn crc16(bytes: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in bytes {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x1021 } else { crc << 1 };
        }
    }
    crc
}

pub fn encode_id(id: u32) -> [bool; CODE_BITS] {
    let word = ((id as u64) << 16) | crc16(&id.to_be_bytes()) as u64;
    let mut bits = [false; CODE_BITS];
    for (i, b) in bits.iter_mut().enumerate() {
        *b = (word >> (CODE_BITS - 1 - i)) & 1 == 1;
    }
    bits
}

fn check(bits: &[bool; CODE_BITS]) -> Option<u32> {
    let word = bits.iter().fold(0u64, |w, &b| (w << 1) | b as u64);
    let id = (word >> 16) as u32;
    (crc16(&id.to_be_bytes()) == (word & 0xFFFF) as u16).then_some(id)
}

/// Recover an id from per-bit soft scores (positive = 1).
/// If the straight reading fails its checksum, retry with up to two of the
/// eight least certain bits flipped. Returns the id and how many bits were flipped.
pub fn decode_soft(soft: &[f32; CODE_BITS]) -> Option<(u32, usize)> {
    let mut bits = [false; CODE_BITS];
    for i in 0..CODE_BITS {
        bits[i] = soft[i] > 0.0;
    }
    if let Some(id) = check(&bits) {
        return Some((id, 0));
    }
    let mut order: Vec<usize> = (0..CODE_BITS).collect();
    order.sort_by(|&a, &b| soft[a].abs().total_cmp(&soft[b].abs()));
    let weak = &order[..8];
    for &a in weak {
        let mut t = bits;
        t[a] = !t[a];
        if let Some(id) = check(&t) {
            return Some((id, 1));
        }
    }
    for (i, &a) in weak.iter().enumerate() {
        for &b in &weak[i + 1..] {
            let mut t = bits;
            t[a] = !t[a];
            t[b] = !t[b];
            if let Some(id) = check(&t) {
                return Some((id, 2));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_repair() {
        for id in [0u32, 1, 42, 0xDEAD_BEEF, u32::MAX] {
            let bits = encode_id(id);
            let mut soft = [0f32; CODE_BITS];
            for i in 0..CODE_BITS {
                soft[i] = if bits[i] { 5.0 } else { -5.0 };
            }
            assert_eq!(decode_soft(&soft), Some((id, 0)));
            soft[3] = -soft[3] * 0.1;
            soft[40] = -soft[40] * 0.1;
            assert_eq!(decode_soft(&soft), Some((id, 2)));
        }
    }
}
