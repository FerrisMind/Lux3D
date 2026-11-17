//! 30-bit voxel sequence codes used by the o_voxel `.vxz` container.
//!
//! Port of the official TRELLIS.2 `o-voxel` serialization layer:
//!
//! - `o_voxel/serialize.py:7-68` — `encode_seq` / `decode_seq` with the
//!   `permute` argument and the `z_order` / `hilbert` modes.
//! - `src/serialize/z_order.cu:10-62` — the bit-interleave z-order codec
//!   (10 bits per axis, 30-bit code).
//! - `src/serialize/hilbert.cu:33-129` — the Hilbert codec (Skilling's
//!   excess-work + Gray-code transform, 10 bits per axis).
//! - `src/io/svo.cpp:17-138` — the sparse voxel octree (SVO) byte stream that
//!   packs a sorted list of chunk-local z-order codes into one byte per node.
//!
//! Coordinates are cast through `int` in the official Python layer; values
//! outside the 10-bit range are silently truncated by the bit operations, so
//! this module reproduces that behavior instead of rejecting them.

use crate::{error::Lux3dError, Result};

/// Sequence-code mode accepted by [`encode_seq`] / [`decode_seq`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqMode {
    /// Interleaved Morton code (official `mode='z_order'`, the default).
    ZOrder,
    /// Hilbert curve code (official `mode='hilbert'`).
    Hilbert,
}

/// The identity permutation used by the official default `permute=[0, 1, 2]`.
pub const DEFAULT_PERMUTE: [usize; 3] = [0, 1, 2];

fn invalid_permute(permute: [usize; 3]) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: format!(
            "o_voxel coordinate permutation {permute:?} must be a permutation of [0, 1, 2]"
        ),
    }
}

fn inverse_permute(permute: [usize; 3]) -> Result<[usize; 3]> {
    let mut inverse = [usize::MAX; 3];
    for (axis, &source) in permute.iter().enumerate() {
        if source >= 3 || inverse[source] != usize::MAX {
            return Err(invalid_permute(permute));
        }
        inverse[source] = axis;
    }
    Ok(inverse)
}

/// Expands a 10-bit integer into 30 bits by inserting two zeros after each bit.
///
/// Verbatim port of `expandBits` (`src/serialize/z_order.cu:11-18`).
#[inline]
pub fn expand_bits(mut value: u32) -> u32 {
    value = value.wrapping_mul(0x0001_0001) & 0xFF00_00FF;
    value = value.wrapping_mul(0x0000_0101) & 0x0F00_F00F;
    value = value.wrapping_mul(0x0000_0011) & 0xC30C_30C3;
    value = value.wrapping_mul(0x0000_0005) & 0x4924_9249;
    value
}

/// Removes the two zeros after each bit of a 30-bit integer.
///
/// Verbatim port of `extractBits` (`src/serialize/z_order.cu:22-30`).
#[inline]
pub fn extract_bits(mut value: u32) -> u32 {
    value &= 0x4924_9249;
    value = (value ^ (value >> 2)) & 0x030C_30C3;
    value = (value ^ (value >> 4)) & 0x0300_F00F;
    value = (value ^ (value >> 8)) & 0x0300_00FF;
    value = (value ^ (value >> 16)) & 0x0000_03FF;
    value
}

/// Z-order encode one coordinate triple (`CPU::z_order_encode`, lines 33-47).
#[inline]
pub fn z_order_encode(x: u32, y: u32, z: u32) -> u32 {
    expand_bits(x) * 4 + expand_bits(y) * 2 + expand_bits(z)
}

/// Z-order decode one 30-bit code (`CPU::z_order_decode`, lines 50-62).
#[inline]
pub fn z_order_decode(code: u32) -> [u32; 3] {
    [
        extract_bits(code >> 2),
        extract_bits(code >> 1),
        extract_bits(code),
    ]
}

/// Hilbert encode one coordinate triple (`CPU::hilbert_encode`, lines 33-84).
#[inline]
pub fn hilbert_encode(x: u32, y: u32, z: u32) -> u32 {
    let mut point = [x, y, z];
    let m: u32 = 1 << 9;

    // Inverse undo excess work.
    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..3 {
            if point[i] & q != 0 {
                point[0] ^= p; // invert
            } else {
                let t = (point[0] ^ point[i]) & p;
                point[0] ^= t;
                point[i] ^= t;
            }
        }
        q >>= 1;
    }

    // Gray encode.
    for i in 1..3 {
        point[i] ^= point[i - 1];
    }
    let mut t = 0;
    let mut q = m;
    while q > 1 {
        if point[2] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for value in &mut point {
        *value ^= t;
    }

    z_order_encode(point[0], point[1], point[2])
}

/// Hilbert decode one 30-bit code (`CPU::hilbert_decode`, lines 87-129).
#[inline]
pub fn hilbert_decode(code: u32) -> [u32; 3] {
    let mut point = [
        extract_bits(code >> 2),
        extract_bits(code >> 1),
        extract_bits(code),
    ];
    let m: u32 = 2 << 9;

    // Gray decode by H ^ (H/2).
    let mut t = point[2] >> 1;
    for i in (1..3).rev() {
        point[i] ^= point[i - 1];
    }
    point[0] ^= t;

    // Undo excess work.
    let mut q = 2;
    while q != m {
        let p = q - 1;
        for i in (0..3).rev() {
            if point[i] & q != 0 {
                point[0] ^= p;
            } else {
                t = (point[0] ^ point[i]) & p;
                point[0] ^= t;
                point[i] ^= t;
            }
        }
        q <<= 1;
    }

    point
}

/// Encode `[N, 3]` integer coordinates with the official `encode_seq`.
///
/// `permute` selects which input axis feeds the code's x/y/z components; the
/// identity permutation is [`DEFAULT_PERMUTE`]. Coordinates are truncated to
/// 10 bits per axis exactly like the official C++ kernels.
pub fn encode_seq(coords: &[[i32; 3]], permute: [usize; 3], mode: SeqMode) -> Result<Vec<u32>> {
    inverse_permute(permute)?;
    let mut codes = Vec::with_capacity(coords.len());
    for coord in coords {
        let x = coord[permute[0]] as u32;
        let y = coord[permute[1]] as u32;
        let z = coord[permute[2]] as u32;
        codes.push(match mode {
            SeqMode::ZOrder => z_order_encode(x, y, z),
            SeqMode::Hilbert => hilbert_encode(x, y, z),
        });
    }
    Ok(codes)
}

/// Decode `[N]` 30-bit codes with the official `decode_seq`.
///
/// The inverse of `permute` is applied so that
/// `decode_seq(encode_seq(coords, p), p)` restores the input axis order
/// (`serialize.py:65-67`).
pub fn decode_seq(codes: &[u32], permute: [usize; 3], mode: SeqMode) -> Result<Vec<[i32; 3]>> {
    let inverse = inverse_permute(permute)?;
    let mut coords = Vec::with_capacity(codes.len());
    for &code in codes {
        let raw = match mode {
            SeqMode::ZOrder => z_order_decode(code),
            SeqMode::Hilbert => hilbert_decode(code),
        };
        coords.push([
            raw[inverse[0]] as i32,
            raw[inverse[1]] as i32,
            raw[inverse[2]] as i32,
        ]);
    }
    Ok(coords)
}

fn svo_error(message: impl Into<String>) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: message.into(),
    }
}

/// Pack sorted chunk-local z-order codes into the official SVO byte stream.
///
/// Port of `encode_sparse_voxel_octree_cpu` (`src/io/svo.cpp:17-76`). The
/// caller must pass ascending codes; each node is one byte whose bits mark the
/// occupied children. `depth` is `log2(chunk_size)`.
pub fn encode_sparse_voxel_octree(codes: &[u32], depth: u32) -> Result<Vec<u8>> {
    if depth == 0 {
        return Err(svo_error(
            "o_voxel SVO depth must be at least 1 (chunk_size >= 2)",
        ));
    }
    let depth = depth as usize;
    let mut svo = vec![0u8]; // root node
    let mut stack = vec![0u8; depth - 1];
    let mut insert_stack = vec![0u32; depth];
    let mut stack_ptr = vec![0usize; depth];
    stack_ptr[0] = 0;

    for (index, &code) in codes.iter().enumerate() {
        for (level, slot) in insert_stack.iter_mut().enumerate() {
            *slot = (code >> (3 * (depth - 1 - level))) & 0x7;
        }

        let insert_from = if index == 0 {
            0
        } else {
            let mut found = depth - 1;
            for level in 0..depth - 1 {
                if insert_stack[level] != u32::from(stack[level]) {
                    found = level;
                    break;
                }
            }
            found
        };

        for level in insert_from..depth {
            if level > insert_from {
                svo.push(0u8);
                stack_ptr[level] = svo.len() - 1;
            }
            svo[stack_ptr[level]] |= 1u8 << insert_stack[level];
            if level < depth - 1 {
                stack[level] = insert_stack[level] as u8;
            }
        }
    }

    Ok(svo)
}

fn decode_svo_node(
    svo: &[u8],
    depth: usize,
    ptr: &mut usize,
    stack: &mut Vec<u8>,
    codes: &mut Vec<u32>,
) -> Result<()> {
    let node = *svo
        .get(*ptr)
        .ok_or_else(|| svo_error("o_voxel SVO stream is truncated"))?;
    if stack.len() == depth - 1 {
        let mut base = 0u32;
        for (level, &octant) in stack.iter().enumerate() {
            base |= u32::from(octant) << (3 * (depth - 1 - level));
        }
        for octant in 0..8u32 {
            if node & (1 << octant) != 0 {
                codes.push((base & !0x7) | octant);
            }
        }
        *ptr += 1;
    } else {
        *ptr += 1;
        for octant in 0..8u8 {
            if node & (1 << octant) != 0 {
                stack.push(octant);
                decode_svo_node(svo, depth, ptr, stack, codes)?;
                stack.pop();
            }
        }
    }
    Ok(())
}

/// Unpack an official SVO byte stream into ascending chunk-local z-order codes.
///
/// Port of `decode_sparse_voxel_octree_cpu` (`src/io/svo.cpp:79-138`).
pub fn decode_sparse_voxel_octree(svo: &[u8], depth: u32) -> Result<Vec<u32>> {
    if depth == 0 {
        return Err(svo_error(
            "o_voxel SVO depth must be at least 1 (chunk_size >= 2)",
        ));
    }
    if svo.is_empty() {
        return Err(svo_error("o_voxel SVO stream is empty (missing root node)"));
    }
    let mut codes = Vec::new();
    let mut stack = Vec::new();
    let mut ptr = 0;
    decode_svo_node(svo, depth as usize, &mut ptr, &mut stack, &mut codes)?;
    if ptr != svo.len() {
        return Err(svo_error(format!(
            "o_voxel SVO stream has {} trailing bytes",
            svo.len() - ptr
        )));
    }
    Ok(codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_order_matches_the_official_bit_layout() {
        // expandBits(1) = 1, expandBits(2) = 8, expandBits(3) = 9.
        assert_eq!(z_order_encode(1, 2, 3), 4 + 16 + 9);
        assert_eq!(z_order_decode(29), [1, 2, 3]);
        for value in 0..1024u32 {
            assert_eq!(extract_bits(expand_bits(value)), value);
        }
        assert_eq!(z_order_decode(z_order_encode(1023, 0, 512)), [1023, 0, 512]);
    }

    #[test]
    fn z_order_round_trips_and_permute_is_symmetric() {
        let coords = [[0, 1, 2], [5, 7, 11], [1023, 512, 3]];
        let codes = encode_seq(&coords, DEFAULT_PERMUTE, SeqMode::ZOrder).unwrap();
        let decoded = decode_seq(&codes, DEFAULT_PERMUTE, SeqMode::ZOrder).unwrap();
        assert_eq!(decoded, coords);

        let permute = [2, 0, 1];
        let codes = encode_seq(&coords, permute, SeqMode::ZOrder).unwrap();
        let raw_x = coords[0][2] as u32;
        let raw_y = coords[0][0] as u32;
        let raw_z = coords[0][1] as u32;
        assert_eq!(codes[0], z_order_encode(raw_x, raw_y, raw_z));
        let decoded = decode_seq(&codes, permute, SeqMode::ZOrder).unwrap();
        assert_eq!(decoded, coords);
    }

    #[test]
    fn hilbert_round_trips_and_matches_official_golden_codes() {
        // Golden codes produced by a line-by-line Python transcription of
        // `CPU::hilbert_encode` from `src/serialize/hilbert.cu:33-84`.
        let cases = [
            ([0u32, 0, 0], 0u32),
            ([1, 0, 0], 7),
            ([0, 1, 0], 3),
            ([1, 1, 0], 4),
            ([0, 0, 1], 1),
            ([1, 0, 1], 6),
            ([0, 1, 1], 2),
            ([1, 1, 1], 5),
            ([2, 3, 4], 467),
            ([7, 5, 3], 165),
            ([1023, 1023, 1023], 766_958_445),
            ([512, 256, 128], 1_030_825_106),
        ];
        for (point, code) in cases {
            assert_eq!(hilbert_encode(point[0], point[1], point[2]), code);
            assert_eq!(hilbert_decode(code), point);
        }

        let coords = [[0, 0, 0], [1, 2, 3], [1023, 0, 1], [17, 999, 4]];
        let codes = encode_seq(&coords, DEFAULT_PERMUTE, SeqMode::Hilbert).unwrap();
        assert_eq!(
            decode_seq(&codes, DEFAULT_PERMUTE, SeqMode::Hilbert).unwrap(),
            coords
        );
        let permute = [1, 2, 0];
        let codes = encode_seq(&coords, permute, SeqMode::Hilbert).unwrap();
        assert_eq!(
            decode_seq(&codes, permute, SeqMode::Hilbert).unwrap(),
            coords
        );
    }

    #[test]
    fn rejects_bad_permutations() {
        let coords = [[1, 2, 3]];
        assert!(encode_seq(&coords, [0, 0, 2], SeqMode::ZOrder).is_err());
        assert!(encode_seq(&coords, [0, 1, 3], SeqMode::ZOrder).is_err());
        assert!(decode_seq(&[0], [2, 2, 0], SeqMode::Hilbert).is_err());
    }

    #[test]
    fn svo_round_trips_sorted_codes() {
        for depth in 1..=4u32 {
            let limit = 1u32 << (3 * depth);
            let mut codes: Vec<u32> = (0..limit)
                .filter(|code| code % 7 == 0 || *code == 0 || *code + 1 == limit)
                .collect();
            codes.sort_unstable();
            let svo = encode_sparse_voxel_octree(&codes, depth).unwrap();
            let decoded = decode_sparse_voxel_octree(&svo, depth).unwrap();
            assert_eq!(decoded, codes, "depth {depth}");
        }
    }

    #[test]
    fn svo_rejects_depth_zero_and_truncated_streams() {
        assert!(encode_sparse_voxel_octree(&[0], 0).is_err());
        assert!(decode_sparse_voxel_octree(&[0], 0).is_err());
        assert!(decode_sparse_voxel_octree(&[], 2).is_err());
        // One partial branch: root marks child 0, but that child node is absent.
        assert!(decode_sparse_voxel_octree(&[0b0000_0001], 2).is_err());
    }
}
