//! Decode-only raw LZMA2 support for o_voxel `.vxz` files.
//!
//! The official writer defaults to `lzma.FORMAT_RAW` with a single
//! `FILTER_LZMA2` (`o_voxel/io/vxz.py:72-74, 89-91, 229`). No already-locked
//! workspace crate implements raw LZMA2 (the lockfile has no `lzma-rs`,
//! `xz2`, `ruzstd` or `zstd`), so this module ships a minimal decode-only
//! implementation of the LZMA2 framing and the LZMA state machine. It follows
//! the public xz format specification; the liblzma decoder
//! (`xz-5.2/src/liblzma/lzma/{lzma_decoder,lzma2_decoder}.c`) was used as the
//! behavioral reference.
//!
//! Supported: all LZMA2 chunk kinds (`0x00` end, `0x01`/`0x02` uncompressed,
//! `0x80..=0xFF` LZMA), dictionary/state/properties resets, and persistence of
//! the LZMA state and dictionary position across LZMA chunks. Encoding is not
//! provided: `.vxz` writes use `none` or raw `deflate` only.

use crate::{error::Lux3dError, Result};

const NUM_STATES: usize = 12;
const NUM_POS_BITS_MAX: usize = 4;
const NUM_LEN_TO_POS_STATES: usize = 4;
const MATCH_LEN_MIN: usize = 2;
const PROB_INIT: u16 = 1 << 10;

const LITERAL_NEXT_STATES: [usize; NUM_STATES] = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 4, 5];

fn lzma_error(message: impl Into<String>) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: format!("o_voxel raw LZMA2 decode failed: {}", message.into()),
    }
}

#[derive(Debug, Clone)]
struct LenDecoder {
    choice: u16,
    choice2: u16,
    low: [[u16; 8]; 1 << NUM_POS_BITS_MAX],
    mid: [[u16; 8]; 1 << NUM_POS_BITS_MAX],
    high: [u16; 256],
}

impl LenDecoder {
    fn new() -> Self {
        Self {
            choice: PROB_INIT,
            choice2: PROB_INIT,
            low: [[PROB_INIT; 8]; 1 << NUM_POS_BITS_MAX],
            mid: [[PROB_INIT; 8]; 1 << NUM_POS_BITS_MAX],
            high: [PROB_INIT; 256],
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn decode(&mut self, range: &mut RangeDecoder<'_>, pos_state: usize) -> Result<usize> {
        if range.decode_bit(&mut self.choice)? == 0 {
            decode_bit_tree(range, &mut self.low[pos_state], 3).map(|value| value as usize)
        } else if range.decode_bit(&mut self.choice2)? == 0 {
            decode_bit_tree(range, &mut self.mid[pos_state], 3).map(|value| value as usize + 8)
        } else {
            decode_bit_tree(range, &mut self.high, 8).map(|value| value as usize + 16)
        }
    }
}

struct LzmaState {
    lc: u32,
    lp: u32,
    pb: u32,
    literal_probs: Vec<u16>,
    pos_slot: [[u16; 64]; NUM_LEN_TO_POS_STATES],
    pos_decoders: [u16; 115],
    pos_align: [u16; 16],
    is_match: [u16; NUM_STATES * (1 << NUM_POS_BITS_MAX)],
    is_rep: [u16; NUM_STATES],
    is_rep_g0: [u16; NUM_STATES],
    is_rep_g1: [u16; NUM_STATES],
    is_rep_g2: [u16; NUM_STATES],
    is_rep_0long: [u16; NUM_STATES * (1 << NUM_POS_BITS_MAX)],
    state: usize,
    /// Match distances minus one, matching the xz reference layout.
    rep: [usize; 4],
    match_len: LenDecoder,
    rep_len: LenDecoder,
}

impl LzmaState {
    fn new() -> Self {
        Self {
            lc: 0,
            lp: 0,
            pb: 0,
            literal_probs: Vec::new(),
            pos_slot: [[PROB_INIT; 64]; NUM_LEN_TO_POS_STATES],
            pos_decoders: [PROB_INIT; 115],
            pos_align: [PROB_INIT; 16],
            is_match: [PROB_INIT; NUM_STATES * (1 << NUM_POS_BITS_MAX)],
            is_rep: [PROB_INIT; NUM_STATES],
            is_rep_g0: [PROB_INIT; NUM_STATES],
            is_rep_g1: [PROB_INIT; NUM_STATES],
            is_rep_g2: [PROB_INIT; NUM_STATES],
            is_rep_0long: [PROB_INIT; NUM_STATES * (1 << NUM_POS_BITS_MAX)],
            state: 0,
            rep: [0; 4],
            match_len: LenDecoder::new(),
            rep_len: LenDecoder::new(),
        }
    }

    fn reset(&mut self, properties: Option<(u32, u32, u32)>) -> Result<()> {
        if let Some((lc, lp, pb)) = properties {
            if lc > 8 || lp > 4 || pb > 4 || lc + lp > 4 {
                return Err(lzma_error(format!(
                    "invalid LZMA properties lc={lc} lp={lp} pb={pb}"
                )));
            }
            self.lc = lc;
            self.lp = lp;
            self.pb = pb;
            self.literal_probs = vec![PROB_INIT; 0x300 << (lc + lp)];
        } else if self.literal_probs.is_empty() {
            return Err(lzma_error(
                "LZMA chunk requires properties before the first state reset",
            ));
        }
        self.literal_probs.fill(PROB_INIT);
        for tree in &mut self.pos_slot {
            tree.fill(PROB_INIT);
        }
        self.pos_decoders.fill(PROB_INIT);
        self.pos_align.fill(PROB_INIT);
        self.is_match.fill(PROB_INIT);
        self.is_rep.fill(PROB_INIT);
        self.is_rep_g0.fill(PROB_INIT);
        self.is_rep_g1.fill(PROB_INIT);
        self.is_rep_g2.fill(PROB_INIT);
        self.is_rep_0long.fill(PROB_INIT);
        self.state = 0;
        self.rep = [0; 4];
        self.match_len.reset();
        self.rep_len.reset();
        Ok(())
    }

    fn decode_literal(
        &mut self,
        range: &mut RangeDecoder<'_>,
        output: &[u8],
        dict_position: usize,
    ) -> Result<u8> {
        let prev_byte = if dict_position == 0 {
            0u32
        } else {
            u32::from(output[output.len() - 1])
        };
        let literal_pos_mask = (1usize << self.lp) - 1;
        let lit_state =
            ((dict_position & literal_pos_mask) << self.lc) + (prev_byte >> (8 - self.lc)) as usize;
        let base = lit_state * 0x300;
        let mut result = 1usize;

        if self.state >= 7 {
            let distance = self.rep[0] + 1;
            if distance > dict_position {
                return Err(lzma_error("literal match distance exceeds output"));
            }
            let mut match_byte = u32::from(output[output.len() - distance]);
            while result < 0x100 {
                let match_bit = (match_byte >> 7) & 1;
                match_byte <<= 1;
                let index = base + ((1 + match_bit as usize) << 8) + result;
                let bit = range.decode_bit(&mut self.literal_probs[index])? as usize;
                result = (result << 1) | bit;
                if match_bit as usize != bit {
                    break;
                }
            }
        }

        while result < 0x100 {
            let bit = range.decode_bit(&mut self.literal_probs[base + result])? as usize;
            result = (result << 1) | bit;
        }

        self.state = LITERAL_NEXT_STATES[self.state];
        Ok((result - 0x100) as u8)
    }

    fn decode_distance(
        &mut self,
        range: &mut RangeDecoder<'_>,
        length_symbol: usize,
    ) -> Result<usize> {
        let len_state = length_symbol.min(NUM_LEN_TO_POS_STATES - 1);
        let pos_slot = decode_bit_tree(range, &mut self.pos_slot[len_state], 6)? as usize;
        if pos_slot < 4 {
            return Ok(pos_slot);
        }

        let num_direct_bits = ((pos_slot >> 1) - 1) as u32;
        let base = (2 | (pos_slot & 1)) << num_direct_bits;
        if pos_slot < 14 {
            // The xz reference decodes at `pos_decoders[base - pos_slot - 1 + m]`
            // with `m` starting at one, i.e. the first probability is at
            // `base - pos_slot`.
            let first_index = base - pos_slot;
            let extra = decode_reverse_bit_tree(
                range,
                &mut self.pos_decoders,
                first_index as isize - 1,
                num_direct_bits,
            )? as usize;
            Ok(base + extra)
        } else {
            let high = range.decode_direct_bits(num_direct_bits - 4)? as usize;
            let low = decode_reverse_bit_tree(range, &mut self.pos_align, 0, 4)? as usize;
            Ok(base + (high << 4) + low)
        }
    }
}

fn decode_bit_tree(range: &mut RangeDecoder<'_>, probs: &mut [u16], bits: u32) -> Result<u32> {
    let mut symbol = 1usize;
    for _ in 0..bits {
        let bit = range.decode_bit(&mut probs[symbol])? as usize;
        symbol = (symbol << 1) | bit;
    }
    Ok((symbol - (1 << bits)) as u32)
}

/// Reverse bit tree where the first probability is `probs[offset + 1]`.
///
/// `offset` is the (possibly `-1`) xz `dist - pos_slot - 1` bias.
fn decode_reverse_bit_tree(
    range: &mut RangeDecoder<'_>,
    probs: &mut [u16],
    offset: isize,
    bits: u32,
) -> Result<u32> {
    let mut symbol = 1usize;
    let mut result = 0u32;
    for bit_index in 0..bits {
        let index = offset + symbol as isize;
        let index = usize::try_from(index)
            .map_err(|_| lzma_error("negative reverse bit-tree probability index"))?;
        let bit = range.decode_bit(&mut probs[index])?;
        symbol = (symbol << 1) | bit as usize;
        result |= bit << bit_index;
    }
    Ok(result)
}

struct RangeDecoder<'a> {
    data: &'a [u8],
    position: usize,
    range: u32,
    code: u32,
}

impl<'a> RangeDecoder<'a> {
    fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 5 {
            return Err(lzma_error("compressed LZMA chunk is shorter than 5 bytes"));
        }
        // The first byte is always zero; the following four are the initial code.
        let code = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
        Ok(Self {
            data,
            position: 5,
            range: 0xFFFF_FFFF,
            code,
        })
    }

    fn next_byte(&mut self) -> Result<u8> {
        let byte = *self
            .data
            .get(self.position)
            .ok_or_else(|| lzma_error("range decoder ran past the compressed chunk"))?;
        self.position += 1;
        Ok(byte)
    }

    #[inline]
    fn normalize(&mut self) -> Result<()> {
        if self.range < 0x0100_0000 {
            self.range <<= 8;
            self.code = (self.code << 8) | u32::from(self.next_byte()?);
        }
        Ok(())
    }

    #[inline]
    fn decode_bit(&mut self, probability: &mut u16) -> Result<u32> {
        let bound = (self.range >> 11) * u32::from(*probability);
        let bit;
        if self.code < bound {
            self.range = bound;
            *probability += (2048 - *probability) >> 5;
            bit = 0;
        } else {
            self.range -= bound;
            self.code -= bound;
            *probability -= *probability >> 5;
            bit = 1;
        }
        self.normalize()?;
        Ok(bit)
    }

    fn decode_direct_bits(&mut self, bits: u32) -> Result<u32> {
        let mut result = 0u32;
        for _ in 0..bits {
            self.range >>= 1;
            self.code = self.code.wrapping_sub(self.range);
            let mask = 0u32.wrapping_sub(self.code >> 31);
            self.code = self.code.wrapping_add(self.range & mask);
            self.normalize()?;
            result = (result << 1) + mask.wrapping_add(1);
        }
        Ok(result)
    }
}

fn copy_match(output: &mut Vec<u8>, distance: usize, length: usize) -> Result<()> {
    let start = output
        .len()
        .checked_sub(distance)
        .ok_or_else(|| lzma_error("match distance exceeds decoded history"))?;
    for offset in 0..length {
        let byte = output[start + offset];
        output.push(byte);
    }
    Ok(())
}

fn decode_lzma_chunk(
    state: &mut LzmaState,
    block: &[u8],
    unpacked_size: usize,
    output: &mut Vec<u8>,
    dict_start: usize,
) -> Result<()> {
    let target = output.len() + unpacked_size;
    let mut range = RangeDecoder::new(block)?;

    while output.len() < target {
        let remaining = target - output.len();
        let dict_position = output.len() - dict_start;
        let pos_state = dict_position & ((1usize << state.pb) - 1);
        let index = state.state * 16 + pos_state;

        if range.decode_bit(&mut state.is_match[index])? == 0 {
            let byte = state.decode_literal(&mut range, output, dict_position)?;
            output.push(byte);
            continue;
        }

        let length;
        if range.decode_bit(&mut state.is_rep[state.state])? != 0 {
            let rep_index;
            if range.decode_bit(&mut state.is_rep_g0[state.state])? == 0 {
                if range.decode_bit(&mut state.is_rep_0long[index])? == 0 {
                    // Short rep: one byte at distance rep0 + 1.
                    state.state = if state.state < 7 { 9 } else { 11 };
                    let distance = state.rep[0] + 1;
                    if distance > dict_position {
                        return Err(lzma_error("short rep distance exceeds decoded history"));
                    }
                    let byte = output[output.len() - distance];
                    output.push(byte);
                    continue;
                }
                rep_index = 0;
            } else if range.decode_bit(&mut state.is_rep_g1[state.state])? == 0 {
                rep_index = 1;
            } else if range.decode_bit(&mut state.is_rep_g2[state.state])? == 0 {
                rep_index = 2;
            } else {
                rep_index = 3;
            }

            if rep_index != 0 {
                let distance = state.rep[rep_index];
                for slot in (0..rep_index).rev() {
                    state.rep[slot + 1] = state.rep[slot];
                }
                state.rep[0] = distance;
            }
            let length_symbol = state.rep_len.decode(&mut range, pos_state)?;
            state.state = if state.state < 7 { 8 } else { 11 };
            length = length_symbol + MATCH_LEN_MIN;
        } else {
            state.rep[3] = state.rep[2];
            state.rep[2] = state.rep[1];
            state.rep[1] = state.rep[0];
            let length_symbol = state.match_len.decode(&mut range, pos_state)?;
            state.state = if state.state < 7 { 7 } else { 10 };
            length = length_symbol + MATCH_LEN_MIN;
            if length > remaining {
                return Err(lzma_error(
                    "match length exceeds the declared uncompressed chunk size",
                ));
            }
            state.rep[0] = state.decode_distance(&mut range, length_symbol)?;
        }

        if length > remaining {
            return Err(lzma_error(
                "match length exceeds the declared uncompressed chunk size",
            ));
        }
        let distance = state.rep[0] + 1;
        if distance > dict_position {
            return Err(lzma_error("match distance exceeds decoded history"));
        }
        copy_match(output, distance, length)?;
    }

    Ok(())
}

/// Decode one standalone raw LZMA2 stream (as stored in one `.vxz` chunk
/// section) into bytes.
pub(crate) fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut dict_start = 0usize;
    let mut state = LzmaState::new();
    let mut have_state = false;
    let mut need_properties = true;
    let mut position = 0usize;

    let take = |position: &mut usize, count: usize| -> Result<&[u8]> {
        let end = position
            .checked_add(count)
            .ok_or_else(|| lzma_error("LZMA2 chunk size overflow"))?;
        let slice = data.get(*position..end).ok_or_else(|| {
            lzma_error(format!(
                "LZMA2 stream truncated: need {count} bytes at offset {}",
                *position
            ))
        })?;
        *position = end;
        Ok(slice)
    };

    loop {
        let control = *data
            .get(position)
            .ok_or_else(|| lzma_error("LZMA2 stream is missing the end marker"))?;
        position += 1;

        match control {
            0x00 => break,
            0x01 | 0x02 => {
                let size_bytes = take(&mut position, 2)?;
                let unpacked_size =
                    ((usize::from(size_bytes[0]) << 8) | usize::from(size_bytes[1])) + 1;
                if control == 0x01 {
                    dict_start = output.len();
                    need_properties = true;
                }
                let bytes = take(&mut position, unpacked_size)?;
                output.extend_from_slice(bytes);
            }
            control if control >= 0x80 => {
                let reset_kind = (control >> 5) & 0x3;
                let reset_dictionary = reset_kind == 3;
                let reset_state = reset_kind != 0;
                let reset_properties = reset_kind >= 2;

                let size_bytes = take(&mut position, 2)?;
                let unpacked_size = ((usize::from(control & 0x1F) << 16)
                    | (usize::from(size_bytes[0]) << 8)
                    | usize::from(size_bytes[1]))
                    + 1;
                let packed_bytes = take(&mut position, 2)?;
                let packed_size =
                    ((usize::from(packed_bytes[0]) << 8) | usize::from(packed_bytes[1])) + 1;

                if reset_dictionary {
                    dict_start = output.len();
                    need_properties = true;
                }

                let properties = if reset_properties {
                    let byte = take(&mut position, 1)?[0];
                    if byte >= 225 {
                        return Err(lzma_error(format!("invalid LZMA2 properties byte {byte}")));
                    }
                    let lc = u32::from(byte % 9);
                    let remainder = byte / 9;
                    let lp = u32::from(remainder % 5);
                    let pb = u32::from(remainder / 5);
                    need_properties = false;
                    Some((lc, lp, pb))
                } else {
                    if need_properties {
                        return Err(lzma_error(
                            "LZMA2 chunk must set properties after a dictionary reset",
                        ));
                    }
                    None
                };

                if reset_state || !have_state {
                    state.reset(properties)?;
                    have_state = true;
                }

                let block = take(&mut position, packed_size)?;
                decode_lzma_chunk(&mut state, block, unpacked_size, &mut output, dict_start)?;
            }
            other => {
                return Err(lzma_error(format!(
                    "invalid LZMA2 control byte 0x{other:02x}"
                )));
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_and_invalid_streams() {
        assert!(decompress(&[]).is_err());
        assert!(decompress(&[0x00]).unwrap().is_empty());
        assert!(decompress(&[0x01, 0x00]).is_err());
        assert!(decompress(&[0x03]).is_err());
        assert!(decompress(&[0x80, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn decodes_uncompressed_chunks() {
        // control 0x02, size-1 = 3, "test", end marker
        let stream = [0x02, 0x00, 0x03, b't', b'e', b's', b't', 0x00];
        assert_eq!(decompress(&stream).unwrap(), b"test");
    }
}
