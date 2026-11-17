//! Official TRELLIS.2 o_voxel `.vxz` voxel container.
//!
//! Layout ported from `o_voxel/io/vxz.py`:
//!
//! ```text
//! 'VXZ' | version u8 (0) | big-endian u32 binary start | JSON structure | binary
//! ```
//!
//! The JSON structure holds the voxel count, chunk size, filter, compression,
//! attribute schema and per-chunk offsets (`vxz.py:22-54, 257-269, 344-365`).
//! Every chunk stores an SVO byte stream followed by independently compressed
//! attribute streams; section offsets are relative to the chunk while `ptr` is
//! relative to the start of the binary section (`vxz.py:327-347`).
//!
//! Compression support matches the locked TR8a scope:
//!
//! - write: `none`, `deflate` (raw DEFLATE, `wbits=-15`); default `deflate`
//!   (documented deviation: the official writer defaults to `lzma`,
//!   `vxz.py:229`).
//! - read: `none`, `deflate`, `lzma` (raw LZMA2, mandatory for official-written
//!   files). `zstd` is rejected with a clear error because no zstd decoder
//!   crate is present in the workspace lockfile.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::serialize::{self, SeqMode, DEFAULT_PERMUTE};
use super::{VoxelAttr, VoxelGrid};
use crate::{error::Lux3dError, Result};

/// Default chunk size (`write_vxz(chunk_size=256)`, `vxz.py:225`).
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// Compression algorithm names understood by the `.vxz` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxzCompression {
    /// Uncompressed sections.
    None,
    /// Raw DEFLATE (`zlib.compressobj(level, wbits=-15)`).
    Deflate,
    /// Raw LZMA2 (`lzma.FORMAT_RAW` + `FILTER_LZMA2`).
    Lzma,
    /// Zstandard frame.
    Zstd,
}

impl VxzCompression {
    /// Header token (`vxz.py:56-78`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Deflate => "deflate",
            Self::Lzma => "lzma",
            Self::Zstd => "zstd",
        }
    }

    /// Parse a header token.
    pub fn from_str(value: &str) -> Result<Self> {
        match value {
            "none" => Ok(Self::None),
            "deflate" => Ok(Self::Deflate),
            "lzma" => Ok(Self::Lzma),
            "zstd" => Ok(Self::Zstd),
            other => Err(vxz_error(format!(
                "unsupported .vxz compression `{other}`; expected none, deflate, lzma or zstd"
            ))),
        }
    }

    fn default_level(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Deflate => 9,
            Self::Lzma => 9,
            Self::Zstd => 22,
        }
    }
}

/// Attribute stream interleaving (`write_vxz(attr_interleave=...)`, `vxz.py:231`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxzInterleave {
    /// One stream per attribute channel, named `{attr}_{channel}`.
    None,
    /// One stream per attribute, row-major channels (`as_is`).
    AsIs,
    /// One stream with every attribute channel concatenated per row.
    All,
}

impl VxzInterleave {
    /// Header token.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::AsIs => "as_is",
            Self::All => "all",
        }
    }

    /// Parse a header token.
    pub fn from_str(value: &str) -> Result<Self> {
        match value {
            "none" => Ok(Self::None),
            "as_is" => Ok(Self::AsIs),
            "all" => Ok(Self::All),
            other => Err(vxz_error(format!(
                "unsupported .vxz attr_interleave `{other}`; expected none, as_is or all"
            ))),
        }
    }
}

/// Options accepted by [`write_vxz`].
#[derive(Debug, Clone)]
pub struct VxzWriteOptions {
    /// Voxels per chunk edge; must be a power of two (`vxz.py:273-275`).
    pub chunk_size: usize,
    /// Compression algorithm. Only [`VxzCompression::None`] and
    /// [`VxzCompression::Deflate`] can be written by this build.
    pub compression: VxzCompression,
    /// Compression level; `None` uses the official defaults (deflate 9,
    /// `vxz.py:56-61`).
    pub compression_level: Option<u32>,
    /// Attribute stream layout.
    pub attr_interleave: VxzInterleave,
}

impl Default for VxzWriteOptions {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            compression: VxzCompression::Deflate,
            compression_level: None,
            attr_interleave: VxzInterleave::AsIs,
        }
    }
}

/// Parsed `.vxz` header, mirroring the official JSON structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VxzInfo {
    /// Total number of voxels.
    pub num_voxel: usize,
    /// Chunk edge length.
    pub chunk_size: usize,
    /// Attribute filter (`none`, `parent`, `neighbor`).
    pub filter: String,
    /// Compression token.
    pub compression: String,
    /// Compression level recorded by the writer.
    pub compression_level: i64,
    /// Uncompressed payload size in bytes.
    pub raw_size: u64,
    /// Compressed binary section size in bytes.
    pub compressed_size: u64,
    /// `raw_size / compressed_size`.
    pub compress_ratio: f64,
    /// Attribute interleave token.
    pub attr_interleave: String,
    /// Attribute schema as `(name, channels)` pairs.
    pub attr: Vec<(String, usize)>,
    /// Per-chunk offsets and identifiers.
    pub chunks: Vec<VxzChunkInfo>,
}

/// One chunk entry in the `.vxz` header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VxzChunkInfo {
    /// Chunk coordinate (multiply by `chunk_size` to get global coordinates).
    pub idx: [i32; 3],
    /// SVO section offset and length relative to this chunk.
    pub svo: [u64; 2],
    /// Chunk offset and length relative to the binary section.
    pub ptr: [u64; 2],
    /// Attribute stream offsets and lengths relative to this chunk.
    #[serde(flatten)]
    pub sections: BTreeMap<String, [u64; 2]>,
}

fn vxz_error(message: impl Into<String>) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: format!("o_voxel VXZ: {}", message.into()),
    }
}

fn io_error(path: &Path, source: std::io::Error) -> Lux3dError {
    Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    }
}

fn parse_header(bytes: &[u8]) -> Result<(VxzInfo, usize)> {
    if bytes.len() < 8 {
        return Err(vxz_error("file is shorter than the 8-byte header"));
    }
    if &bytes[0..3] != b"VXZ" {
        return Err(vxz_error("invalid file magic (expected `VXZ`)"));
    }
    if bytes[3] != 0 {
        return Err(vxz_error(format!(
            "unsupported version {} (expected 0)",
            bytes[3]
        )));
    }
    let binary_start = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if binary_start < 8 || binary_start > bytes.len() {
        return Err(vxz_error(format!(
            "binary start offset {binary_start} is outside the file ({} bytes)",
            bytes.len()
        )));
    }
    let info: VxzInfo = serde_json::from_slice(&bytes[8..binary_start]).map_err(|error| {
        vxz_error(format!(
            "failed to parse the JSON structure at bytes 8..{binary_start}: {error}"
        ))
    })?;
    Ok((info, binary_start))
}

/// Read only the `.vxz` header (`read_vxz_info`, `vxz.py:98-120`).
pub fn read_vxz_info(path: &Path) -> Result<VxzInfo> {
    let bytes = std::fs::read(path).map_err(|source| io_error(path, source))?;
    let (info, _) = parse_header(&bytes)?;
    Ok(info)
}

fn compress_section(
    data: &[u8],
    compression: VxzCompression,
    level: Option<u32>,
) -> Result<Vec<u8>> {
    match compression {
        VxzCompression::None => Ok(data.to_vec()),
        VxzCompression::Deflate => {
            let level = level.unwrap_or(compression.default_level());
            if level > 9 {
                return Err(vxz_error(format!(
                    "deflate compression level {level} is outside 0..=9"
                )));
            }
            let mut encoder =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
            encoder
                .write_all(data)
                .map_err(|error| vxz_error(format!("raw deflate compression failed: {error}")))?;
            encoder
                .finish()
                .map_err(|error| vxz_error(format!("raw deflate compression failed: {error}")))
        }
        VxzCompression::Lzma => Err(vxz_error(
            "writing lzma-compressed .vxz files is not supported by this build; \
             use `none` or `deflate`",
        )),
        VxzCompression::Zstd => Err(vxz_error(
            "writing zstd-compressed .vxz files is not supported by this build; \
             use `none` or `deflate`",
        )),
    }
}

fn decompress_section(data: &[u8], compression: &str) -> Result<Vec<u8>> {
    match compression {
        "none" => Ok(data.to_vec()),
        "deflate" => {
            let mut decoder = flate2::read::DeflateDecoder::new(data);
            let mut output = Vec::new();
            decoder
                .read_to_end(&mut output)
                .map_err(|error| vxz_error(format!("raw deflate decompression failed: {error}")))?;
            Ok(output)
        }
        "lzma" => super::lzma2::decompress(data),
        "zstd" => Err(vxz_error(
            "zstd-compressed sections require a zstd decoder; this build only reads \
             none, deflate and lzma",
        )),
        other => Err(vxz_error(format!(
            "unsupported .vxz compression `{other}`; expected none, deflate, lzma or zstd"
        ))),
    }
}

fn chunk_depth(chunk_size: usize) -> Result<u32> {
    if chunk_size < 2 || !chunk_size.is_power_of_two() {
        return Err(vxz_error(format!(
            "chunk_size must be a power of two >= 2, got {chunk_size}"
        )));
    }
    // Chunk-local coordinates feed the official 30-bit sequence codec (10 bits
    // per axis); larger chunks would silently alias voxels.
    if chunk_size > 1024 {
        return Err(vxz_error(format!(
            "chunk_size must not exceed 1024 (30-bit sequence code), got {chunk_size}"
        )));
    }
    Ok(chunk_size.trailing_zeros())
}

fn validate_attr_schema(attrs: &BTreeMap<String, VoxelAttr>) -> Result<()> {
    for (name, attr) in attrs {
        if name.is_empty() {
            return Err(vxz_error("attribute names must not be empty"));
        }
        if attr.channels == 0 {
            return Err(vxz_error(format!("attribute `{name}` has zero channels")));
        }
        if matches!(name.as_str(), "svo" | "ptr" | "idx") {
            return Err(vxz_error(format!(
                "attribute name `{name}` collides with a .vxz chunk header key"
            )));
        }
    }
    Ok(())
}

fn sorted_chunk_order(codes: &[u32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..codes.len()).collect();
    order.sort_by_key(|&index| codes[index]);
    order
}

/// Encode a voxel grid into `.vxz` bytes.
pub fn encode_vxz(grid: &VoxelGrid, options: &VxzWriteOptions) -> Result<Vec<u8>> {
    grid.validate()?;
    validate_attr_schema(&grid.attrs)?;
    match options.compression {
        VxzCompression::None => {}
        VxzCompression::Deflate => {
            let level = options
                .compression_level
                .unwrap_or(options.compression.default_level());
            if level > 9 {
                return Err(vxz_error(format!(
                    "deflate compression level {level} is outside 0..=9"
                )));
            }
        }
        VxzCompression::Lzma => {
            return Err(vxz_error(
                "writing lzma-compressed .vxz files is not supported by this build; \
                 use `none` or `deflate`",
            ));
        }
        VxzCompression::Zstd => {
            return Err(vxz_error(
                "writing zstd-compressed .vxz files is not supported by this build; \
                 use `none` or `deflate`",
            ));
        }
    }
    let depth = chunk_depth(options.chunk_size)?;
    let chunk_size = options.chunk_size;

    let mut chunks: BTreeMap<[i32; 3], Vec<usize>> = BTreeMap::new();
    for (index, coord) in grid.coords.iter().enumerate() {
        let chunk = [
            coord[0].div_euclid(chunk_size as i32),
            coord[1].div_euclid(chunk_size as i32),
            coord[2].div_euclid(chunk_size as i32),
        ];
        chunks.entry(chunk).or_default().push(index);
    }

    let attr_schema: Vec<(String, usize)> = grid
        .attrs
        .iter()
        .map(|(name, attr)| (name.clone(), attr.channels))
        .collect();
    let total_attr_channels: usize = attr_schema.iter().map(|(_, channels)| channels).sum();

    let mut binary = Vec::new();
    let mut chunk_infos = Vec::with_capacity(chunks.len());

    for (idx, rows) in &chunks {
        let local: Vec<[i32; 3]> = rows
            .iter()
            .map(|&row| {
                let coord = grid.coords[row];
                [
                    coord[0].rem_euclid(chunk_size as i32),
                    coord[1].rem_euclid(chunk_size as i32),
                    coord[2].rem_euclid(chunk_size as i32),
                ]
            })
            .collect();
        let codes = serialize::encode_seq(&local, DEFAULT_PERMUTE, SeqMode::ZOrder)?;
        let order = sorted_chunk_order(&codes);
        let sorted_rows: Vec<usize> = order.iter().map(|&local_index| rows[local_index]).collect();
        let sorted_codes: Vec<u32> = order
            .iter()
            .map(|&local_index| codes[local_index])
            .collect();
        let svo = serialize::encode_sparse_voxel_octree(&sorted_codes, depth)?;
        let svo_bytes = compress_section(&svo, options.compression, options.compression_level)?;

        let mut section_bytes: Vec<(String, Vec<u8>)> = Vec::new();
        match options.attr_interleave {
            VxzInterleave::AsIs => {
                for (name, attr) in &grid.attrs {
                    let mut bytes = Vec::with_capacity(sorted_rows.len() * attr.channels);
                    for &row in &sorted_rows {
                        bytes.extend_from_slice(
                            &attr.data[row * attr.channels..(row + 1) * attr.channels],
                        );
                    }
                    section_bytes.push((
                        name.clone(),
                        compress_section(&bytes, options.compression, options.compression_level)?,
                    ));
                }
            }
            VxzInterleave::None => {
                for (name, attr) in &grid.attrs {
                    for channel in 0..attr.channels {
                        let bytes: Vec<u8> = sorted_rows
                            .iter()
                            .map(|&row| attr.data[row * attr.channels + channel])
                            .collect();
                        section_bytes.push((
                            format!("{name}_{channel}"),
                            compress_section(
                                &bytes,
                                options.compression,
                                options.compression_level,
                            )?,
                        ));
                    }
                }
            }
            VxzInterleave::All => {
                let mut bytes = Vec::with_capacity(sorted_rows.len() * total_attr_channels);
                for &row in &sorted_rows {
                    for attr in grid.attrs.values() {
                        bytes.extend_from_slice(
                            &attr.data[row * attr.channels..(row + 1) * attr.channels],
                        );
                    }
                }
                section_bytes.push((
                    "attr".to_string(),
                    compress_section(&bytes, options.compression, options.compression_level)?,
                ));
            }
        }

        let mut chunk_data = Vec::new();
        let mut sections: BTreeMap<String, [u64; 2]> = BTreeMap::new();
        let svo_offset = chunk_data.len() as u64;
        chunk_data.extend_from_slice(&svo_bytes);
        let svo_section = [svo_offset, svo_bytes.len() as u64];

        for (name, bytes) in section_bytes {
            let offset = chunk_data.len() as u64;
            chunk_data.extend_from_slice(&bytes);
            if sections
                .insert(name.clone(), [offset, bytes.len() as u64])
                .is_some()
            {
                return Err(vxz_error(format!(
                    "duplicate attribute section `{name}` in one chunk"
                )));
            }
        }

        let ptr = [binary.len() as u64, chunk_data.len() as u64];
        binary.extend_from_slice(&chunk_data);
        chunk_infos.push(VxzChunkInfo {
            idx: *idx,
            svo: svo_section,
            ptr,
            sections,
        });
    }

    let raw_size = (grid.coords.len() * 4) as u64
        + grid
            .attrs
            .values()
            .map(|attr| attr.data.len() as u64)
            .sum::<u64>();
    let compressed_size = binary.len() as u64;
    let compress_ratio = if compressed_size == 0 {
        0.0
    } else {
        raw_size as f64 / compressed_size as f64
    };
    let info = VxzInfo {
        num_voxel: grid.coords.len(),
        chunk_size,
        filter: "none".to_string(),
        compression: options.compression.as_str().to_string(),
        compression_level: options
            .compression_level
            .unwrap_or_else(|| options.compression.default_level())
            as i64,
        raw_size,
        compressed_size,
        compress_ratio,
        attr_interleave: options.attr_interleave.as_str().to_string(),
        attr: attr_schema,
        chunks: chunk_infos,
    };
    let structure = serde_json::to_vec(&info)
        .map_err(|error| vxz_error(format!("failed to serialize the JSON structure: {error}")))?;
    let binary_start = structure
        .len()
        .checked_add(8)
        .ok_or_else(|| vxz_error("JSON structure length overflow"))?;
    let binary_start = u32::try_from(binary_start)
        .map_err(|_| vxz_error("JSON structure is too large for a VXZ header"))?;

    let mut file = Vec::with_capacity(8 + structure.len() + binary.len());
    file.extend_from_slice(b"VXZ\x00");
    file.extend_from_slice(&binary_start.to_be_bytes());
    file.extend_from_slice(&structure);
    file.extend_from_slice(&binary);
    Ok(file)
}

/// Write a voxel grid to a `.vxz` file.
pub fn write_vxz(path: &Path, grid: &VoxelGrid, options: &VxzWriteOptions) -> Result<()> {
    let bytes = encode_vxz(grid, options)?;
    std::fs::write(path, bytes).map_err(|source| io_error(path, source))
}

fn section<'a>(data: &'a [u8], offset: [u64; 2], what: &str) -> Result<&'a [u8]> {
    let start = usize::try_from(offset[0])
        .map_err(|_| vxz_error(format!("{what} offset does not fit in usize")))?;
    let length = usize::try_from(offset[1])
        .map_err(|_| vxz_error(format!("{what} length does not fit in usize")))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| vxz_error(format!("{what} range overflows")))?;
    data.get(start..end)
        .ok_or_else(|| vxz_error(format!("{what} range {start}..{end} exceeds the chunk")))
}

fn decompress_stream(
    chunk: &[u8],
    sections: &BTreeMap<String, [u64; 2]>,
    key: &str,
    compression: &str,
) -> Result<Vec<u8>> {
    let range = sections
        .get(key)
        .copied()
        .ok_or_else(|| vxz_error(format!("chunk is missing the `{key}` section")))?;
    decompress_section(section(chunk, range, key)?, compression)
}

fn decode_chunk_attrs(
    schema: &[(String, usize)],
    sections: &BTreeMap<String, [u64; 2]>,
    chunk: &[u8],
    num_voxel: usize,
    interleave: VxzInterleave,
    compression: &str,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut attrs = BTreeMap::new();
    match interleave {
        VxzInterleave::AsIs => {
            for (name, channels) in schema {
                let bytes = decompress_stream(chunk, sections, name, compression)?;
                let expected = num_voxel
                    .checked_mul(*channels)
                    .ok_or_else(|| vxz_error("attribute stream size overflow"))?;
                if bytes.len() != expected {
                    return Err(vxz_error(format!(
                        "attribute `{name}` has {} bytes; expected {expected} for {num_voxel} \
                         voxels x {channels} channels",
                        bytes.len()
                    )));
                }
                attrs.insert(name.clone(), bytes);
            }
        }
        VxzInterleave::None => {
            for (name, channels) in schema {
                let mut interleaved = vec![0u8; num_voxel * *channels];
                for channel in 0..*channels {
                    let key = format!("{name}_{channel}");
                    let bytes = decompress_stream(chunk, sections, &key, compression)?;
                    if bytes.len() != num_voxel {
                        return Err(vxz_error(format!(
                            "attribute stream `{key}` has {} bytes; expected {num_voxel}",
                            bytes.len()
                        )));
                    }
                    for row in 0..num_voxel {
                        interleaved[row * *channels + channel] = bytes[row];
                    }
                }
                attrs.insert(name.clone(), interleaved);
            }
        }
        VxzInterleave::All => {
            let bytes = decompress_stream(chunk, sections, "attr", compression)?;
            let total_channels: usize = schema.iter().map(|(_, channels)| channels).sum();
            let expected = num_voxel * total_channels;
            if bytes.len() != expected {
                return Err(vxz_error(format!(
                    "attribute stream `attr` has {} bytes; expected {expected}",
                    bytes.len()
                )));
            }
            let mut per_attr: Vec<Vec<u8>> = schema
                .iter()
                .map(|(_, channels)| Vec::with_capacity(num_voxel * channels))
                .collect();
            for row in 0..num_voxel {
                let mut offset = row * total_channels;
                for (index, (_, channels)) in schema.iter().enumerate() {
                    per_attr[index].extend_from_slice(&bytes[offset..offset + channels]);
                    offset += channels;
                }
            }
            for ((name, _), values) in schema.iter().zip(per_attr) {
                attrs.insert(name.clone(), values);
            }
        }
    }
    Ok(attrs)
}

/// Decode `.vxz` bytes into a voxel grid.
pub fn decode_vxz(bytes: &[u8]) -> Result<VoxelGrid> {
    let (info, binary_start) = parse_header(bytes)?;
    let depth = chunk_depth(info.chunk_size)?;
    let chunk_size = info.chunk_size as i32;
    let interleave = VxzInterleave::from_str(&info.attr_interleave)?;
    if info.filter != "none" {
        return Err(vxz_error(format!(
            "filter `{}` is not supported; only `none` is implemented (parent/neighbor need the \
             official SVO filter kernels)",
            info.filter
        )));
    }
    if !matches!(
        info.compression.as_str(),
        "none" | "deflate" | "lzma" | "zstd"
    ) {
        return Err(vxz_error(format!(
            "unsupported .vxz compression `{}`",
            info.compression
        )));
    }
    let binary = &bytes[binary_start..];

    let mut schema = Vec::with_capacity(info.attr.len());
    for (name, channels) in &info.attr {
        if *channels == 0 {
            return Err(vxz_error(format!(
                "attribute `{name}` declares zero channels"
            )));
        }
        schema.push((name.clone(), *channels));
    }

    let mut coords: Vec<[i32; 3]> = Vec::with_capacity(info.num_voxel);
    let mut attrs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for (name, _) in &schema {
        attrs.insert(name.clone(), Vec::new());
    }

    for chunk_info in &info.chunks {
        let chunk = section(binary, chunk_info.ptr, "chunk")?;
        let svo = decompress_section(section(chunk, chunk_info.svo, "svo")?, &info.compression)?;
        let codes = serialize::decode_sparse_voxel_octree(&svo, depth)?;
        let local = serialize::decode_seq(&codes, DEFAULT_PERMUTE, SeqMode::ZOrder)?;
        let num_voxel = local.len();
        for coord in local {
            let global = [
                chunk_info.idx[0]
                    .checked_mul(chunk_size)
                    .and_then(|value| value.checked_add(coord[0]))
                    .ok_or_else(|| vxz_error("voxel x coordinate overflows i32"))?,
                chunk_info.idx[1]
                    .checked_mul(chunk_size)
                    .and_then(|value| value.checked_add(coord[1]))
                    .ok_or_else(|| vxz_error("voxel y coordinate overflows i32"))?,
                chunk_info.idx[2]
                    .checked_mul(chunk_size)
                    .and_then(|value| value.checked_add(coord[2]))
                    .ok_or_else(|| vxz_error("voxel z coordinate overflows i32"))?,
            ];
            coords.push(global);
        }

        let chunk_attrs = decode_chunk_attrs(
            &schema,
            &chunk_info.sections,
            chunk,
            num_voxel,
            interleave,
            &info.compression,
        )?;
        for (name, values) in chunk_attrs {
            attrs.entry(name).or_default().extend(values);
        }
    }

    let mut grid = VoxelGrid {
        coords,
        attrs: BTreeMap::new(),
    };
    for (name, channels) in schema {
        let data = attrs.remove(&name).unwrap_or_default();
        grid.attrs.insert(name, VoxelAttr::new(channels, data)?);
    }
    grid.validate()?;
    Ok(grid)
}

/// Read a `.vxz` file into a voxel grid.
pub fn read_vxz(path: &Path) -> Result<VoxelGrid> {
    let bytes = std::fs::read(path).map_err(|source| io_error(path, source))?;
    decode_vxz(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_grid() -> VoxelGrid {
        let mut grid = VoxelGrid::new();
        grid.coords = vec![
            [0, 0, 0],
            [1, 2, 3],
            [-1, 0, 0],
            [-5, -6, -7],
            [4, 0, 1],
            [255, 3, 2],
            [256, 1, 1],
            [2, 2, 2],
        ];
        let base_color: Vec<u8> = (0..grid.coords.len() * 3)
            .map(|index| (index as u8).wrapping_mul(7))
            .collect();
        let metallic: Vec<u8> = (0..grid.coords.len())
            .map(|index| index as u8 * 11)
            .collect();
        let alpha: Vec<u8> = (0..grid.coords.len())
            .map(|index| 200 - index as u8)
            .collect();
        grid.insert_attr("base_color", VoxelAttr::new(3, base_color).unwrap());
        grid.insert_attr("metallic", VoxelAttr::new(1, metallic).unwrap());
        grid.insert_attr("alpha", VoxelAttr::new(1, alpha).unwrap());
        grid
    }

    /// Canonical row representation used to compare grids independent of the
    /// per-chunk z-order sorting performed by the writer.
    fn normalized(grid: &VoxelGrid) -> Vec<([i32; 3], Vec<(String, Vec<u8>)>)> {
        let mut rows: Vec<_> = grid
            .coords
            .iter()
            .enumerate()
            .map(|(row, coord)| {
                let attrs = grid
                    .attrs
                    .iter()
                    .map(|(name, attr)| (name.clone(), attr.row(row).unwrap().to_vec()))
                    .collect();
                (*coord, attrs)
            })
            .collect();
        rows.sort_by_key(|(coord, _)| *coord);
        rows
    }

    /// Locate a generated o_voxel fixture. `test-assets/` is gitignored in this
    /// repository (see `.gitignore`), so fixture-backed tests skip cleanly when
    /// the fixtures have not been generated locally.
    fn fixture(name: &str) -> Option<std::path::PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("test-assets")
            .join("ovox")
            .join(name);
        path.exists().then_some(path)
    }

    fn patch_header(bytes: &[u8], patch: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let binary_start = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let mut value: serde_json::Value =
            serde_json::from_slice(&bytes[8..binary_start]).expect("header json");
        patch(&mut value);
        let structure = serde_json::to_vec(&value).expect("serialize header");
        let mut patched = Vec::new();
        patched.extend_from_slice(b"VXZ\x00");
        patched.extend_from_slice(&((structure.len() + 8) as u32).to_be_bytes());
        patched.extend_from_slice(&structure);
        patched.extend_from_slice(&bytes[binary_start..]);
        patched
    }

    #[test]
    fn vxz_round_trips_supported_compressions_and_interleaves() {
        let grid = sample_grid();
        for compression in [VxzCompression::None, VxzCompression::Deflate] {
            for interleave in [VxzInterleave::AsIs, VxzInterleave::None, VxzInterleave::All] {
                let options = VxzWriteOptions {
                    chunk_size: 4,
                    compression,
                    compression_level: None,
                    attr_interleave: interleave,
                };
                let bytes = encode_vxz(&grid, &options).expect("encode vxz");
                let (info, binary_start) = parse_header(&bytes).expect("header");
                assert_eq!(info.num_voxel, grid.len());
                assert_eq!(info.chunk_size, 4);
                assert_eq!(info.filter, "none");
                assert_eq!(info.compression, compression.as_str());
                assert_eq!(info.attr_interleave, interleave.as_str());
                assert_eq!(
                    info.attr,
                    vec![
                        ("alpha".to_string(), 1),
                        ("base_color".to_string(), 3),
                        ("metallic".to_string(), 1)
                    ]
                );
                assert_eq!(
                    info.raw_size,
                    (grid.coords.len() * 4) as u64
                        + grid
                            .attrs
                            .values()
                            .map(|attr| attr.data.len() as u64)
                            .sum::<u64>()
                );
                assert_eq!(info.compressed_size, (bytes.len() - binary_start) as u64);
                assert!(info.compress_ratio > 0.0);

                // Chunks are emitted in lexicographic chunk-coordinate order and
                // their `ptr` ranges are contiguous inside the binary section.
                let mut previous_end = 0u64;
                let mut previous_idx = None;
                for chunk in &info.chunks {
                    assert_eq!(chunk.ptr[0], previous_end);
                    previous_end += chunk.ptr[1];
                    if let Some(idx) = previous_idx {
                        assert!(idx < chunk.idx, "chunk order must be lexicographic");
                    }
                    previous_idx = Some(chunk.idx);
                    assert_eq!(chunk.svo[0], 0);
                    for range in chunk.sections.values() {
                        assert!(range[0] + range[1] <= chunk.ptr[1]);
                    }
                }
                assert_eq!(previous_end, info.compressed_size);

                let decoded = decode_vxz(&bytes).expect("decode vxz");
                assert_eq!(normalized(&decoded), normalized(&grid));
            }
        }
    }

    #[test]
    fn vxz_header_matches_the_official_json_shape() {
        let grid = sample_grid();
        let options = VxzWriteOptions {
            chunk_size: 4,
            compression: VxzCompression::Deflate,
            compression_level: Some(9),
            attr_interleave: VxzInterleave::AsIs,
        };
        let bytes = encode_vxz(&grid, &options).expect("encode");
        let binary_start = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        assert_eq!(&bytes[0..4], b"VXZ\x00");
        let value: serde_json::Value =
            serde_json::from_slice(&bytes[8..binary_start]).expect("json");
        for key in [
            "num_voxel",
            "chunk_size",
            "filter",
            "compression",
            "compression_level",
            "raw_size",
            "compressed_size",
            "compress_ratio",
            "attr_interleave",
            "attr",
            "chunks",
        ] {
            assert!(value.get(key).is_some(), "missing header key {key}");
        }
        assert_eq!(value["attr"][0], serde_json::json!(["alpha", 1]));
        let chunk = &value["chunks"][0];
        assert!(chunk.get("idx").is_some());
        assert!(chunk.get("svo").is_some());
        assert!(chunk.get("ptr").is_some());
        assert!(chunk.get("base_color").is_some());
        assert!(chunk.get("metallic").is_some());
        assert!(chunk.get("alpha").is_some());
    }

    #[test]
    fn vxz_empty_grid_round_trips() {
        let options = VxzWriteOptions {
            chunk_size: 8,
            compression: VxzCompression::None,
            compression_level: None,
            attr_interleave: VxzInterleave::AsIs,
        };
        let mut grid = VoxelGrid::new();
        grid.insert_attr("alpha", VoxelAttr::new(1, Vec::new()).unwrap());
        let bytes = encode_vxz(&grid, &options).expect("encode empty");
        let (info, _) = parse_header(&bytes).expect("header");
        assert_eq!(info.num_voxel, 0);
        assert!(info.chunks.is_empty());
        assert_eq!(info.compressed_size, 0);
        assert_eq!(info.compress_ratio, 0.0);
        let decoded = decode_vxz(&bytes).expect("decode empty");
        assert!(decoded.coords.is_empty());
        assert_eq!(decoded.attr("alpha").unwrap().data, Vec::<u8>::new());
    }

    #[test]
    fn vxz_rejects_unsupported_write_compression_and_bad_chunk_sizes() {
        let grid = sample_grid();
        for compression in [VxzCompression::Lzma, VxzCompression::Zstd] {
            let options = VxzWriteOptions {
                compression,
                ..VxzWriteOptions::default()
            };
            assert!(encode_vxz(&grid, &options).is_err());
        }
        let options = VxzWriteOptions {
            chunk_size: 3,
            ..VxzWriteOptions::default()
        };
        assert!(encode_vxz(&grid, &options).is_err());
        let options = VxzWriteOptions {
            chunk_size: 1,
            ..VxzWriteOptions::default()
        };
        assert!(encode_vxz(&grid, &options).is_err());
    }

    #[test]
    fn vxz_rejects_bad_magic_and_unsupported_headers() {
        assert!(decode_vxz(b"not a vxz file").is_err());
        let grid = sample_grid();
        let options = VxzWriteOptions {
            chunk_size: 4,
            compression: VxzCompression::None,
            compression_level: None,
            attr_interleave: VxzInterleave::AsIs,
        };
        let bytes = encode_vxz(&grid, &options).expect("encode");

        let zstd = patch_header(&bytes, |value| {
            value["compression"] = serde_json::json!("zstd");
        });
        let error = decode_vxz(&zstd).expect_err("zstd read must fail");
        assert!(error.to_string().contains("zstd"));

        let filtered = patch_header(&bytes, |value| {
            value["filter"] = serde_json::json!("parent");
        });
        let error = decode_vxz(&filtered).expect_err("parent filter must fail");
        assert!(error.to_string().contains("filter"));

        let interleave = patch_header(&bytes, |value| {
            value["attr_interleave"] = serde_json::json!("all");
        });
        let error = decode_vxz(&interleave).expect_err("missing attr section must fail");
        assert!(error.to_string().contains("attr"));
    }

    fn assert_fixture_grid(grid: &VoxelGrid) {
        let coord = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [2, 1, 0],
            [3, 3, 3],
            [4, 4, 4],
            [5, 0, 0],
            [0, 5, 0],
        ];
        assert_eq!(grid.len(), coord.len());
        for (index, expected_coord) in coord.iter().enumerate() {
            let row = grid
                .coords
                .iter()
                .position(|candidate| candidate == expected_coord)
                .unwrap_or_else(|| panic!("missing coordinate {expected_coord:?}"));
            let base = [
                (index * 3) as u8,
                (index * 3 + 1) as u8,
                (index * 3 + 2) as u8,
            ];
            assert_eq!(
                grid.attr("base_color").unwrap().row(row),
                Some(&base[..]),
                "base_color at {expected_coord:?}"
            );
            assert_eq!(
                grid.attr("metallic").unwrap().row(row),
                Some(&[(index as u8).wrapping_mul(7).wrapping_add(3)][..]),
                "metallic at {expected_coord:?}"
            );
            assert_eq!(
                grid.attr("alpha").unwrap().row(row),
                Some(&[255 - index as u8][..]),
                "alpha at {expected_coord:?}"
            );
        }
    }

    #[test]
    fn vxz_reads_official_format_lzma_fixtures() {
        for name in [
            "official_lzma_as_is.vxz",
            "official_lzma_none_interleave.vxz",
            "official_lzma_all_interleave.vxz",
        ] {
            let Some(path) = fixture(name) else {
                eprintln!(
                    "skipping {name}: run test-assets/ovox/generate_fixtures.py to create the \
                     o_voxel fixture"
                );
                continue;
            };
            let info = read_vxz_info(&path).expect("read header");
            assert_eq!(info.num_voxel, 10, "{name}");
            assert_eq!(info.chunk_size, 4, "{name}");
            assert_eq!(info.compression, "lzma", "{name}");
            assert_eq!(info.chunks.len(), 4, "{name}");
            let grid = read_vxz(&path).expect("read vxz fixture");
            assert_fixture_grid(&grid);
        }
    }

    #[test]
    fn vxz_reads_official_format_deflate_and_none_fixtures() {
        let (Some(deflate), Some(none)) = (
            fixture("official_deflate_as_is.vxz"),
            fixture("official_none_as_is.vxz"),
        ) else {
            eprintln!(
                "skipping official deflate/none fixtures: run \
                 test-assets/ovox/generate_fixtures.py to create them"
            );
            return;
        };
        let grid = read_vxz(&deflate).expect("deflate fixture");
        assert_fixture_grid(&grid);
        let grid = read_vxz(&none).expect("none fixture");
        assert_fixture_grid(&grid);
    }

    #[test]
    fn raw_lzma2_decodes_a_multi_chunk_official_stream() {
        let Some(path) = fixture("lzma2_multichunk.bin") else {
            eprintln!(
                "skipping lzma2_multichunk.bin: run test-assets/ovox/generate_fixtures.py to \
                 create it"
            );
            return;
        };
        let stream = std::fs::read(path).expect("lzma2 fixture");
        // The fixture is a 3 MiB pattern, which liblzma necessarily splits into
        // more than one LZMA2 chunk (2 MiB uncompressed maximum per chunk).
        let mut position = 0usize;
        let mut chunk_count = 0usize;
        while position < stream.len() {
            let control = stream[position];
            position += 1;
            if control == 0 {
                break;
            }
            chunk_count += 1;
            if control == 1 || control == 2 {
                let size = ((stream[position] as usize) << 8 | stream[position + 1] as usize) + 1;
                position += 2 + size;
            } else {
                let packed =
                    ((stream[position + 2] as usize) << 8 | stream[position + 3] as usize) + 1;
                position += 4;
                if control >= 0xC0 {
                    position += 1;
                }
                position += packed;
            }
        }
        assert_eq!(position, stream.len());
        assert!(chunk_count >= 2, "expected multiple LZMA2 chunks");
        let decoded = super::super::lzma2::decompress(&stream).expect("decode lzma2 fixture");
        let expected: Vec<u8> = (0..3 * 1024 * 1024)
            .map(|index| ((index * 37 + 11) & 0xFF) as u8)
            .collect();
        assert_eq!(decoded.len(), expected.len());
        assert_eq!(decoded, expected);
    }
}
