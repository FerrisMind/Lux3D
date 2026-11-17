//! PLY voxel interchange (`o_voxel/io/ply.py`).
//!
//! The official voxel PLY layout is a single `vertex` element whose
//! `x`/`y`/`z` properties are floats and whose remaining scalar properties are
//! named `{attribute}_{channel}`. `write_ply` emits binary little-endian
//! (`plyfile.PlyData.write` default) and `read_ply` rounds coordinates back to
//! integers (`ply.py:26-71`).
//!
//! This build reads ASCII and binary little-endian PLY and rejects big-endian
//! or list-valued properties. Attributes must be integer scalars in `0..=255`
//! because the o_voxel voxel model stores `uint8` rows (`vxz.py:249-252`).

use std::path::Path;

use super::{VoxelAttr, VoxelGrid};
use crate::{error::Lux3dError, Result};

fn ply_error(message: impl Into<String>) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: format!("o_voxel PLY: {}", message.into()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    F64,
}

impl PlyType {
    fn parse(token: &str) -> Option<Self> {
        match token {
            "char" | "int8" => Some(Self::I8),
            "uchar" | "uint8" => Some(Self::U8),
            "short" | "int16" => Some(Self::I16),
            "ushort" | "uint16" => Some(Self::U16),
            "int" | "int32" => Some(Self::I32),
            "uint" | "uint32" => Some(Self::U32),
            "float" | "float32" => Some(Self::F32),
            "double" | "float64" => Some(Self::F64),
            _ => None,
        }
    }

    const fn size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    const fn is_integer(self) -> bool {
        !matches!(self, Self::F32 | Self::F64)
    }

    fn read_binary(self, bytes: &[u8], offset: &mut usize) -> Result<f64> {
        let size = self.size();
        let slice = bytes
            .get(*offset..*offset + size)
            .ok_or_else(|| ply_error("binary body is shorter than its header declares"))?;
        *offset += size;
        Ok(match self {
            Self::U8 => f64::from(slice[0]),
            Self::I8 => f64::from(slice[0] as i8),
            Self::U16 => f64::from(u16::from_le_bytes([slice[0], slice[1]])),
            Self::I16 => f64::from(i16::from_le_bytes([slice[0], slice[1]])),
            Self::U32 => f64::from(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])),
            Self::I32 => f64::from(i32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])),
            Self::F32 => f64::from(f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])),
            Self::F64 => f64::from_le_bytes([
                slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            ]),
        })
    }

    fn parse_ascii(self, token: &str) -> Result<f64> {
        let value = token
            .parse::<f64>()
            .map_err(|_| ply_error(format!("invalid numeric token `{token}`")))?;
        if self.is_integer() && (value.fract() != 0.0 || !value.is_finite()) {
            return Err(ply_error(format!(
                "integer property value `{token}` is not an integer"
            )));
        }
        Ok(value)
    }
}

#[derive(Debug, Clone)]
struct PlyProperty {
    name: String,
    ty: PlyType,
}

#[derive(Debug)]
struct PlyHeader {
    ascii: bool,
    count: usize,
    properties: Vec<PlyProperty>,
    body_offset: usize,
}

fn parse_header(bytes: &[u8]) -> Result<PlyHeader> {
    if !bytes.starts_with(b"ply") {
        return Err(ply_error("not a PLY file (missing `ply` magic)"));
    }
    let mut offset = 0usize;
    let mut ascii = None;
    let mut element_seen = false;
    let mut count = None;
    let mut properties = Vec::new();

    loop {
        let end = bytes[offset..]
            .iter()
            .position(|&byte| byte == b'\n')
            .map(|position| offset + position)
            .ok_or_else(|| ply_error("header is missing an `end_header` line"))?;
        let mut line = &bytes[offset..end];
        offset = end + 1;
        if line.last() == Some(&b'\r') {
            line = &line[..line.len() - 1];
        }
        let line =
            std::str::from_utf8(line).map_err(|_| ply_error("header line is not valid UTF-8"))?;
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            Some("ply") | Some("comment") | Some("obj_info") | None => continue,
            Some("format") => {
                let format = tokens
                    .next()
                    .ok_or_else(|| ply_error("`format` line is missing a format name"))?;
                let version = tokens
                    .next()
                    .ok_or_else(|| ply_error("`format` line is missing a version"))?;
                if version != "1.0" {
                    return Err(ply_error(format!(
                        "unsupported PLY version `{version}`; expected 1.0"
                    )));
                }
                ascii = Some(match format {
                    "ascii" => true,
                    "binary_little_endian" => false,
                    other => {
                        return Err(ply_error(format!(
                            "unsupported PLY format `{other}`; use ascii or binary_little_endian"
                        )));
                    }
                });
            }
            Some("element") => {
                let name = tokens
                    .next()
                    .ok_or_else(|| ply_error("`element` line is missing a name"))?;
                let value = tokens
                    .next()
                    .ok_or_else(|| ply_error("`element` line is missing a count"))?;
                let value = value
                    .parse::<usize>()
                    .map_err(|_| ply_error(format!("invalid element count `{value}`")))?;
                if name != "vertex" {
                    return Err(ply_error(format!(
                        "unsupported element `{name}`; voxel PLY files must contain a single \
                         `vertex` element"
                    )));
                }
                if element_seen {
                    return Err(ply_error(
                        "voxel PLY files must contain exactly one `vertex` element",
                    ));
                }
                element_seen = true;
                count = Some(value);
            }
            Some("property") => {
                let ty = tokens
                    .next()
                    .ok_or_else(|| ply_error("`property` line is missing a type"))?;
                if ty == "list" {
                    return Err(ply_error(
                        "list properties are not supported in voxel PLY files",
                    ));
                }
                let ty = PlyType::parse(ty).ok_or_else(|| {
                    ply_error(format!("unsupported property type `{ty}` in voxel PLY"))
                })?;
                let name = tokens
                    .next()
                    .ok_or_else(|| ply_error("`property` line is missing a name"))?;
                properties.push(PlyProperty {
                    name: name.to_string(),
                    ty,
                });
            }
            Some("end_header") => break,
            Some(other) => {
                return Err(ply_error(format!(
                    "unsupported PLY header keyword `{other}`"
                )));
            }
        }
    }

    let ascii = ascii.ok_or_else(|| ply_error("header has no `format` line"))?;
    let count = count.ok_or_else(|| ply_error("header has no `vertex` element"))?;
    for axis in ["x", "y", "z"] {
        if !properties.iter().any(|property| property.name == axis) {
            return Err(ply_error(format!(
                "vertex element is missing the `{axis}` property"
            )));
        }
    }
    Ok(PlyHeader {
        ascii,
        count,
        properties,
        body_offset: offset,
    })
}

fn attr_prefix(name: &str) -> Result<&str> {
    let (prefix, channel) = name.rsplit_once('_').ok_or_else(|| {
        ply_error(format!(
            "attribute property `{name}` must be named `attr_channel`"
        ))
    })?;
    if prefix.is_empty() || channel.is_empty() {
        return Err(ply_error(format!(
            "attribute property `{name}` has an empty name or channel"
        )));
    }
    if !channel.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(ply_error(format!(
            "attribute property `{name}` channel `{channel}` is not numeric"
        )));
    }
    Ok(prefix)
}

fn build_grid(header: &PlyHeader, rows: &[Vec<f64>]) -> Result<VoxelGrid> {
    let x_index = header
        .properties
        .iter()
        .position(|property| property.name == "x")
        .expect("validated x");
    let y_index = header
        .properties
        .iter()
        .position(|property| property.name == "y")
        .expect("validated y");
    let z_index = header
        .properties
        .iter()
        .position(|property| property.name == "z")
        .expect("validated z");

    let mut coords = Vec::with_capacity(rows.len());
    for row in rows {
        let mut coord = [0i32; 3];
        for (axis, index) in [x_index, y_index, z_index].into_iter().enumerate() {
            let rounded = row[index].round_ties_even();
            if !rounded.is_finite() || rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
                return Err(ply_error("vertex coordinate is outside the int32 range"));
            }
            coord[axis] = rounded as i32;
        }
        coords.push(coord);
    }

    let mut attr_order: Vec<String> = Vec::new();
    let mut channels: std::collections::BTreeMap<String, usize> = Default::default();
    for property in &header.properties {
        if matches!(property.name.as_str(), "x" | "y" | "z") {
            continue;
        }
        if !property.ty.is_integer() {
            return Err(ply_error(format!(
                "attribute property `{}` uses a floating point type; o_voxel attributes are uint8",
                property.name
            )));
        }
        let prefix = attr_prefix(&property.name)?;
        if !channels.contains_key(prefix) {
            attr_order.push(prefix.to_string());
        }
        *channels.entry(prefix.to_string()).or_default() += 1;
    }

    let mut attr_values: std::collections::BTreeMap<String, Vec<u8>> = Default::default();
    for name in &attr_order {
        attr_values.insert(
            name.clone(),
            Vec::with_capacity(rows.len() * channels[name]),
        );
    }
    for row in rows {
        for (index, property) in header.properties.iter().enumerate() {
            if matches!(property.name.as_str(), "x" | "y" | "z") {
                continue;
            }
            let prefix = attr_prefix(&property.name)?;
            let value = row[index];
            if value.fract() != 0.0 || !(0.0..=255.0).contains(&value) {
                return Err(ply_error(format!(
                    "attribute property `{}` value {value} is not an integer in 0..=255",
                    property.name
                )));
            }
            attr_values
                .get_mut(prefix)
                .expect("attribute buffers were initialized")
                .push(value as u8);
        }
    }

    let mut grid = VoxelGrid {
        coords,
        attrs: Default::default(),
    };
    for name in attr_order {
        let data = attr_values.remove(&name).unwrap_or_default();
        grid.insert_attr(name.clone(), VoxelAttr::new(channels[&name], data)?);
    }
    grid.validate()?;
    Ok(grid)
}

/// Read a voxel PLY into a [`VoxelGrid`].
pub fn read_ply(path: &Path) -> Result<VoxelGrid> {
    let bytes = std::fs::read(path).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })?;
    let header = parse_header(&bytes)?;
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(header.count);

    if header.ascii {
        let body = std::str::from_utf8(&bytes[header.body_offset..])
            .map_err(|_| ply_error("ASCII body is not valid UTF-8"))?;
        let mut tokens = body.split_whitespace();
        for _ in 0..header.count {
            let mut row = Vec::with_capacity(header.properties.len());
            for property in &header.properties {
                let token = tokens
                    .next()
                    .ok_or_else(|| ply_error("ASCII body has fewer vertices than declared"))?;
                row.push(property.ty.parse_ascii(token)?);
            }
            rows.push(row);
        }
    } else {
        let mut offset = header.body_offset;
        for _ in 0..header.count {
            let mut row = Vec::with_capacity(header.properties.len());
            for property in &header.properties {
                row.push(property.ty.read_binary(&bytes, &mut offset)?);
            }
            rows.push(row);
        }
    }

    build_grid(&header, &rows)
}

/// Write a voxel PLY (`binary_little_endian 1.0`) with `float x/y/z` and
/// `uchar {attr}_{channel}` properties.
pub fn write_ply(path: &Path, grid: &VoxelGrid) -> Result<()> {
    grid.validate()?;
    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str("format binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {}\n", grid.coords.len()));
    header.push_str("property float x\nproperty float y\nproperty float z\n");
    for (name, attr) in &grid.attrs {
        for channel in 0..attr.channels {
            header.push_str(&format!("property uchar {name}_{channel}\n"));
        }
    }
    header.push_str("end_header\n");

    let mut body = Vec::with_capacity(
        grid.coords.len() * (12 + grid.attrs.values().map(|attr| attr.channels).sum::<usize>()),
    );
    for (row, coord) in grid.coords.iter().enumerate() {
        for value in coord {
            body.extend_from_slice(&(*value as f32).to_le_bytes());
        }
        for attr in grid.attrs.values() {
            body.extend_from_slice(&attr.data[row * attr.channels..(row + 1) * attr.channels]);
        }
    }

    let mut file = Vec::with_capacity(header.len() + body.len());
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&body);
    std::fs::write(path, file).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(extension: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "lux3d-ovox-ply-{}-{nonce}.{extension}",
            std::process::id()
        ))
    }

    fn sample_grid() -> VoxelGrid {
        let mut grid = VoxelGrid::new();
        grid.coords = vec![[0, 1, 2], [-4, 5, 6]];
        grid.insert_attr(
            "base_color",
            VoxelAttr::new(3, vec![1, 2, 3, 4, 5, 6]).unwrap(),
        );
        grid.insert_attr("alpha", VoxelAttr::new(1, vec![7, 8]).unwrap());
        grid
    }

    #[test]
    fn ply_binary_round_trips_coords_and_attributes() {
        let path = temp_path("ply");
        let grid = sample_grid();
        write_ply(&path, &grid).expect("write ply");
        let loaded = read_ply(&path).expect("read ply");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded, grid);
    }

    #[test]
    fn ply_reads_ascii_voxel_files() {
        let path = temp_path("ply");
        let text = "ply\nformat ascii 1.0\nelement vertex 2\n\
                    property float x\nproperty float y\nproperty float z\n\
                    property uchar base_color_0\nproperty uchar base_color_1\n\
                    property uchar base_color_2\nend_header\n\
                    0 1 2 10 20 30\n3 4 5 40 50 60\n";
        std::fs::write(&path, text).expect("write ascii ply");
        let grid = read_ply(&path).expect("read ascii ply");
        let _ = std::fs::remove_file(&path);
        assert_eq!(grid.coords, vec![[0, 1, 2], [3, 4, 5]]);
        assert_eq!(
            grid.attr("base_color").unwrap().data,
            vec![10, 20, 30, 40, 50, 60]
        );
    }

    #[test]
    fn ply_header_errors_are_actionable() {
        let path = temp_path("ply");
        std::fs::write(&path, "ply\nformat binary_big_endian 1.0\n").unwrap();
        assert!(read_ply(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
