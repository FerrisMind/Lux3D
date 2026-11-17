//! o_voxel asset I/O parity for TRELLIS.2.
//!
//! This module mirrors the official `o-voxel` sparse-voxel interchange layer:
//!
//! - [`serialize`] — 30-bit z-order / Hilbert sequence codes and the sparse
//!   voxel octree byte stream (`serialize.py:7-68`, `src/serialize/*.cu`,
//!   `src/io/svo.cpp`).
//! - [`vxz`] — the chunked, compressed `.vxz` container
//!   (`o_voxel/io/vxz.py`), including a decode-only raw LZMA2 path so files
//!   written with the official default compression can be opened.
//! - [`npz`] / [`ply`] — voxel array interchange matching
//!   `o_voxel/io/{npz,ply}.py`.
//! - [`read_voxels`] / [`write_voxels`] — extension dispatch matching
//!   `o_voxel/io/__init__.py:8-44`.
//!
//! Attributes are stored as `uint8` rows because the official `.vxz` writer
//! asserts `dtype == torch.uint8` (`vxz.py:249-252`); NPZ/PLY readers accept
//! narrower integer dtypes and reject lossy conversions.

pub mod commands;
pub mod npz;
pub mod ply;
pub mod serialize;
pub mod textured_mesh;
pub mod voxelize;
pub mod vxz;

mod lzma2;

use std::collections::BTreeMap;
use std::path::Path;

use crate::{error::Lux3dError, Result};

/// One `uint8` attribute array with a fixed number of channels per voxel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxelAttr {
    /// Channels per voxel (for example 3 for `base_color`).
    pub channels: usize,
    /// Row-major `[voxel, channel]` bytes.
    pub data: Vec<u8>,
}

impl VoxelAttr {
    /// Build an attribute array, validating the row-major layout.
    pub fn new(channels: usize, data: Vec<u8>) -> Result<Self> {
        if channels == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: "o_voxel attribute arrays need at least one channel".to_string(),
            });
        }
        if data.len() % channels != 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "o_voxel attribute buffer length {} is not a multiple of its {channels} channels",
                    data.len()
                ),
            });
        }
        Ok(Self { channels, data })
    }

    /// Number of voxel rows.
    pub fn len(&self) -> usize {
        self.data.len() / self.channels
    }

    /// Whether the attribute array carries no rows.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Row `index` as a byte slice of `channels` entries.
    pub fn row(&self, index: usize) -> Option<&[u8]> {
        self.data
            .get(index * self.channels..(index + 1) * self.channels)
    }
}

/// Sparse voxel grid in the o_voxel interchange representation.
///
/// `coords` and every attribute row share the same order. The `.vxz` writer
/// re-sorts rows per chunk by z-order code, matching `write_vxz`
/// (`vxz.py:292-299`), so round-trips preserve the set of voxels and their
/// attributes but not necessarily the caller's original row order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VoxelGrid {
    /// Integer voxel coordinates (`int32` in the official reader).
    pub coords: Vec<[i32; 3]>,
    /// Named `uint8` attributes.
    pub attrs: BTreeMap<String, VoxelAttr>,
}

impl VoxelGrid {
    /// Create an empty grid.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of voxels.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Whether the grid has no voxels.
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Insert an attribute array, replacing any existing entry with that name.
    pub fn insert_attr(&mut self, name: impl Into<String>, attr: VoxelAttr) {
        self.attrs.insert(name.into(), attr);
    }

    /// Look up an attribute array by name.
    pub fn attr(&self, name: &str) -> Option<&VoxelAttr> {
        self.attrs.get(name)
    }

    /// Validate the shared row count across coordinates and attributes.
    pub fn validate(&self) -> Result<()> {
        for (name, attr) in &self.attrs {
            if attr.channels == 0 {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: format!("o_voxel attribute `{name}` has zero channels"),
                });
            }
            let expected = self
                .coords
                .len()
                .checked_mul(attr.channels)
                .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                    message: format!("o_voxel attribute `{name}` row count overflows"),
                })?;
            if attr.data.len() != expected {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "o_voxel attribute `{name}` has {} bytes; expected {expected} for {} \
                         voxels x {} channels",
                        attr.data.len(),
                        self.coords.len(),
                        attr.channels
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Voxel artifact formats supported by [`read_voxels`] / [`write_voxels`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OvoxFormat {
    /// NumPy `.npz` with a `coord` array plus named attribute arrays.
    Npz,
    /// Binary little-endian or ASCII PLY point element.
    Ply,
    /// Chunked o_voxel container.
    Vxz,
}

impl OvoxFormat {
    /// Resolve the format from a case-insensitive file extension.
    pub fn from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default();
        Self::from_extension(extension)
    }

    /// Resolve the format from an extension or format name.
    pub fn from_extension(extension: &str) -> Result<Self> {
        match extension
            .trim_start_matches('.')
            .to_ascii_lowercase()
            .as_str()
        {
            "npz" => Ok(Self::Npz),
            "ply" => Ok(Self::Ply),
            "vxz" => Ok(Self::Vxz),
            other => Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "unsupported o_voxel voxel file extension `{other}`; use .npz, .ply or .vxz"
                ),
            }),
        }
    }

    /// Canonical lowercase extension.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Npz => "npz",
            Self::Ply => "ply",
            Self::Vxz => "vxz",
        }
    }
}

/// Read a voxel artifact, dispatching on the file extension
/// (`o_voxel/io/__init__.py:8-26`).
pub fn read_voxels(path: &Path) -> Result<VoxelGrid> {
    match OvoxFormat::from_path(path)? {
        OvoxFormat::Npz => npz::read_npz(path),
        OvoxFormat::Ply => ply::read_ply(path),
        OvoxFormat::Vxz => vxz::read_vxz(path),
    }
}

/// Write a voxel artifact, dispatching on the file extension
/// (`o_voxel/io/__init__.py:29-45`).
pub fn write_voxels(path: &Path, grid: &VoxelGrid) -> Result<OvoxFormat> {
    let format = OvoxFormat::from_path(path)?;
    match format {
        OvoxFormat::Npz => npz::write_npz(path, grid, true)?,
        OvoxFormat::Ply => ply::write_ply(path, grid)?,
        OvoxFormat::Vxz => vxz::write_vxz(path, grid, &vxz::VxzWriteOptions::default())?,
    }
    Ok(format)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_grid_validates_row_counts() {
        let mut grid = VoxelGrid::new();
        grid.coords = vec![[0, 0, 0], [1, 2, 3]];
        grid.insert_attr(
            "base_color",
            VoxelAttr::new(3, vec![1, 2, 3, 4, 5, 6]).unwrap(),
        );
        assert!(grid.validate().is_ok());
        assert_eq!(
            grid.attr("base_color").unwrap().row(1),
            Some(&[4, 5, 6][..])
        );

        grid.insert_attr("broken", VoxelAttr::new(1, vec![1, 2, 3]).unwrap());
        assert!(grid.validate().is_err());
        assert!(VoxelAttr::new(0, Vec::new()).is_err());
        assert!(VoxelAttr::new(3, vec![1, 2]).is_err());
    }

    #[test]
    fn format_dispatch_is_extension_driven() {
        assert_eq!(
            OvoxFormat::from_path(Path::new("a.VXZ")).unwrap(),
            OvoxFormat::Vxz
        );
        assert_eq!(
            OvoxFormat::from_path(Path::new("a.npz")).unwrap(),
            OvoxFormat::Npz
        );
        assert!(OvoxFormat::from_path(Path::new("a.bin")).is_err());
    }
}
