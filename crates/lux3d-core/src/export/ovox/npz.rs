//! NumPy `.npz` voxel interchange (`o_voxel/io/npz.py`).
//!
//! Official semantics:
//!
//! - `write_npz` stores `coord` as `uint16` and every attribute as its own
//!   array, compressed by default (`np.savez_compressed`; `npz.py:29-43`).
//! - `read_npz` loads `coord`, casts it to `int32`, and treats every other
//!   array as an attribute (`npz.py:12-26`).
//!
//! This build stores attributes as `uint8` rows like the `.vxz` writer, so an
//! attribute array with a wider dtype is rejected instead of being truncated.

use std::path::Path;

use ndarray::{Array2, ArrayD};
use ndarray_npy::{NpzReader, NpzWriter, ReadableElement};

use super::{VoxelAttr, VoxelGrid};
use crate::{error::Lux3dError, Result};

fn npz_error(message: impl Into<String>) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: format!("o_voxel NPZ: {}", message.into()),
    }
}

trait CoordScalar: Copy + ReadableElement {
    fn to_i64(self) -> Option<i64>;
}

macro_rules! impl_coord_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl CoordScalar for $ty {
                fn to_i64(self) -> Option<i64> {
                    i64::try_from(self).ok()
                }
            }
        )*
    };
}

impl_coord_scalar!(u8, u16, u32, i8, i16, i32, i64);

impl CoordScalar for f32 {
    fn to_i64(self) -> Option<i64> {
        if self.is_finite() {
            Some(self as i64)
        } else {
            None
        }
    }
}

impl CoordScalar for f64 {
    fn to_i64(self) -> Option<i64> {
        if self.is_finite() {
            Some(self as i64)
        } else {
            None
        }
    }
}

fn convert_coord<T: CoordScalar>(array: ArrayD<T>) -> Result<Vec<[i32; 3]>> {
    if array.ndim() != 2 || array.shape()[1] != 3 {
        return Err(npz_error(format!(
            "`coord` must have shape [N, 3], got {:?}",
            array.shape()
        )));
    }
    let rows = array.shape()[0];
    let mut coords = Vec::with_capacity(rows);
    let values: Vec<T> = array.iter().copied().collect();
    for row in 0..rows {
        let mut coord = [0i32; 3];
        for axis in 0..3 {
            let value = values[row * 3 + axis]
                .to_i64()
                .ok_or_else(|| npz_error("`coord` contains a non-finite or out-of-range value"))?;
            coord[axis] = i32::try_from(value)
                .map_err(|_| npz_error("`coord` value does not fit in int32"))?;
        }
        coords.push(coord);
    }
    Ok(coords)
}

fn try_read_coord<T: CoordScalar>(
    npz: &mut NpzReader<std::fs::File>,
    name: &str,
) -> Option<Result<Vec<[i32; 3]>>> {
    let array: ArrayD<T> = npz.by_name(name).ok()?;
    Some(convert_coord(array))
}

fn read_attr(array: ArrayD<u8>, coords_len: usize, name: &str) -> Result<(usize, Vec<u8>)> {
    let channels = match array.ndim() {
        1 => 1,
        2 => array.shape()[1],
        other => {
            return Err(npz_error(format!(
                "attribute `{name}` must be 1-D or 2-D, got {other} dimensions"
            )));
        }
    };
    if channels == 0 {
        return Err(npz_error(format!("attribute `{name}` has zero channels")));
    }
    let rows = array.shape()[0];
    if rows != coords_len {
        return Err(npz_error(format!(
            "attribute `{name}` has {rows} rows but `coord` has {coords_len}"
        )));
    }
    Ok((channels, array.iter().copied().collect()))
}

/// Read a `.npz` voxel artifact into a [`VoxelGrid`].
pub fn read_npz(path: &Path) -> Result<VoxelGrid> {
    let file = std::fs::File::open(path).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut npz = NpzReader::new(file)
        .map_err(|error| npz_error(format!("failed to open `{}`: {error}", path.display())))?;
    let names = npz
        .names()
        .map_err(|error| npz_error(format!("failed to list arrays: {error}")))?;
    if !names.iter().any(|name| name == "coord") {
        return Err(npz_error(
            "archive has no `coord` array (official read_npz requires it)",
        ));
    }

    let mut coords = None;
    macro_rules! try_coord {
        ($($ty:ty),* $(,)?) => {
            $(
                if coords.is_none() {
                    coords = try_read_coord::<$ty>(&mut npz, "coord");
                }
            )*
        };
    }
    try_coord!(u16, i32, i64, u32, i16, u8, i8, f32, f64);
    let coords = coords.ok_or_else(|| {
        npz_error("`coord` has an unsupported dtype; expected an integer or floating point array")
    })??;

    let mut grid = VoxelGrid {
        coords,
        attrs: Default::default(),
    };
    for name in names {
        if name == "coord" {
            continue;
        }
        let array: ArrayD<u8> = npz.by_name(&name).map_err(|error| {
            npz_error(format!(
                "attribute `{name}` must be a uint8 array (o_voxel attributes are uint8): {error}"
            ))
        })?;
        let (channels, data) = read_attr(array, grid.coords.len(), &name)?;
        grid.insert_attr(name, VoxelAttr::new(channels, data)?);
    }
    grid.validate()?;
    Ok(grid)
}

/// Write a `.npz` voxel artifact.
///
/// `compress` selects `np.savez_compressed` (official default) vs
/// `np.savez`.
pub fn write_npz(path: &Path, grid: &VoxelGrid, compress: bool) -> Result<()> {
    grid.validate()?;
    let file = std::fs::File::create(path).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = if compress {
        NpzWriter::new_compressed(file)
    } else {
        NpzWriter::new(file)
    };

    let coord_values: Vec<u16> = grid
        .coords
        .iter()
        .flat_map(|coord| coord.iter().map(|&value| value as u16))
        .collect();
    let coord_array = Array2::from_shape_vec((grid.coords.len(), 3), coord_values)
        .map_err(|error| npz_error(format!("failed to build the `coord` array: {error}")))?;
    writer
        .add_array("coord", &coord_array)
        .map_err(|error| npz_error(format!("failed to write `coord`: {error}")))?;

    for (name, attr) in &grid.attrs {
        let array = Array2::from_shape_vec((grid.coords.len(), attr.channels), attr.data.clone())
            .map_err(|error| {
            npz_error(format!("failed to build attribute `{name}`: {error}"))
        })?;
        writer
            .add_array(name.as_str(), &array)
            .map_err(|error| npz_error(format!("failed to write attribute `{name}`: {error}")))?;
    }

    writer
        .finish()
        .map(|_| ())
        .map_err(|error| npz_error(format!("failed to finish `{}`: {error}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(extension: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "lux3d-ovox-npz-{}-{nonce}.{extension}",
            std::process::id()
        ))
    }

    fn sample_grid() -> VoxelGrid {
        let mut grid = VoxelGrid::new();
        grid.coords = vec![[0, 1, 2], [4, 5, 6], [7, 8, 9]];
        grid.insert_attr(
            "base_color",
            VoxelAttr::new(3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap(),
        );
        grid.insert_attr("alpha", VoxelAttr::new(1, vec![10, 11, 12]).unwrap());
        grid
    }

    #[test]
    fn npz_round_trips_coords_and_attributes() {
        let path = temp_path("npz");
        let grid = sample_grid();
        write_npz(&path, &grid, true).expect("write npz");
        let loaded = read_npz(&path).expect("read npz");
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded, grid);
    }

    #[test]
    fn npz_writer_rejects_inconsistent_grids() {
        let path = temp_path("npz");
        let mut grid = sample_grid();
        grid.coords.pop();
        assert!(write_npz(&path, &grid, true).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn one_dimensional_attribute_reads_as_one_channel() {
        let array: ArrayD<u8> = Array1::from(vec![1u8, 2, 3]).into_dyn();
        let (channels, data) = read_attr(array, 3, "alpha").expect("1-D attr");
        assert_eq!(channels, 1);
        assert_eq!(data, vec![1, 2, 3]);
    }
}
