//! o_voxel command operations: `mesh2ovox`, `ovox2mesh`, `ovox2glb`.
//!
//! Core entry points for the three official example workflows. The CLI
//! subcommand tree is wired elsewhere; these functions take input/output
//! paths plus options and return structured statistics so the CLI layer stays
//! thin.
//!
//! Official reference chain:
//!
//! * `examples/mesh2ovox.py:9-57` — normalize to the unit cube, geometry
//!   voxelization (`mesh_to_flexible_dual_grid` with `face_weight=1.0`,
//!   `boundary_weight=0.2`, `regularization_weight=1e-2`), material
//!   voxelization (`textured_mesh_to_volumetric_attr`), sort both by z-order,
//!   pack `dual_vertices` / `intersected`, write `.vxz`.
//! * `examples/ovox2mesh.py:1-45` — read `.vxz`, unpack, extract the mesh
//!   (`flexible_dual_grid_to_mesh` with `split_weight=None`) and export `.ply`
//!   with base-color vertex colors.
//! * `examples/ovox2glb.py:1-52` — same extraction, then
//!   `postprocess.to_glb(decimation_target=100000, texture_size=2048,
//!   grid_size=RES, aabb=[-0.5, 0.5])` and a textured `.glb` export.
//!
//! The `.vxz` container does not store the grid size or the aabb (official
//! `vxz.py` header fields), so the decode commands expose both explicitly and
//! default to the examples' `RES = 512` / `[-0.5, 0.5]^3`.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::Serialize;

use crate::error::{Lux3dError, Result};

use super::serialize::{encode_seq, SeqMode, DEFAULT_PERMUTE};
use super::textured_mesh::load_textured_mesh;
use super::voxelize::{
    quantize_attr, resolve_voxelization_settings, textured_mesh_to_volumetric_attr,
    MaterialVoxelization, VoxelizationSettings,
};
use super::vxz::{VxzCompression, VxzWriteOptions};
use super::{npz, ply, vxz, OvoxFormat, VoxelAttr, VoxelGrid};

/// Default voxelization resolution of the shipped examples (`RES = 512`).
pub const OVOX_DEFAULT_RESOLUTION: u32 = 512;
/// Default voxel box of the shipped examples (`[[-0.5; 3], [0.5; 3]]`).
pub const OVOX_DEFAULT_AABB: [[f32; 3]; 2] = [[-0.5; 3], [0.5; 3]];
/// Default texture-mip offset (`textured_mesh_to_volumetric_attr`
/// `mip_level_offset = 0.0`).
pub const OVOX_DEFAULT_MIP_LEVEL_OFFSET: f32 = 0.0;
/// Default decimation target of the `ovox2glb` example
/// (`examples/ovox2glb.py:46`).
pub const OVOX_DEFAULT_DECIMATION_TARGET: usize = 100_000;
/// Default bake texture size of the `ovox2glb` example
/// (`examples/ovox2glb.py:47`).
pub const OVOX_DEFAULT_TEXTURE_SIZE: usize = 2048;
/// Upstream `to_glb` initial `fill_holes(max_hole_perimeter=3e-2)`
/// (`postprocess.py:110`).
pub const OVOX_DEFAULT_FILL_HOLE_PERIMETER: f32 = 3e-2;
/// Upstream `to_glb` `remove_small_connected_components(1e-5)`
/// (`postprocess.py:143`).
pub const OVOX_DEFAULT_MIN_COMPONENT_AREA: f32 = 1e-5;

/// Options for [`mesh2ovox`].
#[derive(Debug, Clone, PartialEq)]
pub struct MeshToOvoxOptions {
    /// Target grid resolution per axis (`[512; 3]`, the example `RES`).
    /// Mutually optional with [`Self::voxel_size`], like the upstream wrapper.
    pub grid_size: Option<[u32; 3]>,
    /// Explicit voxel size per axis (upstream `voxel_size` form). When set,
    /// the kernel uses it verbatim while the grid range comes from
    /// [`Self::grid_size`] (or `round(extent / voxel_size)`).
    pub voxel_size: Option<[f32; 3]>,
    /// Explicit voxel box. `None` resolves to
    /// [`OVOX_DEFAULT_AABB`] after normalization (the example's explicit
    /// `[-0.5, 0.5]^3`); with `normalize_to_unit_cube = false` it falls back
    /// to the upstream auto-padding formula.
    pub aabb: Option<[[f32; 3]; 2]>,
    /// Apply the example's unit-cube normalization
    /// (`mesh2ovox.py:10-14`: center + `0.99999 / max_extent`), true.
    pub normalize_to_unit_cube: bool,
    /// Texture mip-level offset (upstream `mip_level_offset`, 0.0).
    pub mip_level_offset: f32,
    /// QEF face-term gate (`face_weight`, example 1.0).
    pub face_weight: f32,
    /// QEF boundary-term weight (`boundary_weight`, example 0.2).
    pub boundary_weight: f32,
    /// QEF regularization toward the mean intersection
    /// (`regularization_weight`, example 1e-2).
    pub regularization_weight: f32,
    /// `.vxz` compression when the output extension is `.vxz` (deflate; the
    /// TR8a writer supports `none` / `deflate`).
    pub compression: VxzCompression,
}

impl Default for MeshToOvoxOptions {
    fn default() -> Self {
        Self {
            grid_size: Some([OVOX_DEFAULT_RESOLUTION; 3]),
            voxel_size: None,
            aabb: None,
            normalize_to_unit_cube: true,
            mip_level_offset: OVOX_DEFAULT_MIP_LEVEL_OFFSET,
            face_weight: 1.0,
            boundary_weight: 0.2,
            regularization_weight: 1e-2,
            compression: VxzCompression::Deflate,
        }
    }
}

impl MeshToOvoxOptions {
    /// Validate the option set before loading the mesh.
    pub fn validate(&self) -> Result<()> {
        if self.grid_size.is_none() && self.voxel_size.is_none() {
            return Err(Lux3dError::InvalidInput(
                "o_voxel mesh2ovox needs either --grid-size or --voxel-size",
            ));
        }
        if let Some(grid_size) = &self.grid_size {
            for (axis, size) in grid_size.iter().enumerate() {
                if *size < 2 || *size > 4096 {
                    return Err(Lux3dError::InvalidInput(
                        "o_voxel mesh2ovox grid_size must be in 2..=4096 per axis",
                    ));
                }
                let _ = axis;
            }
        }
        if let Some(voxel_size) = &self.voxel_size {
            for size in voxel_size {
                if !size.is_finite() || *size <= 0.0 {
                    return Err(Lux3dError::InvalidInput(
                        "o_voxel mesh2ovox voxel_size must be finite and positive per axis",
                    ));
                }
            }
        }
        if let Some(aabb) = &self.aabb {
            for axis in 0..3 {
                if !aabb[0][axis].is_finite()
                    || !aabb[1][axis].is_finite()
                    || !(aabb[0][axis] < aabb[1][axis])
                {
                    return Err(Lux3dError::InvalidInput(
                        "o_voxel mesh2ovox aabb must be finite with min < max per axis",
                    ));
                }
            }
        }
        if !self.mip_level_offset.is_finite() {
            return Err(Lux3dError::InvalidInput(
                "o_voxel mesh2ovox mip_level_offset must be finite",
            ));
        }
        for weight in [
            self.face_weight,
            self.boundary_weight,
            self.regularization_weight,
        ] {
            if !weight.is_finite() || weight < 0.0 {
                return Err(Lux3dError::InvalidInput(
                    "o_voxel mesh2ovox QEF weights must be finite and non-negative",
                ));
            }
        }
        if !matches!(
            self.compression,
            VxzCompression::None | VxzCompression::Deflate
        ) {
            return Err(Lux3dError::InvalidInput(
                "o_voxel .vxz writing supports only `none` and `deflate` compression",
            ));
        }
        Ok(())
    }
}

/// Statistics of one [`mesh2ovox`] run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MeshToOvoxStats {
    /// Input vertices / triangles.
    pub input_vertices: usize,
    /// Input triangles.
    pub input_faces: usize,
    /// Degenerate triangles dropped before voxelization (official would
    /// propagate `NaN`; see [`super::textured_mesh`]).
    pub dropped_degenerate_triangles: usize,
    /// Voxels written (geometry and material sets are aligned).
    pub voxel_count: usize,
    /// Resolved grid resolution per axis.
    pub grid_size: [u32; 3],
    /// Resolved voxel size per axis.
    pub voxel_size: [f32; 3],
    /// Resolved voxel box.
    pub aabb: [[f32; 3]; 2],
    /// Written attribute name -> channel count (`base_color` 3, `metallic` 1,
    /// `roughness` 1, `emissive` 3, `alpha` 1, `normal` 3, `dual_vertices` 3,
    /// `intersected` 1).
    pub attribute_channels: BTreeMap<String, usize>,
    /// Written container format (`vxz`, `npz`, `ply`).
    pub output_format: &'static str,
    /// Output file size in bytes.
    pub output_bytes: u64,
}

/// Voxelize a textured mesh into a sparse PBR `.vxz` volume.
///
/// Equivalent to `python examples/mesh2ovox.py` with `--res` / options.
pub fn mesh2ovox(
    input: &Path,
    output: &Path,
    options: &MeshToOvoxOptions,
) -> Result<MeshToOvoxStats> {
    options.validate()?;
    let mut mesh = load_textured_mesh(input)?;
    let input_vertices = mesh.vertex_count();
    let input_faces = mesh.face_count();
    if options.normalize_to_unit_cube {
        mesh.normalize_to_unit_cube()?;
    }
    let dropped_degenerate_triangles = mesh.drop_degenerate_triangles();
    if mesh.faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "o_voxel mesh2ovox: every input triangle is degenerate",
        ));
    }
    let bounds = mesh.bounds().ok_or(Lux3dError::InvalidInput(
        "o_voxel mesh2ovox needs a non-empty mesh",
    ))?;
    let aabb = match options.aabb {
        Some(aabb) => Some(aabb),
        None if options.normalize_to_unit_cube => Some(OVOX_DEFAULT_AABB),
        None => None,
    };
    let settings =
        resolve_voxelization_settings(bounds, options.voxel_size, options.grid_size, aabb)?;
    if settings.grid_size.iter().any(|size| *size < 2) {
        return Err(Lux3dError::InvalidInput(
            "o_voxel geometry voxelization needs grid_size >= 2 per axis",
        ));
    }

    let geometry = crate::geometry::mesh_to_flexible_dual_grid(
        &mesh
            .vertices
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<f32>>(),
        &mesh.faces,
        &crate::geometry::MeshToDualGridOptions {
            grid_size: settings.grid_size,
            voxel_size: Some(settings.voxel_size),
            aabb: Some(settings.aabb),
            face_weight: options.face_weight,
            boundary_weight: options.boundary_weight,
            regularization_weight: options.regularization_weight,
        },
    )?;

    let corner_normals = mesh.corner_normals();
    let material = textured_mesh_to_volumetric_attr(
        &mesh.vertices,
        &mesh.faces,
        &mesh.corner_uvs,
        &corner_normals,
        &mesh.material_ids,
        &mesh.materials,
        &settings,
        options.mip_level_offset,
    )?;

    let (coords, dual_vertices, intersected, attrs) =
        align_and_pack(&geometry, &material, &settings)?;
    let grid = VoxelGrid { coords, attrs };
    grid.validate()?;

    let output_format = OvoxFormat::from_path(output)?;
    match output_format {
        OvoxFormat::Vxz => vxz::write_vxz(
            output,
            &grid,
            &VxzWriteOptions {
                compression: options.compression,
                ..VxzWriteOptions::default()
            },
        )?,
        OvoxFormat::Npz => npz::write_npz(output, &grid, true)?,
        OvoxFormat::Ply => ply::write_ply(output, &grid)?,
    }
    let _ = (dual_vertices, intersected);

    let output_bytes = std::fs::metadata(output)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let attribute_channels = grid
        .attrs
        .iter()
        .map(|(name, attr)| (name.clone(), attr.channels))
        .collect();
    Ok(MeshToOvoxStats {
        input_vertices,
        input_faces,
        dropped_degenerate_triangles,
        voxel_count: grid.len(),
        grid_size: settings.grid_size,
        voxel_size: settings.voxel_size,
        aabb: settings.aabb,
        attribute_channels,
        output_format: output_format.extension(),
        output_bytes,
    })
}

/// Sort the geometry and material voxel sets by z-order (the examples'
/// `argsort(encode_seq(...))` step), verify they describe the same set, and
/// pack the `.vxz` attribute rows.
#[allow(clippy::type_complexity)]
fn align_and_pack(
    geometry: &crate::geometry::DualGridVoxelization,
    material: &MaterialVoxelization,
    settings: &VoxelizationSettings,
) -> Result<(
    Vec<[i32; 3]>,
    Vec<[f32; 3]>,
    Vec<[bool; 3]>,
    BTreeMap<String, VoxelAttr>,
)> {
    let codes = encode_seq(&geometry.voxel_indices, DEFAULT_PERMUTE, SeqMode::ZOrder)?;
    let mut order: Vec<usize> = (0..geometry.voxel_indices.len()).collect();
    order.sort_by_key(|&index| codes[index]);

    let material_map: HashMap<[i32; 3], usize> = material
        .coords
        .iter()
        .enumerate()
        .map(|(index, coord)| (*coord, index))
        .collect();
    if material_map.len() != material.coords.len() {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: "o_voxel material voxelization produced duplicate coordinates".to_string(),
        });
    }
    if material.coords.len() != geometry.voxel_indices.len() {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "o_voxel geometry and material voxel sets differ ({} vs {} voxels); the official \
                 example assumes they match",
                geometry.voxel_indices.len(),
                material.coords.len()
            ),
        });
    }

    let count = order.len();
    let mut coords = Vec::with_capacity(count);
    let mut base_color = Vec::with_capacity(count * 3);
    let mut metallic = Vec::with_capacity(count);
    let mut roughness = Vec::with_capacity(count);
    let mut emissive = Vec::with_capacity(count * 3);
    let mut alpha = Vec::with_capacity(count);
    let mut normal = Vec::with_capacity(count * 3);
    let mut dual_packed = Vec::with_capacity(count * 3);
    let mut intersected_packed = Vec::with_capacity(count);

    for &index in &order {
        let coord = geometry.voxel_indices[index];
        let source =
            *material_map
                .get(&coord)
                .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "o_voxel geometry voxel {coord:?} has no material attribute row"
                    ),
                })?;
        coords.push(coord);
        for channel in 0..3 {
            base_color.push(quantize_attr(material.base_color[source][channel]));
        }
        metallic.push(quantize_attr(material.metallic[source]));
        roughness.push(quantize_attr(material.roughness[source]));
        for channel in 0..3 {
            emissive.push(quantize_attr(material.emissive[source][channel]));
        }
        alpha.push(quantize_attr(material.alpha[source]));
        for channel in 0..3 {
            // Official `volumetic_attr.py:299` maps the world-space normal to
            // `[0, 1]` before the uint8 cast.
            normal.push(quantize_attr(material.normal[source][channel] * 0.5 + 0.5));
        }
        let dual = geometry.dual_vertices[index];
        for axis in 0..3 {
            let fractional = dual[axis] * settings.grid_size[axis] as f32 - coord[axis] as f32;
            dual_packed.push((fractional.clamp(0.0, 1.0) * 255.0) as u8);
        }
        intersected_packed.push(
            u8::from(geometry.intersected[index][0])
                + 2 * u8::from(geometry.intersected[index][1])
                + 4 * u8::from(geometry.intersected[index][2]),
        );
    }

    let mut attrs = BTreeMap::new();
    attrs.insert("base_color".to_string(), VoxelAttr::new(3, base_color)?);
    attrs.insert("metallic".to_string(), VoxelAttr::new(1, metallic)?);
    attrs.insert("roughness".to_string(), VoxelAttr::new(1, roughness)?);
    attrs.insert("emissive".to_string(), VoxelAttr::new(3, emissive)?);
    attrs.insert("alpha".to_string(), VoxelAttr::new(1, alpha)?);
    attrs.insert("normal".to_string(), VoxelAttr::new(3, normal)?);
    attrs.insert("dual_vertices".to_string(), VoxelAttr::new(3, dual_packed)?);
    attrs.insert(
        "intersected".to_string(),
        VoxelAttr::new(1, intersected_packed)?,
    );

    let dual_vertices = order
        .iter()
        .map(|&index| geometry.dual_vertices[index])
        .collect();
    let intersected = order
        .iter()
        .map(|&index| geometry.intersected[index])
        .collect();
    Ok((coords, dual_vertices, intersected, attrs))
}

/// Options for the two decode commands.
#[derive(Debug, Clone, PartialEq)]
pub struct OvoxDecodeOptions {
    /// Voxel grid resolution the asset was written with (`[512; 3]`).
    pub grid_size: [u32; 3],
    /// Voxel box the asset was written with (`[-0.5, 0.5]^3`).
    pub aabb: [[f32; 3]; 2],
}

impl Default for OvoxDecodeOptions {
    fn default() -> Self {
        Self {
            grid_size: [OVOX_DEFAULT_RESOLUTION; 3],
            aabb: OVOX_DEFAULT_AABB,
        }
    }
}

impl OvoxDecodeOptions {
    /// Validate the decode placement.
    pub fn validate(&self) -> Result<()> {
        for size in self.grid_size {
            if size == 0 {
                return Err(Lux3dError::InvalidInput(
                    "o_voxel decode grid_size must be positive per axis",
                ));
            }
        }
        for axis in 0..3 {
            if !self.aabb[0][axis].is_finite()
                || !self.aabb[1][axis].is_finite()
                || !(self.aabb[0][axis] < self.aabb[1][axis])
            {
                return Err(Lux3dError::InvalidInput(
                    "o_voxel decode aabb must be finite with min < max per axis",
                ));
            }
        }
        Ok(())
    }
}

/// Statistics of one [`ovox2mesh`] run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OvoxToMeshStats {
    /// Input voxel count.
    pub voxel_count: usize,
    /// Reconstructed vertex count (one per active voxel, like the reference).
    pub vertices: usize,
    /// Reconstructed triangle count.
    pub faces: usize,
    /// Reconstructed vertex bounding box.
    pub bbox_min: [f32; 3],
    /// Reconstructed vertex bounding box.
    pub bbox_max: [f32; 3],
    /// Whether base-color vertex colors were written.
    pub vertex_colors: bool,
    /// Output PLY size in bytes.
    pub output_bytes: u64,
}

/// Decoded `.vxz`/`.npz`/`.ply` voxel asset, unpacked for mesh extraction.
pub struct DecodedOvox {
    /// Voxel coordinates.
    pub coords: Vec<[i32; 3]>,
    /// Per-voxel dual offsets in cell units (`[0, 1]` after the official
    /// `uint8 / 255` unpack).
    pub dual_vertices: Vec<[f32; 3]>,
    /// Per-axis quad emission flags.
    pub intersected: Vec<[bool; 3]>,
    /// `[N, 3]` base color bytes, when present.
    pub base_color: Option<Vec<u8>>,
    /// `[N, 6]` PBR attributes in the official layout
    /// (`base_color 0:3, metallic 3, roughness 4, alpha 5`), when all four
    /// attribute arrays are present.
    pub pbr6: Option<Vec<f32>>,
}

/// Read and unpack an o_voxel asset (`.vxz`, `.npz` or `.ply`).
pub fn decode_ovox(path: &Path, options: &OvoxDecodeOptions) -> Result<DecodedOvox> {
    options.validate()?;
    let grid = super::read_voxels(path)?;
    grid.validate()?;
    let dual_attr =
        grid.attr("dual_vertices")
            .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "o_voxel asset `{}` has no `dual_vertices` attribute",
                    path.display()
                ),
            })?;
    if dual_attr.channels != 3 {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "o_voxel asset `{}` `dual_vertices` must have 3 channels",
                path.display()
            ),
        });
    }
    let intersected_attr =
        grid.attr("intersected")
            .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "o_voxel asset `{}` has no `intersected` attribute",
                    path.display()
                ),
            })?;
    if intersected_attr.channels != 1 {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "o_voxel asset `{}` `intersected` must have 1 channel",
                path.display()
            ),
        });
    }

    let mut coords = Vec::with_capacity(grid.len());
    for coord in &grid.coords {
        for axis in 0..3 {
            if coord[axis] < 0 || coord[axis] >= options.grid_size[axis] as i32 {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "o_voxel asset `{}` voxel {coord:?} is outside the supplied grid_size {:?}",
                        path.display(),
                        options.grid_size
                    ),
                });
            }
        }
        coords.push(*coord);
    }

    let dual_vertices: Vec<[f32; 3]> = dual_attr
        .data
        .chunks_exact(3)
        .map(|chunk| {
            [
                chunk[0] as f32 / 255.0,
                chunk[1] as f32 / 255.0,
                chunk[2] as f32 / 255.0,
            ]
        })
        .collect();
    let intersected: Vec<[bool; 3]> = intersected_attr
        .data
        .iter()
        .map(|&packed| {
            [
                packed % 2 == 1,
                (packed / 2) % 2 == 1,
                (packed / 4) % 2 == 1,
            ]
        })
        .collect();

    let base_color = grid.attr("base_color").and_then(|attr| {
        if attr.channels == 3 {
            Some(attr.data.clone())
        } else {
            None
        }
    });
    let pbr6 = match (
        grid.attr("base_color"),
        grid.attr("metallic"),
        grid.attr("roughness"),
        grid.attr("alpha"),
    ) {
        (Some(base), Some(metallic), Some(roughness), Some(alpha))
            if base.channels == 3
                && metallic.channels == 1
                && roughness.channels == 1
                && alpha.channels == 1 =>
        {
            let mut values = Vec::with_capacity(grid.len() * 6);
            for row in 0..grid.len() {
                values.push(base.data[row * 3] as f32 / 255.0);
                values.push(base.data[row * 3 + 1] as f32 / 255.0);
                values.push(base.data[row * 3 + 2] as f32 / 255.0);
                values.push(metallic.data[row] as f32 / 255.0);
                values.push(roughness.data[row] as f32 / 255.0);
                values.push(alpha.data[row] as f32 / 255.0);
            }
            Some(values)
        }
        _ => None,
    };

    Ok(DecodedOvox {
        coords,
        dual_vertices,
        intersected,
        base_color,
        pbr6,
    })
}

/// Reconstruct a mesh from a `.vxz`/`.npz`/`.ply` o_voxel asset and write a
/// PLY with base-color vertex colors — `examples/ovox2mesh.py`.
pub fn ovox2mesh(
    input: &Path,
    output: &Path,
    options: &OvoxDecodeOptions,
) -> Result<OvoxToMeshStats> {
    let extension = output
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if extension != "ply" {
        return Err(Lux3dError::InvalidInput(
            "o_voxel ovox2mesh writes `.ply` only (the official example's format)",
        ));
    }
    let decoded = decode_ovox(input, options)?;
    let mesh = crate::runtime::flexible_dual_grid_to_mesh_auto_split(
        &decoded.coords,
        &decoded.dual_vertices,
        &decoded.intersected,
        [
            options.grid_size[0] as usize,
            options.grid_size[1] as usize,
            options.grid_size[2] as usize,
        ],
        options.aabb,
    );
    if mesh.faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "o_voxel ovox2mesh reconstruction produced no triangles; check the grid_size/aabb \
             settings",
        ));
    }
    let vertex_colors = decoded.base_color.as_ref().map(|colors| {
        colors
            .iter()
            .map(|value| *value as f32 / 255.0)
            .collect::<Vec<f32>>()
    });
    let (bbox_min, bbox_max) = vertex_bounds(&mesh.vertices);
    let view = crate::export::MeshView {
        vertices: &mesh.vertices,
        faces: &mesh.faces,
        colors: vertex_colors.as_deref(),
    };
    crate::export::mesh::write_ply(&view, output)?;
    let output_bytes = std::fs::metadata(output)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    Ok(OvoxToMeshStats {
        voxel_count: decoded.coords.len(),
        vertices: mesh.vertices.len() / 3,
        faces: mesh.faces.len() / 3,
        bbox_min,
        bbox_max,
        vertex_colors: vertex_colors.is_some(),
        output_bytes,
    })
}

/// Options for [`ovox2glb`] (the official example defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct OvoxToGlbOptions {
    /// Decode placement (see [`OvoxDecodeOptions`]).
    pub decode: OvoxDecodeOptions,
    /// Post-process decimation target (`examples/ovox2glb.py:46`: 100000).
    pub decimation_target: usize,
    /// Bake texture edge length (`examples/ovox2glb.py:47`: 2048).
    pub texture_size: usize,
    /// Upstream `to_glb` hole-filling perimeter (`postprocess.py:110`,
    /// `3e-2`).
    pub fill_hole_perimeter: f32,
    /// Upstream small-component area threshold (`postprocess.py:143`, `1e-5`).
    pub min_component_area: f32,
}

impl Default for OvoxToGlbOptions {
    fn default() -> Self {
        Self {
            decode: OvoxDecodeOptions::default(),
            decimation_target: OVOX_DEFAULT_DECIMATION_TARGET,
            texture_size: OVOX_DEFAULT_TEXTURE_SIZE,
            fill_hole_perimeter: OVOX_DEFAULT_FILL_HOLE_PERIMETER,
            min_component_area: OVOX_DEFAULT_MIN_COMPONENT_AREA,
        }
    }
}

impl OvoxToGlbOptions {
    /// Validate the options.
    pub fn validate(&self) -> Result<()> {
        self.decode.validate()?;
        if self.texture_size < 8 || self.texture_size > 8192 || self.texture_size % 8 != 0 {
            return Err(Lux3dError::InvalidInput(
                "o_voxel ovox2glb texture_size must be a multiple of 8 in 8..=8192",
            ));
        }
        if !self.fill_hole_perimeter.is_finite() || self.fill_hole_perimeter < 0.0 {
            return Err(Lux3dError::InvalidInput(
                "o_voxel ovox2glb fill_hole_perimeter must be finite and non-negative",
            ));
        }
        if !self.min_component_area.is_finite() || self.min_component_area < 0.0 {
            return Err(Lux3dError::InvalidInput(
                "o_voxel ovox2glb min_component_area must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Statistics of one [`ovox2glb`] run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OvoxToGlbStats {
    /// Input voxel count.
    pub voxel_count: usize,
    /// Reconstructed (pre-postprocess) vertex / triangle counts.
    pub extracted_vertices: usize,
    /// Reconstructed (pre-postprocess) triangle count.
    pub extracted_faces: usize,
    /// Post-processed vertex / triangle counts.
    pub postprocessed_vertices: usize,
    /// Post-processed triangle count.
    pub postprocessed_faces: usize,
    /// Atlas vertex count written into the GLB.
    pub atlas_vertices: usize,
    /// Atlas triangle count written into the GLB.
    pub atlas_faces: usize,
    /// UV charts produced by the unwrap.
    pub chart_count: u32,
    /// Bake texture edge length.
    pub texture_size: usize,
    /// Texels covered by a rasterized UV triangle before inpainting.
    pub valid_texels: usize,
    /// Full material double-sided flag (`true` for the non-remeshed path).
    pub double_sided: bool,
    /// Output GLB size in bytes.
    pub output_bytes: u64,
}

/// Reconstruct a mesh, clean/UV-unwrap/bake it and write a textured `.glb` —
/// the official `examples/ovox2glb.py` workflow (`o_voxel.postprocess.to_glb`
/// with `remesh = False`, the `to_glb` signature default, because the example
/// does not pass a remesh flag).
pub fn ovox2glb(input: &Path, output: &Path, options: &OvoxToGlbOptions) -> Result<OvoxToGlbStats> {
    options.validate()?;
    let extension = output
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if extension != "glb" {
        return Err(Lux3dError::InvalidInput(
            "o_voxel ovox2glb writes `.glb` only",
        ));
    }
    let decoded = decode_ovox(input, &options.decode)?;
    let pbr6 = decoded
        .pbr6
        .as_ref()
        .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "o_voxel asset `{}` is missing one of base_color/metallic/roughness/alpha, which \
                 ovox2glb needs for the PBR bake",
                input.display()
            ),
        })?;
    let grid_size = [
        options.decode.grid_size[0] as usize,
        options.decode.grid_size[1] as usize,
        options.decode.grid_size[2] as usize,
    ];
    let extracted = crate::runtime::flexible_dual_grid_to_mesh_auto_split(
        &decoded.coords,
        &decoded.dual_vertices,
        &decoded.intersected,
        grid_size,
        options.decode.aabb,
    );
    if extracted.faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "o_voxel ovox2glb reconstruction produced no triangles; check the grid_size/aabb \
             settings",
        ));
    }

    // `to_glb` first fills small holes, then builds the BVH used for the bake
    // snap-back from that filled mesh (`postprocess.py:110-122`).
    let extracted_faces: Vec<u32> = extracted.faces.iter().map(|&index| index as u32).collect();
    let (pre_vertices, pre_faces) = crate::geometry::fill_holes_with_centroids(
        &extracted.vertices,
        &extracted_faces,
        options.fill_hole_perimeter,
    );
    let original_surface = crate::geometry::MeshBvh::build(&pre_vertices, &pre_faces)?;

    // Non-remesh `to_glb` branch: simplify 3x, cleanup, simplify to target,
    // cleanup, unify orientations (`postprocess.py:134-162`).
    let postprocessed = crate::geometry::postprocess_mesh(
        &pre_vertices,
        &pre_faces,
        options.fill_hole_perimeter,
        options.decimation_target,
        options.min_component_area,
    );
    if postprocessed.faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "o_voxel ovox2glb post-processing removed every triangle",
        ));
    }
    let vertices: Vec<[f32; 3]> = postprocessed
        .vertices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    let atlas =
        crate::geometry::uv_unwrap(&vertices, &postprocessed.faces, options.texture_size as u32)?;
    // Official order (`postprocess.py:215-216`): compute vertex normals on the
    // PRE-unwrap mesh, then gather them through the atlas vertex map. Computing
    // area-weighted normals on `atlas.vertices/atlas.indices` instead would give
    // seam-duplicated vertices only the normals of their own chart, producing
    // lighting seams at chart boundaries.
    let normals = gather_atlas_normals(&vertices, &postprocessed.faces, &atlas.vmapping)?;
    let baked = crate::geometry::bake_pbr_texture_with_snap_in_grid(
        &atlas.vertices,
        &atlas.uvs,
        &atlas.indices,
        &original_surface,
        &decoded.coords,
        pbr6,
        crate::geometry::PbrSampleGrid {
            aabb: options.decode.aabb,
            grid_size: options.decode.grid_size,
        },
        options.texture_size,
    )?;

    let textured = crate::export::TexturedGlbMesh {
        vertices: atlas.vertices.iter().flatten().copied().collect(),
        normals: normals.iter().flatten().copied().collect(),
        uvs: atlas.uvs.iter().flatten().copied().collect(),
        faces: atlas.indices.clone(),
        base_color_rgba: baked.base_color_rgba,
        metallic_roughness: baked.metallic_roughness,
        texture_size: options.texture_size,
        alpha_mode: crate::export::TexturedAlphaMode::Opaque,
        double_sided: true,
        already_gltf_axes: false,
    };
    crate::export::write_textured_glb(&textured, output)?;
    let output_bytes = std::fs::metadata(output)
        .map(|metadata| metadata.len())
        .unwrap_or(0);

    Ok(OvoxToGlbStats {
        voxel_count: decoded.coords.len(),
        extracted_vertices: extracted.vertices.len() / 3,
        extracted_faces: extracted.faces.len() / 3,
        postprocessed_vertices: postprocessed.vertices.len() / 3,
        postprocessed_faces: postprocessed.faces.len() / 3,
        atlas_vertices: atlas.vertices.len(),
        atlas_faces: atlas.indices.len() / 3,
        chart_count: atlas.chart_count,
        texture_size: options.texture_size,
        valid_texels: baked.stats.valid_texels,
        double_sided: true,
        output_bytes,
    })
}

fn vertex_bounds(vertices: &[f32]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices.chunks_exact(3) {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    if vertices.is_empty() {
        ([0.0; 3], [0.0; 3])
    } else {
        (min, max)
    }
}

/// Area-weighted vertex normals on the pre-unwrap topology, gathered through
/// the UV atlas `vmapping` (`normals = mesh.read_vertex_normals()[vmaps]`,
/// `postprocess.py:215-216`).
fn gather_atlas_normals(
    vertices: &[[f32; 3]],
    faces: &[u32],
    vmapping: &[u32],
) -> Result<Vec<[f32; 3]>> {
    let pre_unwrap = crate::geometry::area_weighted_vertex_normals(vertices, faces);
    let mut normals = Vec::new();
    normals.try_reserve_exact(vmapping.len()).map_err(|_| {
        Lux3dError::CanonicalWeightsValidation {
            message: "o_voxel ovox2glb: out of memory while gathering atlas normals".to_string(),
        }
    })?;
    for &original in vmapping {
        let normal = pre_unwrap.get(original as usize).copied().ok_or_else(|| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "o_voxel ovox2glb: UV atlas vmapping points at missing vertex {original} \
                     (pre-unwrap mesh has {} vertices)",
                    pre_unwrap.len()
                ),
            }
        })?;
        normals.push(normal);
    }
    Ok(normals)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube welded to 8 shared vertices and 12 outward triangles. The
    /// hard 90-degree edges force the UV unwrap to duplicate seam vertices.
    fn welded_cube() -> (Vec<[f32; 3]>, Vec<u32>) {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let faces = vec![
            0, 2, 1, 0, 3, 2, // bottom (-z)
            4, 5, 6, 4, 6, 7, // top (+z)
            0, 1, 5, 0, 5, 4, // front (-y)
            2, 3, 7, 2, 7, 6, // back (+y)
            0, 4, 7, 0, 7, 3, // left (-x)
            1, 2, 6, 1, 6, 5, // right (+x)
        ];
        (vertices, faces)
    }

    #[test]
    fn gathered_atlas_normals_use_pre_unwrap_topology_at_seams() {
        let (vertices, faces) = welded_cube();
        let pre_unwrap = crate::geometry::area_weighted_vertex_normals(&vertices, &faces);
        let atlas = crate::geometry::uv_unwrap(&vertices, &faces, 64).expect("uv unwrap");
        assert!(
            atlas.vmapping.len() > vertices.len(),
            "the cube's hard edges must split atlas vertices (atlas {} vs source {})",
            atlas.vmapping.len(),
            vertices.len()
        );

        let gathered =
            gather_atlas_normals(&vertices, &faces, &atlas.vmapping).expect("gather normals");
        assert_eq!(gathered.len(), atlas.vmapping.len());
        for (index, &original) in atlas.vmapping.iter().enumerate() {
            assert_eq!(
                gathered[index], pre_unwrap[original as usize],
                "atlas vertex {index} must reuse the pre-unwrap normal of source vertex {original}"
            );
        }

        // Every atlas copy of an original vertex shares one normal, and at
        // least one copy differs from the chart-local area-weighted normal
        // (which sees only its own face), i.e. the old computation would have
        // produced a seam.
        let mut duplicates = 0usize;
        let chart_normals =
            crate::geometry::area_weighted_vertex_normals(&atlas.vertices, &atlas.indices);
        let mut differs = false;
        for original in 0..vertices.len() as u32 {
            let copies: Vec<usize> = atlas
                .vmapping
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| (value == original).then_some(index))
                .collect();
            if copies.len() < 2 {
                continue;
            }
            duplicates += 1;
            for &copy in &copies {
                assert_eq!(gathered[copy], pre_unwrap[original as usize]);
                if chart_normals[copy] != gathered[copy] {
                    differs = true;
                }
            }
        }
        assert!(duplicates > 0, "expected seam-duplicated source vertices");
        assert!(
            differs,
            "chart-local normals should disagree with the gathered normal at some seam copy"
        );
    }

    #[test]
    fn gathered_atlas_normals_reject_out_of_range_vmapping() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![0, 1, 2];
        let error = gather_atlas_normals(&vertices, &faces, &[0, 1, 2, 3])
            .expect_err("vmapping index 3 is out of range");
        assert!(error.to_string().contains("missing vertex"));
    }
}
