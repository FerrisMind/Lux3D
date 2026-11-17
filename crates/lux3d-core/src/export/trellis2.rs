//! TRELLIS.2 export stage: dual-grid mesh writer with optional vertex
//! colors, the PBR voxel sidecar, and the upstream-parity textured GLB.
//!
//! OBJ output stays positions-only (the shared OBJ writer has no color
//! surface; the reference exports textured OBJs via UV atlases, out of
//! scope). PLY/GLB carry the per-vertex base colors sampled by the texture
//! stage when present. The `<stem>.pbrvox.bin` sidecar serializes the
//! sparse PBR voxel grid (magic `L3DPBR1\0`, u32 version, u32 resolution,
//! u64 voxel count, then one little-endian record per active voxel: 3
//! packed i32 coords + 6 f32 attributes) and the optional
//! `<stem>.pbr.json` describes the layout and value ranges.
//!
//! Textured GLB (upstream `to_glb`): when the pipeline's texture-export
//! stage ran (`Trellis2Stages::textured`), [`Trellis2ExportStage::
//! write_textured_glb`] writes a glTF 2.0 binary with POSITION/NORMAL/
//! TEXCOORD_0 + u32 indices, embedded PNG base-color / metallic-roughness
//! textures and the reference PBR material (`metallicFactor = 1`,
//! `roughnessFactor = 1`, `alphaMode = "OPAQUE"`, `doubleSided = false` for
//! remeshed meshes and `true` otherwise, `postprocess.py:303`).
//! See `export::glb_textured` for the conversion details (axis rotation
//! `(x, y, z) -> (x, z, -y)` on positions and normals, UV V flip) — the
//! same conversions the reference `to_glb` epilogue applies
//! (`postprocess.py:307-315`).
//!
//! Coordinate frames (upstream `to_glb`, `postprocess.py:306-315`): the GLB
//! writer applies the upstream glTF-only axis conversion
//! `(x, y, z) -> (x, z, -y)` (swap Y/Z, invert Y) to the vertex positions
//! via [`to_glb_axes`]; vertex colors pass through unchanged (a per-vertex
//! attribute) and face winding stays valid (the transform is a rotation
//! with determinant +1, so the index buffer is unchanged). OBJ/PLY keep the
//! decoded reference `[-0.5, 0.5]^3` frame with no axis remap. The shared
//! GLB writer emits no normals (clients compute flat normals), so the
//! upstream normal conversion has no counterpart there — the textured GLB
//! writer DOES convert the baked area-weighted normals. OBJ formatting
//! follows the shared trimesh style (8 decimals, 1-based face indices).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::{
    contracts::{ExportPlan, ExportStage, Trellis2Mesh},
    error::Lux3dError,
    geometry::Trellis2MeshBuffers,
    runtime::Trellis2TexturedMesh,
    Result,
};

use super::mesh::{write_mesh, write_obj, MeshFormat, MeshView, ObjStyle};

/// PBR voxel sidecar constants (`L3DPBR1\0` magic, format version 1).
pub mod pbrvox {
    /// File magic: `L3DPBR1\0` (8 bytes).
    pub const MAGIC: [u8; 8] = *b"L3DPBR1\0";
    /// Sidecar format version.
    pub const VERSION: u32 = 1;
    /// Per-record attribute channel count (base_color 3 + metallic +
    /// roughness + alpha).
    pub const CHANNELS: usize = 6;
}

/// Export-stage descriptor for the TRELLIS.2 family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Trellis2ExportStage;

impl ExportStage<Trellis2Mesh> for Trellis2ExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &Trellis2Mesh) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "glb".to_string(),
            alternate_extensions: vec!["obj".to_string(), "ply".to_string()],
            utility_notes: vec![
                "GLB is the primary artifact; format follows the output extension".to_string(),
                "OBJ: v lines carry positions only with 8 decimals; f indices are 1-based"
                    .to_string(),
                "PLY/GLB: per-vertex colors sampled from the decoded PBR attributes".to_string(),
                "GLB: upstream glTF axis conversion (x, y, z) -> (x, z, -y) applied on export"
                    .to_string(),
                "textured GLB (upstream parity): POSITION/NORMAL/TEXCOORD_0 + embedded PNG \
                 base-color and metallic-roughness textures, alphaMode OPAQUE, doubleSided"
                    .to_string(),
                "OBJ/PLY: reference coordinate frame [-0.5, 0.5]^3 as decoded, no axis remap"
                    .to_string(),
                "optional <stem>.pbrvox.bin sidecar + <stem>.pbr.json layout descriptor"
                    .to_string(),
            ],
        })
    }
}

/// The upstream GLB-only axis conversion (`to_glb`,
/// `postprocess.py:313`): swap Y/Z and invert Y, i.e.
/// `(x, y, z) -> (x, z, -y)` — as a matrix `[[1,0,0],[0,0,1],[0,-1,0]]`, a
/// proper rotation with determinant +1, so triangle winding stays valid and
/// the face index buffer is unchanged.
///
/// Only the GLB writer applies this; OBJ/PLY keep the decoded reference
/// frame. Vertex colors are per-vertex attributes and pass through
/// unchanged.
pub fn to_glb_axes(vertices: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(vertices.len());
    for vertex in vertices.chunks_exact(3) {
        out.push(vertex[0]);
        out.push(vertex[2]);
        out.push(-vertex[1]);
    }
    // Defensive: keep any trailing partial vertex untouched (mesh buffers
    // are validated to be vertex triples before reaching this point).
    let converted = vertices.len() - vertices.len() % 3;
    out.extend_from_slice(&vertices[converted..]);
    out
}

/// Shared sidecar guard: the payload must carry the 6 PBR channels and one
/// coordinate row per attribute row (the header `count` is derived from the
/// attributes, so a mismatched coordinate buffer would corrupt the records).
fn validate_pbr_payload(mesh: &Trellis2MeshBuffers) -> Result<()> {
    mesh.validate()?;
    if mesh.voxel_attr_channels != pbrvox::CHANNELS {
        return Err(Lux3dError::ModelAssetResolution {
            message: format!(
                "PBR sidecar needs {} attribute channels, mesh carries {}",
                pbrvox::CHANNELS,
                mesh.voxel_attr_channels
            ),
        });
    }
    if mesh.voxel_coords.len() / 3 != mesh.voxel_count() {
        return Err(Lux3dError::ModelAssetResolution {
            message: format!(
                "PBR sidecar coord count {} does not match the attribute row count {}",
                mesh.voxel_coords.len() / 3,
                mesh.voxel_count()
            ),
        });
    }
    Ok(())
}

impl Trellis2ExportStage {
    /// Write the textured atlas mesh as a glTF 2.0 binary (`.glb`) with
    /// embedded base-color / metallic-roughness PNG textures (upstream
    /// `to_glb` textured export). Applies the upstream glTF axis conversion
    /// `(x, y, z) -> (x, z, -y)` to positions AND normals and the UV V flip
    /// `v' = 1 - v` (the baked textures use the nvdiffrast row-0 = v = 1
    /// layout). Material: `metallicFactor = 1`, `roughnessFactor = 1`,
    /// `alphaMode = "OPAQUE"`, and `doubleSided = double_sided` — the
    /// upstream rule is `false` for remeshed meshes, `true` otherwise
    /// (`postprocess.py:303`). The plain [`Self::write_mesh`] behavior for
    /// non-textured exports is unchanged.
    pub fn write_textured_glb(
        &self,
        textured: &Trellis2TexturedMesh,
        path: &Path,
        double_sided: bool,
    ) -> Result<()> {
        let glb_mesh = super::glb_textured::TexturedGlbMesh::from_textured_mesh(
            textured,
            super::glb_textured::TexturedAlphaMode::Opaque,
            double_sided,
        );
        super::glb_textured::write_textured_glb(&glb_mesh, path)
    }

    /// Write the mesh buffers in the format selected by `path`'s extension.
    /// PLY/GLB include the sampled vertex colors when the mesh carries
    /// them; OBJ stays positions-only. GLB output applies the upstream
    /// `to_glb` axis conversion ([`to_glb_axes`]); OBJ/PLY keep the decoded
    /// reference frame.
    pub fn write_mesh(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<MeshFormat> {
        mesh.validate()?;
        let format = MeshFormat::from_path(path)?;
        let glb_vertices;
        let view = match format {
            MeshFormat::Glb => {
                glb_vertices = to_glb_axes(&mesh.vertices);
                MeshView {
                    vertices: &glb_vertices,
                    faces: &mesh.faces,
                    colors: mesh.vertex_colors.as_deref(),
                }
            }
            _ => MeshView {
                vertices: &mesh.vertices,
                faces: &mesh.faces,
                colors: mesh.vertex_colors.as_deref(),
            },
        };
        write_mesh(&view, path, ObjStyle::TRIPOSG)
    }

    /// Write the mesh buffers as UTF-8 ASCII OBJ (positions only).
    pub fn write_obj(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<()> {
        mesh.validate()?;
        let view = MeshView {
            vertices: &mesh.vertices,
            faces: &mesh.faces,
            colors: None,
        };
        write_obj(&view, path, ObjStyle::TRIPOSG)
    }

    /// Write the mesh plus the PBR voxel sidecar pair:
    /// `<stem>.pbrvox.bin` (sparse grid records) and `<stem>.pbr.json`
    /// (layout descriptor). Returns the sidecar path.
    pub fn write_mesh_with_pbr(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<PathBuf> {
        self.write_mesh(mesh, path)?;
        let stem = path.parent().unwrap_or_else(|| Path::new(".")).join(
            path.file_stem()
                .unwrap_or_else(|| std::ffi::OsStr::new("mesh")),
        );
        self.write_pbr_vox(mesh, &stem)
    }

    /// Write the `<stem>.pbrvox.bin` + `<stem>.pbr.json` sidecar pair for
    /// the mesh's PBR voxel payload. Fails when the mesh does not carry the
    /// 6-channel PBR attributes (geometry-only runs export the mesh alone).
    pub fn write_pbr_vox(&self, mesh: &Trellis2MeshBuffers, stem: &Path) -> Result<PathBuf> {
        self.write_pbrvox_bin(mesh, &stem.with_extension("pbrvox.bin"))?;
        self.write_pbr_json(mesh, &stem.with_extension("pbr.json"))?;
        Ok(stem.with_extension("pbrvox.bin"))
    }

    /// The binary PBR voxel sidecar. Layout (little-endian throughout):
    /// magic `L3DPBR1\0`, u32 version 1, u32 resolution, u64 voxel count,
    /// then `count` records of 3 packed i32 coords + 6 f32 attributes.
    pub fn write_pbrvox_bin(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<()> {
        validate_pbr_payload(mesh)?;
        let count = mesh.voxel_count();
        let file = File::create(path).map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&pbrvox::MAGIC).map_err(pbr_io(path))?;
        writer
            .write_all(&pbrvox::VERSION.to_le_bytes())
            .map_err(pbr_io(path))?;
        writer
            .write_all(&mesh.resolution.to_le_bytes())
            .map_err(pbr_io(path))?;
        writer
            .write_all(&(count as u64).to_le_bytes())
            .map_err(pbr_io(path))?;
        for voxel in 0..count {
            let coord = voxel * 3;
            for axis in 0..3 {
                writer
                    .write_all(&mesh.voxel_coords[coord + axis].to_le_bytes())
                    .map_err(pbr_io(path))?;
            }
            let attr = voxel * pbrvox::CHANNELS;
            for channel in 0..pbrvox::CHANNELS {
                writer
                    .write_all(&mesh.voxel_attrs[attr + channel].to_le_bytes())
                    .map_err(pbr_io(path))?;
            }
        }
        writer.flush().map_err(pbr_io(path))?;
        Ok(())
    }

    /// The JSON sidecar describing the PBR voxel layout and observed value
    /// ranges (min/max per channel over the active voxels, or `null` for an
    /// empty grid).
    pub fn write_pbr_json(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<()> {
        validate_pbr_payload(mesh)?;
        let count = mesh.voxel_count();
        let mut ranges: [Option<(f32, f32)>; pbrvox::CHANNELS] = Default::default();
        for voxel in 0..count {
            let attr = voxel * pbrvox::CHANNELS;
            for (channel, range) in ranges.iter_mut().enumerate() {
                let value = mesh.voxel_attrs[attr + channel];
                match range {
                    Some((min, max)) => {
                        *min = min.min(value);
                        *max = max.max(value);
                    }
                    None => *range = Some((value, value)),
                }
            }
        }
        // Debug formatting keeps the decimal point ("0.0"), so every range
        // entry parses as a JSON float.
        let range_json = |range: &Option<(f32, f32)>| match range {
            Some((min, max)) => format!("[{min:?}, {max:?}]"),
            None => "null".to_string(),
        };
        let json = format!(
            "{{\n  \"format\": \"pbrvox\",\n  \"version\": {},\n  \"resolution\": {},\n  \
             \"voxel_count\": {},\n  \"channels\": {},\n  \"layout\": {{\n    \
             \"base_color\": [0, 3],\n    \"metallic\": 3,\n    \"roughness\": 4,\n    \
             \"alpha\": 5\n  }},\n  \"ranges\": [\n    {},\n    {},\n    {},\n    {},\n    \
             {},\n    {}\n  ]\n}}\n",
            pbrvox::VERSION,
            mesh.resolution,
            count,
            pbrvox::CHANNELS,
            range_json(&ranges[0]),
            range_json(&ranges[1]),
            range_json(&ranges[2]),
            range_json(&ranges[3]),
            range_json(&ranges[4]),
            range_json(&ranges[5]),
        );
        std::fs::write(path, json).map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })
    }
}

/// Closure factory for the sidecar IO error mapping.
fn pbr_io(path: &Path) -> impl Fn(std::io::Error) -> Lux3dError + '_ {
    move |source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Trellis2MeshBuffers;
    use crate::runtime::{build_trellis2_textured_mesh, Trellis2TextureExportOptions};

    /// Synthetic textured cube: reference-frame `[-0.5, 0.5]^3` unit cube
    /// (12 triangles, outward winding) over a full 2x2x2 PBR voxel grid at
    /// resolution 2 (each voxel 6 channels: base color ramps per axis in
    /// `[0.5, 1]`, metallic 0.5, roughness 0.25, alpha 1). Vertex colors are
    /// attached with the reference sparse trilinear, exactly like
    /// `attach_texture` does.
    fn textured_cube_buffers() -> Trellis2MeshBuffers {
        let vertices: Vec<f32> = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
        .iter()
        .flatten()
        .copied()
        .collect();
        let faces: Vec<i64> = [
            [0, 2, 1],
            [0, 3, 2], // bottom (-z)
            [4, 5, 6],
            [4, 6, 7], // top (+z)
            [0, 1, 5],
            [0, 5, 4], // front (-y)
            [2, 3, 7],
            [2, 7, 6], // back (+y)
            [0, 4, 7],
            [0, 7, 3], // left (-x)
            [1, 2, 6],
            [1, 6, 5], // right (+x)
        ]
        .iter()
        .flatten()
        .map(|&index| index as i64)
        .collect();
        let mut coords = Vec::new();
        let mut attrs = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    coords.push([x, y, z]);
                    attrs.extend([
                        0.5 + 0.5 * x as f32,
                        0.5 + 0.5 * y as f32,
                        0.5 + 0.5 * z as f32,
                        0.5,  // metallic
                        0.25, // roughness
                        1.0,  // alpha
                    ]);
                }
            }
        }
        let vertex_rows: Vec<[f32; 3]> = vertices
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        let coord_rows = coords.clone();
        let vertex_colors =
            crate::runtime::query_vertex_colors(&vertex_rows, &coord_rows, &attrs, 2)
                .expect("vertex colors");
        Trellis2MeshBuffers {
            resolution: 2,
            vertices,
            faces,
            voxel_coords: coords.iter().flatten().copied().collect(),
            voxel_attrs: attrs,
            voxel_attr_channels: 6,
            vertex_colors: Some(vertex_colors),
        }
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "lux3d-textured-cube-{}-{label}-{nonce}.glb",
            std::process::id()
        ))
    }

    /// Full atlas -> bake -> textured GLB integration: the synthetic cube
    /// runs through `build_trellis2_textured_mesh` and
    /// `Trellis2ExportStage::write_textured_glb`, and the GLB is parsed back
    /// (header, JSON, embedded PNGs) with material and buffer assertions.
    #[test]
    fn textured_cube_runs_atlas_bake_glb_pipeline() {
        let buffers = textured_cube_buffers();
        buffers.validate().expect("fixture valid");
        let options = Trellis2TextureExportOptions {
            enabled: true,
            texture_size: 256,
            uv_unwrap: true,
        };
        let textured = build_trellis2_textured_mesh(&buffers, &options)
            .expect("build succeeds")
            .expect("PBR payload present, so the textured mesh is built");

        // Atlas products: charts exist, UVs are in range, the atlas mesh is
        // index-valid and keeps every cube face, normals are unit length.
        assert!(textured.chart_count > 0);
        assert_eq!(256, textured.atlas_size);
        let atlas_vertices = textured.uvs.len() / 2;
        assert!(atlas_vertices >= 8, "seams duplicate vertices");
        assert_eq!(0, textured.faces.len() % 3);
        assert!(textured.faces.len() >= 36, "all 12 cube faces kept");
        for uv in textured.uvs.chunks_exact(2) {
            assert!(uv[0] >= 0.0 && uv[0] <= 1.0);
            assert!(uv[1] >= 0.0 && uv[1] <= 1.0);
        }
        for &index in &textured.faces {
            assert!((index as usize) < atlas_vertices);
        }
        for normal in textured.normals.chunks_exact(3) {
            let length =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-4 || length == 0.0);
        }

        // Baked textures: metallic/roughness planes are the constant voxel
        // values everywhere (valid AND inpainted texels), base color has no
        // black seams (every channel >= 0.5 * 255 - 1).
        let size = textured.texture.size;
        assert_eq!(256, size);
        assert!(textured.texture.stats.valid_texels > 0);
        for texel in 0..size * size {
            let mr = &textured.texture.metallic_roughness[texel * 4..texel * 4 + 4];
            assert_eq!([0, 63, 127, 255], mr, "mr texel {texel}");
            let rgba = &textured.texture.base_color_rgba[texel * 4..texel * 4 + 4];
            for channel in 0..3 {
                assert!(
                    rgba[channel] >= 126,
                    "base color texel {texel} channel {channel} = {} (black seam?)",
                    rgba[channel]
                );
            }
            assert_eq!(255, rgba[3], "alpha is 1.0 everywhere");
        }

        // Export and parse back.
        let stage = Trellis2ExportStage;
        let path = temp_path("cube");
        stage
            .write_textured_glb(&textured, &path, true)
            .expect("write glb");
        let bytes = std::fs::read(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);

        assert_eq!(&bytes[..4], b"glTF");
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let json: serde_json::Value =
            serde_json::from_slice(&bytes[20..20 + json_len]).expect("glb json");
        assert_eq!(json["images"].as_array().unwrap().len(), 2);
        assert_eq!(json["textures"].as_array().unwrap().len(), 2);
        assert_eq!(
            json["materials"][0]["pbrMetallicRoughness"]["metallicFactor"],
            1.0
        );
        assert_eq!(
            json["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"],
            1.0
        );
        assert_eq!(json["materials"][0]["alphaMode"], "OPAQUE");
        assert_eq!(json["materials"][0]["doubleSided"], true);
        let attributes = &json["meshes"][0]["primitives"][0]["attributes"];
        assert_eq!(attributes["POSITION"], 0);
        assert_eq!(attributes["NORMAL"], 1);
        assert_eq!(attributes["TEXCOORD_0"], 2);
        assert_eq!(json["meshes"][0]["primitives"][0]["indices"], 3);

        // Decode the embedded base-color PNG: 256x256, and its texels equal
        // the baked buffer row 0 (the writer flips UV V, not the image).
        let bin_offset = 20 + json_len + 8;
        let bin_len = u32::from_le_bytes(bytes[bin_offset - 8..bin_offset - 4].try_into().unwrap());
        let bin = &bytes[bin_offset..bin_offset + bin_len as usize];
        let image0 = &json["images"][0];
        let view = image0["bufferView"].as_u64().unwrap() as usize;
        let byte_offset = json["bufferViews"][view]["byteOffset"].as_u64().unwrap() as usize;
        let byte_length = json["bufferViews"][view]["byteLength"].as_u64().unwrap() as usize;
        let decoded = image::load_from_memory(&bin[byte_offset..byte_offset + byte_length])
            .expect("decode embedded png");
        assert_eq!((256, 256), (decoded.width(), decoded.height()));
        let rgba = decoded.to_rgba8();
        assert_eq!(
            textured.texture.base_color_rgba[0..4],
            rgba.get_pixel(0, 0).0
        );
    }

    /// Clean skips (no error, `None`): disabled options, disabled unwrap,
    /// non-PBR channel counts, and geometry-only payloads.
    #[test]
    fn textured_export_skips_cleanly() {
        let buffers = textured_cube_buffers();
        let base = Trellis2TextureExportOptions::default();

        let disabled = Trellis2TextureExportOptions {
            enabled: false,
            ..base.clone()
        };
        assert_eq!(
            None,
            build_trellis2_textured_mesh(&buffers, &disabled).expect("skip")
        );

        let no_unwrap = Trellis2TextureExportOptions {
            uv_unwrap: false,
            ..base.clone()
        };
        assert_eq!(
            None,
            build_trellis2_textured_mesh(&buffers, &no_unwrap).expect("skip")
        );

        let mut shape_channels = buffers.clone();
        shape_channels.voxel_attr_channels = 7;
        assert_eq!(
            None,
            build_trellis2_textured_mesh(&shape_channels, &base).expect("skip")
        );

        let mut geometry_only = buffers.clone();
        geometry_only.voxel_attr_channels = 0;
        geometry_only.voxel_attrs.clear();
        geometry_only.vertex_colors = None;
        assert_eq!(
            None,
            build_trellis2_textured_mesh(&geometry_only, &base).expect("skip")
        );

        let mut empty = buffers;
        empty.faces.clear();
        assert_eq!(
            None,
            build_trellis2_textured_mesh(&empty, &base).expect("skip")
        );
    }

    /// Invalid texture sizes fail validation before any work happens.
    #[test]
    fn texture_export_options_validate_size() {
        let base = Trellis2TextureExportOptions::default();
        base.validate().expect("defaults valid");
        assert_eq!(2048, base.texture_size);
        assert!(base.enabled);
        assert!(base.uv_unwrap);

        for size in [0usize, 4, 12, 2047, 8200] {
            let options = Trellis2TextureExportOptions {
                texture_size: size,
                ..base.clone()
            };
            assert!(options.validate().is_err(), "size {size} must be rejected");
            let mut disabled = options;
            disabled.enabled = false;
            disabled.validate().expect("disabled skips checks");
        }
    }
}
