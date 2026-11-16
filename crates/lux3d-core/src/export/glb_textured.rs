//! Textured glTF 2.0 binary (`.glb`) writer: the upstream-parity TRELLIS.2
//! textured export (`to_glb`, `postprocess.py:272-323`).
//!
//! One mesh, one triangle primitive with `POSITION` + `NORMAL` +
//! `TEXCOORD_0` + u32 indices, and a single `pbrMetallicRoughness` material
//! referencing two embedded PNG textures:
//!
//! * `baseColorTexture` — RGBA PNG (`base_color_rgba`),
//! * `metallicRoughnessTexture` — PNG packed `[0, roughness, metallic, 255]`
//!   (glTF channel convention: G = roughness, B = metallic),
//! * `metallicFactor = 1.0`, `roughnessFactor = 1.0`,
//! * `alphaMode = "OPAQUE"` (default; `"MASK"` with cutoff and `"BLEND"`
//!   supported), `doubleSided = true` (the reference
//!   `PBRMaterial(..., doubleSided=True)` for the non-remeshed path).
//!
//! Upstream export-time conversions applied HERE (not upstream of the
//! writer), exactly like the reference `to_glb` epilogue:
//!
//! * positions AND normals go through the glTF axis conversion
//!   `(x, y, z) -> (x, z, -y)` ([`to_glb_axes`]; a proper rotation with
//!   determinant +1, so triangle winding is unchanged) unless
//!   [`TexturedGlbMesh::already_gltf_axes`] is set,
//! * UVs get the V flip `v' = 1 - v` ALWAYS: the baked texture images use
//!   the nvdiffrast image layout (row 0 = `v = 1`, see
//!   `geometry::pbr_bake`), while glTF samples `v' = 0` at the image top —
//!   the same flip the reference applies to its exported UVs.
//!
//! Binary layout: tightly packed f32 `POSITION` / `NORMAL` /
//! `TEXCOORD_0` buffer views (with explicit `byteStride`), a u32 index
//! buffer view (34-byte-aligned accessors: every accessor byteOffset is a
//! multiple of 4), then the two PNG payload buffer views (4-byte aligned,
//! `mimeType: "image/png"`). Samplers clamp to the edge
//! (`CLAMP_TO_EDGE`, atlas requirement) with linear filtering.

use std::io::Cursor;
use std::path::Path;

use serde_json::json;

use super::mesh::write_glb_container;
use super::trellis2::to_glb_axes;
use crate::{Result, error::Lux3dError};

/// glTF material alpha mode for the textured export.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TexturedAlphaMode {
    /// glTF `OPAQUE` (the reference `alphaMode = 'OPAQUE'`).
    #[default]
    Opaque,
    /// glTF `MASK` with the given `alphaCutoff` in [0, 1].
    Mask(f32),
    /// glTF `BLEND`.
    Blend,
}

impl TexturedAlphaMode {
    /// glTF alphaMode string.
    pub const fn gltf_name(self) -> &'static str {
        match self {
            Self::Opaque => "OPAQUE",
            Self::Mask(_) => "MASK",
            Self::Blend => "BLEND",
        }
    }

    /// glTF `alphaCutoff` (only for `MASK`).
    pub const fn alpha_cutoff(self) -> Option<f32> {
        match self {
            Self::Mask(cutoff) => Some(cutoff),
            _ => None,
        }
    }
}

/// Textured triangle mesh ready for the GLB writer. Buffers live in the
/// decoded REFERENCE frame (`[-0.5, 0.5]^3` positions + matching normals,
/// atlas UVs with row 0 at `v = 1`) — the writer applies the upstream glTF
/// axis conversion and the UV V flip. Set `already_gltf_axes = true` to skip
/// the axis conversion (UVs are always V-flipped; pre-flip them and this
/// writer is not for you).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TexturedGlbMesh {
    /// Flat `[x, y, z, ...]` positions, reference frame.
    pub vertices: Vec<f32>,
    /// Flat `[nx, ny, nz, ...]` normals, reference frame.
    pub normals: Vec<f32>,
    /// Flat `[u, v, ...]` atlas UVs in `[0, 1]` (pre V-flip).
    pub uvs: Vec<f32>,
    /// Flat zero-based triangle indices (u32).
    pub faces: Vec<u32>,
    /// RGBA base color texture, `texture_size * texture_size * 4` bytes.
    pub base_color_rgba: Vec<u8>,
    /// `[0, roughness, metallic, 255]` texture, `texture_size * texture_size
    /// * 4` bytes.
    pub metallic_roughness: Vec<u8>,
    /// Texture edge length (`size * size` texels).
    pub texture_size: usize,
    /// Material alpha mode (glTF `OPAQUE` default).
    pub alpha_mode: TexturedAlphaMode,
    /// Skip the glTF axis conversion (positions/normals already converted).
    pub already_gltf_axes: bool,
}

impl TexturedGlbMesh {
    /// Build the writer input from a pipeline [`Trellis2TexturedMesh`]
    /// (clones the buffers — the textured mesh stays reusable; the copies
    /// are transient, roughly `vertices + normals + uvs` bytes).
    pub fn from_textured_mesh(
        mesh: &crate::runtime::Trellis2TexturedMesh,
        alpha_mode: TexturedAlphaMode,
    ) -> Self {
        Self {
            vertices: mesh.vertices.clone(),
            normals: mesh.normals.clone(),
            uvs: mesh.uvs.clone(),
            faces: mesh.faces.clone(),
            base_color_rgba: mesh.texture.base_color_rgba.clone(),
            metallic_roughness: mesh.texture.metallic_roughness.clone(),
            texture_size: mesh.texture.size,
            alpha_mode,
            already_gltf_axes: false,
        }
    }
}

/// Write a textured binary glTF 2.0 (`.glb`) file.
pub fn write_textured_glb(mesh: &TexturedGlbMesh, path: &Path) -> Result<()> {
    validate(mesh)?;

    // Upstream export conversions: axis rotation (unless already applied)
    // and the UV V flip (always — the baked images use the nvdiffrast row
    // layout). Winding stays valid (determinant +1).
    let (positions, normals) = if mesh.already_gltf_axes {
        (mesh.vertices.clone(), mesh.normals.clone())
    } else {
        (to_glb_axes(&mesh.vertices), to_glb_axes(&mesh.normals))
    };
    let texcoords: Vec<f32> = mesh
        .uvs
        .chunks_exact(2)
        .flat_map(|uv| [uv[0], 1.0 - uv[1]])
        .collect();

    let vertex_count = mesh.vertices.len() / 3;
    let face_count = mesh.faces.len() / 3;
    let size = mesh.texture_size;

    let positions_png = encode_png(&mesh.base_color_rgba, size, "base color")?;
    let metallic_roughness_png = encode_png(&mesh.metallic_roughness, size, "metallic roughness")?;

    let mut binary = Vec::with_capacity(
        vertex_count * 12 * 2
            + vertex_count * 8
            + face_count * 12
            + positions_png.len()
            + metallic_roughness_png.len()
            + 8,
    );
    let positions_offset = 0;
    binary.extend_from_slice(&f32_slice_to_le(&positions));
    let normals_offset = binary.len();
    binary.extend_from_slice(&f32_slice_to_le(&normals));
    let texcoords_offset = binary.len();
    binary.extend_from_slice(&f32_slice_to_le(&texcoords));
    let indices_offset = binary.len();
    for index in &mesh.faces {
        binary.extend_from_slice(&index.to_le_bytes());
    }
    let base_color_offset = align4(binary.len());
    binary.resize(base_color_offset, 0);
    binary.extend_from_slice(&positions_png);
    let metallic_roughness_offset = align4(binary.len());
    binary.resize(metallic_roughness_offset, 0);
    binary.extend_from_slice(&metallic_roughness_png);
    let binary_len = align4(binary.len());
    binary.resize(binary_len, 0);

    let (min, max) = position_bounds(&positions);
    let buffer_views = vec![
        json!({
            "buffer": 0,
            "byteOffset": positions_offset,
            "byteLength": vertex_count * 12,
            "byteStride": 12,
            "target": 34962,
        }),
        json!({
            "buffer": 0,
            "byteOffset": normals_offset,
            "byteLength": vertex_count * 12,
            "byteStride": 12,
            "target": 34962,
        }),
        json!({
            "buffer": 0,
            "byteOffset": texcoords_offset,
            "byteLength": vertex_count * 8,
            "byteStride": 8,
            "target": 34962,
        }),
        json!({
            "buffer": 0,
            "byteOffset": indices_offset,
            "byteLength": face_count * 12,
            "target": 34963,
        }),
        json!({
            "buffer": 0,
            "byteOffset": base_color_offset,
            "byteLength": positions_png.len(),
        }),
        json!({
            "buffer": 0,
            "byteOffset": metallic_roughness_offset,
            "byteLength": metallic_roughness_png.len(),
        }),
    ];
    let accessors = vec![
        json!({
            "bufferView": 0,
            "componentType": 5126,
            "count": vertex_count,
            "type": "VEC3",
            "min": [min[0], min[1], min[2]],
            "max": [max[0], max[1], max[2]],
        }),
        json!({
            "bufferView": 1,
            "componentType": 5126,
            "count": vertex_count,
            "type": "VEC3",
        }),
        json!({
            "bufferView": 2,
            "componentType": 5126,
            "count": vertex_count,
            "type": "VEC2",
        }),
        json!({
            "bufferView": 3,
            "componentType": 5125,
            "count": face_count * 3,
            "type": "SCALAR",
        }),
    ];
    let mut material = json!({
        "name": "trellis2_pbr",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "baseColorTexture": { "index": 0 },
            "metallicRoughnessTexture": { "index": 1 },
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
        },
        "alphaMode": mesh.alpha_mode.gltf_name(),
        "doubleSided": true,
    });
    if let Some(cutoff) = mesh.alpha_mode.alpha_cutoff() {
        material["alphaCutoff"] = json!(cutoff);
    }
    let document = json!({
        "asset": {
            "version": "2.0",
            "generator": "lux3d",
        },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{
            "name": "mesh",
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2,
                },
                "indices": 3,
                "material": 0,
                "mode": 4,
            }],
        }],
        "materials": [material],
        "textures": [
            { "sampler": 0, "source": 0 },
            { "sampler": 0, "source": 1 },
        ],
        "samplers": [{
            "magFilter": 9729,
            "minFilter": 9987,
            "wrapS": 33071,
            "wrapT": 33071,
        }],
        "images": [
            { "bufferView": 4, "mimeType": "image/png" },
            { "bufferView": 5, "mimeType": "image/png" },
        ],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{ "byteLength": binary_len }],
    });
    let json =
        serde_json::to_vec(&document).map_err(|error| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to serialize textured GLB JSON: {error}"),
        })?;
    write_glb_container(path, &json, &binary)
}

/// Structural validation of the writer input (descriptive errors, no
/// panics).
fn validate(mesh: &TexturedGlbMesh) -> Result<()> {
    if mesh.vertices.len() % 3 != 0 || mesh.normals.len() != mesh.vertices.len() {
        return Err(Lux3dError::InvalidInput(
            "textured GLB needs matching flat vertex and normal buffers (3 channels each)",
        ));
    }
    if mesh.uvs.len() != mesh.vertices.len() / 3 * 2 {
        return Err(Lux3dError::InvalidInput(
            "textured GLB UV buffer must hold two channels per vertex",
        ));
    }
    if mesh.faces.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "textured GLB face buffer length must be a multiple of 3",
        ));
    }
    if mesh.faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "textured GLB export requires at least one triangle",
        ));
    }
    let vertex_count = mesh.vertices.len() / 3;
    for value in mesh.vertices.iter().chain(mesh.normals.iter()) {
        if !value.is_finite() {
            return Err(Lux3dError::InvalidInput(
                "textured GLB export rejected a non-finite vertex or normal",
            ));
        }
    }
    for uv in mesh.uvs.chunks_exact(2) {
        if !uv[0].is_finite() || !uv[1].is_finite() {
            return Err(Lux3dError::InvalidInput(
                "textured GLB export rejected a non-finite UV",
            ));
        }
    }
    for &index in &mesh.faces {
        if index as usize >= vertex_count {
            return Err(Lux3dError::InvalidInput(
                "textured GLB export rejected a face index outside the vertex buffer",
            ));
        }
    }
    if mesh.texture_size == 0 || mesh.texture_size > 8192 {
        return Err(Lux3dError::InvalidInput(
            "textured GLB texture_size must be in 1..=8192",
        ));
    }
    let expected = mesh.texture_size * mesh.texture_size * 4;
    if mesh.base_color_rgba.len() != expected || mesh.metallic_roughness.len() != expected {
        return Err(Lux3dError::InvalidInput(
            "textured GLB texture buffers must hold texture_size^2 RGBA texels",
        ));
    }
    if let TexturedAlphaMode::Mask(cutoff) = mesh.alpha_mode {
        if !cutoff.is_finite() || !(0.0..=1.0).contains(&cutoff) {
            return Err(Lux3dError::InvalidInput(
                "textured GLB alpha cutoff must be a finite value in [0, 1]",
            ));
        }
    }
    Ok(())
}

/// Encode a tightly packed RGBA byte buffer as a PNG (the `image` crate
/// encoder; embedded directly into the GLB binary chunk).
fn encode_png(bytes: &[u8], size: usize, label: &str) -> Result<Vec<u8>> {
    let image = image::RgbaImage::from_raw(size as u32, size as u32, bytes.to_vec()).ok_or(
        Lux3dError::InvalidInput("textured GLB texture buffer does not fit size*size RGBA"),
    )?;
    let mut out = Cursor::new(Vec::new());
    image
        .write_to(&mut out, image::ImageFormat::Png)
        .map_err(|error| Lux3dError::CanonicalWeightsValidation {
            message: format!("textured GLB {label} PNG encoding failed: {error}"),
        })?;
    Ok(out.into_inner())
}

/// Flatten f32 rows into little-endian bytes.
fn f32_slice_to_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Round up to the 4-byte alignment every glTF accessor (and, defensively,
/// every image buffer view) needs.
const fn align4(offset: usize) -> usize {
    (offset + 3) & !3
}

fn position_bounds(vertices: &[f32]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices.chunks_exact(3) {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One textured quad (2 triangles) with a flat red base color and
    /// mid-gray roughness/metallic on an 8x8 texture.
    fn fixture() -> TexturedGlbMesh {
        let size = 8usize;
        TexturedGlbMesh {
            vertices: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            normals: [[0.0, 0.0, 1.0]].repeat(4).into_iter().flatten().collect(),
            uvs: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            faces: vec![0, 1, 2, 0, 2, 3],
            base_color_rgba: vec![255, 0, 0, 255].repeat(size * size),
            metallic_roughness: vec![0, 128, 64, 255].repeat(size * size),
            texture_size: size,
            alpha_mode: TexturedAlphaMode::Opaque,
            already_gltf_axes: false,
        }
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "lux3d-textured-glb-{}-{label}-{nonce}.glb",
            std::process::id()
        ))
    }

    /// Parse a GLB file into (json, bin) parts.
    fn parse_glb(bytes: &[u8]) -> (serde_json::Value, &[u8]) {
        assert_eq!(&bytes[..4], b"glTF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
        let total = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(total, bytes.len());
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(
            u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            0x4E4F_534A
        );
        let json: serde_json::Value =
            serde_json::from_slice(&bytes[20..20 + json_len]).expect("glb json");
        let bin_offset = 20 + json_len + 8;
        let bin_len = u32::from_le_bytes(bytes[bin_offset - 8..bin_offset - 4].try_into().unwrap());
        assert_eq!(
            u32::from_le_bytes(bytes[bin_offset - 4..bin_offset].try_into().unwrap()),
            0x004E_4942
        );
        (json, &bytes[bin_offset..bin_offset + bin_len as usize])
    }

    #[test]
    fn textured_glb_carries_material_textures_and_conversions() {
        let mesh = fixture();
        let path = temp_path("full");
        write_textured_glb(&mesh, &path).expect("write textured glb");
        let bytes = std::fs::read(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);
        let (json, bin) = parse_glb(&bytes);

        // Material: upstream PBRMaterial fields.
        let material = &json["materials"][0];
        assert_eq!(material["alphaMode"], "OPAQUE");
        assert_eq!(material["doubleSided"], true);
        assert_eq!(material["pbrMetallicRoughness"]["metallicFactor"], 1.0);
        assert_eq!(material["pbrMetallicRoughness"]["roughnessFactor"], 1.0);
        assert_eq!(
            material["pbrMetallicRoughness"]["baseColorTexture"]["index"],
            0
        );
        assert_eq!(
            material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"],
            1
        );

        // Two textures + two embedded PNG images + one sampler.
        assert_eq!(json["textures"].as_array().unwrap().len(), 2);
        assert_eq!(json["images"].as_array().unwrap().len(), 2);
        for image in json["images"].as_array().unwrap() {
            assert_eq!(image["mimeType"], "image/png");
        }
        assert_eq!(json["samplers"].as_array().unwrap().len(), 1);
        assert_eq!(json["samplers"][0]["wrapS"], 33071);
        assert_eq!(json["samplers"][0]["wrapT"], 33071);

        // Primitive attributes + indices + material.
        let primitive = &json["meshes"][0]["primitives"][0];
        assert_eq!(primitive["attributes"]["POSITION"], 0);
        assert_eq!(primitive["attributes"]["NORMAL"], 1);
        assert_eq!(primitive["attributes"]["TEXCOORD_0"], 2);
        assert_eq!(primitive["indices"], 3);
        assert_eq!(primitive["material"], 0);
        assert_eq!(primitive["mode"], 4);

        // Accessor bookkeeping.
        let accessors = json["accessors"].as_array().unwrap();
        assert_eq!(accessors.len(), 4);
        assert_eq!(accessors[0]["type"], "VEC3");
        assert_eq!(accessors[0]["count"], 4);
        assert_eq!(accessors[1]["type"], "VEC3");
        assert_eq!(accessors[2]["type"], "VEC2");
        assert_eq!(accessors[3]["componentType"], 5125);
        assert_eq!(accessors[3]["count"], 6);
        // Axis conversion on the fixture: (x, y, z) -> (x, z, -y); every
        // converted Y is 0 (all fixture Z are 0) and Z spans [-1, 0].
        assert_eq!(accessors[0]["min"], json!([0.0, 0.0, -1.0]));
        assert_eq!(accessors[0]["max"], json!([1.0, 0.0, 0.0]));

        // bufferView alignment: every byteOffset is 4-byte aligned and the
        // vertex views carry byteStride.
        let views = json["bufferViews"].as_array().unwrap();
        assert_eq!(views.len(), 6);
        for view in views {
            let offset = view["byteOffset"].as_u64().unwrap();
            assert_eq!(offset % 4, 0, "bufferView byteOffset alignment");
        }
        assert_eq!(views[0]["byteStride"], 12);
        assert_eq!(views[1]["byteStride"], 12);
        assert_eq!(views[2]["byteStride"], 8);
        assert_eq!(views[3]["target"], 34963);
        // PNG payloads exist and the buffer length covers the binary chunk.
        assert!(views[4]["byteLength"].as_u64().unwrap() > 0);
        assert!(views[5]["byteLength"].as_u64().unwrap() > 0);
        assert_eq!(
            json["buffers"][0]["byteLength"].as_u64().unwrap(),
            bin.len() as u64
        );

        // Decode both embedded PNGs and check size + known pixels (row 0 of
        // the image == baked buffer row 0: the writer flips UV V, not the
        // image).
        let image0_view = &views[4];
        let image0_bytes = &bin[image0_view["byteOffset"].as_u64().unwrap() as usize
            ..(image0_view["byteOffset"].as_u64().unwrap()
                + image0_view["byteLength"].as_u64().unwrap()) as usize];
        let decoded = image::load_from_memory(image0_bytes).expect("decode base color png");
        assert_eq!((8, 8), (decoded.width(), decoded.height()));
        let rgba = decoded.to_rgba8();
        assert_eq!([255, 0, 0, 255], rgba.get_pixel(0, 0).0);
        assert_eq!([255, 0, 0, 255], rgba.get_pixel(7, 7).0);

        let image1_view = &views[5];
        let image1_bytes = &bin[image1_view["byteOffset"].as_u64().unwrap() as usize
            ..(image1_view["byteOffset"].as_u64().unwrap()
                + image1_view["byteLength"].as_u64().unwrap()) as usize];
        let decoded_mr = image::load_from_memory(image1_bytes).expect("decode mr png");
        assert_eq!((8, 8), (decoded_mr.width(), decoded_mr.height()));
        // First texel: baked [0, 128, 64, 255] — R stays 0, A stays 255.
        assert_eq!([0, 128, 64, 255], decoded_mr.to_rgba8().get_pixel(0, 0).0);

        // Raw accessor bytes: the exported positions/normals are axis
        // converted, the texcoords V-flipped.
        let pos_offset = views[0]["byteOffset"].as_u64().unwrap() as usize;
        let px = f32::from_le_bytes(bin[pos_offset..pos_offset + 4].try_into().unwrap());
        let py = f32::from_le_bytes(bin[pos_offset + 4..pos_offset + 8].try_into().unwrap());
        let pz = f32::from_le_bytes(bin[pos_offset + 8..pos_offset + 12].try_into().unwrap());
        assert_eq!((px, py, pz), (0.0, 0.0, 0.0));
        let uv_offset = views[2]["byteOffset"].as_u64().unwrap() as usize;
        let u0 = f32::from_le_bytes(bin[uv_offset..uv_offset + 4].try_into().unwrap());
        let v0 = f32::from_le_bytes(bin[uv_offset + 4..uv_offset + 8].try_into().unwrap());
        assert_eq!((u0, v0), (0.0, 1.0), "V flip applied at export");
    }

    #[test]
    fn textured_glb_alpha_modes_and_skip_axes_flag() {
        let mut mesh = fixture();
        mesh.alpha_mode = TexturedAlphaMode::Mask(0.5);
        mesh.already_gltf_axes = true;
        let path = temp_path("mask");
        write_textured_glb(&mesh, &path).expect("write mask glb");
        let bytes = std::fs::read(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);
        let (json, _bin) = parse_glb(&bytes);
        let material = &json["materials"][0];
        assert_eq!(material["alphaMode"], "MASK");
        assert_eq!(material["alphaCutoff"], 0.5);

        // already_gltf_axes skips the conversion: vertex 1 (1, 0, 0) stays.
        let views = json["bufferViews"].as_array().unwrap();
        let pos_offset = views[0]["byteOffset"].as_u64().unwrap() as usize;
        let py = f32::from_le_bytes(bin_from(&bytes, pos_offset + 4).try_into().unwrap());
        assert_eq!(py, 0.0, "unconverted Y kept (fixture Y=0)");
        // A non-symmetric check: vertex 2 (1, 1, 0) stays (1, 1, 0).
        let x2 = f32::from_le_bytes(bin_from(&bytes, pos_offset + 24).try_into().unwrap());
        let y2 = f32::from_le_bytes(bin_from(&bytes, pos_offset + 28).try_into().unwrap());
        let z2 = f32::from_le_bytes(bin_from(&bytes, pos_offset + 32).try_into().unwrap());
        assert_eq!((x2, y2, z2), (1.0, 1.0, 0.0));
    }

    fn bin_from<'a>(bytes: &'a [u8], offset: usize) -> &'a [u8] {
        let (json, bin) = parse_glb(bytes);
        let _ = json;
        &bin[offset..offset + 4]
    }

    #[test]
    fn textured_glb_rejects_invalid_inputs_descriptively() {
        let mut mesh = fixture();
        mesh.normals.push(0.0);
        let path = temp_path("bad-normals");
        let error = write_textured_glb(&mesh, &path).expect_err("normal count mismatch");
        assert!(
            error
                .to_string()
                .contains("matching flat vertex and normal")
        );
        assert!(!path.exists(), "failed writes must not leave a file");

        let mut mesh = fixture();
        mesh.faces[2] = 9;
        let path = temp_path("bad-faces");
        let error = write_textured_glb(&mesh, &path).expect_err("face index out of range");
        assert!(error.to_string().contains("outside the vertex buffer"));

        let mut mesh = fixture();
        mesh.texture_size = 8;
        mesh.base_color_rgba.truncate(10);
        let path = temp_path("bad-texture");
        let error = write_textured_glb(&mesh, &path).expect_err("texture buffer mismatch");
        assert!(error.to_string().contains("texture_size^2 RGBA texels"));
    }
}
