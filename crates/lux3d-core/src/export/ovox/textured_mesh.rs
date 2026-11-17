//! Textured-mesh loading for the o_voxel command surface.
//!
//! The official `mesh2ovox` example starts from a trimesh scene
//! (`examples/utils.py:22-27`, a textured `.glb`) and requires per-material
//! PBR data (`o_voxel/convert/volumetic_attr.py:202-229`). This module loads
//! that surface:
//!
//! * **GLB** — full glTF 2.0 PBR materials: `baseColorFactor` +
//!   `baseColorTexture`, `metallicFactor` / `roughnessFactor` +
//!   `metallicRoughnessTexture`, `emissiveFactor` + `emissiveTexture`,
//!   `normalTexture`, `alphaMode` / `alphaCutoff`; embedded images
//!   (`bufferView`) and `data:` URIs. Per-primitive `material` indices are
//!   preserved, so multi-material scenes voxelize into the right material per
//!   triangle. The trimesh texture convention is preserved: `TEXCOORD_0` is
//!   flipped (`v' = 1 - v`) like trimesh's glTF loader
//!   (`export/import.rs:40-54`), and images decode to top-row-first RGBA like
//!   `np.array(PIL image)`.
//! * **OBJ / PLY / STL** — geometry and (OBJ/GLB-style) complete per-corner
//!   UVs through [`crate::export::import::load_mesh`], with a single default
//!   PBR material (base color white, metallic 1, roughness 1, opaque).
//!   MTF/MTL material and texture parsing is deliberately out of scope for
//!   this wave: the official example asset is a GLB, and the o_voxel
//!   material path is only meaningfully exercised by textures.
//!
//! Texture sampler state is fixed to the official wrapper values: LINEAR
//! filtering with REPEAT wrap (`volumetic_attr.py:264-286` passes
//! `filter = [1] * n`, `wrap = [0] * n` regardless of the glTF sampler), so
//! this loader does not read glTF samplers.
//!
//! The `pbrMetallicRoughness` metallic/roughness channels follow glTF: BLUE
//! is metallic, GREEN is roughness — the exact channels the official kernel
//! extracts (`volumetic_attr.py:214-218`).

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Lux3dError, Result};

use super::voxelize::{OvoxAlphaMode, OvoxMaterial, OvoxTexture};

/// A textured triangle mesh ready for o_voxel voxelization.
#[derive(Debug, Clone, PartialEq)]
pub struct OvoxTexturedMesh {
    /// `[V]` vertex positions.
    pub vertices: Vec<[f32; 3]>,
    /// Flat zero-based triangle indices (`3 * F`).
    pub faces: Vec<u32>,
    /// Per-corner UVs (`3 * F`), trimesh convention (origin lower-left).
    pub corner_uvs: Vec<[f32; 2]>,
    /// Per-face material index into [`OvoxTexturedMesh::materials`].
    pub material_ids: Vec<usize>,
    /// PBR materials (at least one entry).
    pub materials: Vec<OvoxMaterial>,
}

impl OvoxTexturedMesh {
    /// Number of triangles.
    pub fn face_count(&self) -> usize {
        self.faces.len() / 3
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Position bounds `[[min], [max]]`, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<[[f32; 3]; 2]> {
        if self.vertices.is_empty() {
            return None;
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vertex in &self.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        Some([min, max])
    }

    /// Per-corner (face corner) normals, computed from the triangle fan with
    /// area weighting — the trimesh `vertex_normals[faces]` surface the
    /// official voxelizer consumes (`volumetic_attr.py:206`).
    pub fn corner_normals(&self) -> Vec<[f32; 3]> {
        let normals = crate::geometry::area_weighted_vertex_normals(&self.vertices, &self.faces);
        self.faces
            .iter()
            .map(|&index| normals[index as usize])
            .collect()
    }

    /// Scale/translate the mesh into the example's normalized cube
    /// (`examples/mesh2ovox.py:10-14`): translate the bounding-box center to
    /// the origin, then scale by `0.99999 / max_extent`.
    pub fn normalize_to_unit_cube(&mut self) -> Result<()> {
        let bounds = self.bounds().ok_or(Lux3dError::InvalidInput(
            "o_voxel mesh normalization needs a non-empty mesh",
        ))?;
        let mut extent = 0.0f32;
        let mut center = [0.0f32; 3];
        for axis in 0..3 {
            if !bounds[0][axis].is_finite() || !bounds[1][axis].is_finite() {
                return Err(Lux3dError::InvalidInput(
                    "o_voxel mesh normalization requires finite vertex positions",
                ));
            }
            center[axis] = (bounds[0][axis] + bounds[1][axis]) * 0.5;
            extent = extent.max(bounds[1][axis] - bounds[0][axis]);
        }
        if !extent.is_finite() || extent <= 0.0 {
            return Err(Lux3dError::InvalidInput(
                "o_voxel mesh normalization rejected a degenerate bounding box",
            ));
        }
        let scale = 0.99999 / extent;
        for vertex in &mut self.vertices {
            for axis in 0..3 {
                vertex[axis] = (vertex[axis] - center[axis]) * scale;
            }
        }
        Ok(())
    }

    /// Drop index-degenerate and zero-area triangles, keeping UVs and
    /// material ids aligned. The geometry voxelizer rejects such triangles
    /// (`geometry/dual_grid.rs` validation), while the official kernel would
    /// propagate `NaN` normals; dropping them is the documented safe
    /// deviation. Returns the number of dropped triangles.
    pub fn drop_degenerate_triangles(&mut self) -> usize {
        let face_count = self.face_count();
        let mut kept_faces = Vec::with_capacity(self.faces.len());
        let mut kept_uvs = Vec::with_capacity(self.corner_uvs.len());
        let mut kept_materials = Vec::with_capacity(self.material_ids.len());
        let mut dropped = 0usize;
        for face in 0..face_count {
            let base = face * 3;
            let tri = [
                self.faces[base],
                self.faces[base + 1],
                self.faces[base + 2],
            ];
            let v0 = self.vertices[tri[0] as usize];
            let v1 = self.vertices[tri[1] as usize];
            let v2 = self.vertices[tri[2] as usize];
            // Same f32 edge differences the dual-grid kernel uses
            // (`e0 = v1 - v0`, `e1 = v2 - v1`).
            let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
            let normal = [
                e0[1] * e1[2] - e0[2] * e1[1],
                e0[2] * e1[0] - e0[0] * e1[2],
                e0[0] * e1[1] - e0[1] * e1[0],
            ];
            let norm2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] || !(norm2 > 0.0) {
                dropped += 1;
                continue;
            }
            kept_faces.extend_from_slice(&tri);
            kept_uvs.extend_from_slice(&self.corner_uvs[base..base + 3]);
            kept_materials.push(self.material_ids[face]);
        }
        self.faces = kept_faces;
        self.corner_uvs = kept_uvs;
        self.material_ids = kept_materials;
        dropped
    }
}

/// Load a textured mesh for `mesh2ovox`.
///
/// See the module docs for the per-format material coverage.
pub fn load_textured_mesh(path: &Path) -> Result<OvoxTexturedMesh> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .trim_start_matches('.')
        .to_ascii_lowercase();
    let mesh = match extension.as_str() {
        "glb" => read_textured_glb(path)?,
        "obj" | "ply" | "stl" => mesh_from_import(path)?,
        other => {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "unsupported o_voxel mesh input extension `{other}` for `{}`; use .glb, .obj, \
                     .ply or .stl",
                    path.display()
                ),
            });
        }
    };
    validate_textured_mesh(&mesh, path)?;
    Ok(mesh)
}

fn validate_textured_mesh(mesh: &OvoxTexturedMesh, path: &Path) -> Result<()> {
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "o_voxel input `{}` contains no usable triangles",
                path.display()
            ),
        });
    }
    if mesh.faces.len() % 3 != 0
        || mesh.corner_uvs.len() != mesh.faces.len()
        || mesh.material_ids.len() != mesh.faces.len() / 3
    {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "o_voxel input `{}` has inconsistent face, UV or material buffers",
                path.display()
            ),
        });
    }
    if mesh.materials.is_empty() {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "o_voxel input `{}` carries no materials",
                path.display()
            ),
        });
    }
    let vertex_count = mesh.vertices.len();
    for &index in &mesh.faces {
        if index as usize >= vertex_count {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "o_voxel input `{}` references vertex {index} outside the {vertex_count}-vertex \
                     buffer",
                    path.display()
                ),
            });
        }
    }
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        if vertex.iter().any(|value| !value.is_finite()) {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "o_voxel input `{}` vertex {index} has non-finite coordinates",
                    path.display()
                ),
            });
        }
    }
    for (index, uv) in mesh.corner_uvs.iter().enumerate() {
        if uv.iter().any(|value| !value.is_finite()) {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "o_voxel input `{}` UV corner {index} is non-finite",
                    path.display()
                ),
            });
        }
    }
    for &material_id in &mesh.material_ids {
        if material_id >= mesh.materials.len() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "o_voxel input `{}` references material {material_id} outside the material list",
                    path.display()
                ),
            });
        }
    }
    for material in &mesh.materials {
        material.validate()?;
    }
    Ok(())
}

/// OBJ / PLY / STL path: geometry + per-corner UVs through the shared
/// importer, one default material.
fn mesh_from_import(path: &Path) -> Result<OvoxTexturedMesh> {
    let imported = crate::export::import::load_mesh(path)?;
    let vertices: Vec<[f32; 3]> = imported
        .vertices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    let faces = imported.faces;
    let corner_uvs = match &imported.uvs {
        Some(uvs) => uvs
            .chunks_exact(2)
            .map(|chunk| [chunk[0], chunk[1]])
            .collect(),
        // Upstream fills zero UVs for UV-less geometry
        // (`volumetic_attr.py:207-208`).
        None => vec![[0.0, 0.0]; faces.len()],
    };
    Ok(OvoxTexturedMesh {
        vertices,
        material_ids: vec![0; faces.len() / 3],
        faces,
        corner_uvs,
        materials: vec![OvoxMaterial::default()],
    })
}

// ---------------------------------------------------------------------------
// Textured GLB reader
// ---------------------------------------------------------------------------

const GLB_MAGIC: u32 = 0x4654_6C67;
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

#[derive(Debug, Deserialize)]
struct Gltf {
    #[serde(default)]
    scene: Option<usize>,
    #[serde(default)]
    scenes: Vec<GltfScene>,
    #[serde(default)]
    nodes: Vec<GltfNode>,
    #[serde(default)]
    meshes: Vec<GltfMesh>,
    #[serde(default)]
    accessors: Vec<GltfAccessor>,
    #[serde(default, rename = "bufferViews")]
    buffer_views: Vec<GltfBufferView>,
    #[serde(default)]
    materials: Vec<GltfMaterial>,
    #[serde(default)]
    textures: Vec<GltfTexture>,
    #[serde(default)]
    images: Vec<GltfImage>,
}

#[derive(Debug, Default, Deserialize)]
struct GltfScene {
    #[serde(default)]
    nodes: Vec<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct GltfNode {
    #[serde(default)]
    mesh: Option<usize>,
    #[serde(default)]
    children: Vec<usize>,
    #[serde(default)]
    matrix: Option<Vec<f64>>,
    #[serde(default)]
    translation: Option<Vec<f64>>,
    #[serde(default)]
    rotation: Option<Vec<f64>>,
    #[serde(default)]
    scale: Option<Vec<f64>>,
}

#[derive(Debug, Default, Deserialize)]
struct GltfMesh {
    #[serde(default)]
    primitives: Vec<GltfPrimitive>,
}

#[derive(Debug, Default, Deserialize)]
struct GltfPrimitive {
    #[serde(default)]
    attributes: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    indices: Option<usize>,
    #[serde(default)]
    material: Option<usize>,
    #[serde(default)]
    mode: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct GltfAccessor {
    #[serde(default, rename = "bufferView")]
    buffer_view: Option<usize>,
    #[serde(default, rename = "byteOffset")]
    byte_offset: Option<usize>,
    #[serde(default, rename = "componentType")]
    component_type: u32,
    #[serde(default)]
    count: usize,
    #[serde(default, rename = "type")]
    ty: String,
    #[serde(default)]
    sparse: serde_json::Value,
}

#[derive(Debug, Default, Deserialize)]
struct GltfBufferView {
    #[serde(default)]
    buffer: usize,
    #[serde(default, rename = "byteOffset")]
    byte_offset: Option<usize>,
    #[serde(default, rename = "byteLength")]
    byte_length: usize,
    #[serde(default, rename = "byteStride")]
    byte_stride: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct GltfMaterial {
    #[serde(default, rename = "pbrMetallicRoughness")]
    pbr_metallic_roughness: GltfPbr,
    #[serde(default, rename = "emissiveFactor")]
    emissive_factor: Option<Vec<f64>>,
    #[serde(default, rename = "emissiveTexture")]
    emissive_texture: Option<GltfTextureRef>,
    #[serde(default, rename = "normalTexture")]
    normal_texture: Option<GltfTextureRef>,
    #[serde(default, rename = "alphaMode")]
    alpha_mode: Option<String>,
    #[serde(default, rename = "alphaCutoff")]
    alpha_cutoff: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct GltfPbr {
    #[serde(default, rename = "baseColorFactor")]
    base_color_factor: Option<Vec<f64>>,
    #[serde(default, rename = "baseColorTexture")]
    base_color_texture: Option<GltfTextureRef>,
    #[serde(default, rename = "metallicFactor")]
    metallic_factor: Option<f64>,
    #[serde(default, rename = "roughnessFactor")]
    roughness_factor: Option<f64>,
    #[serde(default, rename = "metallicRoughnessTexture")]
    metallic_roughness_texture: Option<GltfTextureRef>,
}

#[derive(Debug, Default, Deserialize)]
struct GltfTextureRef {
    #[serde(default)]
    index: usize,
}

#[derive(Debug, Default, Deserialize)]
struct GltfTexture {
    #[serde(default)]
    source: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct GltfImage {
    #[serde(default)]
    uri: Option<String>,
    #[serde(default, rename = "mimeType")]
    mime_type: Option<String>,
    #[serde(default, rename = "bufferView")]
    buffer_view: Option<usize>,
}

fn glb_error(path: &Path, message: impl std::fmt::Display) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: format!("GLB `{}`: {message}", path.display()),
    }
}

fn read_textured_glb(path: &Path) -> Result<OvoxTexturedMesh> {
    let bytes = std::fs::read(path).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })?;
    let (json, bin) = parse_glb(&bytes, path)?;
    let document: Gltf = serde_json::from_slice(json)
        .map_err(|error| glb_error(path, format!("invalid glTF JSON: {error}")))?;

    let materials = read_materials(&document, bin, path)?;
    // `read_materials` appends the implicit default material at this index.
    let default_material = document.materials.len();

    let mut out = OvoxTexturedMesh {
        vertices: Vec::new(),
        faces: Vec::new(),
        corner_uvs: Vec::new(),
        material_ids: Vec::new(),
        materials,
    };

    // Scene roots (or every mesh at identity when there is no scene graph).
    let roots: Vec<usize> = match document.scene {
        Some(index) if document.scenes.get(index).is_some() => {
            document.scenes[index].nodes.clone()
        }
        _ if !document.scenes.is_empty() => document.scenes[0].nodes.clone(),
        _ => {
            for mesh_index in 0..document.meshes.len() {
                visit_mesh(
                    &document,
                    bin,
                    mesh_index,
                    &glb_identity(),
                    default_material,
                    &mut out,
                    path,
                )?;
            }
            Vec::new()
        }
    };
    let mut in_stack = HashSet::new();
    for root in roots {
        visit_node(
            &document,
            bin,
            root,
            &glb_identity(),
            0,
            default_material,
            &mut in_stack,
            &mut out,
            path,
        )?;
    }
    Ok(out)
}

fn parse_glb<'a>(bytes: &'a [u8], path: &Path) -> Result<(&'a [u8], &'a [u8])> {
    if bytes.len() < 12 {
        return Err(glb_error(path, "file is too short to be a GLB container"));
    }
    if glb_u32(bytes, 0, path)? != GLB_MAGIC {
        return Err(glb_error(path, "not a GLB file (missing `glTF` magic)"));
    }
    if glb_u32(bytes, 4, path)? != 2 {
        return Err(glb_error(path, "only glTF 2.0 GLB containers are supported"));
    }
    let declared = glb_u32(bytes, 8, path)? as usize;
    if declared > bytes.len() {
        return Err(glb_error(
            path,
            format!(
                "container declares {declared} bytes but the file holds {}",
                bytes.len()
            ),
        ));
    }
    let mut cursor = 12usize;
    let mut json_chunk: Option<&[u8]> = None;
    let mut bin_chunk: Option<&[u8]> = None;
    while cursor + 8 <= declared {
        let length = glb_u32(bytes, cursor, path)? as usize;
        let chunk_type = glb_u32(bytes, cursor + 4, path)?;
        let start = cursor + 8;
        let end = start
            .checked_add(length)
            .filter(|end| *end <= declared)
            .ok_or_else(|| glb_error(path, "GLB chunk extends past the container"))?;
        match chunk_type {
            CHUNK_JSON => json_chunk = Some(&bytes[start..end]),
            CHUNK_BIN => bin_chunk = Some(&bytes[start..end]),
            _ => {}
        }
        cursor = end;
    }
    let json = json_chunk.ok_or_else(|| glb_error(path, "GLB has no JSON chunk"))?;
    Ok((json, bin_chunk.unwrap_or(&[])))
}

fn glb_u32(bytes: &[u8], offset: usize, path: &Path) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| glb_error(path, "truncated GLB header"))?;
    Ok(u32::from_le_bytes(slice.try_into().expect("4-byte slice")))
}

fn glb_identity() -> [f64; 16] {
    let mut matrix = [0.0f64; 16];
    matrix[0] = 1.0;
    matrix[5] = 1.0;
    matrix[10] = 1.0;
    matrix[15] = 1.0;
    matrix
}

#[allow(clippy::too_many_arguments)]
fn visit_node(
    document: &Gltf,
    bin: &[u8],
    node_index: usize,
    parent: &[f64; 16],
    depth: usize,
    default_material: usize,
    in_stack: &mut HashSet<usize>,
    out: &mut OvoxTexturedMesh,
    path: &Path,
) -> Result<()> {
    if depth > 256 {
        return Err(glb_error(path, "node tree is nested too deeply"));
    }
    if !in_stack.insert(node_index) {
        return Err(glb_error(
            path,
            format!("node graph contains a cycle at node {node_index}"),
        ));
    }
    let node = document
        .nodes
        .get(node_index)
        .ok_or_else(|| glb_error(path, format!("node index {node_index} is out of range")))?;
    let world = mat4_mul(parent, &node_matrix(node, path)?);
    if let Some(mesh_index) = node.mesh {
        visit_mesh(
            document,
            bin,
            mesh_index,
            &world,
            default_material,
            out,
            path,
        )?;
    }
    for child in &node.children {
        visit_node(
            document,
            bin,
            *child,
            &world,
            depth + 1,
            default_material,
            in_stack,
            out,
            path,
        )?;
    }
    in_stack.remove(&node_index);
    Ok(())
}

fn node_matrix(node: &GltfNode, path: &Path) -> Result<[f64; 16]> {
    if let Some(matrix) = &node.matrix {
        if matrix.len() != 16 {
            return Err(glb_error(path, "node matrix must have 16 values"));
        }
        let mut out = [0.0f64; 16];
        out.copy_from_slice(matrix);
        return Ok(out);
    }
    let scale = node.scale.as_deref().unwrap_or(&[1.0, 1.0, 1.0]);
    if scale.len() != 3 {
        return Err(glb_error(path, "node scale must have 3 values"));
    }
    let mut out = glb_identity();
    out[0] = scale[0];
    out[5] = scale[1];
    out[10] = scale[2];
    if let Some(rotation) = &node.rotation {
        if rotation.len() != 4 {
            return Err(glb_error(path, "node rotation quaternion must have 4 values"));
        }
        let rotation_matrix = quaternion_matrix(rotation[0], rotation[1], rotation[2], rotation[3]);
        for column in 0..3 {
            for row in 0..3 {
                out[column * 4 + row] = rotation_matrix[column * 4 + row] * scale[column];
            }
        }
    }
    let translation = node.translation.as_deref().unwrap_or(&[0.0, 0.0, 0.0]);
    if translation.len() != 3 {
        return Err(glb_error(path, "node translation must have 3 values"));
    }
    out[12] = translation[0];
    out[13] = translation[1];
    out[14] = translation[2];
    Ok(out)
}

fn quaternion_matrix(x: f64, y: f64, z: f64, w: f64) -> [f64; 16] {
    let mut m = [0.0f64; 16];
    m[0] = 1.0 - 2.0 * (y * y + z * z);
    m[1] = 2.0 * (x * y + z * w);
    m[2] = 2.0 * (x * z - y * w);
    m[4] = 2.0 * (x * y - z * w);
    m[5] = 1.0 - 2.0 * (x * x + z * z);
    m[6] = 2.0 * (y * z + x * w);
    m[8] = 2.0 * (x * z + y * w);
    m[9] = 2.0 * (y * z - x * w);
    m[10] = 1.0 - 2.0 * (x * x + y * y);
    m[15] = 1.0;
    m
}

fn mat4_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    for column in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[column * 4 + k];
            }
            out[column * 4 + row] = sum;
        }
    }
    out
}

fn transform_vertex(matrix: &[f64; 16], position: [f32; 3]) -> [f32; 3] {
    let (x, y, z) = (
        f64::from(position[0]),
        f64::from(position[1]),
        f64::from(position[2]),
    );
    [
        (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) as f32,
        (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) as f32,
        (matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]) as f32,
    ]
}

#[allow(clippy::too_many_arguments)]
fn visit_mesh(
    document: &Gltf,
    bin: &[u8],
    mesh_index: usize,
    world: &[f64; 16],
    default_material: usize,
    out: &mut OvoxTexturedMesh,
    path: &Path,
) -> Result<()> {
    let mesh = document
        .meshes
        .get(mesh_index)
        .ok_or_else(|| glb_error(path, format!("mesh index {mesh_index} is out of range")))?;
    for (primitive_index, primitive) in mesh.primitives.iter().enumerate() {
        let context = format!("mesh {mesh_index} primitive {primitive_index}");
        if primitive.mode.unwrap_or(4) != 4 {
            continue;
        }
        let position_accessor = primitive
            .attributes
            .get("POSITION")
            .ok_or_else(|| glb_error(path, format!("{context} has no POSITION attribute")))?;
        let positions = read_positions(document, bin, *position_accessor, world, path, &context)?;
        let indices = match primitive.indices {
            Some(accessor) => {
                read_indices(document, bin, accessor, positions.len(), path, &context)?
            }
            None => {
                if positions.len() % 3 != 0 {
                    return Err(glb_error(
                        path,
                        format!(
                            "{context}: non-indexed primitive needs a POSITION count divisible by 3"
                        ),
                    ));
                }
                (0..positions.len()).collect()
            }
        };
        let uvs = match primitive.attributes.get("TEXCOORD_0") {
            Some(accessor) => {
                let uvs = read_texcoords(document, bin, *accessor, positions.len(), path, &context)?;
                Some(uvs)
            }
            None => None,
        };
        let material = primitive.material.unwrap_or(default_material);
        if material >= out.materials.len() {
            return Err(glb_error(
                path,
                format!("{context}: material index {material} is out of range"),
            ));
        }
        let vertex_base = out.vertices.len() as u32;
        out.vertices.extend(positions);
        for triangle in indices.chunks_exact(3) {
            for &index in triangle {
                out.faces.push(vertex_base + index as u32);
            }
            out.material_ids.push(material);
            match &uvs {
                Some(uvs) => {
                    for &index in triangle {
                        let uv = uvs[index];
                        // glTF TEXCOORD_0 is top-left; trimesh/the o_voxel
                        // wrapper use the lower-left convention.
                        out.corner_uvs.push([uv[0], 1.0 - uv[1]]);
                    }
                }
                // Upstream fills zero UVs for UV-less geometry.
                None => out.corner_uvs.extend_from_slice(&[[0.0, 0.0]; 3]),
            }
        }
    }
    Ok(())
}

fn accessor_span(
    document: &Gltf,
    accessor: &GltfAccessor,
    element_size: usize,
    bin_len: usize,
    path: &Path,
    context: &str,
    label: &str,
) -> Result<(usize, usize)> {
    if accessor.sparse != serde_json::Value::Null
        && accessor
            .sparse
            .as_object()
            .map(|map| !map.is_empty())
            == Some(true)
    {
        return Err(glb_error(
            path,
            format!("{context}: sparse {label} accessors are not supported"),
        ));
    }
    let view_index = accessor
        .buffer_view
        .ok_or_else(|| glb_error(path, format!("{context}: {label} accessor has no bufferView")))?;
    let view = document
        .buffer_views
        .get(view_index)
        .ok_or_else(|| glb_error(path, format!("{context}: bufferView {view_index} is out of range")))?;
    if view.buffer != 0 {
        return Err(glb_error(
            path,
            format!("{context}: only the embedded BIN buffer (buffer 0) is supported"),
        ));
    }
    let stride = view.byte_stride.unwrap_or(element_size);
    if stride < element_size {
        return Err(glb_error(
            path,
            format!("{context}: {label} byteStride {stride} is smaller than the element size"),
        ));
    }
    let start = view
        .byte_offset
        .unwrap_or(0)
        .checked_add(accessor.byte_offset.unwrap_or(0))
        .ok_or_else(|| glb_error(path, format!("{context}: {label} byte offset overflows")))?;
    let last_element = accessor.count.saturating_sub(1);
    let end = last_element
        .checked_mul(stride)
        .and_then(|span| span.checked_add(element_size))
        .and_then(|span| start.checked_add(span))
        .ok_or_else(|| glb_error(path, format!("{context}: {label} accessor extent overflows")))?;
    let view_end = view
        .byte_offset
        .unwrap_or(0)
        .checked_add(view.byte_length)
        .ok_or_else(|| glb_error(path, format!("{context}: bufferView extent overflows")))?;
    if end > bin_len || end > view_end {
        return Err(glb_error(
            path,
            format!("{context}: {label} accessor extends past its buffer view"),
        ));
    }
    Ok((start, stride))
}

fn read_positions(
    document: &Gltf,
    bin: &[u8],
    accessor_index: usize,
    world: &[f64; 16],
    path: &Path,
    context: &str,
) -> Result<Vec<[f32; 3]>> {
    let accessor = document.accessors.get(accessor_index).ok_or_else(|| {
        glb_error(
            path,
            format!("{context}: POSITION accessor {accessor_index} is out of range"),
        )
    })?;
    if accessor.component_type != 5126 || accessor.ty != "VEC3" {
        return Err(glb_error(
            path,
            format!("{context}: POSITION must be a float32 VEC3 accessor"),
        ));
    }
    let (start, stride) = accessor_span(
        document,
        accessor,
        12,
        bin.len(),
        path,
        context,
        "POSITION",
    )?;
    let mut positions = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = start + item * stride;
        let chunk = &bin[offset..offset + 12];
        let position = [
            f32::from_le_bytes(chunk[0..4].try_into().expect("4-byte slice")),
            f32::from_le_bytes(chunk[4..8].try_into().expect("4-byte slice")),
            f32::from_le_bytes(chunk[8..12].try_into().expect("4-byte slice")),
        ];
        positions.push(transform_vertex(world, position));
    }
    Ok(positions)
}

fn read_indices(
    document: &Gltf,
    bin: &[u8],
    accessor_index: usize,
    vertex_count: usize,
    path: &Path,
    context: &str,
) -> Result<Vec<usize>> {
    let accessor = document.accessors.get(accessor_index).ok_or_else(|| {
        glb_error(
            path,
            format!("{context}: index accessor {accessor_index} is out of range"),
        )
    })?;
    if accessor.ty != "SCALAR" {
        return Err(glb_error(
            path,
            format!("{context}: index accessor must be SCALAR"),
        ));
    }
    let element_size = match accessor.component_type {
        5121 => 1,
        5123 => 2,
        5125 => 4,
        other => {
            return Err(glb_error(
                path,
                format!("{context}: unsupported index component type {other}"),
            ));
        }
    };
    let (start, stride) = accessor_span(
        document,
        accessor,
        element_size,
        bin.len(),
        path,
        context,
        "index",
    )?;
    let mut indices = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = start + item * stride;
        let index = match accessor.component_type {
            5121 => bin[offset] as usize,
            5123 => u16::from_le_bytes(bin[offset..offset + 2].try_into().expect("2-byte slice"))
                as usize,
            _ => u32::from_le_bytes(bin[offset..offset + 4].try_into().expect("4-byte slice"))
                as usize,
        };
        if index >= vertex_count {
            return Err(glb_error(
                path,
                format!("{context}: index {index} outside the {vertex_count}-vertex primitive"),
            ));
        }
        indices.push(index);
    }
    if indices.len() % 3 != 0 {
        return Err(glb_error(
            path,
            format!("{context}: triangle index count must be a multiple of 3"),
        ));
    }
    Ok(indices)
}

fn read_texcoords(
    document: &Gltf,
    bin: &[u8],
    accessor_index: usize,
    vertex_count: usize,
    path: &Path,
    context: &str,
) -> Result<Vec<[f32; 2]>> {
    let accessor = document.accessors.get(accessor_index).ok_or_else(|| {
        glb_error(
            path,
            format!("{context}: TEXCOORD_0 accessor {accessor_index} is out of range"),
        )
    })?;
    if accessor.ty != "VEC2" {
        return Err(glb_error(
            path,
            format!("{context}: TEXCOORD_0 must be VEC2"),
        ));
    }
    let element_size = match accessor.component_type {
        5126 => 8,
        5121 => 2,
        5123 => 4,
        other => {
            return Err(glb_error(
                path,
                format!("{context}: unsupported TEXCOORD_0 component type {other}"),
            ));
        }
    };
    let (start, stride) = accessor_span(
        document,
        accessor,
        element_size,
        bin.len(),
        path,
        context,
        "TEXCOORD_0",
    )?;
    if accessor.count < vertex_count {
        return Err(glb_error(
            path,
            format!("{context}: TEXCOORD_0 has fewer entries than POSITION"),
        ));
    }
    let mut uvs = Vec::with_capacity(vertex_count);
    for item in 0..vertex_count {
        let offset = start + item * stride;
        let uv = match accessor.component_type {
            5126 => [
                f32::from_le_bytes(bin[offset..offset + 4].try_into().expect("4-byte slice")),
                f32::from_le_bytes(bin[offset + 4..offset + 8].try_into().expect("4-byte slice")),
            ],
            5121 => [
                bin[offset] as f32 / 255.0,
                bin[offset + 1] as f32 / 255.0,
            ],
            _ => [
                u16::from_le_bytes(bin[offset..offset + 2].try_into().expect("2-byte slice")) as f32
                    / 65535.0,
                u16::from_le_bytes(
                    bin[offset + 2..offset + 4].try_into().expect("2-byte slice"),
                ) as f32
                    / 65535.0,
            ],
        };
        uvs.push(uv);
    }
    Ok(uvs)
}

fn read_materials(document: &Gltf, bin: &[u8], path: &Path) -> Result<Vec<OvoxMaterial>> {
    let mut materials = Vec::with_capacity(document.materials.len() + 1);
    for (index, material) in document.materials.iter().enumerate() {
        let context = format!("material {index}");
        let mut parsed = OvoxMaterial::default();
        if let Some(factor) = &material.pbr_metallic_roughness.base_color_factor {
            if factor.len() != 4 {
                return Err(glb_error(
                    path,
                    format!("{context}: baseColorFactor must have 4 values"),
                ));
            }
            for channel in 0..4 {
                parsed.base_color_factor[channel] = factor[channel] as f32;
            }
        }
        parsed.base_color_texture = match &material.pbr_metallic_roughness.base_color_texture {
            Some(reference) => Some(read_texture(document, bin, reference.index, path, &context)?),
            None => None,
        };
        if let Some(factor) = material.pbr_metallic_roughness.metallic_factor {
            parsed.metallic_factor = factor as f32;
        }
        if let Some(factor) = material.pbr_metallic_roughness.roughness_factor {
            parsed.roughness_factor = factor as f32;
        }
        if let Some(reference) = &material.pbr_metallic_roughness.metallic_roughness_texture {
            let texture = read_texture(document, bin, reference.index, path, &context)?;
            parsed.metallic_texture = Some(texture.clone());
            parsed.roughness_texture = Some(texture);
        }
        if let Some(factor) = &material.emissive_factor {
            if factor.len() != 3 {
                return Err(glb_error(
                    path,
                    format!("{context}: emissiveFactor must have 3 values"),
                ));
            }
            for channel in 0..3 {
                parsed.emissive_factor[channel] = factor[channel] as f32;
            }
        }
        parsed.emissive_texture = match &material.emissive_texture {
            Some(reference) => Some(read_texture(document, bin, reference.index, path, &context)?),
            None => None,
        };
        parsed.alpha_mode = match material.alpha_mode.as_deref().unwrap_or("OPAQUE") {
            "OPAQUE" => OvoxAlphaMode::Opaque,
            "MASK" => OvoxAlphaMode::Mask,
            "BLEND" => OvoxAlphaMode::Blend,
            other => {
                return Err(glb_error(
                    path,
                    format!("{context}: unsupported alphaMode `{other}`"),
                ));
            }
        };
        if let Some(cutoff) = material.alpha_cutoff {
            parsed.alpha_cutoff = cutoff as f32;
        }
        parsed.normal_texture = match &material.normal_texture {
            Some(reference) => Some(read_texture(document, bin, reference.index, path, &context)?),
            None => None,
        };
        if parsed.alpha_mode != OvoxAlphaMode::Opaque {
            parsed.alpha_texture = parsed.base_color_texture.clone();
        }
        materials.push(parsed);
    }
    // An implicit default material for primitives without one (glTF
    // `defaultMaterial`).
    materials.push(OvoxMaterial::default());
    Ok(materials)
}

fn read_texture(
    document: &Gltf,
    bin: &[u8],
    texture_index: usize,
    path: &Path,
    context: &str,
) -> Result<OvoxTexture> {
    let texture = document.textures.get(texture_index).ok_or_else(|| {
        glb_error(
            path,
            format!("{context}: texture index {texture_index} is out of range"),
        )
    })?;
    let image_index = texture.source.ok_or_else(|| {
        glb_error(
            path,
            format!("{context}: texture {texture_index} has no image source"),
        )
    })?;
    let image = document.images.get(image_index).ok_or_else(|| {
        glb_error(
            path,
            format!("{context}: image index {image_index} is out of range"),
        )
    })?;
    let bytes: Vec<u8> = if let Some(view_index) = image.buffer_view {
        let view = document.buffer_views.get(view_index).ok_or_else(|| {
            glb_error(
                path,
                format!("{context}: image bufferView {view_index} is out of range"),
            )
        })?;
        if view.buffer != 0 {
            return Err(glb_error(
                path,
                format!("{context}: only embedded BIN images are supported"),
            ));
        }
        let start = view.byte_offset.unwrap_or(0);
        let end = start + view.byte_length;
        bin.get(start..end)
            .ok_or_else(|| glb_error(path, format!("{context}: image extends past the BIN chunk")))?
            .to_vec()
    } else if let Some(uri) = &image.uri {
        if let Some(payload) = uri.strip_prefix("data:") {
            let (meta, data) = payload.split_once(',').ok_or_else(|| {
                glb_error(path, format!("{context}: malformed data URI"))
            })?;
            if !meta.ends_with(";base64") {
                return Err(glb_error(
                    path,
                    format!("{context}: only base64 data URIs are supported"),
                ));
            }
            base64_decode(data).ok_or_else(|| {
                glb_error(path, format!("{context}: invalid base64 data URI"))
            })?
        } else {
            return Err(glb_error(
                path,
                format!(
                    "{context}: external image URI `{uri}` is not supported; embed the image in \
                     the GLB"
                ),
            ));
        }
    } else {
        return Err(glb_error(
            path,
            format!(
                "{context}: image {image_index} has neither a bufferView nor a data URI \
                 (mime type {:?})",
                image.mime_type
            ),
        ));
    };
    let decoded = image::load_from_memory(&bytes)
        .map_err(|error| glb_error(path, format!("{context}: image decode failed: {error}")))?;
    let rgba = decoded.to_rgba8();
    OvoxTexture::new(rgba.width() as usize, rgba.height() as usize, rgba.into_raw())
}

/// Minimal base64 decoder for `data:` URIs (RFC 4648 standard alphabet,
/// padding optional).
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut values = Vec::with_capacity(input.len());
    for byte in input.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => continue,
            b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => return None,
        };
        values.push(value);
    }
    let mut out = Vec::with_capacity(values.len() * 3 / 4);
    for chunk in values.chunks(4) {
        match chunk.len() {
            4 => {
                let triple = (u32::from(chunk[0]) << 18)
                    | (u32::from(chunk[1]) << 12)
                    | (u32::from(chunk[2]) << 6)
                    | u32::from(chunk[3]);
                out.push((triple >> 16) as u8);
                out.push((triple >> 8) as u8);
                out.push(triple as u8);
            }
            3 => {
                let triple =
                    (u32::from(chunk[0]) << 18) | (u32::from(chunk[1]) << 12) | (u32::from(chunk[2]) << 6);
                out.push((triple >> 16) as u8);
                out.push((triple >> 8) as u8);
            }
            2 => {
                let triple = (u32::from(chunk[0]) << 18) | (u32::from(chunk[1]) << 12);
                out.push((triple >> 16) as u8);
            }
            _ => return None,
        }
    }
    Some(out)
}
