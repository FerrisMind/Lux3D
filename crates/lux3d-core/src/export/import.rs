//! Triangle-mesh import for existing mesh files.
//!
//! The TRELLIS.2 texturing stage textures an *already generated* mesh
//! artifact, so the export module also needs readers. [`load_mesh`] dispatches
//! on the file extension and returns an [`ImportedMesh`] with flat
//! `vertices`/`faces` buffers that mirror the writer types in
//! [`crate::export::mesh`].
//!
//! Supported formats and limits:
//! - **OBJ** (`read_obj`): ASCII Wavefront with `v`/`f` lines. Polygon faces
//!   are fan-triangulated; `v`, `v/vt`, `v//vn` and `v/vt/vn` face tokens are
//!   accepted with 1-based positive and negative relative indices. `vt`, `vn`,
//!   groups, materials and unknown keywords are ignored; multiple objects
//!   merge into one buffer. Line continuations (`\`) are not supported.
//! - **PLY** (`read_ply`): `format ascii 1.0` and
//!   `format binary_little_endian 1.0`. Arbitrary vertex property order;
//!   `x`/`y`/`z` may use any numeric scalar type (`float`/`double` are the
//!   common ones). Faces are read from the `vertex_indices`/`vertex_index`
//!   list property and polygons are fan-triangulated. Big-endian PLY and any
//!   unknown/compressed format variant are rejected with a clear error.
//! - **GLB** (`read_glb`): glTF 2.0 container (JSON + optional BIN chunk).
//!   Triangle primitives (`mode` 4) are gathered from the active scene, or
//!   from every mesh at identity when the file has no scene graph. Static
//!   node transforms (`matrix` or TRS with a unit quaternion) are applied to
//!   positions; winding order is preserved. Indices may be `u8`/`u16`/`u32`;
//!   positions must be `f32`. `KHR_draco_mesh_compression`,
//!   `EXT_meshopt_compression` and sparse accessors are rejected. Skins are
//!   ignored (bind-pose geometry is returned) and non-affine transforms are
//!   applied without perspective divide.
//! - **STL** (`read_stl`): ASCII and binary. Vertices are deduplicated by
//!   exact bit pattern (no epsilon welding; upstream trimesh merges by
//!   proximity anyway), so shared corners become shared indices.
//!
//! All readers reject empty meshes, non-finite coordinates and face indices
//! outside the vertex range. Triangles whose three indices are not pairwise
//! distinct (index-degenerate, zero-area triangles) are dropped; everything
//! else keeps file order — vertices and faces are never reordered.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::Deserialize;

use crate::{Result, error::Lux3dError};

/// A loaded triangle mesh with flat, writer-compatible buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedMesh {
    /// Flat `[x, y, z, ...]` vertex positions in file order.
    pub vertices: Vec<f32>,
    /// Flat zero-based triangle indices in file order.
    pub faces: Vec<u32>,
}

/// Load a triangle mesh, selecting the reader from `path`'s extension.
pub fn load_mesh(path: &Path) -> Result<ImportedMesh> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default();
    match extension
        .trim_start_matches('.')
        .to_ascii_lowercase()
        .as_str()
    {
        "obj" => read_obj(path),
        "ply" => read_ply(path),
        "glb" => read_glb(path),
        "stl" => read_stl(path),
        other => Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "unsupported mesh input extension `{other}` for `{}`; use .glb, .obj, .ply or .stl",
                path.display()
            ),
        }),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn validation(message: String) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation { message }
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })
}

/// Shared end-of-load validation. Drops index-degenerate triangles, checks
/// finiteness and index ranges, and never reorders vertices or faces.
fn finalize_import(
    vertices: Vec<f32>,
    faces: Vec<u32>,
    path: &Path,
    format: &str,
) -> Result<ImportedMesh> {
    if vertices.is_empty() {
        return Err(validation(format!(
            "{format} `{}` contains no vertices",
            path.display()
        )));
    }
    if vertices.len() % 3 != 0 {
        return Err(validation(format!(
            "{format} `{}` vertex buffer length is not a multiple of 3",
            path.display()
        )));
    }
    if vertices.len() / 3 > u32::MAX as usize {
        return Err(validation(format!(
            "{format} `{}` exceeds the u32 vertex index range (> {} vertices)",
            path.display(),
            u32::MAX
        )));
    }
    for (vertex_index, vertex) in vertices.chunks_exact(3).enumerate() {
        if !vertex.iter().all(|value| value.is_finite()) {
            return Err(validation(format!(
                "{format} `{}` vertex {vertex_index} has non-finite coordinates",
                path.display()
            )));
        }
    }
    if faces.is_empty() {
        return Err(validation(format!(
            "{format} `{}` contains no triangles",
            path.display()
        )));
    }
    let vertex_count = vertices.len() / 3;
    for (triangle_index, triangle) in faces.chunks_exact(3).enumerate() {
        for &index in triangle {
            if index as usize >= vertex_count {
                return Err(validation(format!(
                    "{format} `{}` triangle {triangle_index} references vertex {index} outside the {vertex_count}-vertex buffer",
                    path.display()
                )));
            }
        }
    }
    let mut kept = Vec::with_capacity(faces.len());
    let mut dropped = 0usize;
    for triangle in faces.chunks_exact(3) {
        if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[0] == triangle[2] {
            dropped += 1;
            continue;
        }
        kept.extend_from_slice(triangle);
    }
    if kept.is_empty() {
        return Err(validation(format!(
            "{format} `{}` has no usable triangles: all {dropped} triangles were index-degenerate",
            path.display()
        )));
    }
    Ok(ImportedMesh {
        vertices,
        faces: kept,
    })
}

/// Fan-triangulate one polygon face and push it onto `faces`.
fn push_triangulated_face(
    indices: &[usize],
    faces: &mut Vec<u32>,
    vertex_count: Option<usize>,
    error: impl Fn(String) -> Lux3dError,
) -> Result<()> {
    if indices.len() < 3 {
        // Points/lines inside a face element are skipped; they carry no area.
        return Ok(());
    }
    if let Some(vertex_count) = vertex_count {
        for &index in indices {
            if index >= vertex_count {
                return Err(error(format!(
                    "face references vertex {index} outside the {vertex_count}-vertex element"
                )));
            }
        }
    }
    for window in 1..indices.len() - 1 {
        let triangle = [indices[0], indices[window], indices[window + 1]];
        let mut pushed = [0u32; 3];
        for (slot, &index) in triangle.iter().enumerate() {
            pushed[slot] = u32::try_from(index)
                .map_err(|_| error("vertex indices exceed the u32 index range".to_string()))?;
        }
        faces.extend_from_slice(&pushed);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// OBJ
// ---------------------------------------------------------------------------

fn obj_error(path: &Path, line_number: usize, message: impl std::fmt::Display) -> Lux3dError {
    validation(format!(
        "OBJ `{}` line {line_number}: {message}",
        path.display()
    ))
}

/// Read an ASCII Wavefront OBJ mesh.
///
/// `v x y z [w]` defines vertices (the optional `w` is ignored) and `f` lines
/// define faces with `v`, `v/vt`, `v//vn` or `v/vt/vn` tokens using 1-based
/// positive or negative relative indices. Polygons are fan-triangulated;
/// `vt`, `vn`, groups, materials, comments and unknown keywords are ignored,
/// and multiple objects merge into one mesh.
pub fn read_obj(path: &Path) -> Result<ImportedMesh> {
    let bytes = read_file(path)?;
    let text = String::from_utf8(bytes)
        .map_err(|_| validation(format!("OBJ `{}` is not valid UTF-8", path.display())))?;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut tokens = trimmed.split_whitespace();
        match tokens.next() {
            Some("v") => {
                let mut vertex = [0.0f32; 3];
                for axis in 0..3 {
                    let token = tokens.next().ok_or_else(|| {
                        obj_error(path, line_number, "`v` line needs x y z coordinates")
                    })?;
                    vertex[axis] = token.parse::<f32>().map_err(|_| {
                        obj_error(
                            path,
                            line_number,
                            format!("`v` coordinate `{token}` is not a number"),
                        )
                    })?;
                }
                vertices.extend_from_slice(&vertex);
                // A trailing `w` component is accepted and ignored.
            }
            Some("f") => {
                let vertex_count = vertices.len() / 3;
                let mut polygon = Vec::new();
                for token in tokens {
                    let vertex_token = token.split('/').next().unwrap_or_default();
                    let index: i64 = if vertex_token.is_empty() {
                        return Err(obj_error(
                            path,
                            line_number,
                            format!("face token `{token}` has no vertex index"),
                        ));
                    } else {
                        token.parse::<i64>().or_else(|_| {
                            vertex_token.parse::<i64>().map_err(|_| {
                                obj_error(
                                    path,
                                    line_number,
                                    format!("face index `{vertex_token}` is not an integer"),
                                )
                            })
                        })?
                    };
                    let resolved = resolve_obj_index(index, vertex_count).ok_or_else(|| {
                        obj_error(
                            path,
                            line_number,
                            format!("face index {index} is outside the {vertex_count} vertices parsed so far"),
                        )
                    })?;
                    polygon.push(resolved);
                }
                if polygon.len() < 3 {
                    return Err(obj_error(
                        path,
                        line_number,
                        format!(
                            "face has {} vertices; at least 3 are required",
                            polygon.len()
                        ),
                    ));
                }
                let error = |message: String| obj_error(path, line_number, message);
                push_triangulated_face(&polygon, &mut faces, Some(vertex_count), error)?;
            }
            // `vt`, `vn`, `vp`, `g`, `o`, `s`, `usemtl`, `mtllib`, ... ignored.
            _ => {}
        }
    }
    finalize_import(vertices, faces, path, "OBJ")
}

/// Resolve a 1-based OBJ face index (positive or negative-relative).
fn resolve_obj_index(index: i64, vertex_count: usize) -> Option<usize> {
    if index == 0 {
        return None;
    }
    let resolved = if index > 0 {
        usize::try_from(index).ok()?.checked_sub(1)?
    } else {
        let offset = isize::try_from(index).ok()?;
        vertex_count.checked_add_signed(offset)?
    };
    (resolved < vertex_count).then_some(resolved)
}

// ---------------------------------------------------------------------------
// PLY
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyScalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyScalar {
    fn parse(name: &str) -> Option<Self> {
        match name {
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

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Debug)]
enum PlyProperty {
    Scalar {
        name: String,
        ty: PlyScalar,
    },
    List {
        name: String,
        count: PlyScalar,
        item: PlyScalar,
    },
}

impl PlyProperty {
    fn name(&self) -> &str {
        match self {
            Self::Scalar { name, .. } | Self::List { name, .. } => name,
        }
    }
}

#[derive(Debug)]
struct PlyElement {
    name: String,
    count: usize,
    properties: Vec<PlyProperty>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyEncoding {
    Ascii,
    BinaryLittleEndian,
}

#[derive(Debug)]
struct PlyHeader {
    encoding: PlyEncoding,
    elements: Vec<PlyElement>,
    body_offset: usize,
}

#[derive(Debug, Clone, Copy)]
enum PlyValue {
    Int(i64),
    Float(f64),
}

impl PlyValue {
    fn as_coord(self) -> f64 {
        match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }

    fn as_index(self, context: &str, path: &Path) -> Result<usize> {
        let value = match self {
            Self::Int(value) => value,
            Self::Float(value) => {
                if value < 0.0 || value.fract() != 0.0 {
                    return Err(ply_error(
                        path,
                        format!("{context}: index {value} is not a non-negative integer"),
                    ));
                }
                value as i64
            }
        };
        usize::try_from(value)
            .map_err(|_| ply_error(path, format!("{context}: index {value} is out of range")))
    }

    fn as_count(self, context: &str, path: &Path) -> Result<usize> {
        let value = match self {
            Self::Int(value) => value,
            Self::Float(value) => {
                if value < 0.0 || value.fract() != 0.0 {
                    return Err(ply_error(
                        path,
                        format!("{context}: list count {value} is not a non-negative integer"),
                    ));
                }
                value as i64
            }
        };
        usize::try_from(value).map_err(|_| {
            ply_error(
                path,
                format!("{context}: list count {value} is out of range"),
            )
        })
    }
}

fn ply_error(path: &Path, message: impl std::fmt::Display) -> Lux3dError {
    validation(format!("PLY `{}`: {message}", path.display()))
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Read a PLY mesh (ASCII or binary little-endian).
///
/// Vertex coordinates are taken from the `x`/`y`/`z` scalar properties in
/// whatever order they appear, and faces from the `vertex_indices` or
/// `vertex_index` list property. Polygons with more than three vertices are
/// fan-triangulated; face entries with fewer than three indices are skipped.
pub fn read_ply(path: &Path) -> Result<ImportedMesh> {
    let bytes = read_file(path)?;
    let header = parse_ply_header(&bytes, path)?;
    let (vertices, faces) = match header.encoding {
        PlyEncoding::Ascii => ply_parse_ascii(&bytes, &header, path)?,
        PlyEncoding::BinaryLittleEndian => ply_parse_binary(&bytes, &header, path)?,
    };
    finalize_import(vertices, faces, path, "PLY")
}

fn parse_ply_header(bytes: &[u8], path: &Path) -> Result<PlyHeader> {
    if !bytes.starts_with(b"ply") {
        return Err(ply_error(path, "not a PLY file (missing `ply` magic)"));
    }
    let body_offset = find_ply_body_offset(bytes, path)?;
    let header_text = String::from_utf8_lossy(&bytes[..body_offset]);
    let mut encoding: Option<PlyEncoding> = None;
    let mut elements: Vec<PlyElement> = Vec::new();
    for line in header_text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            Some("ply") | Some("comment") | Some("obj_info") => {}
            Some("format") => {
                let which = tokens.next().unwrap_or_default();
                encoding = Some(match which {
                    "ascii" => PlyEncoding::Ascii,
                    "binary_little_endian" => PlyEncoding::BinaryLittleEndian,
                    "binary_big_endian" => {
                        return Err(ply_error(
                            path,
                            "big-endian PLY (`format binary_big_endian`) is not supported; convert the file to `binary_little_endian` or `ascii`",
                        ));
                    }
                    other => {
                        return Err(ply_error(
                            path,
                            format!(
                                "unsupported PLY format `{other}`; supported formats are `ascii 1.0` and `binary_little_endian 1.0` (no compressed PLY variants)"
                            ),
                        ));
                    }
                });
            }
            Some("element") => {
                let name = tokens
                    .next()
                    .ok_or_else(|| ply_error(path, "`element` line needs a name"))?;
                let count_token = tokens
                    .next()
                    .ok_or_else(|| ply_error(path, "`element` line needs a count"))?;
                let count = count_token.parse::<usize>().map_err(|_| {
                    ply_error(
                        path,
                        format!("element count `{count_token}` is not a valid integer"),
                    )
                })?;
                elements.push(PlyElement {
                    name: name.to_string(),
                    count,
                    properties: Vec::new(),
                });
            }
            Some("property") => {
                let element = elements.last_mut().ok_or_else(|| {
                    ply_error(path, "`property` line appears before any `element`")
                })?;
                match tokens.next() {
                    Some("list") => {
                        let count_ty = PlyScalar::parse(tokens.next().unwrap_or_default())
                            .ok_or_else(|| {
                                ply_error(path, "list property has an unknown count type")
                            })?;
                        let item_ty = PlyScalar::parse(tokens.next().unwrap_or_default())
                            .ok_or_else(|| {
                                ply_error(path, "list property has an unknown item type")
                            })?;
                        let name = tokens
                            .next()
                            .ok_or_else(|| ply_error(path, "list property needs a name"))?;
                        element.properties.push(PlyProperty::List {
                            name: name.to_string(),
                            count: count_ty,
                            item: item_ty,
                        });
                    }
                    Some(type_name) => {
                        let ty = PlyScalar::parse(type_name).ok_or_else(|| {
                            ply_error(
                                path,
                                format!("property type `{type_name}` is not supported"),
                            )
                        })?;
                        let name = tokens
                            .next()
                            .ok_or_else(|| ply_error(path, "property needs a name"))?;
                        element.properties.push(PlyProperty::Scalar {
                            name: name.to_string(),
                            ty,
                        });
                    }
                    None => return Err(ply_error(path, "`property` line is incomplete")),
                }
            }
            Some("end_header") => break,
            _ => {}
        }
    }
    let encoding = encoding.ok_or_else(|| ply_error(path, "PLY header has no `format` line"))?;
    Ok(PlyHeader {
        encoding,
        elements,
        body_offset,
    })
}

/// Locate the `end_header` line and return the byte offset just past it.
fn find_ply_body_offset(bytes: &[u8], path: &Path) -> Result<usize> {
    let mut search_from = 0usize;
    loop {
        let relative = find_subslice(&bytes[search_from..], b"end_header")
            .ok_or_else(|| ply_error(path, "header is truncated: no `end_header` line found"))?;
        let start = search_from + relative;
        let after = start + b"end_header".len();
        let at_line_start = start == 0 || bytes[start - 1] == b'\n';
        let body_offset =
            if bytes.get(after) == Some(&b'\r') && bytes.get(after + 1) == Some(&b'\n') {
                Some(after + 2)
            } else if bytes.get(after) == Some(&b'\n') {
                Some(after + 1)
            } else {
                None
            };
        if at_line_start {
            if let Some(offset) = body_offset {
                return Ok(offset);
            }
        }
        search_from = start + 1;
    }
}

fn ply_element_index(header: &PlyHeader, names: &[&str]) -> Option<usize> {
    header
        .elements
        .iter()
        .position(|element| names.contains(&element.name.as_str()))
}

/// Map `x`/`y`/`z` scalar property indices to coordinate axes.
fn ply_vertex_axes(element: &PlyElement, path: &Path) -> Result<[Option<usize>; 3]> {
    let mut axes = [None; 3];
    for (prop_index, property) in element.properties.iter().enumerate() {
        if let PlyProperty::Scalar { name, .. } = property {
            let axis = match name.as_str() {
                "x" => Some(0),
                "y" => Some(1),
                "z" => Some(2),
                _ => None,
            };
            if let Some(axis) = axis {
                axes[axis] = Some(prop_index);
            }
        }
    }
    if axes.iter().any(|axis| axis.is_none()) {
        return Err(ply_error(
            path,
            "vertex element is missing `x`/`y`/`z` scalar properties",
        ));
    }
    Ok(axes)
}

/// Find the face element's vertex-index list property.
fn ply_face_list_property(element: &PlyElement, path: &Path) -> Result<usize> {
    for (prop_index, property) in element.properties.iter().enumerate() {
        if let PlyProperty::List { name, .. } = property {
            if name == "vertex_indices" || name == "vertex_index" {
                return Ok(prop_index);
            }
        }
    }
    // Fall back to the first list property in the face element.
    for (prop_index, property) in element.properties.iter().enumerate() {
        if matches!(property, PlyProperty::List { .. }) {
            return Ok(prop_index);
        }
    }
    Err(ply_error(
        path,
        "face element has no list property (expected `vertex_indices` or `vertex_index`)",
    ))
}

struct AsciiTokenizer<'a> {
    rest: &'a str,
}

impl<'a> AsciiTokenizer<'a> {
    fn next(&mut self) -> Option<&'a str> {
        let rest = self.rest.trim_start();
        self.rest = rest;
        if rest.is_empty() {
            return None;
        }
        let end = rest
            .find(|character: char| character.is_ascii_whitespace())
            .unwrap_or(rest.len());
        let token = &rest[..end];
        self.rest = &rest[end..];
        Some(token)
    }

    fn expect(&mut self, context: &str, path: &Path) -> Result<&'a str> {
        self.next()
            .ok_or_else(|| ply_error(path, format!("{context}: ASCII PLY body ended early")))
    }
}

fn ply_ascii_scalar(
    tokens: &mut AsciiTokenizer<'_>,
    ty: PlyScalar,
    context: &str,
    path: &Path,
) -> Result<PlyValue> {
    let token = tokens.expect(context, path)?;
    match ty {
        PlyScalar::F32 | PlyScalar::F64 => {
            let value: f64 = token
                .parse()
                .map_err(|_| ply_error(path, format!("{context}: `{token}` is not a number")))?;
            Ok(PlyValue::Float(value))
        }
        _ => {
            let value: i64 = token
                .parse()
                .map_err(|_| ply_error(path, format!("{context}: `{token}` is not an integer")))?;
            Ok(PlyValue::Int(value))
        }
    }
}

/// Read one ASCII PLY item, capturing coordinates and face indices when asked.
fn ply_read_ascii_item(
    tokens: &mut AsciiTokenizer<'_>,
    element: &PlyElement,
    axes: &[Option<usize>; 3],
    face_list: Option<usize>,
    context: &str,
    path: &Path,
) -> Result<([f64; 3], Vec<usize>)> {
    let mut coords = [0.0f64; 3];
    let mut list_indices = Vec::new();
    for (prop_index, property) in element.properties.iter().enumerate() {
        let prop_context = format!("{context} property `{}`", property.name());
        match property {
            PlyProperty::Scalar { ty, .. } => {
                let value = ply_ascii_scalar(tokens, *ty, &prop_context, path)?;
                for axis in 0..3 {
                    if axes[axis] == Some(prop_index) {
                        coords[axis] = value.as_coord();
                    }
                }
            }
            PlyProperty::List { count, item, .. } => {
                let count_value = ply_ascii_scalar(tokens, *count, &prop_context, path)?;
                let count = count_value.as_count(&prop_context, path)?;
                if face_list == Some(prop_index) {
                    for _ in 0..count {
                        let value = ply_ascii_scalar(tokens, *item, &prop_context, path)?;
                        list_indices.push(value.as_index(&prop_context, path)?);
                    }
                } else {
                    for _ in 0..count {
                        ply_ascii_scalar(tokens, *item, &prop_context, path)?;
                    }
                }
            }
        }
    }
    Ok((coords, list_indices))
}

fn ply_parse_ascii(bytes: &[u8], header: &PlyHeader, path: &Path) -> Result<(Vec<f32>, Vec<u32>)> {
    let body = std::str::from_utf8(&bytes[header.body_offset..])
        .map_err(|_| ply_error(path, "ASCII PLY body is not valid UTF-8"))?;
    let vertex_element = ply_element_index(header, &["vertex", "vertices"]);
    let face_element = ply_element_index(header, &["face", "faces"]);
    if vertex_element.is_none() {
        return Err(ply_error(path, "PLY has no `vertex` element"));
    }
    if face_element.is_none() {
        return Err(ply_error(
            path,
            "PLY has no `face` element; triangle meshes need a face element",
        ));
    }
    let mut tokens = AsciiTokenizer { rest: body };
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut vertex_count_so_far: Option<usize> = None;
    for (element_index, element) in header.elements.iter().enumerate() {
        let is_vertex = Some(element_index) == vertex_element;
        let is_face = Some(element_index) == face_element;
        let axes = if is_vertex {
            ply_vertex_axes(element, path)?
        } else {
            [None; 3]
        };
        let face_list = if is_face {
            Some(ply_face_list_property(element, path)?)
        } else {
            None
        };
        for item in 0..element.count {
            let context = format!("{} element item {item}", element.name);
            let (coords, list) =
                ply_read_ascii_item(&mut tokens, element, &axes, face_list, &context, path)?;
            if is_vertex {
                vertices.extend_from_slice(&[coords[0] as f32, coords[1] as f32, coords[2] as f32]);
            } else if is_face {
                let error = |message: String| ply_error(path, format!("{context}: {message}"));
                push_triangulated_face(&list, &mut faces, vertex_count_so_far, error)?;
            }
        }
        if is_vertex {
            vertex_count_so_far = Some(element.count);
        }
    }
    Ok((vertices, faces))
}

fn ply_binary_check(
    bytes: &[u8],
    cursor: usize,
    size: usize,
    context: &str,
    path: &Path,
) -> Result<()> {
    let end = cursor
        .checked_add(size)
        .ok_or_else(|| ply_error(path, format!("{context}: item size overflows")))?;
    if end > bytes.len() {
        return Err(ply_error(
            path,
            format!("{context}: binary PLY body is truncated"),
        ));
    }
    Ok(())
}

fn ply_read_binary_scalar(
    bytes: &[u8],
    cursor: &mut usize,
    ty: PlyScalar,
    context: &str,
    path: &Path,
) -> Result<PlyValue> {
    let size = ty.size();
    ply_binary_check(bytes, *cursor, size, context, path)?;
    let start = *cursor;
    *cursor = start + size;
    let chunk = &bytes[start..start + size];
    let value = match ty {
        // Slice lengths below are fixed by `ty.size()`, so the conversions
        // cannot fail.
        PlyScalar::I8 => {
            PlyValue::Int(i8::from_le_bytes(chunk.try_into().expect("1-byte slice")) as i64)
        }
        PlyScalar::U8 => PlyValue::Int(chunk[0] as i64),
        PlyScalar::I16 => {
            PlyValue::Int(i16::from_le_bytes(chunk.try_into().expect("2-byte slice")) as i64)
        }
        PlyScalar::U16 => {
            PlyValue::Int(u16::from_le_bytes(chunk.try_into().expect("2-byte slice")) as i64)
        }
        PlyScalar::I32 => {
            PlyValue::Int(i32::from_le_bytes(chunk.try_into().expect("4-byte slice")) as i64)
        }
        PlyScalar::U32 => {
            PlyValue::Int(u32::from_le_bytes(chunk.try_into().expect("4-byte slice")) as i64)
        }
        PlyScalar::F32 => {
            PlyValue::Float(f32::from_le_bytes(chunk.try_into().expect("4-byte slice")) as f64)
        }
        PlyScalar::F64 => {
            PlyValue::Float(f64::from_le_bytes(chunk.try_into().expect("8-byte slice")))
        }
    };
    Ok(value)
}

/// Read one binary little-endian PLY item, capturing coordinates and face
/// indices when asked; other properties are skipped by byte count.
fn ply_read_binary_item(
    bytes: &[u8],
    cursor: &mut usize,
    element: &PlyElement,
    axes: &[Option<usize>; 3],
    face_list: Option<usize>,
    context: &str,
    path: &Path,
) -> Result<([f64; 3], Vec<usize>)> {
    let mut coords = [0.0f64; 3];
    let mut list_indices = Vec::new();
    for (prop_index, property) in element.properties.iter().enumerate() {
        let prop_context = format!("{context} property `{}`", property.name());
        match property {
            PlyProperty::Scalar { ty, .. } => {
                let value = ply_read_binary_scalar(bytes, cursor, *ty, &prop_context, path)?;
                for axis in 0..3 {
                    if axes[axis] == Some(prop_index) {
                        coords[axis] = value.as_coord();
                    }
                }
            }
            PlyProperty::List { count, item, .. } => {
                let count_value =
                    ply_read_binary_scalar(bytes, cursor, *count, &prop_context, path)?;
                let count = count_value.as_count(&prop_context, path)?;
                if face_list == Some(prop_index) {
                    for _ in 0..count {
                        let value =
                            ply_read_binary_scalar(bytes, cursor, *item, &prop_context, path)?;
                        list_indices.push(value.as_index(&prop_context, path)?);
                    }
                } else {
                    let total = count.checked_mul(item.size()).ok_or_else(|| {
                        ply_error(path, format!("{prop_context}: list length overflows"))
                    })?;
                    ply_binary_check(bytes, *cursor, total, &prop_context, path)?;
                    *cursor += total;
                }
            }
        }
    }
    Ok((coords, list_indices))
}

fn ply_parse_binary(bytes: &[u8], header: &PlyHeader, path: &Path) -> Result<(Vec<f32>, Vec<u32>)> {
    let vertex_element = ply_element_index(header, &["vertex", "vertices"]);
    let face_element = ply_element_index(header, &["face", "faces"]);
    if vertex_element.is_none() {
        return Err(ply_error(path, "PLY has no `vertex` element"));
    }
    if face_element.is_none() {
        return Err(ply_error(
            path,
            "PLY has no `face` element; triangle meshes need a face element",
        ));
    }
    let mut cursor = header.body_offset;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut vertex_count_so_far: Option<usize> = None;
    for (element_index, element) in header.elements.iter().enumerate() {
        let is_vertex = Some(element_index) == vertex_element;
        let is_face = Some(element_index) == face_element;
        let axes = if is_vertex {
            ply_vertex_axes(element, path)?
        } else {
            [None; 3]
        };
        let face_list = if is_face {
            Some(ply_face_list_property(element, path)?)
        } else {
            None
        };
        for item in 0..element.count {
            let context = format!("{} element item {item}", element.name);
            let (coords, list) = ply_read_binary_item(
                bytes,
                &mut cursor,
                element,
                &axes,
                face_list,
                &context,
                path,
            )?;
            if is_vertex {
                vertices.extend_from_slice(&[coords[0] as f32, coords[1] as f32, coords[2] as f32]);
            } else if is_face {
                let error = |message: String| ply_error(path, format!("{context}: {message}"));
                push_triangulated_face(&list, &mut faces, vertex_count_so_far, error)?;
            }
        }
        if is_vertex {
            vertex_count_so_far = Some(element.count);
        }
    }
    Ok((vertices, faces))
}

// ---------------------------------------------------------------------------
// GLB
// ---------------------------------------------------------------------------

const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F_534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E_4942; // "BIN\0"

#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct GltfDocument {
    scene: Option<usize>,
    scenes: Vec<GltfScene>,
    nodes: Vec<GltfNode>,
    meshes: Vec<GltfMesh>,
    accessors: Vec<GltfAccessor>,
    buffer_views: Vec<GltfBufferView>,
    buffers: Vec<GltfBuffer>,
    extensions_used: Vec<String>,
    extensions_required: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GltfScene {
    nodes: Vec<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GltfNode {
    mesh: Option<usize>,
    children: Vec<usize>,
    matrix: Option<Vec<f64>>,
    translation: Option<Vec<f64>>,
    rotation: Option<Vec<f64>>,
    scale: Option<Vec<f64>>,
    // `skin` and other unknown node fields are ignored by serde; skinned
    // meshes are loaded at their bind pose with only the node transform.
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GltfMesh {
    primitives: Vec<GltfPrimitive>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GltfPrimitive {
    attributes: std::collections::BTreeMap<String, usize>,
    indices: Option<usize>,
    mode: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct GltfAccessor {
    buffer_view: Option<usize>,
    byte_offset: usize,
    component_type: u32,
    count: usize,
    #[serde(rename = "type")]
    ty: String,
    sparse: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct GltfBufferView {
    buffer: usize,
    byte_offset: usize,
    byte_stride: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GltfBuffer {
    uri: Option<String>,
}

fn glb_error(path: &Path, message: impl std::fmt::Display) -> Lux3dError {
    validation(format!("GLB `{}`: {message}", path.display()))
}

fn glb_u32(bytes: &[u8], offset: usize, path: &Path) -> Result<u32> {
    let chunk = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| glb_error(path, "GLB container is truncated"))?;
    Ok(u32::from_le_bytes(chunk.try_into().expect("4-byte slice")))
}

/// Read a glTF 2.0 binary (`.glb`) triangle mesh.
///
/// Triangle primitives from the active scene (or every mesh at identity when
/// the file has no scene graph) are concatenated with a per-primitive vertex
/// offset. Static node transforms (`matrix` or TRS) are applied to positions;
/// winding order is preserved. Draco/meshopt compression, sparse accessors,
/// external buffers and non-float32 positions are rejected.
pub fn read_glb(path: &Path) -> Result<ImportedMesh> {
    let bytes = read_file(path)?;
    let (json_chunk, bin_chunk) = parse_glb_container(&bytes, path)?;
    let document: GltfDocument = serde_json::from_slice(json_chunk)
        .map_err(|error| glb_error(path, format!("invalid glTF JSON: {error}")))?;
    glb_reject_unsupported_extensions(&document, path)?;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    glb_collect_meshes(&document, bin_chunk, &mut vertices, &mut faces, path)?;
    finalize_import(vertices, faces, path, "GLB")
}

fn parse_glb_container<'a>(bytes: &'a [u8], path: &Path) -> Result<(&'a [u8], &'a [u8])> {
    if bytes.len() < 12 {
        return Err(glb_error(path, "file is too short to be a GLB container"));
    }
    if glb_u32(bytes, 0, path)? != GLB_MAGIC {
        return Err(glb_error(path, "not a GLB file (missing `glTF` magic)"));
    }
    let version = glb_u32(bytes, 4, path)?;
    if version != 2 {
        return Err(glb_error(
            path,
            format!("unsupported glTF container version {version}; only glTF 2.0 is supported"),
        ));
    }
    let declared_length = glb_u32(bytes, 8, path)? as usize;
    if declared_length > bytes.len() {
        return Err(glb_error(
            path,
            format!(
                "container declares {declared_length} bytes but the file holds {} bytes (truncated GLB)",
                bytes.len()
            ),
        ));
    }
    let limit = declared_length;
    let mut cursor = 12usize;
    let mut json_chunk: Option<&[u8]> = None;
    let mut bin_chunk: Option<&[u8]> = None;
    while cursor + 8 <= limit {
        let length = glb_u32(bytes, cursor, path)? as usize;
        let chunk_type = glb_u32(bytes, cursor + 4, path)?;
        let start = cursor + 8;
        let end = start
            .checked_add(length)
            .ok_or_else(|| glb_error(path, "GLB chunk length overflows"))?;
        if end > limit {
            return Err(glb_error(
                path,
                "GLB chunk extends past the declared container length",
            ));
        }
        let payload = &bytes[start..end];
        match chunk_type {
            CHUNK_JSON => {
                if json_chunk.is_some() {
                    return Err(glb_error(path, "GLB contains more than one JSON chunk"));
                }
                json_chunk = Some(payload);
            }
            CHUNK_BIN => {
                if bin_chunk.is_some() {
                    return Err(glb_error(path, "GLB contains more than one BIN chunk"));
                }
                bin_chunk = Some(payload);
            }
            _ => {}
        }
        cursor = end;
    }
    let json = json_chunk.ok_or_else(|| glb_error(path, "GLB has no JSON chunk"))?;
    Ok((json, bin_chunk.unwrap_or(&[])))
}

fn glb_reject_unsupported_extensions(document: &GltfDocument, path: &Path) -> Result<()> {
    const REJECTED: [&str; 2] = ["KHR_draco_mesh_compression", "EXT_meshopt_compression"];
    for extension in document
        .extensions_used
        .iter()
        .chain(&document.extensions_required)
    {
        if REJECTED.contains(&extension.as_str()) {
            return Err(glb_error(
                path,
                format!(
                    "uses the `{extension}` compression extension, which is not supported; export an uncompressed GLB instead"
                ),
            ));
        }
    }
    Ok(())
}

const GLB_IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn glb_collect_meshes(
    document: &GltfDocument,
    bin: &[u8],
    vertices: &mut Vec<f32>,
    faces: &mut Vec<u32>,
    path: &Path,
) -> Result<()> {
    if document
        .buffers
        .first()
        .map(|buffer| buffer.uri.is_some())
        .unwrap_or(false)
    {
        return Err(glb_error(
            path,
            "references an external buffer URI; only self-contained GLB files with an embedded BIN chunk are supported",
        ));
    }
    let mut in_stack = HashSet::new();
    match document.scene {
        Some(scene_index) if document.scenes.get(scene_index).is_some() => {
            let roots = document.scenes[scene_index].nodes.clone();
            for node_index in roots {
                glb_visit_node(
                    document,
                    bin,
                    node_index,
                    &GLB_IDENTITY,
                    0,
                    &mut in_stack,
                    vertices,
                    faces,
                    path,
                )?;
            }
        }
        _ if !document.scenes.is_empty() => {
            let roots = document.scenes[0].nodes.clone();
            for node_index in roots {
                glb_visit_node(
                    document,
                    bin,
                    node_index,
                    &GLB_IDENTITY,
                    0,
                    &mut in_stack,
                    vertices,
                    faces,
                    path,
                )?;
            }
        }
        // No scene graph: every mesh is loaded at identity.
        _ => {
            for mesh_index in 0..document.meshes.len() {
                glb_visit_mesh(
                    document,
                    bin,
                    mesh_index,
                    &GLB_IDENTITY,
                    vertices,
                    faces,
                    path,
                )?;
            }
        }
    }
    Ok(())
}

fn glb_visit_node(
    document: &GltfDocument,
    bin: &[u8],
    node_index: usize,
    parent: &[f64; 16],
    depth: usize,
    in_stack: &mut HashSet<usize>,
    vertices: &mut Vec<f32>,
    faces: &mut Vec<u32>,
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
    let local = glb_node_matrix(node, path)?;
    let world = mat4_mul(parent, &local);
    if let Some(mesh_index) = node.mesh {
        glb_visit_mesh(document, bin, mesh_index, &world, vertices, faces, path)?;
    }
    for child in &node.children {
        glb_visit_node(
            document,
            bin,
            *child,
            &world,
            depth + 1,
            in_stack,
            vertices,
            faces,
            path,
        )?;
    }
    in_stack.remove(&node_index);
    Ok(())
}

fn glb_visit_mesh(
    document: &GltfDocument,
    bin: &[u8],
    mesh_index: usize,
    world: &[f64; 16],
    vertices: &mut Vec<f32>,
    faces: &mut Vec<u32>,
    path: &Path,
) -> Result<()> {
    let mesh = document
        .meshes
        .get(mesh_index)
        .ok_or_else(|| glb_error(path, format!("mesh index {mesh_index} is out of range")))?;
    for (prim_index, primitive) in mesh.primitives.iter().enumerate() {
        glb_read_primitive(
            document, bin, primitive, mesh_index, prim_index, world, vertices, faces, path,
        )?;
    }
    Ok(())
}

/// Column-major 4x4 matrix for a node: explicit `matrix` field, or TRS
/// composed as `T * R * S` (rotation quaternion is used as-is, per the glTF
/// requirement of unit quaternions).
fn glb_node_matrix(node: &GltfNode, path: &Path) -> Result<[f64; 16]> {
    if let Some(matrix) = &node.matrix {
        if matrix.len() != 16 {
            return Err(glb_error(
                path,
                format!("node matrix must have 16 values, found {}", matrix.len()),
            ));
        }
        let mut out = [0.0f64; 16];
        out.copy_from_slice(matrix);
        return Ok(out);
    }
    let scale = node.scale.as_deref().unwrap_or(&[1.0, 1.0, 1.0]);
    if scale.len() != 3 {
        return Err(glb_error(
            path,
            format!("node scale must have 3 values, found {}", scale.len()),
        ));
    }
    let mut out = [0.0f64; 16];
    out[0] = scale[0];
    out[5] = scale[1];
    out[10] = scale[2];
    out[15] = 1.0;
    if let Some(rotation) = &node.rotation {
        if rotation.len() != 4 {
            return Err(glb_error(
                path,
                format!(
                    "node rotation quaternion must have 4 values, found {}",
                    rotation.len()
                ),
            ));
        }
        let rotation_matrix = quaternion_matrix(rotation[0], rotation[1], rotation[2], rotation[3]);
        // out = R * S: scale the rotation columns.
        for column in 0..3 {
            for row in 0..3 {
                out[column * 4 + row] = rotation_matrix[column * 4 + row] * scale[column];
            }
        }
    }
    let translation = node.translation.as_deref().unwrap_or(&[0.0, 0.0, 0.0]);
    if translation.len() != 3 {
        return Err(glb_error(
            path,
            format!(
                "node translation must have 3 values, found {}",
                translation.len()
            ),
        ));
    }
    out[12] = translation[0];
    out[13] = translation[1];
    out[14] = translation[2];
    Ok(out)
}

/// Column-major rotation matrix from a `(x, y, z, w)` quaternion.
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

/// Multiply two column-major 4x4 matrices (`a * b`).
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

/// Apply a column-major affine 4x4 matrix to a position (w assumed 1).
fn glb_transform_vertex(matrix: &[f64; 16], x: f32, y: f32, z: f32) -> [f32; 3] {
    let (x, y, z) = (f64::from(x), f64::from(y), f64::from(z));
    [
        (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) as f32,
        (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) as f32,
        (matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]) as f32,
    ]
}

fn glb_read_primitive(
    document: &GltfDocument,
    bin: &[u8],
    primitive: &GltfPrimitive,
    mesh_index: usize,
    prim_index: usize,
    world: &[f64; 16],
    vertices: &mut Vec<f32>,
    faces: &mut Vec<u32>,
    path: &Path,
) -> Result<()> {
    // Only TRIANGLES (mode 4, the default) are loaded; points, lines and
    // strips carry no closed-surface geometry for texturing.
    if primitive.mode.unwrap_or(4) != 4 {
        return Ok(());
    }
    let context = format!("mesh {mesh_index} primitive {prim_index}");
    let position_accessor_index = primitive
        .attributes
        .get("POSITION")
        .ok_or_else(|| glb_error(path, format!("{context} has no POSITION attribute")))?;
    let position_accessor = document
        .accessors
        .get(*position_accessor_index)
        .ok_or_else(|| {
            glb_error(
                path,
                format!("{context}: POSITION accessor {position_accessor_index} is out of range"),
            )
        })?;
    if position_accessor.component_type != 5126 {
        return Err(glb_error(
            path,
            format!(
                "{context}: POSITION accessor must use componentType 5126 (float32), found {}",
                position_accessor.component_type
            ),
        ));
    }
    if position_accessor.ty != "VEC3" {
        return Err(glb_error(
            path,
            format!(
                "{context}: POSITION accessor must be VEC3, found `{}`",
                position_accessor.ty
            ),
        ));
    }
    if position_accessor.sparse.is_some() {
        return Err(glb_error(
            path,
            format!("{context}: sparse accessors are not supported"),
        ));
    }
    let vertex_base = vertices.len() / 3;
    glb_read_positions(
        document,
        bin,
        position_accessor,
        world,
        vertices,
        path,
        &context,
    )?;
    let position_count = position_accessor.count;
    let indices = match primitive.indices {
        Some(accessor_index) => glb_read_indices(
            document,
            bin,
            accessor_index,
            position_count,
            path,
            &context,
        )?,
        None => {
            if position_count % 3 != 0 {
                return Err(glb_error(
                    path,
                    format!(
                        "{context}: non-indexed triangle primitive needs a POSITION count that is a multiple of 3, found {position_count}"
                    ),
                ));
            }
            (0..position_count).collect::<Vec<usize>>()
        }
    };
    for triangle in indices.chunks_exact(3) {
        let mut pushed = [0u32; 3];
        for (slot, &index) in triangle.iter().enumerate() {
            pushed[slot] = u32::try_from(vertex_base + index).map_err(|_| {
                glb_error(
                    path,
                    format!("{context}: vertex count exceeds the u32 index range"),
                )
            })?;
        }
        faces.extend_from_slice(&pushed);
    }
    Ok(())
}

fn glb_read_positions(
    document: &GltfDocument,
    bin: &[u8],
    accessor: &GltfAccessor,
    world: &[f64; 16],
    vertices: &mut Vec<f32>,
    path: &Path,
    context: &str,
) -> Result<()> {
    let element_size = 12usize; // VEC3 of float32
    let (base, stride) = glb_accessor_span(
        document,
        bin,
        accessor,
        element_size,
        path,
        context,
        "POSITION",
    )?;
    vertices.reserve(accessor.count * 3);
    for item in 0..accessor.count {
        let start = base + item * stride;
        let chunk = &bin[start..start + element_size];
        let x = f32::from_le_bytes(chunk[0..4].try_into().expect("4-byte slice"));
        let y = f32::from_le_bytes(chunk[4..8].try_into().expect("4-byte slice"));
        let z = f32::from_le_bytes(chunk[8..12].try_into().expect("4-byte slice"));
        vertices.extend_from_slice(&glb_transform_vertex(world, x, y, z));
    }
    Ok(())
}

fn glb_read_indices(
    document: &GltfDocument,
    bin: &[u8],
    accessor_index: usize,
    position_count: usize,
    path: &Path,
    context: &str,
) -> Result<Vec<usize>> {
    let accessor = document.accessors.get(accessor_index).ok_or_else(|| {
        glb_error(
            path,
            format!("{context}: indices accessor {accessor_index} is out of range"),
        )
    })?;
    if accessor.ty != "SCALAR" {
        return Err(glb_error(
            path,
            format!(
                "{context}: indices accessor must be SCALAR, found `{}`",
                accessor.ty
            ),
        ));
    }
    if accessor.sparse.is_some() {
        return Err(glb_error(
            path,
            format!("{context}: sparse accessors are not supported"),
        ));
    }
    let element_size = match accessor.component_type {
        5121 => 1usize, // u8
        5123 => 2usize, // u16
        5125 => 4usize, // u32
        other => {
            return Err(glb_error(
                path,
                format!(
                    "{context}: index componentType {other} is not supported; use 5121 (u8), 5123 (u16) or 5125 (u32)"
                ),
            ));
        }
    };
    if accessor.count % 3 != 0 {
        return Err(glb_error(
            path,
            format!(
                "{context}: triangle indices count must be a multiple of 3, found {}",
                accessor.count
            ),
        ));
    }
    let (base, stride) = glb_accessor_span(
        document,
        bin,
        accessor,
        element_size,
        path,
        context,
        "indices",
    )?;
    let mut indices = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let start = base + item * stride;
        let chunk = &bin[start..start + element_size];
        let raw = match element_size {
            1 => chunk[0] as u64,
            2 => u16::from_le_bytes(chunk.try_into().expect("2-byte slice")) as u64,
            _ => u32::from_le_bytes(chunk.try_into().expect("4-byte slice")) as u64,
        };
        let index = usize::try_from(raw).map_err(|_| {
            glb_error(
                path,
                format!("{context}: index value {raw} is out of range"),
            )
        })?;
        if index >= position_count {
            return Err(glb_error(
                path,
                format!(
                    "{context}: index {index} is outside the primitive's {position_count} vertices"
                ),
            ));
        }
        indices.push(index);
    }
    Ok(indices)
}

/// Validate a float/int accessor's buffer view and return the absolute byte
/// offset of its first element plus the per-element stride.
fn glb_accessor_span(
    document: &GltfDocument,
    bin: &[u8],
    accessor: &GltfAccessor,
    element_size: usize,
    path: &Path,
    context: &str,
    label: &str,
) -> Result<(usize, usize)> {
    if document.buffers.is_empty() {
        return Err(glb_error(
            path,
            format!(
                "{context}: {label} accessor needs data but the GLB has no embedded BIN buffer"
            ),
        ));
    }
    let view_index = accessor.buffer_view.ok_or_else(|| {
        glb_error(
            path,
            format!("{context}: {label} accessor has no bufferView"),
        )
    })?;
    let view = document.buffer_views.get(view_index).ok_or_else(|| {
        glb_error(
            path,
            format!("{context}: bufferView {view_index} is out of range"),
        )
    })?;
    if view.buffer != 0 {
        return Err(glb_error(
            path,
            format!(
                "{context}: geometry must reference the embedded buffer 0, found buffer {}",
                view.buffer
            ),
        ));
    }
    let stride = match view.byte_stride {
        Some(stride) if stride != 0 => {
            if stride < element_size {
                return Err(glb_error(
                    path,
                    format!(
                        "{context}: bufferView byteStride {stride} is smaller than the {element_size}-byte element"
                    ),
                ));
            }
            stride
        }
        _ => element_size,
    };
    let base = view
        .byte_offset
        .checked_add(accessor.byte_offset)
        .ok_or_else(|| glb_error(path, format!("{context}: buffer offsets overflow")))?;
    if accessor.count == 0 {
        return Ok((base, stride));
    }
    let span = (accessor.count - 1)
        .checked_mul(stride)
        .and_then(|last| last.checked_add(element_size))
        .ok_or_else(|| glb_error(path, format!("{context}: accessor size overflows")))?;
    if base
        .checked_add(span)
        .ok_or_else(|| glb_error(path, format!("{context}: accessor size overflows")))?
        > bin.len()
    {
        return Err(glb_error(
            path,
            format!("{context}: {label} accessor reads past the end of the BIN chunk"),
        ));
    }
    Ok((base, stride))
}

// ---------------------------------------------------------------------------
// STL
// ---------------------------------------------------------------------------

fn stl_error(path: &Path, message: impl std::fmt::Display) -> Lux3dError {
    validation(format!("STL `{}`: {message}", path.display()))
}

fn stl_binary_length_matches(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let count = u32::from_le_bytes(bytes[80..84].try_into().expect("4-byte slice")) as u64;
    84 + count * 50 == bytes.len() as u64
}

fn stl_is_ascii(bytes: &[u8]) -> bool {
    bytes.starts_with(b"solid")
        && find_subslice(&bytes[..bytes.len().min(1024)], b"facet").is_some()
}

/// Read an ASCII or binary STL mesh.
///
/// Vertices are deduplicated by exact bit pattern, so corners shared between
/// triangles collapse into shared indices. Binary files are detected by the
/// `84 + 50 * count` length formula; files that start with `solid` and
/// contain a `facet` keyword are parsed as ASCII.
pub fn read_stl(path: &Path) -> Result<ImportedMesh> {
    let bytes = read_file(path)?;
    if stl_binary_length_matches(&bytes) {
        stl_read_binary(&bytes, path)
    } else if stl_is_ascii(&bytes) {
        stl_read_ascii(&bytes, path)
    } else if bytes.len() >= 84 {
        stl_read_binary(&bytes, path)
    } else {
        stl_read_ascii(&bytes, path)
    }
}

fn stl_push_vertex(
    vertices: &mut Vec<f32>,
    dedup: &mut HashMap<[u32; 3], u32>,
    vertex: [f32; 3],
    path: &Path,
) -> Result<u32> {
    let key = [
        vertex[0].to_bits(),
        vertex[1].to_bits(),
        vertex[2].to_bits(),
    ];
    if let Some(&index) = dedup.get(&key) {
        return Ok(index);
    }
    let index = u32::try_from(vertices.len() / 3)
        .map_err(|_| stl_error(path, "mesh exceeds the u32 vertex index range"))?;
    vertices.extend_from_slice(&vertex);
    dedup.insert(key, index);
    Ok(index)
}

fn stl_read_binary(bytes: &[u8], path: &Path) -> Result<ImportedMesh> {
    if bytes.len() < 84 {
        return Err(stl_error(
            path,
            "binary STL is too short (needs an 80-byte header and a 4-byte triangle count)",
        ));
    }
    let count = u32::from_le_bytes(bytes[80..84].try_into().expect("4-byte slice")) as u64;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut dedup: HashMap<[u32; 3], u32> = HashMap::new();
    for triangle in 0..count {
        let start = 84u64 + triangle * 50;
        if start + 50 > bytes.len() as u64 {
            return Err(stl_error(
                path,
                format!(
                    "binary STL declares {count} triangles but the file only holds {} complete records",
                    (bytes.len() as u64 - 84) / 50
                ),
            ));
        }
        let start = start as usize;
        let record = &bytes[start..start + 50];
        let mut ids = [0u32; 3];
        for corner in 0..3 {
            let base = 12 + corner * 12; // skip the 12-byte facet normal
            let vertex = [
                f32::from_le_bytes(record[base..base + 4].try_into().expect("4-byte slice")),
                f32::from_le_bytes(record[base + 4..base + 8].try_into().expect("4-byte slice")),
                f32::from_le_bytes(
                    record[base + 8..base + 12]
                        .try_into()
                        .expect("4-byte slice"),
                ),
            ];
            ids[corner] = stl_push_vertex(&mut vertices, &mut dedup, vertex, path)?;
        }
        faces.extend_from_slice(&ids);
        // The 2-byte attribute count at the end of the record is ignored.
    }
    finalize_import(vertices, faces, path, "STL")
}

fn stl_read_ascii(bytes: &[u8], path: &Path) -> Result<ImportedMesh> {
    let text = String::from_utf8_lossy(bytes);
    let mut pending: Vec<f32> = Vec::new();
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut dedup: HashMap<[u32; 3], u32> = HashMap::new();
    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let mut tokens = line.trim().split_whitespace();
        match tokens.next() {
            Some("vertex") => {
                for _ in 0..3 {
                    let token = tokens.next().ok_or_else(|| {
                        stl_error(
                            path,
                            format!("ASCII line {line_number}: `vertex` needs x y z coordinates"),
                        )
                    })?;
                    let value = token.parse::<f32>().map_err(|_| {
                        stl_error(
                            path,
                            format!("ASCII line {line_number}: `{token}` is not a number"),
                        )
                    })?;
                    pending.push(value);
                }
                if pending.len() == 9 {
                    let mut ids = [0u32; 3];
                    for corner in 0..3 {
                        let vertex = [
                            pending[corner * 3],
                            pending[corner * 3 + 1],
                            pending[corner * 3 + 2],
                        ];
                        ids[corner] = stl_push_vertex(&mut vertices, &mut dedup, vertex, path)?;
                    }
                    faces.extend_from_slice(&ids);
                    pending.clear();
                }
            }
            // `facet`, `normal`, `outer`, `loop`, `endloop`, `endfacet`,
            // `solid`, `endsolid` and unknown lines carry no positions.
            _ => {}
        }
    }
    if !pending.is_empty() {
        return Err(stl_error(
            path,
            "ASCII STL ends with an incomplete facet (vertex count is not a multiple of 3)",
        ));
    }
    finalize_import(vertices, faces, path, "STL")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::mesh::{
        MeshView, ObjStyle, write_glb, write_glb_container, write_ply, write_stl,
    };

    fn temp_path(tag: &str, extension: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "lux3d-import-{tag}-{}-{nonce}.{extension}",
            std::process::id()
        ))
    }

    fn write_temp(tag: &str, extension: &str, contents: &[u8]) -> std::path::PathBuf {
        let path = temp_path(tag, extension);
        std::fs::write(&path, contents).expect("write temp file");
        path
    }

    // OBJ -------------------------------------------------------------------

    #[test]
    fn obj_reads_triangle_quad_attributes_and_crlf() {
        let obj = "# comment\r\n\
                   v 0.0 0.0 0.0\r\n\
                   v 1.0 0.0 0.0\r\n\
                   v 1.0 1.0 0.0\r\n\
                   v 0.0 1.0 0.0\r\n\
                   vt 0 0\r\n\
                   vn 0 0 1\r\n\
                   o merged\r\n\
                   g group\r\n\
                   usemtl none\r\n\
                   f 1/1/1 2/2/1 3/3/1 4//1\r\n";
        let path = write_temp("obj-quad", "obj", obj.as_bytes());
        let mesh = read_obj(&path).expect("read obj");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            mesh.vertices,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(mesh.faces, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn obj_negative_indices_and_merged_objects() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\n\
                   f -3 -2 -1\n\
                   o second\n\
                   v 5 5 5\n\
                   f -3 -2 -1\n";
        let path = write_temp("obj-negative", "obj", obj.as_bytes());
        let mesh = read_obj(&path).expect("read obj");
        let _ = std::fs::remove_file(&path);
        // Second file face: 4 vertices so -3 -> 1, -2 -> 2, -1 -> 3.
        assert_eq!(mesh.vertices.len(), 12);
        assert_eq!(mesh.faces, vec![0, 1, 2, 1, 2, 3]);
    }

    #[test]
    fn obj_rejects_malformed_index_with_line_context() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n";
        let path = write_temp("obj-malformed", "obj", obj.as_bytes());
        let error = read_obj(&path).expect_err("out-of-range index must fail");
        let _ = std::fs::remove_file(&path);
        let message = error.to_string();
        assert!(message.contains("line 4"), "unexpected message: {message}");
        assert!(message.contains("99"), "unexpected message: {message}");

        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 x\n";
        let path = write_temp("obj-nonnumeric", "obj", obj.as_bytes());
        let error = read_obj(&path).expect_err("non-numeric index must fail");
        let _ = std::fs::remove_file(&path);
        assert!(error.to_string().contains("not an integer"));
    }

    #[test]
    fn obj_drops_index_degenerate_triangles() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nf 1 1 2\n";
        let path = write_temp("obj-degenerate", "obj", obj.as_bytes());
        let mesh = read_obj(&path).expect("read obj");
        let _ = std::fs::remove_file(&path);
        assert_eq!(mesh.faces, vec![0, 1, 2]);
    }

    // PLY -------------------------------------------------------------------

    #[test]
    fn ply_ascii_reordered_properties_and_quad_fan() {
        let ply = "ply\n\
                   format ascii 1.0\n\
                   comment synthetic\n\
                   element vertex 4\n\
                   property double z\n\
                   property uchar red\n\
                   property float y\n\
                   property float x\n\
                   element face 1\n\
                   property list uchar int vertex_indices\n\
                   end_header\n\
                   0 255 0 0\n\
                   0 255 0 1\n\
                   0 255 0 2\n\
                   0 255 0 3\n\
                   4 0 1 2 3\n";
        let path = write_temp("ply-ascii", "ply", ply.as_bytes());
        let mesh = read_ply(&path).expect("read ascii ply");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            mesh.vertices,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]
        );
        assert_eq!(mesh.faces, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn ply_binary_little_endian_with_face_list() {
        let mut body: Vec<u8> = b"ply\n\
            format binary_little_endian 1.0\n\
            element vertex 4\n\
            property float x\n\
            property float y\n\
            property float z\n\
            property uchar red\n\
            property uchar green\n\
            property uchar blue\n\
            element face 2\n\
            property list uchar int vertex_indices\n\
            end_header\n"
            .to_vec();
        for vertex in [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ] {
            for component in vertex {
                body.extend_from_slice(&component.to_le_bytes());
            }
            body.extend_from_slice(&[255u8, 128, 0]);
        }
        body.push(4u8); // first face: quad 0 1 2 3
        for index in [0i32, 1, 2, 3] {
            body.extend_from_slice(&index.to_le_bytes());
        }
        body.push(3u8); // second face: triangle 0 2 3
        for index in [0i32, 2, 3] {
            body.extend_from_slice(&index.to_le_bytes());
        }
        let path = write_temp("ply-binary", "ply", &body);
        let mesh = read_ply(&path).expect("read binary ply");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            mesh.vertices,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(mesh.faces, vec![0, 1, 2, 0, 2, 3, 0, 2, 3]);
    }

    #[test]
    fn ply_rejects_big_endian_and_missing_face_element() {
        let ply = "ply\nformat binary_big_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        let path = write_temp("ply-big", "ply", ply.as_bytes());
        let error = read_ply(&path).expect_err("big-endian must fail");
        let _ = std::fs::remove_file(&path);
        assert!(error.to_string().contains("big-endian"));

        let point_cloud = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n";
        let path = write_temp("ply-points", "ply", point_cloud.as_bytes());
        let error = read_ply(&path).expect_err("point cloud without faces must fail");
        let _ = std::fs::remove_file(&path);
        assert!(error.to_string().contains("no `face` element"));
    }

    #[test]
    fn ply_rejects_truncated_binary_body() {
        let mut body: Vec<u8> = b"ply\n\
            format binary_little_endian 1.0\n\
            element vertex 3\n\
            property float x\n\
            property float y\n\
            property float z\n\
            element face 1\n\
            property list uchar int vertex_indices\n\
            end_header\n"
            .to_vec();
        body.extend_from_slice(&0.0f32.to_le_bytes()); // one component only
        let path = write_temp("ply-truncated", "ply", &body);
        let error = read_ply(&path).expect_err("truncated body must fail");
        let _ = std::fs::remove_file(&path);
        assert!(error.to_string().contains("truncated"));
    }

    // GLB -------------------------------------------------------------------

    fn triangle_mesh_view() -> (Vec<f32>, Vec<i64>) {
        (
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0, 1, 2],
        )
    }

    #[test]
    fn glb_roundtrip_with_writer_matches_exactly() {
        let (vertices, faces) = triangle_mesh_view();
        let colors = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let path = temp_path("glb-roundtrip", "glb");
        write_glb(
            &MeshView {
                vertices: &vertices,
                faces: &faces,
                colors: Some(&colors),
            },
            &path,
        )
        .expect("write glb");
        let mesh = read_glb(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);
        assert_eq!(mesh.vertices, vertices);
        assert_eq!(mesh.faces, vec![0u32, 1, 2]);
    }

    #[test]
    fn glb_handbuilt_u16_indices_with_byte_stride() {
        // Interleaved positions: 12 bytes of position + 4 bytes padding.
        let mut bin = Vec::new();
        for vertex in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for component in vertex {
                bin.extend_from_slice(&component.to_le_bytes());
            }
            bin.extend_from_slice(&[0u8; 4]);
        }
        debug_assert_eq!(bin.len(), 48);
        for index in [0u16, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        bin.extend_from_slice(&[0u8; 2]); // pad to 4 bytes
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 },
                    "indices": 1,
                    "mode": 4,
                }],
            }],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" },
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 48, "byteStride": 16 },
                { "buffer": 0, "byteOffset": 48, "byteLength": 6 },
            ],
            "buffers": [{ "byteLength": 56 }],
        });
        let path = temp_path("glb-stride", "glb");
        write_glb_container(&path, &serde_json::to_vec(&json).expect("json"), &bin)
            .expect("write container");
        let mesh = read_glb(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            mesh.vertices,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(mesh.faces, vec![0u32, 1, 2]);
    }

    fn glb_with_nodes(nodes: serde_json::Value) -> std::path::PathBuf {
        glb_with_nodes_and_extensions(nodes, &[])
    }

    fn glb_with_nodes_and_extensions(
        nodes: serde_json::Value,
        extensions_used: &[&str],
    ) -> std::path::PathBuf {
        let mut bin = Vec::new();
        for vertex in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for component in vertex {
                bin.extend_from_slice(&component.to_le_bytes());
            }
        }
        for index in [0u32, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [nodes],
            "extensionsUsed": extensions_used,
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 },
                    "indices": 1,
                    "mode": 4,
                }],
            }],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 12 },
            ],
            "buffers": [{ "byteLength": 48 }],
        });
        let path = temp_path("glb-nodes", "glb");
        write_glb_container(&path, &serde_json::to_vec(&json).expect("json"), &bin)
            .expect("write container");
        path
    }

    #[test]
    fn glb_applies_node_matrix_translation() {
        // glTF matrices are column-major; translation lives in elements 12..14.
        let path = glb_with_nodes(serde_json::json!({
            "mesh": 0,
            "matrix": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 20.0, 30.0, 1.0],
        }));
        let mesh = read_glb(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            mesh.vertices,
            vec![10.0, 20.0, 30.0, 11.0, 20.0, 30.0, 10.0, 21.0, 30.0]
        );
        assert_eq!(mesh.faces, vec![0u32, 1, 2]);
    }

    #[test]
    fn glb_applies_trs_translation() {
        let path = glb_with_nodes(serde_json::json!({
            "mesh": 0,
            "translation": [1.0, 0.0, 0.0],
        }));
        let mesh = read_glb(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            mesh.vertices,
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        );
    }

    #[test]
    fn glb_rejects_draco_and_meshopt() {
        for extension in ["KHR_draco_mesh_compression", "EXT_meshopt_compression"] {
            let path =
                glb_with_nodes_and_extensions(serde_json::json!({ "mesh": 0 }), &[extension]);
            let error = read_glb(&path).expect_err("compressed GLB must fail");
            let _ = std::fs::remove_file(&path);
            let message = error.to_string();
            assert!(
                message.contains("compression") && message.contains("not supported"),
                "unexpected message: {message}"
            );
        }
    }

    #[test]
    fn glb_loads_all_meshes_when_scene_is_missing() {
        let mut bin = Vec::new();
        for vertex in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for component in vertex {
                bin.extend_from_slice(&component.to_le_bytes());
            }
        }
        for vertex in [[2.0f32, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]] {
            for component in vertex {
                bin.extend_from_slice(&component.to_le_bytes());
            }
        }
        for index in [0u32, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        // glTF indices are primitive-local: the second mesh re-indexes 0..2
        // and the reader shifts them by its vertex offset.
        for index in [0u32, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "meshes": [
                { "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }] },
                { "primitives": [{ "attributes": { "POSITION": 2 }, "indices": 3 }] },
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
                { "bufferView": 0, "byteOffset": 36, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "byteOffset": 12, "componentType": 5125, "count": 3, "type": "SCALAR" },
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 72 },
                { "buffer": 0, "byteOffset": 72, "byteLength": 24 },
            ],
            "buffers": [{ "byteLength": 96 }],
        });
        let path = temp_path("glb-noscene", "glb");
        write_glb_container(&path, &serde_json::to_vec(&json).expect("json"), &bin)
            .expect("write container");
        let mesh = read_glb(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);
        assert_eq!(mesh.vertices.len(), 18);
        assert_eq!(mesh.faces, vec![0u32, 1, 2, 3, 4, 5]);
    }

    // STL -------------------------------------------------------------------

    #[test]
    fn stl_binary_roundtrip_deduplicates_vertices() {
        let vertices = vec![
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        ];
        let faces = vec![0i64, 1, 2, 0, 2, 3]; // two triangles sharing edge 0-2
        let path = temp_path("stl-binary", "stl");
        write_stl(
            &MeshView {
                vertices: &vertices,
                faces: &faces,
                colors: None,
            },
            &path,
        )
        .expect("write stl");
        let mesh = read_stl(&path).expect("read stl");
        let _ = std::fs::remove_file(&path);
        // Shared corners collapse; winding and exact bits are preserved.
        assert_eq!(mesh.vertices.len(), 12);
        assert_eq!(mesh.vertices, vertices);
        assert_eq!(mesh.faces, vec![0u32, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn stl_ascii_reads_facets() {
        let stl = "solid test\n\
                   facet normal 0 0 1\n\
                   outer loop\n\
                   vertex 0 0 0\n\
                   vertex 1 0 0\n\
                   vertex 0 1 0\n\
                   endloop\n\
                   endfacet\n\
                   facet normal 0 0 1\n\
                   outer loop\n\
                   vertex 1 0 0\n\
                   vertex 1 1 0\n\
                   vertex 0 1 0\n\
                   endloop\n\
                   endfacet\n\
                   endsolid test\n";
        let path = write_temp("stl-ascii", "stl", stl.as_bytes());
        let mesh = read_stl(&path).expect("read ascii stl");
        let _ = std::fs::remove_file(&path);
        // Second triangle re-uses corners (1,0,0) and (0,1,0).
        assert_eq!(mesh.vertices.len(), 12);
        assert_eq!(mesh.faces, vec![0u32, 1, 2, 1, 3, 2]);
    }

    // Dispatch --------------------------------------------------------------

    #[test]
    fn load_mesh_dispatches_by_extension() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let path = write_temp("dispatch-obj", "obj", obj.as_bytes());
        let mesh = load_mesh(&path).expect("load obj");
        let _ = std::fs::remove_file(&path);
        assert_eq!(mesh.faces, vec![0u32, 1, 2]);

        let vertices = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let faces = vec![0i64, 1, 2];
        let path = temp_path("dispatch-ply", "ply");
        write_ply(
            &MeshView {
                vertices: &vertices,
                faces: &faces,
                colors: None,
            },
            &path,
        )
        .expect("write ply");
        let mesh = load_mesh(&path).expect("load ply");
        let _ = std::fs::remove_file(&path);
        assert_eq!(mesh.vertices, vertices);
        assert_eq!(mesh.faces, vec![0u32, 1, 2]);

        // Case-insensitive GLB dispatch through the writer.
        let path = temp_path("dispatch-glb", "GLB");
        write_glb(
            &MeshView {
                vertices: &vertices,
                faces: &faces,
                colors: None,
            },
            &path,
        )
        .expect("write glb");
        let mesh = load_mesh(&path).expect("load glb");
        let _ = std::fs::remove_file(&path);
        assert_eq!(mesh.vertices, vertices);
    }

    #[test]
    fn load_mesh_rejects_unsupported_extension() {
        let path = write_temp("dispatch-unknown", "txt", b"not a mesh");
        let error = load_mesh(&path).expect_err("unsupported extension must fail");
        let _ = std::fs::remove_file(&path);
        let message = error.to_string();
        assert!(
            message.contains("unsupported mesh input extension") && message.contains(".glb"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn load_mesh_rejects_empty_and_garbage_files() {
        let path = write_temp("empty-obj", "obj", b"");
        let error = load_mesh(&path).expect_err("empty file must fail");
        let _ = std::fs::remove_file(&path);
        assert!(error.to_string().contains("no vertices"));

        let path = write_temp("garbage-glb", "glb", b"\x00\x01\x02\x03garbage");
        let error = load_mesh(&path).expect_err("garbage glb must fail");
        let _ = std::fs::remove_file(&path);
        assert!(error.to_string().contains("GLB"));
    }
}
