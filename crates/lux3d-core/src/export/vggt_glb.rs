//! VGGT demo GLB scene export (`visual_util.predictions_to_glb`).
//!
//! The official Gradio demo exports its reconstruction as one glTF binary
//! containing a colored point cloud plus one camera-cone mesh per frame
//! (`demo_gradio.py:225-242`). This module ports that artifact:
//!
//! * Point cloud: positions from `world_points` (pointmap branch) or
//!   `world_points_from_depth` (depth branch), colors from the preprocessed
//!   `[S, 3, H, W]` images, filtered by the confidence percentile
//!   (`visual_util.py:68-193`). The official scene uses `trimesh.PointCloud`,
//!   which exports as a glTF `POINTS` primitive (mode 0) with `POSITION` +
//!   `COLOR_0`; there is no per-point sphere geometry in the reference.
//! * Camera cones: `trimesh.creation.cone(width, height, sections=4)` scaled by
//!   the scene scale, stacked with 0.95x and 2-degree rotated copies, wired
//!   through `compute_camera_faces`, transformed by
//!   `camera_to_world @ opengl_conversion @ rot45`
//!   (`visual_util.py:218-260, 329-362`).
//! * Scene alignment: every geometry node carries the same
//!   `inv(extrinsics[0]) @ opengl_conversion @ Ry(180deg)` matrix
//!   (`visual_util.py:263-301`), exactly like trimesh's GLB export.
//!
//! The GLB container writer is shared with the mesh exporters
//! ([`super::mesh::write_glb_container`]); the whole scene lives in one
//! embedded buffer.
//!
//! Deliberate scope: `mask_sky` (ONNX skyseg, `visual_util.py:87-135`) is not
//! ported. The black/white background masks are ported as [`VggtGlbOptions`]
//! fields but are not exposed as CLI flags (the CLI surface is the documented
//! `--glb*` set). The demo's "Show Points from Frame" selector
//! (`visual_util.py:60-66, 137-141`) is not exposed either.

use std::{collections::HashMap, path::Path};

use serde_json::{json, Value};

use super::mesh::write_glb_container;
use crate::{error::Lux3dError, Result};

/// Official `predictions_to_glb` confidence slider default: 50% of the
/// low-confidence points are filtered out (`visual_util.py:20`,
/// `demo_gradio.py:475`).
pub const DEFAULT_GLB_CONF_PERCENTILE: f64 = 50.0;

/// Options for the VGGT GLB scene export.
///
/// The defaults mirror the `predictions_to_glb` signature
/// (`visual_util.py:18-28`): 50% confidence percentile, camera cones shown,
/// no background masks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VggtGlbOptions {
    /// Percentage of low-confidence points to filter out (`conf_thres`);
    /// `0.0` disables the percentile and keeps every `conf > 1e-5` point
    /// (`visual_util.py:56-57, 151-158`). Must be finite and in `[0, 100]`.
    pub conf_percentile: f64,
    /// Include the per-frame camera-cone meshes (`show_cam`).
    pub show_cam: bool,
    /// Filter black background pixels (`mask_black_bg`,
    /// `visual_util.py:160-162`).
    pub mask_black_bg: bool,
    /// Filter white background pixels (`mask_white_bg`,
    /// `visual_util.py:164-168`).
    pub mask_white_bg: bool,
}

impl Default for VggtGlbOptions {
    fn default() -> Self {
        Self {
            conf_percentile: DEFAULT_GLB_CONF_PERCENTILE,
            show_cam: true,
            mask_black_bg: false,
            mask_white_bg: false,
        }
    }
}

/// CPU prediction arrays consumed by [`build_vggt_glb_scene`].
///
/// The layouts are the batch-squeezed demo layouts: `world_points`
/// `[S, H, W, 3]`, `world_points_conf` `[S, H, W]`, `images` `[S, 3, H, W]`
/// (the NCHW branch of `visual_util.py:144-149`) and `extrinsics` `[S, 3, 4]`
/// world-to-camera matrices.
#[derive(Debug, Clone, Copy)]
pub struct VggtGlbPredictions<'a> {
    /// Flat `[S, H, W, 3]` world points.
    pub world_points: &'a [f32],
    /// Flat `[S, H, W]` world-point confidences.
    pub world_points_conf: &'a [f32],
    /// Flat `[S, 3, H, W]` preprocessed RGB images in `[0, 1]`.
    pub images: &'a [f32],
    /// Flat `[S, 3, 4]` world-to-camera extrinsics.
    pub extrinsics: &'a [f32],
    /// Frame count `S`.
    pub frames: usize,
    /// Frame height `H`.
    pub height: usize,
    /// Frame width `W`.
    pub width: usize,
}

/// One camera-cone mesh, already in pre-alignment world coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct VggtGlbCameraMesh {
    /// Flat `[x, y, z, ...]` positions (14 vertices for the fixed sections=4
    /// cone after trimesh's referenced-vertex merge).
    pub vertices: Vec<f32>,
    /// Flat zero-based triangle indices (48 triangles, double-sided).
    pub faces: Vec<u32>,
    /// Uniform per-camera RGB color (the `gist_rainbow` sample).
    pub color: [u8; 3],
}

/// A parsed VGGT GLB scene ready for [`write_vggt_glb`].
#[derive(Debug, Clone, PartialEq)]
pub struct VggtGlbScene {
    /// Flat `[x, y, z, ...]` point-cloud positions, pre-alignment.
    pub points: Vec<f32>,
    /// Flat `[r, g, b, ...]` point-cloud colors (`[0, 255]`).
    pub point_colors: Vec<u8>,
    /// Per-frame camera meshes (empty when `show_cam` is false).
    pub cameras: Vec<VggtGlbCameraMesh>,
    /// Row-major `inv(extrinsics[0]) @ opengl @ Ry(180deg)` scene alignment
    /// applied to every node (`visual_util.py:263-301`).
    pub alignment: [[f64; 4]; 4],
}

impl VggtGlbScene {
    /// Number of exported points.
    pub fn point_count(&self) -> usize {
        self.points.len() / 3
    }

    /// Number of exported camera meshes.
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }
}

/// Build the VGGT GLB scene from CPU predictions.
///
/// This is the exact pre-GLB half of `predictions_to_glb`
/// (`visual_util.py:68-215`): confidence percentile, background masks,
/// scene-scale estimation, `gist_rainbow` camera colors, camera cones and the
/// first-camera alignment matrix.
pub fn build_vggt_glb_scene(
    predictions: &VggtGlbPredictions<'_>,
    options: &VggtGlbOptions,
) -> Result<VggtGlbScene> {
    validate_inputs(predictions, options)?;
    let VggtGlbPredictions {
        world_points,
        world_points_conf,
        images,
        extrinsics,
        frames,
        height,
        width,
    } = *predictions;
    let pixel_count = frames * height * width;

    // `colors_rgb = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255)
    // .astype(np.uint8)` (`visual_util.py:143-149`).
    let mut colors = Vec::with_capacity(pixel_count * 3);
    let frame_plane = height * width;
    for pixel in 0..pixel_count {
        let frame = pixel / frame_plane;
        let offset = pixel % frame_plane;
        for channel in 0..3 {
            let value = images[(frame * 3 + channel) * frame_plane + offset];
            colors.push((value * 255.0) as u8);
        }
    }

    // `conf_threshold = np.percentile(conf, conf_thres)` unless conf_thres is
    // exactly zero (`visual_util.py:151-156`).
    let threshold = if options.conf_percentile == 0.0 {
        0.0
    } else {
        let mut confidences: Vec<f64> = world_points_conf.iter().map(|&v| f64::from(v)).collect();
        percentile_sorted(&mut confidences, options.conf_percentile)
    };

    let mut points = Vec::new();
    let mut point_colors = Vec::new();
    for index in 0..pixel_count {
        let conf = f64::from(world_points_conf[index]);
        // `conf_mask = (conf >= conf_threshold) & (conf > 1e-5)`
        // (`visual_util.py:158`).
        let mut keep = conf >= threshold && conf > 1e-5;
        if keep && options.mask_black_bg {
            // `black_bg_mask = colors_rgb.sum(axis=1) >= 16`
            // (`visual_util.py:161`).
            let sum = u32::from(colors[index * 3])
                + u32::from(colors[index * 3 + 1])
                + u32::from(colors[index * 3 + 2]);
            keep = sum >= 16;
        }
        if keep && options.mask_white_bg {
            // `~((r > 240) & (g > 240) & (b > 240))` (`visual_util.py:167`).
            let white = colors[index * 3] > 240
                && colors[index * 3 + 1] > 240
                && colors[index * 3 + 2] > 240;
            keep = !white;
        }
        if !keep {
            continue;
        }
        points.extend_from_slice(&world_points[index * 3..index * 3 + 3]);
        point_colors.extend_from_slice(&colors[index * 3..index * 3 + 3]);
    }

    let scene_scale = if points.is_empty() {
        // `vertices_3d = [[1, 0, 0]]; colors_rgb = [[255, 255, 255]];
        // scene_scale = 1` (`visual_util.py:173-177`).
        points.extend_from_slice(&[1.0, 0.0, 0.0]);
        point_colors.extend_from_slice(&[255, 255, 255]);
        1.0
    } else {
        // `norm(percentile(points, 95, axis=0) - percentile(points, 5,
        // axis=0))` (`visual_util.py:178-183`).
        let mut axes: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for point in points.chunks_exact(3) {
            for axis in 0..3 {
                axes[axis].push(f64::from(point[axis]));
            }
        }
        let mut delta = [0.0f64; 3];
        for (axis, values) in axes.iter_mut().enumerate() {
            let lower = percentile_sorted(values, 5.0);
            let upper = percentile_sorted(values, 95.0);
            delta[axis] = upper - lower;
        }
        (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
    };

    let mut cameras = Vec::new();
    if options.show_cam {
        for index in 0..frames {
            // `rgba_color = colormap(i / num_cameras)` (`visual_util.py:206`).
            let color = gist_rainbow_rgb(index as f64 / frames as f64);
            let extrinsic = &extrinsics[index * 12..index * 12 + 12];
            cameras.push(build_camera_mesh(extrinsic, scene_scale, color)?);
        }
    }

    let alignment = scene_alignment(&extrinsics[..12])?;

    Ok(VggtGlbScene {
        points,
        point_colors,
        cameras,
        alignment,
    })
}

/// Write the scene as a binary glTF 2.0 (`.glb`) file.
///
/// One embedded buffer holds the point-cloud `POSITION`/`COLOR_0` arrays, each
/// camera mesh's `POSITION`/`COLOR_0` and the shared 48-triangle cone index
/// buffer. Point clouds use primitive mode 0 (`POINTS`), cameras mode 4
/// (`TRIANGLES`), matching the trimesh GLB export of the official scene.
pub fn write_vggt_glb(scene: &VggtGlbScene, path: &Path) -> Result<()> {
    validate_scene(scene)?;
    let point_count = scene.point_count();
    let matrix: Vec<f64> = (0..4)
        .flat_map(|column| (0..4).map(move |row| scene.alignment[row][column]))
        .collect();
    let matrix = json!(matrix);

    let mut binary = Vec::new();
    let mut views: Vec<Value> = Vec::new();
    let mut accessors: Vec<Value> = Vec::new();

    let point_positions = push_f32_view(&mut binary, &mut views, &scene.points, 34962);
    let (min, max) = position_bounds(&scene.points);
    let point_position_accessor =
        push_vec3_accessor(&mut accessors, point_positions, point_count, min, max);
    let point_rgba = rgba_colors(&scene.point_colors);
    let point_color_view = push_u8_view(&mut binary, &mut views, &point_rgba, 34962);
    let point_color_accessor = push_color_accessor(&mut accessors, point_color_view, point_count);

    let mut meshes: Vec<Value> = vec![json!({
        "name": "point_cloud",
        "primitives": [{
            "attributes": {
                "POSITION": point_position_accessor,
                "COLOR_0": point_color_accessor,
            },
            "mode": 0,
        }],
    })];
    let mut nodes: Vec<Value> = vec![json!({
        "name": "point_cloud",
        "mesh": 0,
        "matrix": matrix,
    })];

    // All camera cones share the same triangle list
    // (`compute_camera_faces`, `visual_util.py:329-362`), so the index accessor
    // is emitted once and referenced by every camera mesh, exactly like
    // trimesh's deduplicated GLB export.
    if let Some(first) = scene.cameras.first() {
        if scene
            .cameras
            .iter()
            .any(|camera| camera.faces != first.faces)
        {
            return Err(Lux3dError::InvalidInput(
                "VGGT GLB camera meshes must share one cone topology",
            ));
        }
        let indices_view = push_u32_view(&mut binary, &mut views, &first.faces, 34963);
        let indices_accessor =
            push_scalar_index_accessor(&mut accessors, indices_view, first.faces.len());
        for (index, camera) in scene.cameras.iter().enumerate() {
            let vertex_count = camera.vertices.len() / 3;
            let positions = push_f32_view(&mut binary, &mut views, &camera.vertices, 34962);
            let (min, max) = position_bounds(&camera.vertices);
            let position_accessor =
                push_vec3_accessor(&mut accessors, positions, vertex_count, min, max);
            let colors = vec![camera.color; vertex_count];
            let color_rgba = rgba_colors(&colors.into_iter().flatten().collect::<Vec<u8>>());
            let color_view = push_u8_view(&mut binary, &mut views, &color_rgba, 34962);
            let color_accessor = push_color_accessor(&mut accessors, color_view, vertex_count);
            let mesh_index = meshes.len();
            meshes.push(json!({
                "name": format!("camera_{index}"),
                "primitives": [{
                    "attributes": {
                        "POSITION": position_accessor,
                        "COLOR_0": color_accessor,
                    },
                    "indices": indices_accessor,
                    "mode": 4,
                }],
            }));
            nodes.push(json!({
                "name": format!("camera_{index}"),
                "mesh": mesh_index,
                "matrix": matrix,
            }));
        }
    }

    let scene_nodes: Vec<usize> = (0..nodes.len()).collect();
    let document = json!({
        "asset": {
            "version": "2.0",
            "generator": "lux3d",
        },
        "scene": 0,
        "scenes": [{ "nodes": scene_nodes }],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{ "byteLength": binary.len() }],
    });
    let json =
        serde_json::to_vec(&document).map_err(|error| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to serialize VGGT GLB JSON: {error}"),
        })?;
    write_glb_container(path, &json, &binary)
}

fn validate_inputs(predictions: &VggtGlbPredictions<'_>, options: &VggtGlbOptions) -> Result<()> {
    if !options.conf_percentile.is_finite() || !(0.0..=100.0).contains(&options.conf_percentile) {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB confidence percentile must be a finite value in [0, 100]",
        ));
    }
    if predictions.frames == 0 || predictions.height == 0 || predictions.width == 0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB export requires at least one non-empty frame",
        ));
    }
    let pixel_count = predictions
        .frames
        .checked_mul(predictions.height)
        .and_then(|value| value.checked_mul(predictions.width))
        .ok_or(Lux3dError::InvalidInput(
            "VGGT GLB frame dimensions overflow the address space",
        ))?;
    let expected_points = pixel_count.checked_mul(3).ok_or(Lux3dError::InvalidInput(
        "VGGT GLB world_points length overflows the address space",
    ))?;
    let expected_images = pixel_count.checked_mul(3).ok_or(Lux3dError::InvalidInput(
        "VGGT GLB image length overflows the address space",
    ))?;
    let expected_extrinsics =
        predictions
            .frames
            .checked_mul(12)
            .ok_or(Lux3dError::InvalidInput(
                "VGGT GLB extrinsics length overflows the address space",
            ))?;
    if predictions.world_points.len() != expected_points {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB world_points must hold [S, H, W, 3] values",
        ));
    }
    if predictions.world_points_conf.len() != pixel_count {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB world_points_conf must hold [S, H, W] values",
        ));
    }
    if predictions.images.len() != expected_images {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB images must hold [S, 3, H, W] values",
        ));
    }
    if predictions.extrinsics.len() != expected_extrinsics {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB extrinsics must hold [S, 3, 4] values",
        ));
    }
    let finite = predictions
        .world_points
        .iter()
        .chain(predictions.world_points_conf.iter())
        .chain(predictions.images.iter())
        .chain(predictions.extrinsics.iter())
        .all(|value| value.is_finite());
    if !finite {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB export rejected non-finite prediction values",
        ));
    }
    Ok(())
}

fn validate_scene(scene: &VggtGlbScene) -> Result<()> {
    if scene.points.len() % 3 != 0 || scene.points.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB point buffer must hold at least one xyz position",
        ));
    }
    if scene.point_colors.len() != scene.points.len() {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB point colors must match the point positions",
        ));
    }
    if scene.points.iter().any(|value| !value.is_finite()) {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB export rejected a non-finite point position",
        ));
    }
    if scene
        .alignment
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB export rejected a non-finite scene alignment matrix",
        ));
    }
    let point_count = scene.point_count();
    if point_count > u32::MAX as usize {
        return Err(Lux3dError::InvalidInput(
            "VGGT GLB export supports at most u32::MAX points",
        ));
    }
    for camera in &scene.cameras {
        if camera.vertices.len() % 3 != 0 || camera.vertices.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "VGGT GLB camera mesh must hold at least one xyz vertex",
            ));
        }
        if camera.faces.len() % 3 != 0 || camera.faces.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "VGGT GLB camera mesh must hold at least one triangle",
            ));
        }
        if camera.vertices.iter().any(|value| !value.is_finite()) {
            return Err(Lux3dError::InvalidInput(
                "VGGT GLB export rejected a non-finite camera vertex",
            ));
        }
        let vertex_count = camera.vertices.len() / 3;
        if camera
            .faces
            .iter()
            .any(|&index| index as usize >= vertex_count)
        {
            return Err(Lux3dError::InvalidInput(
                "VGGT GLB export rejected a camera face index outside the vertex buffer",
            ));
        }
    }
    Ok(())
}

fn push_f32_view(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    values: &[f32],
    target: u32,
) -> usize {
    align_binary(binary);
    let offset = binary.len();
    for value in values {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    let index = views.len();
    views.push(json!({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": values.len() * 4,
        "target": target,
    }));
    index
}

fn push_u8_view(binary: &mut Vec<u8>, views: &mut Vec<Value>, values: &[u8], target: u32) -> usize {
    align_binary(binary);
    let offset = binary.len();
    binary.extend_from_slice(values);
    let index = views.len();
    views.push(json!({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": values.len(),
        "target": target,
    }));
    index
}

fn push_u32_view(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    values: &[u32],
    target: u32,
) -> usize {
    align_binary(binary);
    let offset = binary.len();
    for value in values {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    let index = views.len();
    views.push(json!({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": values.len() * 4,
        "target": target,
    }));
    index
}

fn align_binary(binary: &mut Vec<u8>) {
    while binary.len() % 4 != 0 {
        binary.push(0);
    }
}

fn push_vec3_accessor(
    accessors: &mut Vec<Value>,
    view: usize,
    count: usize,
    min: [f32; 3],
    max: [f32; 3],
) -> usize {
    let index = accessors.len();
    accessors.push(json!({
        "bufferView": view,
        "componentType": 5126,
        "count": count,
        "type": "VEC3",
        "min": [min[0], min[1], min[2]],
        "max": [max[0], max[1], max[2]],
    }));
    index
}

fn push_color_accessor(accessors: &mut Vec<Value>, view: usize, count: usize) -> usize {
    let index = accessors.len();
    accessors.push(json!({
        "bufferView": view,
        "componentType": 5121,
        "normalized": true,
        "count": count,
        "type": "VEC4",
    }));
    index
}

fn push_scalar_index_accessor(accessors: &mut Vec<Value>, view: usize, count: usize) -> usize {
    let index = accessors.len();
    accessors.push(json!({
        "bufferView": view,
        "componentType": 5125,
        "count": count,
        "type": "SCALAR",
    }));
    index
}

fn rgba_colors(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for color in rgb.chunks_exact(3) {
        rgba.extend_from_slice(&[color[0], color[1], color[2], 255]);
    }
    rgba
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

/// NumPy-compatible linear percentile (`np.percentile(..., method="linear")`).
/// The slice is sorted in place; an empty slice yields `NaN` (never reached for
/// valid VGGT predictions, which always have at least one pixel).
fn percentile_sorted(values: &mut [f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    if values.len() == 1 {
        return values[0];
    }
    let rank = (values.len() - 1) as f64 * percentile / 100.0;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let fraction = rank - lower as f64;
        values[lower] + (values[upper] - values[lower]) * fraction
    }
}

/// Matplotlib `gist_rainbow` anchors (`matplotlib._cm._gist_rainbow_data`):
/// `(position, (r, g, b))` in `[0, 1]`.
const GIST_RAINBOW: [(f64, [f64; 3]); 8] = [
    (0.0, [1.0, 0.0, 0.16]),
    (0.03, [1.0, 0.0, 0.0]),
    (0.215, [1.0, 1.0, 0.0]),
    (0.4, [0.0, 1.0, 0.0]),
    (0.586, [0.0, 1.0, 1.0]),
    (0.77, [0.0, 0.0, 1.0]),
    (0.954, [1.0, 0.0, 1.0]),
    (1.0, [1.0, 0.0, 0.75]),
];

/// One `gist_rainbow` sample exactly as the demo colorizes camera `i` of `n`:
/// `colormap(i / n)` with matplotlib's 256-entry linear LUT quantization, then
/// `tuple(int(255 * x) for x in rgba[:3])` (`visual_util.py:206-207`).
fn gist_rainbow_rgb(fraction: f64) -> [u8; 3] {
    const N: usize = 256;
    let scaled = fraction * N as f64;
    let index = if scaled >= N as f64 {
        N - 1
    } else if scaled <= 0.0 {
        0
    } else {
        scaled as usize
    };
    // `LinearSegmentedColormap` samples its anchors at x = i / (N - 1).
    let position = index as f64 / (N - 1) as f64;
    let mut rgb = [0u8; 3];
    for (channel, value) in rgb.iter_mut().enumerate() {
        let mut sample = GIST_RAINBOW[GIST_RAINBOW.len() - 1].1[channel];
        for pair in GIST_RAINBOW.windows(2) {
            let (start, start_rgb) = pair[0];
            let (end, end_rgb) = pair[1];
            if position <= end {
                let fraction = if end > start {
                    (position - start) / (end - start)
                } else {
                    0.0
                };
                let fraction = fraction.clamp(0.0, 1.0);
                sample = start_rgb[channel] + (end_rgb[channel] - start_rgb[channel]) * fraction;
                break;
            }
        }
        *value = (255.0 * sample) as u8;
    }
    rgb
}

type Mat4 = [[f64; 4]; 4];

const fn identity4() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(left: &Mat4, right: &Mat4) -> Mat4 {
    let mut out = [[0.0f64; 4]; 4];
    for (row, out_row) in out.iter_mut().enumerate() {
        for (column, out_cell) in out_row.iter_mut().enumerate() {
            *out_cell = (0..4).map(|k| left[row][k] * right[k][column]).sum::<f64>();
        }
    }
    out
}

/// General 4x4 inverse (Gauss-Jordan with partial pivoting), the equivalent of
/// `np.linalg.inv` for the demo's extrinsic matrices (`visual_util.py:205,
/// 282`).
fn mat4_inverse(matrix: &Mat4) -> Result<Mat4> {
    let mut work = *matrix;
    let mut inverse = identity4();
    for column in 0..4 {
        let pivot_row = (column..4)
            .max_by(|&left, &right| {
                work[left][column]
                    .abs()
                    .total_cmp(&work[right][column].abs())
            })
            .expect("non-empty pivot range");
        if work[pivot_row][column] == 0.0 {
            return Err(Lux3dError::InvalidInput(
                "VGGT GLB export received a singular camera extrinsic matrix",
            ));
        }
        if pivot_row != column {
            work.swap(pivot_row, column);
            inverse.swap(pivot_row, column);
        }
        let pivot = work[column][column];
        for cell in 0..4 {
            work[column][cell] /= pivot;
            inverse[column][cell] /= pivot;
        }
        for row in 0..4 {
            if row == column {
                continue;
            }
            let factor = work[row][column];
            if factor == 0.0 {
                continue;
            }
            for cell in 0..4 {
                work[row][cell] -= factor * work[column][cell];
                inverse[row][cell] -= factor * inverse[column][cell];
            }
        }
    }
    Ok(inverse)
}

fn transform_point(matrix: &Mat4, point: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|row| {
        matrix[row][0] * point[0]
            + matrix[row][1] * point[1]
            + matrix[row][2] * point[2]
            + matrix[row][3]
    })
}

fn extrinsic4(extrinsic: &[f32]) -> Mat4 {
    [
        [
            f64::from(extrinsic[0]),
            f64::from(extrinsic[1]),
            f64::from(extrinsic[2]),
            f64::from(extrinsic[3]),
        ],
        [
            f64::from(extrinsic[4]),
            f64::from(extrinsic[5]),
            f64::from(extrinsic[6]),
            f64::from(extrinsic[7]),
        ],
        [
            f64::from(extrinsic[8]),
            f64::from(extrinsic[9]),
            f64::from(extrinsic[10]),
            f64::from(extrinsic[11]),
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rotation_z(degrees: f64) -> Mat4 {
    let radians = degrees.to_radians();
    let (sin, cos) = radians.sin_cos();
    let mut rotation = identity4();
    rotation[0][0] = cos;
    rotation[0][1] = -sin;
    rotation[1][0] = sin;
    rotation[1][1] = cos;
    rotation
}

/// `inv(extrinsics[0]) @ opengl_conversion @ Ry(180deg)`
/// (`visual_util.py:275-283`).
fn scene_alignment(first_extrinsic: &[f32]) -> Result<Mat4> {
    let extrinsic = mat4_inverse(&extrinsic4(first_extrinsic))?;
    let mut opengl = identity4();
    opengl[1][1] = -1.0;
    opengl[2][2] = -1.0;
    let mut align = identity4();
    let radians = std::f64::consts::PI;
    let (sin, cos) = radians.sin_cos();
    align[0][0] = cos;
    align[0][2] = sin;
    align[2][0] = -sin;
    align[2][2] = cos;
    Ok(mat4_mul(&mat4_mul(&extrinsic, &opengl), &align))
}

/// The reference `trimesh.creation.cone(radius, height, sections=4)` vertices:
/// base center at the origin, base rim on `z = 0`, apex on `+z`.
fn base_cone(width: f64, height: f64) -> ([[f64; 3]; 6], [[usize; 3]; 8]) {
    (
        [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [0.0, 0.0, height],
            [0.0, width, 0.0],
            [-width, 0.0, 0.0],
            [0.0, -width, 0.0],
        ],
        [
            [1, 0, 3],
            [1, 3, 2],
            [3, 0, 4],
            [3, 4, 2],
            [4, 0, 5],
            [4, 5, 2],
            [5, 0, 1],
            [5, 1, 2],
        ],
    )
}

/// `integrate_camera_into_scene` (`visual_util.py:218-260`) plus the trimesh
/// `Trimesh(vertices, faces)` constructor behavior: unreferenced vertices are
/// dropped and duplicate vertices merged (digits = 8, trimesh's
/// `tol.merge = 1e-8`), which fixes the cone at 14 vertices / 48 triangles.
fn build_camera_mesh(
    extrinsic: &[f32],
    scene_scale: f64,
    color: [u8; 3],
) -> Result<VggtGlbCameraMesh> {
    let cam_width = scene_scale * 0.05;
    let cam_height = scene_scale * 0.1;

    let mut rot_45 = rotation_z(45.0);
    rot_45[2][3] = -cam_height;
    let mut opengl = identity4();
    opengl[1][1] = -1.0;
    opengl[2][2] = -1.0;
    let camera_to_world = mat4_inverse(&extrinsic4(extrinsic))?;
    let complete = mat4_mul(&mat4_mul(&camera_to_world, &opengl), &rot_45);

    let (cone_vertices, cone_faces) = base_cone(cam_width, cam_height);
    let slight_rotation = rotation_z(2.0);
    let mut combined = Vec::with_capacity(18);
    for vertex in cone_vertices {
        combined.push(vertex);
    }
    for vertex in cone_vertices {
        combined.push(vertex.map(|value| value * 0.95));
    }
    for vertex in cone_vertices {
        combined.push(transform_point(&slight_rotation, vertex));
    }
    let transformed: Vec<[f64; 3]> = combined
        .iter()
        .map(|&vertex| transform_point(&complete, vertex))
        .collect();

    // `compute_camera_faces` (`visual_util.py:329-362`): skip the base-center
    // faces, stitch the three cone copies, then append the reversed list.
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(48);
    for face in cone_faces {
        if face.contains(&0) {
            continue;
        }
        let [v1, v2, v3] = face;
        let (v1_offset, v2_offset, v3_offset) = (v1 + 6, v2 + 6, v3 + 6);
        let (v1_offset_2, v2_offset_2, v3_offset_2) = (v1 + 12, v2 + 12, v3 + 12);
        faces.push([v1 as u32, v2 as u32, v2_offset as u32]);
        faces.push([v1 as u32, v1_offset as u32, v3 as u32]);
        faces.push([v3_offset as u32, v2 as u32, v3 as u32]);
        faces.push([v1 as u32, v2 as u32, v2_offset_2 as u32]);
        faces.push([v1 as u32, v1_offset_2 as u32, v3 as u32]);
        faces.push([v3_offset_2 as u32, v2 as u32, v3 as u32]);
    }
    let reversed: Vec<[u32; 3]> = faces.iter().map(|[a, b, c]| [*c, *b, *a]).collect();
    faces.extend(reversed);

    let (vertices, faces) = merge_camera_vertices(&transformed, &faces);
    Ok(VggtGlbCameraMesh {
        vertices: vertices
            .iter()
            .flat_map(|vertex| vertex.map(|value| value as f32))
            .collect(),
        faces,
        color,
    })
}

/// trimesh's `Trimesh(process=True)` drops unreferenced vertices and groups the
/// referenced ones by position rounded to 8 decimals (`tol.merge = 1e-8`,
/// `grouping.merge_vertices`), keeping the first occurrence order.
fn merge_camera_vertices(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> (Vec<[f64; 3]>, Vec<u32>) {
    let mut referenced = vec![false; vertices.len()];
    for face in faces {
        for &index in face {
            referenced[index as usize] = true;
        }
    }
    let mut remap = vec![u32::MAX; vertices.len()];
    let mut merged: Vec<[f64; 3]> = Vec::new();
    let mut positions: HashMap<(i64, i64, i64), u32> = HashMap::new();
    for (index, vertex) in vertices.iter().enumerate() {
        if !referenced[index] {
            continue;
        }
        let key = (
            (vertex[0] * 1e8).round_ties_even() as i64,
            (vertex[1] * 1e8).round_ties_even() as i64,
            (vertex[2] * 1e8).round_ties_even() as i64,
        );
        remap[index] = *positions.entry(key).or_insert_with(|| {
            merged.push(*vertex);
            (merged.len() - 1) as u32
        });
    }
    let faces = faces
        .iter()
        .flat_map(|[a, b, c]| [remap[*a as usize], remap[*b as usize], remap[*c as usize]])
        .collect();
    (merged, faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_predictions(
        frames: usize,
        height: usize,
        width: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let pixels = frames * height * width;
        let world_points = (0..pixels)
            .flat_map(|index| [index as f32, index as f32 * 0.5, index as f32 * 0.25])
            .collect();
        let world_points_conf = (0..pixels).map(|index| index as f32 + 1.0).collect();
        // Proper NCHW planes [S, 3, H, W]: each value encodes its plane and
        // pixel offset so a wrong channel order changes the exported color.
        let mut images = Vec::with_capacity(pixels * 3);
        for plane in 0..frames * 3 {
            for offset in 0..height * width {
                images.push((offset + 1 + plane * height * width) as f32 / (pixels * 3) as f32);
            }
        }
        let mut extrinsics = Vec::with_capacity(frames * 12);
        for frame in 0..frames {
            extrinsics.extend_from_slice(&[
                1.0,
                0.0,
                0.0,
                frame as f32 * 0.5,
                0.0,
                1.0,
                0.0,
                -(frame as f32) * 0.25,
                0.0,
                0.0,
                1.0,
                2.0 + frame as f32,
            ]);
        }
        (world_points, world_points_conf, images, extrinsics)
    }

    fn temp_path() -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("lux3d-vggt-glb-{}-{nonce}.glb", std::process::id()))
    }

    fn parse_glb(bytes: &[u8]) -> (serde_json::Value, Vec<u8>) {
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
        let bin_len =
            u32::from_le_bytes(bytes[bin_offset - 8..bin_offset - 4].try_into().unwrap()) as usize;
        assert_eq!(
            u32::from_le_bytes(bytes[bin_offset - 4..bin_offset].try_into().unwrap()),
            0x004E_4942
        );
        (json, bytes[bin_offset..bin_offset + bin_len].to_vec())
    }

    #[test]
    fn percentile_matches_numpy_linear_method() {
        let mut values = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(percentile_sorted(&mut values, 50.0), 1.5);
        assert_eq!(percentile_sorted(&mut values, 0.0), 0.0);
        assert_eq!(percentile_sorted(&mut values, 100.0), 3.0);
        assert!((percentile_sorted(&mut values, 5.0) - 0.15).abs() < 1e-12);
        assert!((percentile_sorted(&mut values, 95.0) - 2.85).abs() < 1e-12);
    }

    #[test]
    fn gist_rainbow_matches_matplotlib_samples() {
        // Reference values from matplotlib 3.x `gist_rainbow` LUT after the
        // demo's `int(255 * x)` conversion.
        assert_eq!(gist_rainbow_rgb(0.0), [255, 0, 40]);
        assert_eq!(gist_rainbow_rgb(0.1), [255, 93, 0]);
        assert_eq!(gist_rainbow_rgb(0.2), [255, 234, 0]);
        assert_eq!(gist_rainbow_rgb(0.3), [140, 255, 0]);
        assert_eq!(gist_rainbow_rgb(0.5), [0, 255, 139]);
        assert_eq!(gist_rainbow_rgb(1.0), [255, 0, 191]);
    }

    /// Golden values generated by running the official `visual_util.py` with
    /// `conf_thres=50`, empty points (scene_scale = 1) and an identity first
    /// camera: the cone mesh has the trimesh-merged 14 vertices / 48 faces.
    #[test]
    fn camera_mesh_matches_official_cone_geometry() {
        let extrinsic = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mesh = build_camera_mesh(&extrinsic, 1.0, [255, 0, 40]).expect("camera mesh");
        assert_eq!(mesh.vertices.len(), 14 * 3);
        assert_eq!(mesh.faces.len(), 48 * 3);

        let expected_vertices: [[f64; 3]; 14] = [
            [0.035355339059327376, -0.035355339059327383, 0.1],
            [0.0, 0.0, 0.0],
            [-0.035355339059327376, -0.035355339059327376, 0.1],
            [-0.035355339059327383, 0.035355339059327376, 0.1],
            [0.035355339059327376, 0.035355339059327383, 0.1],
            [0.033587572106361006, -0.033587572106361013, 0.1],
            [0.0, 0.0, 0.0050000000000000044],
            [-0.033587572106361006, -0.033587572106361006, 0.1],
            [-0.033587572106361013, 0.033587572106361006, 0.1],
            [0.033587572106361006, 0.033587572106361013, 0.1],
            [0.034099918003124931, -0.036567685080958527, 0.1],
            [-0.036567685080958527, -0.034099918003124931, 0.1],
            [-0.034099918003124931, 0.036567685080958527, 0.1],
            [0.036567685080958527, 0.034099918003124931, 0.1],
        ];
        for (vertex, expected) in mesh.vertices.chunks_exact(3).zip(expected_vertices) {
            for axis in 0..3 {
                assert!(
                    (f64::from(vertex[axis]) - expected[axis]).abs() < 1e-7,
                    "vertex mismatch: {vertex:?} vs {expected:?}"
                );
            }
        }

        let expected_faces: [u32; 48 * 3] = [
            0, 2, 7, 0, 5, 1, 6, 2, 1, 0, 2, 11, 0, 10, 1, 1, 2, 1, 2, 3, 8, 2, 7, 1, 6, 3, 1, 2,
            3, 12, 2, 11, 1, 1, 3, 1, 3, 4, 9, 3, 8, 1, 6, 4, 1, 3, 4, 13, 3, 12, 1, 1, 4, 1, 4, 0,
            5, 4, 9, 1, 6, 0, 1, 4, 0, 10, 4, 13, 1, 1, 0, 1, 7, 2, 0, 1, 5, 0, 1, 2, 6, 11, 2, 0,
            1, 10, 0, 1, 2, 1, 8, 3, 2, 1, 7, 2, 1, 3, 6, 12, 3, 2, 1, 11, 2, 1, 3, 1, 9, 4, 3, 1,
            8, 3, 1, 4, 6, 13, 4, 3, 1, 12, 3, 1, 4, 1, 5, 0, 4, 1, 9, 4, 1, 0, 6, 10, 0, 4, 1, 13,
            4, 1, 0, 1,
        ];
        assert_eq!(mesh.faces, expected_faces);
        assert_eq!(mesh.color, [255, 0, 40]);
    }

    /// Golden for a rotated second camera (E1 = Ry(90deg) + translation),
    /// generated by the official code with scene_scale = 1.
    #[test]
    fn camera_mesh_applies_extrinsic_transform_verbatim() {
        let extrinsic = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0, -1.0, 0.0, 0.0, 3.0];
        let mesh = build_camera_mesh(&extrinsic, 1.0, [0, 255, 139]).expect("camera mesh");
        let expected: [[f64; 3]; 14] = [
            [2.9, -2.0353553390593273, -0.9646446609406726],
            [3.0, -2.0, -1.0],
            [2.9, -2.0353553390593273, -1.0353553390593273],
            [2.9, -1.9646446609406727, -1.0353553390593273],
            [2.9, -1.9646446609406727, -0.9646446609406726],
            [2.9, -2.033587572106361, -0.966412427893639],
            [2.995, -2.0, -1.0],
            [2.9, -2.033587572106361, -1.033587572106361],
            [2.9, -1.966412427893639, -1.033587572106361],
            [2.9, -1.966412427893639, -0.966412427893639],
            [2.9, -2.0365676850809584, -0.965900081996875],
            [2.9, -2.034099918003125, -1.0365676850809584],
            [2.9, -1.9634323149190416, -1.034099918003125],
            [2.9, -1.965900081996875, -0.9634323149190415],
        ];
        for (vertex, expected) in mesh.vertices.chunks_exact(3).zip(expected) {
            for axis in 0..3 {
                assert!(
                    (f64::from(vertex[axis]) - expected[axis]).abs() < 1e-6,
                    "vertex mismatch: {vertex:?} vs {expected:?}"
                );
            }
        }
    }

    #[test]
    fn scene_alignment_matches_official_first_camera_math() {
        let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let alignment = scene_alignment(&identity).expect("alignment");
        // inv(I) @ diag(1, -1, -1) @ Ry(180deg) = diag(-1, -1, 1); sin(pi)
        // leaves the same 1.2e-16 cross terms numpy produces.
        assert_eq!(alignment[0][0], -1.0);
        assert_eq!(alignment[1][1], -1.0);
        assert_eq!(alignment[2][2], 1.0);
        assert!(alignment[0][2].abs() < 1e-15);
        assert!(alignment[2][0].abs() < 1e-15);
        assert_eq!(alignment[3], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            [alignment[0][3], alignment[1][3], alignment[2][3]],
            [0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn build_scene_filters_by_confidence_percentile() {
        let (points, conf, images, extrinsics) = synthetic_predictions(1, 1, 4);
        let predictions = VggtGlbPredictions {
            world_points: &points,
            world_points_conf: &conf,
            images: &images,
            extrinsics: &extrinsics,
            frames: 1,
            height: 1,
            width: 4,
        };
        // conf = [1, 2, 3, 4]; percentile 50 -> threshold 2.5 -> 2 points.
        let options = VggtGlbOptions {
            conf_percentile: 50.0,
            show_cam: false,
            ..Default::default()
        };
        let scene = build_vggt_glb_scene(&predictions, &options).expect("scene");
        assert_eq!(scene.point_count(), 2);
        assert_eq!(scene.camera_count(), 0);

        // Percentile 0 disables the percentile gate: every `conf > 1e-5`
        // point survives. Confidences are 1..4, so all four pass.
        let options = VggtGlbOptions {
            conf_percentile: 0.0,
            show_cam: false,
            ..Default::default()
        };
        let scene = build_vggt_glb_scene(&predictions, &options).expect("scene");
        assert_eq!(scene.point_count(), 4);
    }

    #[test]
    fn build_scene_inserts_dummy_point_for_empty_cloud() {
        let (points, _, images, extrinsics) = synthetic_predictions(1, 1, 2);
        let conf = vec![0.0f32, 0.0];
        let predictions = VggtGlbPredictions {
            world_points: &points,
            world_points_conf: &conf,
            images: &images,
            extrinsics: &extrinsics,
            frames: 1,
            height: 1,
            width: 2,
        };
        let scene = build_vggt_glb_scene(
            &predictions,
            &VggtGlbOptions {
                show_cam: true,
                ..Default::default()
            },
        )
        .expect("scene");
        assert_eq!(scene.point_count(), 1);
        assert_eq!(scene.points, vec![1.0, 0.0, 0.0]);
        assert_eq!(scene.point_colors, vec![255, 255, 255]);
        assert_eq!(scene.camera_count(), 1);
        // scene_scale = 1 -> cone height 0.1; the synthetic first camera sits
        // at (0, 0, -2), so the apex is z = -2 and the rim is z = -1.9.
        let mesh = &scene.cameras[0];
        assert_eq!(mesh.vertices.len(), 14 * 3);
        let zs: Vec<f32> = mesh.vertices.chunks_exact(3).map(|v| v[2]).collect();
        let min = zs.iter().copied().fold(f32::INFINITY, f32::min);
        let max = zs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (min - (-2.0)).abs() < 1e-6,
            "apex at the camera center: {min}"
        );
        assert!((max - (-1.9)).abs() < 1e-6, "cone height 0.1: {max}");
    }

    #[test]
    fn build_scene_applies_background_masks() {
        let pixels = 3;
        let points = (0..pixels)
            .flat_map(|index| [index as f32, 0.0, 0.0])
            .collect::<Vec<f32>>();
        let conf = vec![1.0f32; pixels];
        // NCHW: pixel 0 black, pixel 1 white, pixel 2 mid-gray.
        let mut images = vec![0.0f32; pixels * 3];
        for channel in 0..3 {
            images[channel * pixels] = 0.0;
            images[channel * pixels + 1] = 1.0;
            images[channel * pixels + 2] = 0.5;
        }
        let extrinsics = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let predictions = VggtGlbPredictions {
            world_points: &points,
            world_points_conf: &conf,
            images: &images,
            extrinsics: &extrinsics,
            frames: 1,
            height: 1,
            width: pixels,
        };
        let base = VggtGlbOptions {
            conf_percentile: 0.0,
            show_cam: false,
            ..Default::default()
        };
        assert_eq!(
            build_vggt_glb_scene(&predictions, &base)
                .expect("scene")
                .point_count(),
            3
        );
        let black = VggtGlbOptions {
            mask_black_bg: true,
            ..base
        };
        assert_eq!(
            build_vggt_glb_scene(&predictions, &black)
                .expect("scene")
                .point_count(),
            2
        );
        let white = VggtGlbOptions {
            mask_white_bg: true,
            ..base
        };
        assert_eq!(
            build_vggt_glb_scene(&predictions, &white)
                .expect("scene")
                .point_count(),
            2
        );
    }

    #[test]
    fn glb_writer_emits_point_cloud_and_camera_nodes() {
        let (points, conf, images, extrinsics) = synthetic_predictions(2, 2, 2);
        let predictions = VggtGlbPredictions {
            world_points: &points,
            world_points_conf: &conf,
            images: &images,
            extrinsics: &extrinsics,
            frames: 2,
            height: 2,
            width: 2,
        };
        let scene = build_vggt_glb_scene(
            &predictions,
            &VggtGlbOptions {
                conf_percentile: 50.0,
                show_cam: true,
                mask_black_bg: false,
                mask_white_bg: false,
            },
        )
        .expect("scene");
        // conf 1..=8, threshold 4.5 -> 4 points, 2 cameras.
        assert_eq!(scene.point_count(), 4);
        assert_eq!(scene.camera_count(), 2);

        let path = temp_path();
        write_vggt_glb(&scene, &path).expect("write glb");
        let bytes = std::fs::read(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);
        let (json, bin) = parse_glb(&bytes);

        assert_eq!(json["asset"]["version"], "2.0");
        assert_eq!(json["nodes"].as_array().unwrap().len(), 3);
        assert_eq!(json["meshes"].as_array().unwrap().len(), 3);
        assert_eq!(json["scenes"][0]["nodes"], json!([0, 1, 2]));
        // 2 point accessors + 2 per camera + one shared index accessor.
        assert_eq!(json["accessors"].as_array().unwrap().len(), 7);
        assert_eq!(json["bufferViews"].as_array().unwrap().len(), 7);
        assert_eq!(
            json["buffers"][0]["byteLength"].as_u64().unwrap() as usize,
            bin.len()
        );

        // Point cloud: POINTS primitive, POSITION + COLOR_0, no indices.
        let point_primitive = &json["meshes"][0]["primitives"][0];
        assert_eq!(point_primitive["mode"], 0);
        assert!(point_primitive.get("indices").is_none());
        let point_positions = &json["accessors"]
            [point_primitive["attributes"]["POSITION"].as_u64().unwrap() as usize];
        assert_eq!(point_positions["count"], 4);
        assert_eq!(point_positions["type"], "VEC3");
        let point_colors =
            &json["accessors"][point_primitive["attributes"]["COLOR_0"].as_u64().unwrap() as usize];
        assert_eq!(point_colors["count"], 4);
        assert_eq!(point_colors["componentType"], 5121);
        assert_eq!(point_colors["normalized"], true);

        // Camera meshes: shared index accessor, 14 vertices / 48 triangles.
        let shared_indices = json["meshes"][1]["primitives"][0]["indices"].clone();
        for mesh_index in [1usize, 2] {
            let primitive = &json["meshes"][mesh_index]["primitives"][0];
            assert_eq!(primitive["mode"], 4);
            assert_eq!(primitive["indices"], shared_indices);
            let positions =
                &json["accessors"][primitive["attributes"]["POSITION"].as_u64().unwrap() as usize];
            assert_eq!(positions["count"], 14);
            let colors =
                &json["accessors"][primitive["attributes"]["COLOR_0"].as_u64().unwrap() as usize];
            assert_eq!(colors["count"], 14);
        }
        let indices = &json["accessors"][shared_indices.as_u64().unwrap() as usize];
        assert_eq!(indices["componentType"], 5125);
        assert_eq!(indices["count"], 48 * 3);

        // Node matrices carry the first-camera alignment (identity extrinsic
        // -> diag(-1, -1, 1) with the sin(pi) cross terms).
        for node in json["nodes"].as_array().unwrap() {
            let matrix = node["matrix"].as_array().unwrap();
            assert_eq!(matrix.len(), 16);
            assert_eq!(matrix[0], -1.0);
            assert_eq!(matrix[5], -1.0);
            assert_eq!(matrix[10], 1.0);
            assert_eq!(matrix[15], 1.0);
        }

        // Point colors are the NCHW -> NHWC colors of the surviving pixels,
        // truncated by `(value * 255) as u8` like numpy's uint8 cast. With
        // conf 1..=8 and a 4.5 threshold the survivors are pixels 4..=7
        // (frame 1, offsets 0..=3); value = (offset + 1 + plane * 4) / 24 for
        // planes 3, 4, 5 of the synthetic images.
        let color_view =
            &json["bufferViews"][point_colors["bufferView"].as_u64().unwrap() as usize];
        let color_offset = color_view["byteOffset"].as_u64().unwrap() as usize;
        let color_bytes =
            &bin[color_offset..color_offset + color_view["byteLength"].as_u64().unwrap() as usize];
        let expected_colors: [[u8; 4]; 4] = [
            [138, 180, 223, 255],
            [148, 191, 233, 255],
            [159, 201, 244, 255],
            [170, 212, 255, 255],
        ];
        assert_eq!(color_bytes.len(), expected_colors.len() * 4);
        for (pixel, expected) in color_bytes.chunks_exact(4).zip(expected_colors) {
            assert_eq!(pixel, expected);
        }
    }
}
