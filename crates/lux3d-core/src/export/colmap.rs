//! COLMAP binary sparse-model export for the VGGT feedforward path.
//!
//! Mirrors the upstream `demo_colmap.py` feedforward mode
//! (`batch_np_matrix_to_pycolmap_wo_track` + `rename_colmap_recons_and_rescale_camera`):
//! a confidence mask `depth_conf >= conf_threshold`, at most 100000 3D points,
//! pixel-grid 2D associations with track length 1, one PINHOLE camera per
//! image, and camera/point rescaling back to the original image resolution
//! through the per-frame [`VggtOriginalCoords`].
//!
//! Differences from upstream, all deliberate and documented:
//! * The 3D-point source defaults to the depth-unprojected
//!   [`VggtInferenceOutput::world_points_from_depth`] (upstream
//!   `unproject_depth_map_to_point_map`); [`VggtColmapPointSource::Pointmap`]
//!   selects the point-head `world_points` instead.
//! * Point limiting without a seed is a deterministic evenly-spaced subsample
//!   of the ascending candidate list. With [`VggtColmapOptions::seed`] the
//!   subset is drawn with a seeded SplitMix64 Fisher-Yates shuffle: uniform
//!   without replacement, distribution-equivalent to upstream's
//!   `np.random.choice(..., replace=False)` but not draw-identical to numpy.
//! * `sparse/points.ply` next to the `.bin` files is the ASCII PLY written by
//!   this crate (upstream delegates to trimesh, which defaults to binary).
//! * The binary layout matches COLMAP's sparse model format as specified for
//!   this task (COLMAP 3.8-era `images.bin` still stores the per-point 2D
//!   `point3D_id`; newer COLMAP releases omit it). Integer ids are written as
//!   little-endian unsigned values, which is byte-identical to the task's
//!   signed spec for the positive 1-based ids used here.

use std::{fs, path::Path};

use candle_core::{Device, Tensor};

use super::point_cloud::{write_ply, PointCloudView};
use crate::{
    error::Lux3dError,
    geometry::{rescale_intrinsics_to_original, shift_point2d_to_original},
    preprocess::{VggtOriginalCoords, VggtPreparedInputs},
    runtime::VggtInferenceOutput,
    Result,
};

/// COLMAP camera model id for `PINHOLE` (params: fx, fy, cx, cy).
pub const COLMAP_CAMERA_MODEL_PINHOLE: i32 = 1;
/// Upstream demo `conf_thres_value` default for the feedforward mode.
pub const DEFAULT_CONF_THRESHOLD: f32 = 5.0;
/// Upstream demo `max_points_for_colmap`.
pub const DEFAULT_MAX_POINTS: usize = 100_000;

/// Source of the exported 3D points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VggtColmapPointSource {
    /// Depth unprojection `unproject_depth_map_to_point_map`
    /// ([`VggtInferenceOutput::world_points_from_depth`]) — the upstream
    /// `demo_colmap.py` feedforward source (`demo_colmap.py:140, 216`).
    #[default]
    Depth,
    /// Point-head `world_points` ([`VggtInferenceOutput::world_points`]).
    Pointmap,
}

#[derive(Debug, Clone)]
pub struct VggtColmapOptions {
    /// Confidence gate `depth_conf >= conf_threshold` (upstream uses `>=` in
    /// the COLMAP demo, versus the PLY export's strict `>`).
    pub conf_threshold: f32,
    /// Cap on the number of exported 3D points.
    pub max_points: usize,
    /// Optional additional gate `world_points_conf >= value` applied on top of
    /// `conf_threshold`. Upstream only gates on depth confidence.
    pub confidence: Option<f32>,
    /// 3D-point source; `Depth` is the upstream feedforward semantics.
    pub point_source: VggtColmapPointSource,
    /// Seeded random point limiting, mirroring upstream `--seed` +
    /// `randomly_limit_trues` (`demo_colmap.py:45, 98-100`;
    /// `vggt/utils/helper.py:10-30`). `None` keeps the deterministic
    /// evenly-spaced subsample.
    pub seed: Option<u32>,
}

impl Default for VggtColmapOptions {
    fn default() -> Self {
        Self {
            conf_threshold: DEFAULT_CONF_THRESHOLD,
            max_points: DEFAULT_MAX_POINTS,
            confidence: None,
            point_source: VggtColmapPointSource::Depth,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColmapExportStats {
    pub camera_count: usize,
    pub image_count: usize,
    pub point3d_count: usize,
    /// Total 2D points written across all images (track length is always 1 in
    /// the feedforward mode, so this equals `point3d_count`).
    pub point2d_count: usize,
    /// Candidates that passed the confidence gate before point limiting.
    pub confidence_candidates: usize,
}

/// Write a COLMAP binary sparse model (`cameras.bin`, `images.bin`,
/// `points3D.bin`) plus the sibling `points.ply` visualization cloud for the
/// VGGT feedforward reconstruction.
///
/// * 3D points: [`VggtColmapOptions::point_source`] (depth unprojection by
///   default, matching upstream) filtered by the confidence gate.
/// * 2D points: the pixel grid `(x, y)` of each sampled pixel, shifted to the
///   original resolution via [`shift_point2d_to_original`].
/// * Colors: prepared RGB frames resampled to the model resolution.
/// * Cameras: one PINHOLE camera per image, intrinsics rescaled to the
///   original resolution via [`rescale_intrinsics_to_original`].
/// * Ids are 1-based (COLMAP convention); files are deterministic unless a
///   random seed is set.
pub fn write_feedforward_colmap(
    dir: &Path,
    output: &VggtInferenceOutput,
    prepared: &VggtPreparedInputs,
    image_names: &[String],
    options: &VggtColmapOptions,
) -> Result<ColmapExportStats> {
    let (s, h, w) = resolve_grid_dims(output)?;
    if prepared.original_coords.len() != s {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT COLMAP export needs {} per-frame original coords, found {}",
                s,
                prepared.original_coords.len()
            ),
        });
    }
    let coords: &[VggtOriginalCoords] = &prepared.original_coords;
    let img_size = f64::from(prepared.target_size.width.max(prepared.target_size.height));
    if img_size <= 0.0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT prepared target size must be non-zero",
        ));
    }

    // Points, colors, confidences (all flattened to CPU row-major vectors).
    let points = resolve_point_source(output, options.point_source, s, h, w)?;
    let rgb = rgb_frames_at_model_resolution(prepared, s, h, w)?;
    let depth_conf = tensor_to_cpu_vec::<f32>(&squeeze_grid(&output.depth_conf)?, "depth conf")?;
    let world_conf = match options.confidence {
        Some(_) => Some(tensor_to_cpu_vec::<f32>(
            &squeeze_grid(&output.world_points_conf)?,
            "world points conf",
        )?),
        None => None,
    };

    // Confidence mask (candidates kept in ascending grid order, so no
    // duplicates are possible and the output order is deterministic).
    let candidates: Vec<usize> = (0..s * h * w)
        .filter(|&idx| {
            depth_conf[idx] >= options.conf_threshold
                && world_conf.as_ref().is_none_or(|values| {
                    values[idx] >= options.confidence.unwrap_or(f32::NEG_INFINITY)
                })
        })
        .collect();
    let selected = match options.seed {
        Some(seed) => limit_candidates_seeded(&candidates, options.max_points, seed),
        None => limit_candidates(&candidates, options.max_points),
    };

    let intrinsics_rescaled = rescale_intrinsics_to_original(&output.intrinsics, coords, img_size)?;
    let extrinsics =
        tensor_to_cpu_vec::<f64>(&squeeze_leading_batch_3d(&output.extrinsics)?, "extrinsics")?;

    // Build the 3D points plus their single-element tracks.
    struct ExportPoint3D {
        point3d_id: usize,
        xyz: [f64; 3],
        rgb: [u8; 3],
        frame: usize,
        point2d_idx: usize,
    }
    let mut per_frame_points2d: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); s];
    let mut points3d = Vec::with_capacity(selected.len());
    // `sparse/points.ply` payload: the same selected points/colors as the
    // `points3D.bin` entries, in export order.
    let mut ply_points = Vec::with_capacity(selected.len() * 3);
    let mut ply_colors = Vec::with_capacity(selected.len() * 3);
    for (rank, &idx) in selected.iter().enumerate() {
        let frame = idx / (h * w);
        let rem = idx % (h * w);
        let y = rem / w;
        let x = rem % w;
        let xyz = [
            f64::from(points[idx * 3]),
            f64::from(points[idx * 3 + 1]),
            f64::from(points[idx * 3 + 2]),
        ];
        let color = [
            rgb_u8(rgb[(frame * 3) * h * w + y * w + x]),
            rgb_u8(rgb[(frame * 3 + 1) * h * w + y * w + x]),
            rgb_u8(rgb[(frame * 3 + 2) * h * w + y * w + x]),
        ];
        ply_points.extend_from_slice(&[points[idx * 3], points[idx * 3 + 1], points[idx * 3 + 2]]);
        ply_colors.extend_from_slice(&color);
        let (px, py) = shift_point2d_to_original(x as f64, y as f64, &coords[frame], img_size);
        let point2d_idx = per_frame_points2d[frame].len();
        per_frame_points2d[frame].push((rank + 1, px, py));
        points3d.push(ExportPoint3D {
            point3d_id: rank + 1,
            xyz,
            rgb: color,
            frame,
            point2d_idx,
        });
    }

    // cameras.bin: u64 count; per camera: u32 camera_id, i32 model_id,
    // u64 width, u64 height, 4 x f64 params (fx, fy, cx, cy).
    let mut cameras_bin = Vec::new();
    put_u64(&mut cameras_bin, s as u64);
    for frame in 0..s {
        let k = &intrinsics_rescaled[frame];
        put_u32(&mut cameras_bin, (frame + 1) as u32);
        put_i32(&mut cameras_bin, COLMAP_CAMERA_MODEL_PINHOLE);
        put_u64(&mut cameras_bin, u64::from(coords[frame].width));
        put_u64(&mut cameras_bin, u64::from(coords[frame].height));
        put_f64(&mut cameras_bin, k[0][0]);
        put_f64(&mut cameras_bin, k[1][1]);
        put_f64(&mut cameras_bin, k[0][2]);
        put_f64(&mut cameras_bin, k[1][2]);
    }

    // images.bin: u64 count; per image: u32 image_id, 4 x f64 qvec (w, x, y,
    // z), 3 x f64 tvec, u32 camera_id, NUL-terminated name, u64 num_points2D,
    // per point: 2 x f64 xy + u64 point3D_id.
    let mut images_bin = Vec::new();
    put_u64(&mut images_bin, s as u64);
    for frame in 0..s {
        let rotation = extrinsics_rotation(&extrinsics, frame);
        let translation = extrinsics_translation(&extrinsics, frame);
        let qvec = rotation_to_quat_wxyz(&rotation);
        put_u32(&mut images_bin, (frame + 1) as u32);
        for value in &qvec {
            put_f64(&mut images_bin, *value);
        }
        for value in &translation {
            put_f64(&mut images_bin, *value);
        }
        put_u32(&mut images_bin, (frame + 1) as u32);
        let fallback = format!("image_{}", frame + 1);
        let name = image_names
            .get(frame)
            .map(String::as_str)
            .filter(|name| !name.is_empty())
            .unwrap_or(&fallback);
        put_nul_terminated(&mut images_bin, name);
        let points2d = &per_frame_points2d[frame];
        put_u64(&mut images_bin, points2d.len() as u64);
        for (point3d_id, px, py) in points2d {
            put_f64(&mut images_bin, *px);
            put_f64(&mut images_bin, *py);
            put_u64(&mut images_bin, *point3d_id as u64);
        }
    }

    // points3D.bin: u64 count; per point: u64 point3D_id, 3 x f64 xyz,
    // 3 x u8 rgb, f64 error (0.0), u64 track_len, per track element:
    // u32 image_id + u32 point2D_idx.
    let mut points3d_bin = Vec::new();
    put_u64(&mut points3d_bin, points3d.len() as u64);
    for point in &points3d {
        put_u64(&mut points3d_bin, point.point3d_id as u64);
        for value in &point.xyz {
            put_f64(&mut points3d_bin, *value);
        }
        for channel in &point.rgb {
            put_u8(&mut points3d_bin, *channel);
        }
        put_f64(&mut points3d_bin, 0.0);
        put_u64(&mut points3d_bin, 1); // track length (feedforward: no tracks)
        put_u32(&mut points3d_bin, (point.frame + 1) as u32);
        put_u32(&mut points3d_bin, point.point2d_idx as u32);
    }

    fs::create_dir_all(dir).map_err(|source| Lux3dError::ModelAssetIo {
        path: dir.to_path_buf(),
        source,
    })?;
    for (name, bytes) in [
        ("cameras.bin", cameras_bin),
        ("images.bin", images_bin),
        ("points3D.bin", points3d_bin),
    ] {
        let path = dir.join(name);
        fs::write(&path, bytes).map_err(|source| Lux3dError::ModelAssetIo { path, source })?;
    }
    // Upstream writes the visualization cloud next to the sparse model
    // (`demo_colmap.py:249`).
    let ply_path = dir.join("points.ply");
    let ply_view = PointCloudView {
        points: &ply_points,
        colors: &ply_colors,
    };
    write_ply(&ply_view, &ply_path)?;

    Ok(ColmapExportStats {
        camera_count: s,
        image_count: s,
        point3d_count: points3d.len(),
        point2d_count: points3d.len(),
        confidence_candidates: candidates.len(),
    })
}

/// Deterministic point cap: keep an evenly spaced subset of the ascending
/// candidate list (`candidates[j * len / max]` for `j in 0..max`), which
/// preserves order and uniqueness. Replaces upstream's seeded random choice
/// when [`VggtColmapOptions::seed`] is `None`.
fn limit_candidates(candidates: &[usize], max_points: usize) -> Vec<usize> {
    let count = candidates.len();
    if count <= max_points {
        return candidates.to_vec();
    }
    (0..max_points)
        .map(|j| candidates[j * count / max_points])
        .collect()
}

/// Seeded point cap: `max_points` distinct candidate positions drawn
/// uniformly without replacement (partial Fisher-Yates over a SplitMix64
/// stream), returned in ascending candidate order so the output stays
/// deterministic for a fixed seed. This is distribution-equivalent to
/// upstream `np.random.choice(true_indices, replace=False)`
/// (`helper.py:10-30`); the draw stream itself differs from numpy.
fn limit_candidates_seeded(candidates: &[usize], max_points: usize, seed: u32) -> Vec<usize> {
    let count = candidates.len();
    if count <= max_points {
        return candidates.to_vec();
    }
    let mut state = u64::from(seed);
    let mut positions: Vec<usize> = (0..count).collect();
    for slot in 0..max_points {
        let remaining = count - slot;
        let offset = (splitmix64_next(&mut state) % remaining as u64) as usize;
        positions.swap(slot, slot + offset);
    }
    let mut picked = positions[..max_points].to_vec();
    picked.sort_unstable();
    picked
        .into_iter()
        .map(|position| candidates[position])
        .collect()
}

/// One SplitMix64 output; the same scrambler used by the deterministic
/// sampling RNGs elsewhere in the crate (no external RNG dependency).
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Flatten the configured 3D-point source to CPU row-major `[x, y, z, ...]`
/// values for the `(s, h, w)` grid.
fn resolve_point_source(
    output: &VggtInferenceOutput,
    point_source: VggtColmapPointSource,
    s: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>> {
    let points = match point_source {
        VggtColmapPointSource::Depth => output.world_points_from_depth()?,
        VggtColmapPointSource::Pointmap => output.world_points.clone(),
    };
    let dims = points.dims();
    let grid_ok = match dims.len() {
        4 => dims == [s, h, w, 3],
        5 => dims == [1, s, h, w, 3],
        _ => false,
    };
    if !grid_ok {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT {point_source:?} COLMAP point source shape {dims:?} does not match grid {s}x{h}x{w}"
            ),
        });
    }
    tensor_to_cpu_vec::<f32>(&points, "world points")
}

/// Resolve the `(frames, height, width)` grid from the output tensors and
/// validate that `depth`/`depth_conf` agree with `world_points`.
fn resolve_grid_dims(output: &VggtInferenceOutput) -> Result<(usize, usize, usize)> {
    let world_dims = output.world_points.dims();
    let (s, h, w) = match world_dims.len() {
        5 if world_dims[0] == 1 && world_dims[4] == 3 => {
            (world_dims[1], world_dims[2], world_dims[3])
        }
        4 if world_dims[3] == 3 => (world_dims[0], world_dims[1], world_dims[2]),
        _ => {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "VGGT world_points shape {world_dims:?} must be [1,S,H,W,3] or [S,H,W,3]"
                ),
            });
        }
    };

    let depth_dims = output.depth.dims();
    let depth_ok = match depth_dims.len() {
        5 => depth_dims == [1, s, h, w, 1],
        4 => depth_dims == [s, h, w, 1],
        3 => depth_dims == [s, h, w],
        _ => false,
    };
    if !depth_ok {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT depth shape {depth_dims:?} does not match world_points grid {s}x{h}x{w}"
            ),
        });
    }

    let conf_dims = squeeze_grid(&output.depth_conf)?.dims().to_vec();
    if conf_dims != vec![s, h, w] {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT depth_conf shape {conf_dims:?} does not match world_points grid {s}x{h}x{w}"
            ),
        });
    }
    Ok((s, h, w))
}

/// Squeeze a confidence grid `[1, S, H, W]` -> `[S, H, W]`.
fn squeeze_grid(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() == 4 && dims[0] == 1 {
        tensor
            .squeeze(0)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to squeeze VGGT confidence grid: {source}"),
            })
    } else {
        Ok(tensor.clone())
    }
}

/// Squeeze `[1, S, 3, 4]` -> `[S, 3, 4]` (no-op for already-squeezed input).
fn squeeze_leading_batch_3d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() == 4 && dims[0] == 1 {
        tensor
            .squeeze(0)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to squeeze VGGT extrinsics batch dim: {source}"),
            })
    } else {
        Ok(tensor.clone())
    }
}

fn tensor_to_cpu_vec<S: candle_core::WithDType>(tensor: &Tensor, name: &str) -> Result<Vec<S>> {
    tensor
        .to_device(&Device::Cpu)
        .and_then(|tensor| tensor.to_dtype(S::DTYPE))
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1::<S>())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to read VGGT {name} for COLMAP export: {source}"),
        })
}

/// Prepared RGB frames flattened as `[S, 3, H, W]` f32 in `[0, 1]`, resampled
/// to the model grid when the prepared size differs (approximate torch
/// bilinear; exact in the normal case where both sizes already match).
fn rgb_frames_at_model_resolution(
    prepared: &VggtPreparedInputs,
    s: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>> {
    let dims = prepared.rgb_frames.dims();
    if dims.len() != 4 || dims[0] != s || dims[1] != 3 {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT rgb_frames shape {dims:?} must be [{s}, 3, H, W] for COLMAP export"
            ),
        });
    }
    let flat = tensor_to_cpu_vec::<f32>(&prepared.rgb_frames, "rgb frames")?;
    let (src_h, src_w) = (dims[2], dims[3]);
    if (src_h, src_w) == (h, w) {
        return Ok(flat);
    }
    Ok(resize_chw_bilinear(&flat, s * 3, src_h, src_w, h, w))
}

/// Bilinear CHW resize (torch `F.interpolate(align_corners=False)` formula:
/// source coordinate `(dst + 0.5) * src / dst - 0.5`, clamped).
fn resize_chw_bilinear(
    src: &[f32],
    planes: usize,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let scale_y = src_h as f64 / dst_h as f64;
    let scale_x = src_w as f64 / dst_w as f64;
    let mut out = vec![0.0_f32; planes * dst_h * dst_w];
    for dy in 0..dst_h {
        let sy = ((dy as f64 + 0.5) * scale_y - 0.5).clamp(0.0, (src_h - 1) as f64);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = (sy - y0 as f64) as f32;
        for dx in 0..dst_w {
            let sx = ((dx as f64 + 0.5) * scale_x - 0.5).clamp(0.0, (src_w - 1) as f64);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = (sx - x0 as f64) as f32;
            for plane in 0..planes {
                let top = src[plane * src_h * src_w + y0 * src_w + x0] * (1.0 - wx)
                    + src[plane * src_h * src_w + y0 * src_w + x1] * wx;
                let bottom = src[plane * src_h * src_w + y1 * src_w + x0] * (1.0 - wx)
                    + src[plane * src_h * src_w + y1 * src_w + x1] * wx;
                out[plane * dst_h * dst_w + dy * dst_w + dx] = top * (1.0 - wy) + bottom * wy;
            }
        }
    }
    out
}

fn rgb_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn extrinsics_rotation(extrinsics: &[f64], frame: usize) -> [[f64; 3]; 3] {
    let base = frame * 12;
    [
        [extrinsics[base], extrinsics[base + 1], extrinsics[base + 2]],
        [
            extrinsics[base + 4],
            extrinsics[base + 5],
            extrinsics[base + 6],
        ],
        [
            extrinsics[base + 8],
            extrinsics[base + 9],
            extrinsics[base + 10],
        ],
    ]
}

fn extrinsics_translation(extrinsics: &[f64], frame: usize) -> [f64; 3] {
    let base = frame * 12;
    [
        extrinsics[base + 3],
        extrinsics[base + 7],
        extrinsics[base + 11],
    ]
}

/// Rotation matrix -> normalized COLMAP quaternion `(w, x, y, z)`
/// (Shepperd's method), matching `pycolmap.Rotation3d(extrinsics[:, :3, :3])`.
fn rotation_to_quat_wxyz(r: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let (w, x, y, z) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        )
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        (
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        )
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        (
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        )
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        (
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        )
    };
    let norm = (w * w + x * x + y * y + z * z).sqrt();
    [w / norm, x / norm, y / norm, z / norm]
}

fn put_u8(buf: &mut Vec<u8>, value: u8) {
    buf.push(value);
}

fn put_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_i32(buf: &mut Vec<u8>, value: i32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_f64(buf: &mut Vec<u8>, value: f64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_nul_terminated(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(value.as_bytes());
    buf.push(0);
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::*;
    use crate::preprocess::VggtPreprocessMode;

    /// Minimal test-only binary reader (mirrors the writer layout).
    mod reader {
        pub struct Cursor<'a> {
            data: &'a [u8],
            pos: usize,
        }

        impl<'a> Cursor<'a> {
            pub fn new(data: &'a [u8]) -> Self {
                Self { data, pos: 0 }
            }

            fn take(&mut self, n: usize) -> &'a [u8] {
                let slice = &self.data[self.pos..self.pos + n];
                self.pos += n;
                slice
            }

            pub fn u8(&mut self) -> u8 {
                self.take(1)[0]
            }

            pub fn u32(&mut self) -> u32 {
                u32::from_le_bytes(self.take(4).try_into().expect("u32"))
            }

            pub fn i32(&mut self) -> i32 {
                i32::from_le_bytes(self.take(4).try_into().expect("i32"))
            }

            pub fn u64(&mut self) -> u64 {
                u64::from_le_bytes(self.take(8).try_into().expect("u64"))
            }

            pub fn f64(&mut self) -> f64 {
                f64::from_le_bytes(self.take(8).try_into().expect("f64"))
            }

            pub fn f64_array(&mut self, n: usize) -> Vec<f64> {
                (0..n).map(|_| self.f64()).collect()
            }

            pub fn string(&mut self) -> String {
                let start = self.pos;
                while self.data[self.pos] != 0 {
                    self.pos += 1;
                }
                let value =
                    String::from_utf8(self.data[start..self.pos].to_vec()).expect("utf-8 name");
                self.pos += 1; // NUL
                value
            }

            pub fn expect_end(&mut self) {
                assert_eq!(self.pos, self.data.len(), "trailing binary bytes");
            }
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct TestCamera {
            pub camera_id: u32,
            pub model_id: i32,
            pub width: u64,
            pub height: u64,
            pub params: Vec<f64>,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct TestPoint2D {
            pub x: f64,
            pub y: f64,
            pub point3d_id: u64,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct TestImage {
            pub image_id: u32,
            pub qvec: Vec<f64>,
            pub tvec: Vec<f64>,
            pub camera_id: u32,
            pub name: String,
            pub points2d: Vec<TestPoint2D>,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct TestTrackElement {
            pub image_id: u32,
            pub point2d_idx: u32,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct TestPoint3D {
            pub point3d_id: u64,
            pub xyz: Vec<f64>,
            pub rgb: [u8; 3],
            pub error: f64,
            pub track: Vec<TestTrackElement>,
        }

        pub fn parse_cameras(data: &[u8]) -> Vec<TestCamera> {
            let mut cursor = Cursor::new(data);
            let count = cursor.u64() as usize;
            let cameras = (0..count)
                .map(|_| TestCamera {
                    camera_id: cursor.u32(),
                    model_id: cursor.i32(),
                    width: cursor.u64(),
                    height: cursor.u64(),
                    params: cursor.f64_array(4),
                })
                .collect();
            cursor.expect_end();
            cameras
        }

        pub fn parse_images(data: &[u8]) -> Vec<TestImage> {
            let mut cursor = Cursor::new(data);
            let count = cursor.u64() as usize;
            let images = (0..count)
                .map(|_| {
                    let image_id = cursor.u32();
                    let qvec = cursor.f64_array(4);
                    let tvec = cursor.f64_array(3);
                    let camera_id = cursor.u32();
                    let name = cursor.string();
                    let num_points2d = cursor.u64() as usize;
                    let points2d = (0..num_points2d)
                        .map(|_| TestPoint2D {
                            x: cursor.f64(),
                            y: cursor.f64(),
                            point3d_id: cursor.u64(),
                        })
                        .collect();
                    TestImage {
                        image_id,
                        qvec,
                        tvec,
                        camera_id,
                        name,
                        points2d,
                    }
                })
                .collect();
            cursor.expect_end();
            images
        }

        pub fn parse_points3d(data: &[u8]) -> Vec<TestPoint3D> {
            let mut cursor = Cursor::new(data);
            let count = cursor.u64() as usize;
            let points = (0..count)
                .map(|_| {
                    let point3d_id = cursor.u64();
                    let xyz = cursor.f64_array(3);
                    let rgb = [cursor.u8(), cursor.u8(), cursor.u8()];
                    let error = cursor.f64();
                    let track_len = cursor.u64() as usize;
                    let track = (0..track_len)
                        .map(|_| TestTrackElement {
                            image_id: cursor.u32(),
                            point2d_idx: cursor.u32(),
                        })
                        .collect();
                    TestPoint3D {
                        point3d_id,
                        xyz,
                        rgb,
                        error,
                        track,
                    }
                })
                .collect();
            cursor.expect_end();
            points
        }
    }

    /// Quaternion (w, x, y, z) -> rotation matrix, for verifying `qvec`.
    fn quat_to_mat_wxyz(q: &[f64; 4]) -> [[f64; 3]; 3] {
        let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    struct Fixture {
        output: VggtInferenceOutput,
        prepared: VggtPreparedInputs,
        candidates: Vec<usize>,
        selected: Vec<usize>,
        img_size: f64,
    }

    /// 2-frame, 4x4 synthetic reconstruction. Confidence values
    /// `((i * 7) % 10) / 4` give 13 candidates at threshold 1.5; the cap of
    /// 10 keeps the evenly spaced subset.
    fn fixture() -> Fixture {
        let device = Device::Cpu;
        let (s, h, w) = (2usize, 4usize, 4usize);
        let pixels = s * h * w;

        let depth_conf: Vec<f32> = (0..pixels).map(|i| ((i * 7) % 10) as f32 / 4.0).collect();
        let depth_conf_tensor =
            Tensor::from_vec(depth_conf.clone(), (1, s, h, w), &device).expect("depth conf");
        let depth = Tensor::from_vec(
            (0..pixels).map(|i| 2.0 + 0.5 * (i % 5) as f32).collect(),
            (1, s, h, w, 1),
            &device,
        )
        .expect("depth");
        let world_points = Tensor::from_vec(
            (0..pixels * 3).map(|i| (i + 1) as f32 * 0.25).collect(),
            (1, s, h, w, 3),
            &device,
        )
        .expect("world points");
        let world_points_conf =
            Tensor::ones((1, s, h, w), DType::F32, &device).expect("world points conf");

        // Frame 0: 90-degree z rotation; frame 1: identity.
        let extrinsics = vec![
            0.0f32, -1.0, 0.0, 0.5, //
            1.0, 0.0, 0.0, -0.3, //
            0.0, 0.0, 1.0, 2.0, //
            1.0, 0.0, 0.0, 0.25, //
            0.0, 1.0, 0.0, 0.5, //
            0.0, 0.0, 1.0, 1.0,
        ];
        let extrinsics = Tensor::from_vec(extrinsics, (1, s, 3, 4), &device).expect("extrinsics");
        let intrinsics = Tensor::from_vec(
            vec![
                50.0f32, 0.0, 2.0, //
                0.0, 60.0, 1.5, //
                0.0, 0.0, 1.0, //
                50.0, 0.0, 2.0, //
                0.0, 60.0, 1.5, //
                0.0, 0.0, 1.0,
            ],
            (1, s, 3, 3),
            &device,
        )
        .expect("intrinsics");

        let pose_enc = Tensor::zeros((1, s, 9), DType::F32, &device).expect("pose enc");
        let rgb_frames = Tensor::from_vec(
            (0..s * 3 * h * w)
                .map(|i| (i % 251) as f32 / 255.0)
                .collect(),
            (s, 3, h, w),
            &device,
        )
        .expect("rgb frames");
        let export_mask = Tensor::ones((1, s, h, w), DType::U8, &device).expect("mask");

        let output = VggtInferenceOutput {
            world_points,
            world_points_conf,
            depth,
            depth_conf: depth_conf_tensor,
            pose_enc,
            pose_enc_list: vec![Tensor::zeros((1, s, 9), DType::F32, &device).expect("pose")],
            extrinsics,
            intrinsics,
            rgb_frames: rgb_frames.clone(),
            export_mask,
        };

        let prepared = VggtPreparedInputs {
            interval: 1,
            original_size: crate::contracts::SpatialSize::new(1024, 768),
            target_size: crate::contracts::SpatialSize::new(4, 4),
            preprocess_mode: VggtPreprocessMode::Crop,
            rgb_frames,
            normalized_frames: Tensor::zeros((1, s, 3, h, w), DType::F32, &device)
                .expect("normalized frames"),
            original_coords: vec![
                VggtOriginalCoords {
                    width: 1024,
                    height: 768,
                    offset_x: 1,
                    offset_y: 2,
                    content_width: 4,
                    content_height: 4,
                    mode: VggtPreprocessMode::Crop,
                    square: None,
                },
                VggtOriginalCoords {
                    width: 500,
                    height: 500,
                    offset_x: 0,
                    offset_y: 0,
                    content_width: 4,
                    content_height: 4,
                    mode: VggtPreprocessMode::Crop,
                    square: None,
                },
            ],
        };

        let candidates: Vec<usize> = (0..pixels).filter(|&i| depth_conf[i] >= 1.5).collect();
        let selected = limit_candidates(&candidates, 10);
        Fixture {
            output,
            prepared,
            candidates,
            selected,
            img_size: 4.0,
        }
    }

    #[test]
    fn vggt_colmap_write_and_parse_feedforward_reconstruction() {
        let fixture = fixture();
        let dir = std::env::temp_dir().join(format!(
            "lux3d-vggt-colmap-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let image_names = vec!["frame_a.png".to_string(), "frame_b.png".to_string()];
        let options = VggtColmapOptions {
            conf_threshold: 1.5,
            max_points: 10,
            confidence: None,
            point_source: VggtColmapPointSource::Pointmap,
            seed: None,
        };

        let stats = write_feedforward_colmap(
            &dir,
            &fixture.output,
            &fixture.prepared,
            &image_names,
            &options,
        )
        .expect("write colmap");

        assert_eq!(stats.camera_count, 2);
        assert_eq!(stats.image_count, 2);
        assert_eq!(stats.confidence_candidates, fixture.candidates.len());
        assert_eq!(stats.confidence_candidates, 13, "expected 13 candidates");
        assert_eq!(stats.point3d_count, 10);
        assert_eq!(stats.point2d_count, 10);
        assert_eq!(fixture.selected.len(), 10);

        let cameras =
            reader::parse_cameras(&fs::read(dir.join("cameras.bin")).expect("cameras.bin"));
        let images = reader::parse_images(&fs::read(dir.join("images.bin")).expect("images.bin"));
        let points =
            reader::parse_points3d(&fs::read(dir.join("points3D.bin")).expect("points3D.bin"));

        // --- cameras: PINHOLE, original resolution, rescaled params ---
        assert_eq!(cameras.len(), 2);
        let ratio0 = fixture.prepared.original_coords[0].resize_ratio(fixture.img_size);
        let ratio1 = fixture.prepared.original_coords[1].resize_ratio(fixture.img_size);
        assert_eq!(
            cameras[0],
            reader::TestCamera {
                camera_id: 1,
                model_id: 1,
                width: 1024,
                height: 768,
                params: vec![50.0 * ratio0, 60.0 * ratio0, 512.0, 384.0],
            }
        );
        assert_eq!(
            cameras[1],
            reader::TestCamera {
                camera_id: 2,
                model_id: 1,
                width: 500,
                height: 500,
                params: vec![50.0 * ratio1, 60.0 * ratio1, 250.0, 250.0],
            }
        );

        // --- images: names, qvec/tvec, 2D associations ---
        assert_eq!(images.len(), 2);
        for (frame, image) in images.iter().enumerate() {
            assert_eq!(image.image_id, frame as u32 + 1, "1-based image id");
            assert_eq!(image.camera_id, frame as u32 + 1);
        }
        assert_eq!(images[0].name, "frame_a.png");
        assert_eq!(images[1].name, "frame_b.png");

        // qvec is normalized and represents the camera-from-world rotation R
        // (COLMAP world-to-camera convention, as upstream passes extrinsics R
        // straight into pycolmap.Rotation3d).
        for (frame, expected_r) in [
            (
                0usize,
                [[0.0f64, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ),
            (
                1usize,
                [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ),
        ] {
            let q = &images[frame].qvec;
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!((norm - 1.0).abs() < 1e-9, "qvec not normalized");
            let q4 = [q[0], q[1], q[2], q[3]];
            let rebuilt = quat_to_mat_wxyz(&q4);
            for row in 0..3 {
                for col in 0..3 {
                    assert!(
                        (rebuilt[row][col] - expected_r[row][col]).abs() < 1e-9,
                        "qvec rotation mismatch at frame {frame}"
                    );
                }
            }
        }
        // tvec is the extrinsics translation (f32 source -> f64 storage).
        for (frame, expected_t) in [(0usize, [0.5f64, -0.3, 2.0]), (1usize, [0.25, 0.5, 1.0])] {
            for (axis, value) in images[frame].tvec.iter().enumerate() {
                assert!(
                    (value - expected_t[axis]).abs() < 1e-6,
                    "tvec mismatch at frame {frame} axis {axis}"
                );
            }
        }

        // 2D points per image, shifted to the original resolution.
        for (frame, image) in images.iter().enumerate() {
            let expected_ids: Vec<u64> = fixture
                .selected
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, idx)| idx / (4 * 4) == frame)
                .map(|(rank, _)| (rank + 1) as u64)
                .collect();
            assert_eq!(
                image.points2d.len(),
                expected_ids.len(),
                "frame {frame} point count"
            );
            let coords = &fixture.prepared.original_coords[frame];
            for (point, expected_id) in image.points2d.iter().zip(&expected_ids) {
                assert_eq!(point.point3d_id, *expected_id, "frame {frame} ids");
                // Reconstruct the source pixel from the stored point3D id.
                let rank = *expected_id as usize - 1;
                let idx = fixture.selected[rank];
                let x = (idx % 4) as f64;
                let y = (idx / 4 % 4) as f64;
                let (expected_x, expected_y) =
                    shift_point2d_to_original(x, y, coords, fixture.img_size);
                assert!((point.x - expected_x).abs() < 1e-9);
                assert!((point.y - expected_y).abs() < 1e-9);
            }
        }

        // --- points3D: ids, xyz, rgb, error, single-element tracks ---
        assert_eq!(points.len(), 10);
        let world_flat = fixture
            .output
            .world_points
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("world points vec");
        for (rank, point) in points.iter().enumerate() {
            assert_eq!(point.point3d_id, rank as u64 + 1, "1-based point3D ids");
            let idx = fixture.selected[rank];
            for (axis, value) in point.xyz.iter().enumerate() {
                assert!(
                    (value - f64::from(world_flat[idx * 3 + axis])).abs() < 1e-9,
                    "xyz mismatch at point {rank}"
                );
            }
            assert_eq!(point.error, 0.0);
            assert_eq!(point.track.len(), 1, "feedforward track length is 1");
            let frame = idx / (4 * 4);
            let element = &point.track[0];
            assert_eq!(element.image_id, frame as u32 + 1);
            // point2D_idx is the 0-based position inside that image's list.
            let position = images[frame as usize]
                .points2d
                .iter()
                .position(|p| p.point3d_id == point.point3d_id)
                .expect("point2D in image") as u32;
            assert_eq!(element.point2d_idx, position);
        }

        // --- points.ply sibling (`demo_colmap.py:249`) ---
        let ply = fs::read_to_string(dir.join("points.ply")).expect("points.ply");
        let mut ply_lines = ply.lines();
        for expected in [
            "ply",
            "format ascii 1.0",
            "element vertex 10",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ] {
            assert_eq!(ply_lines.next(), Some(expected), "points.ply header");
        }
        let ply_rows: Vec<&str> = ply_lines.collect();
        assert_eq!(ply_rows.len(), 10, "points.ply vertex rows");
        for (rank, row) in ply_rows.iter().enumerate() {
            let fields: Vec<&str> = row.split_whitespace().collect();
            assert_eq!(fields.len(), 9, "points.ply row fields");
            let point = &points[rank];
            for axis in 0..3 {
                let value: f64 = fields[axis].parse().expect("points.ply coordinate");
                assert!(
                    (value - point.xyz[axis]).abs() < 1e-6,
                    "points.ply xyz mismatch at rank {rank}"
                );
                let normal: f64 = fields[3 + axis].parse().expect("points.ply normal");
                assert_eq!(normal, 0.0, "points.ply normals stay zero");
            }
            for channel in 0..3 {
                let value: u8 = fields[6 + channel].parse().expect("points.ply color");
                assert_eq!(value, point.rgb[channel], "points.ply rgb mismatch");
            }
        }

        // Deterministic output: a second write is byte-identical.
        let dir2 = dir.with_extension("again");
        write_feedforward_colmap(
            &dir2,
            &fixture.output,
            &fixture.prepared,
            &image_names,
            &options,
        )
        .expect("write colmap again");
        for name in ["cameras.bin", "images.bin", "points3D.bin", "points.ply"] {
            assert_eq!(
                fs::read(dir.join(name)).expect("first write"),
                fs::read(dir2.join(name)).expect("second write"),
                "{name} not deterministic"
            );
        }

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&dir2);
    }

    #[test]
    fn vggt_colmap_confidence_gate_and_point_cap() {
        let fixture = fixture();
        let dir = std::env::temp_dir().join(format!(
            "lux3d-vggt-colmap-conf-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));

        // Additional world_points_conf gate (values are all 1.0): threshold
        // 2.0 removes every candidate.
        let stats = write_feedforward_colmap(
            &dir,
            &fixture.output,
            &fixture.prepared,
            &[],
            &VggtColmapOptions {
                conf_threshold: 0.5,
                max_points: 100_000,
                confidence: Some(2.0),
                point_source: VggtColmapPointSource::Depth,
                seed: None,
            },
        )
        .expect("write colmap with conf gate");
        assert_eq!(stats.confidence_candidates, 0);
        assert_eq!(stats.point3d_count, 0);
        assert_eq!(stats.point2d_count, 0);

        // Empty names fall back to `image_{i+1}`; max_points 0 writes an
        // empty reconstruction.
        let stats = write_feedforward_colmap(
            &dir,
            &fixture.output,
            &fixture.prepared,
            &[],
            &VggtColmapOptions {
                conf_threshold: 0.5,
                max_points: 0,
                confidence: None,
                point_source: VggtColmapPointSource::Depth,
                seed: None,
            },
        )
        .expect("write colmap with zero cap");
        assert_eq!(stats.point3d_count, 0);
        let images = reader::parse_images(&fs::read(dir.join("images.bin")).expect("images.bin"));
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].name, "image_1");
        assert_eq!(images[0].points2d.len(), 0);
        let points =
            reader::parse_points3d(&fs::read(dir.join("points3D.bin")).expect("points.bin"));
        assert!(points.is_empty());
        // The visualization cloud is written even when empty.
        let ply = fs::read_to_string(dir.join("points.ply")).expect("points.ply");
        assert!(ply.contains("element vertex 0"));

        let _ = fs::remove_dir_all(&dir);
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "lux3d-vggt-colmap-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ))
    }

    /// The default point source is the upstream depth unprojection
    /// (`unproject_depth_map_to_point_map`), not the point head.
    #[test]
    fn vggt_colmap_default_point_source_unprojects_depth() {
        assert_eq!(
            VggtColmapPointSource::Depth,
            VggtColmapOptions::default().point_source
        );
        assert_eq!(None, VggtColmapOptions::default().seed);

        let fixture = fixture();
        let dir = temp_dir("depth-source");
        let options = VggtColmapOptions {
            conf_threshold: 1.5,
            max_points: 100_000,
            confidence: None,
            point_source: VggtColmapPointSource::Depth,
            seed: None,
        };
        let stats =
            write_feedforward_colmap(&dir, &fixture.output, &fixture.prepared, &[], &options)
                .expect("write colmap depth source");
        assert_eq!(stats.confidence_candidates, 13);
        assert_eq!(stats.point3d_count, 13);

        let points =
            reader::parse_points3d(&fs::read(dir.join("points3D.bin")).expect("points3D.bin"));
        let depth_points = fixture
            .output
            .world_points_from_depth()
            .expect("depth unprojection")
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("depth points vec");
        let head_points = fixture
            .output
            .world_points
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("point-head vec");
        let mut differs_from_head = false;
        for (rank, point) in points.iter().enumerate() {
            let idx = fixture.candidates[rank];
            for axis in 0..3 {
                let expected = f64::from(depth_points[idx * 3 + axis]);
                assert!(
                    (point.xyz[axis] - expected).abs() < 1e-6,
                    "depth-source xyz mismatch at rank {rank}"
                );
                if (expected - f64::from(head_points[idx * 3 + axis])).abs() > 1e-6 {
                    differs_from_head = true;
                }
            }
        }
        assert!(
            differs_from_head,
            "fixture must distinguish depth unprojection from the point head"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Seeded limiting is reproducible for one seed and selects a different
    /// subset for another seed; both stay subsets of the confidence mask.
    #[test]
    fn vggt_colmap_seeded_limit_is_reproducible_and_seed_dependent() {
        let fixture = fixture();
        let write = |dir: &std::path::Path, seed: u32| {
            write_feedforward_colmap(
                dir,
                &fixture.output,
                &fixture.prepared,
                &[],
                &VggtColmapOptions {
                    conf_threshold: 1.5,
                    max_points: 5,
                    confidence: None,
                    point_source: VggtColmapPointSource::Pointmap,
                    seed: Some(seed),
                },
            )
            .expect("write seeded colmap")
        };

        let dir_a = temp_dir("seed7a");
        let dir_b = temp_dir("seed7b");
        let dir_c = temp_dir("seed11");
        write(&dir_a, 7);
        write(&dir_b, 7);
        write(&dir_c, 11);
        for name in ["cameras.bin", "images.bin", "points3D.bin", "points.ply"] {
            assert_eq!(
                fs::read(dir_a.join(name)).expect("seed 7 first"),
                fs::read(dir_b.join(name)).expect("seed 7 second"),
                "{name} must be reproducible for the same seed"
            );
        }

        let points_a =
            reader::parse_points3d(&fs::read(dir_a.join("points3D.bin")).expect("points3D.bin"));
        let points_c =
            reader::parse_points3d(&fs::read(dir_c.join("points3D.bin")).expect("points3D.bin"));
        assert_eq!(points_a.len(), 5);
        assert_eq!(points_c.len(), 5);

        let xyz_a: Vec<Vec<f64>> = points_a.iter().map(|point| point.xyz.clone()).collect();
        let xyz_c: Vec<Vec<f64>> = points_c.iter().map(|point| point.xyz.clone()).collect();
        assert_ne!(xyz_a, xyz_c, "different seeds must draw different subsets");

        // Every seeded point comes from the confidence-filtered candidates and
        // keeps the ascending candidate order.
        let world_flat = fixture
            .output
            .world_points
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("world points vec");
        let candidate_xyz: Vec<[f64; 3]> = fixture
            .candidates
            .iter()
            .map(|&idx| {
                [
                    f64::from(world_flat[idx * 3]),
                    f64::from(world_flat[idx * 3 + 1]),
                    f64::from(world_flat[idx * 3 + 2]),
                ]
            })
            .collect();
        let mut previous = None;
        for xyz in &xyz_a {
            let position = candidate_xyz
                .iter()
                .position(|candidate| {
                    (candidate[0] - xyz[0]).abs() < 1e-9
                        && (candidate[1] - xyz[1]).abs() < 1e-9
                        && (candidate[2] - xyz[2]).abs() < 1e-9
                })
                .expect("seeded point is a confidence candidate");
            if let Some(previous) = previous {
                assert!(position > previous, "seeded points keep candidate order");
            }
            previous = Some(position);
        }

        for dir in [&dir_a, &dir_b, &dir_c] {
            let _ = fs::remove_dir_all(dir);
        }
    }
}
