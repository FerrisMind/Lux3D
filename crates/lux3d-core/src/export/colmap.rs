//! COLMAP binary sparse-model export for the VGGT feedforward path.
//!
//! Mirrors the upstream `demo_colmap.py` feedforward mode
//! (`batch_np_matrix_to_pycolmap_wo_track` + `rename_colmap_recons_and_rescale_camera`):
//! a confidence mask `depth_conf >= conf_threshold`, at most 100000 3D points,
//! pixel-grid 2D associations with track length 1, one PINHOLE camera per
//! image, and camera/point rescaling back to the original image resolution
//! through the per-frame [`VggtOriginalCoords`]. Crop-mode rescaling is the
//! exact inverse of the `load_fn.py:162-175` recipe (per-axis
//! `width / 518` and `height / resized_height`, adding `crop_start_y` back on
//! the y axis); pad and square paths keep their existing mappings.
//!
//! It also implements the track-based reconstruction path
//! [`write_track_colmap`] (upstream `batch_np_matrix_to_pycolmap`,
//! `vggt/dependency/np_to_pycolmap.py:12-145`): per-frame tracks `[S, N, 2]`
//! plus a visibility mask (`vis > vis_thresh`, official default 0.2), the
//! predicted world points as the initial 3D positions (official feeds
//! `pred_points_3d`, sampled from the point map at each query pixel — there is
//! no triangulation solver), a reprojection-error mask (`diff <
//! max_reproj_error`, official default 8.0), the official per-frame inlier
//! count gate (`min_inlier_per_frame`, official default 64), the
//! `inlier_num >= 2` per-track gate, the `< 3000` point-coordinate gate, and
//! `SIMPLE_PINHOLE` cameras. No bundle adjustment is run (explicit non-goal —
//! the export is COLMAP-ready input for external BA).
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
//! * Track-mode reprojection errors are evaluated after the official
//!   square-mode intrinsic rescale (`intrinsic[:, :2, :] *= 1024 / 518`,
//!   `demo_colmap.py:168-169`): our tracks are predicted in model-input
//!   pixels, so they are multiplied by the same `load_size / model_size`
//!   factor before the threshold test. Crop/pad frames have no official
//!   COLMAP counterpart; their reprojection errors stay in model-input
//!   pixels.

use std::{fs, path::Path};

use candle_core::{Device, Tensor};

use super::point_cloud::{PointCloudView, write_ply};
use crate::{
    Result,
    error::Lux3dError,
    geometry::{rescale_intrinsics_to_original, shift_point2d_to_original},
    preprocess::{VggtOriginalCoords, VggtPreparedInputs, VggtPreprocessMode},
    runtime::{VggtAutoTrackResult, VggtInferenceOutput},
};

/// COLMAP camera model id for `PINHOLE` (params: fx, fy, cx, cy).
pub const COLMAP_CAMERA_MODEL_PINHOLE: i32 = 1;
/// COLMAP camera model id for `SIMPLE_PINHOLE` (params: f, cx, cy) — the
/// track/BA path model (`_build_pycolmap_intri`, `np_to_pycolmap.py:310-312`).
pub const COLMAP_CAMERA_MODEL_SIMPLE_PINHOLE: i32 = 2;
/// Upstream demo `conf_thres_value` default for the feedforward mode.
pub const DEFAULT_CONF_THRESHOLD: f32 = 5.0;
/// Upstream demo `max_points_for_colmap`.
pub const DEFAULT_MAX_POINTS: usize = 100_000;
/// Upstream `--vis_thresh` default (`demo_colmap.py:53`).
pub const DEFAULT_VIS_THRESHOLD: f32 = 0.2;
/// Upstream `--max_reproj_error` default (`demo_colmap.py:49`).
pub const DEFAULT_MAX_REPROJ_ERROR: f32 = 8.0;
/// Upstream builder `min_inlier_per_frame` default (`np_to_pycolmap.py:24`).
pub const DEFAULT_MIN_INLIER_PER_FRAME: usize = 64;
/// Upstream builder `max_points3D_val` default (`np_to_pycolmap.py:20`).
pub const DEFAULT_MAX_POINTS3D_VAL: f64 = 3000.0;

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

/// COLMAP export mode (upstream `demo_colmap.py` feedforward branch vs the
/// track/BA-input branch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VggtColmapMode {
    /// `batch_np_matrix_to_pycolmap_wo_track` (feedforward): PINHOLE cameras,
    /// one single-observation track per exported pixel.
    #[default]
    Feedforward,
    /// `batch_np_matrix_to_pycolmap` with the bundle-adjustment optimization
    /// left out (see [`write_track_colmap`]): SIMPLE_PINHOLE cameras and
    /// multi-frame tracks.
    Track,
}

/// Track-mode export options. Mirrors the official builder arguments
/// (`np_to_pycolmap.py:12-25`) plus the demo's visibility threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct VggtTrackColmapOptions {
    /// Strict track visibility gate `pred_vis_scores > vis_thresh`
    /// (`demo_colmap.py:170`, default 0.2).
    pub vis_thresh: f32,
    /// Reprojection-error gate `||project(points3d) - tracks|| <
    /// max_reproj_error` (`np_to_pycolmap.py:56-60`, demo default 8.0). When
    /// the preprocess mode is `square` both the projection and the tracks are
    /// in the loaded 1024-pixel space, matching the official
    /// `intrinsic[:, :2, :] *= 1024 / 518` step.
    pub max_reproj_error: f32,
    /// Share one camera object across all frames (`np_to_pycolmap.py:93-101`).
    /// Unused by the feedforward path, where upstream forces it to `False`.
    pub shared_camera: bool,
    /// Official `min_inlier_per_frame`: every frame must keep at least this
    /// many inlier observations or the official builder returns no
    /// reconstruction (`np_to_pycolmap.py:71-73`). Default 64.
    pub min_inlier_per_frame: usize,
    /// Official `max_points3D_val`: points with any coordinate `>=` this value
    /// stay in the reconstruction but receive no 2D observations
    /// (`np_to_pycolmap.py:120`). Default 3000.
    pub max_points3d_val: f64,
}

impl Default for VggtTrackColmapOptions {
    fn default() -> Self {
        Self {
            vis_thresh: DEFAULT_VIS_THRESHOLD,
            max_reproj_error: DEFAULT_MAX_REPROJ_ERROR,
            shared_camera: false,
            min_inlier_per_frame: DEFAULT_MIN_INLIER_PER_FRAME,
            max_points3d_val: DEFAULT_MAX_POINTS3D_VAL,
        }
    }
}

/// Flat track payload for [`write_track_colmap`]: row-major
/// `points3d [N, 3]`, `points_rgb [N, 3]`, `tracks [S, N, 2]` (pixel
/// coordinates in the model-input space, e.g. 518x518) and
/// `visibility [S, N]`.
#[derive(Debug, Clone, Copy)]
pub struct VggtTrackPack<'a> {
    /// Predicted initial 3D point per track (official `pred_points_3d`; no
    /// triangulation).
    pub points3d: &'a [f32],
    /// Per-track RGB color `[0, 255]` (official `pred_colors`).
    pub points_rgb: &'a [u8],
    /// Track pixel coordinates `[S, N, 2]` in model-input resolution.
    pub tracks: &'a [f32],
    /// Track visibility `[S, N]` in `[0, 1]`.
    pub visibility: &'a [f32],
}

/// Per-track initialization data sampled from the prediction point map at the
/// query pixels (official `pred_points_3d` / `pred_colors`,
/// `track_predict.py:171-179`).
#[derive(Debug, Clone, PartialEq)]
pub struct VggtTrackSeeds {
    /// Row-major `[N, 3]` world points.
    pub points3d: Vec<f32>,
    /// Row-major `[N, 3]` RGB colors.
    pub points_rgb: Vec<u8>,
}

/// Result counters for [`write_track_colmap`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VggtTrackColmapStats {
    pub camera_count: usize,
    pub image_count: usize,
    /// Points with at least two inlier observations (official `valid_mask`,
    /// `np_to_pycolmap.py:78-88`). Includes points skipped by the
    /// `max_points3D_val` observation gate, which keep an empty track.
    pub point3d_count: usize,
    /// Total 2D observations written across all images.
    pub point2d_count: usize,
    /// Tracks fed into the builder (`N`), including tracks dropped by the
    /// `inlier_num >= 2` gate and the `points.ply` payload.
    pub track_count: usize,
    /// Smallest per-frame inlier count after the visibility + reprojection
    /// masks (the value checked against `min_inlier_per_frame`).
    pub min_inliers_per_frame: usize,
}

/// Sample the per-track 3D initialization points and colors from the predicted
/// point map and RGB frames at the query pixels — exactly the official
/// `pred_points_3d` / `pred_colors` computation (`track_predict.py:171-179`):
/// round the model-resolution query pixel, index the query frame's point map,
/// and take the RGB frame value there.
///
/// `point_source` selects the official depth-unprojected map
/// ([`VggtColmapPointSource::Depth`], what `demo_colmap.py` passes to
/// `predict_tracks` at `:140,155-164`) or the point-head map.
pub fn sample_track_seeds(
    output: &VggtInferenceOutput,
    prepared: &VggtPreparedInputs,
    query_frames: &[usize],
    query_points: &[[f32; 2]],
    point_source: VggtColmapPointSource,
) -> Result<VggtTrackSeeds> {
    if query_frames.len() != query_points.len() {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT track seeds need one query frame per point, got {} frames for {} points",
                query_frames.len(),
                query_points.len()
            ),
        });
    }
    let (s, h, w) = resolve_grid_dims(output)?;
    let points = resolve_point_source(output, point_source, s, h, w)?;
    let rgb = rgb_frames_at_model_resolution(prepared, s, h, w)?;
    let mut seeds = VggtTrackSeeds {
        points3d: Vec::with_capacity(query_points.len() * 3),
        points_rgb: Vec::with_capacity(query_points.len() * 3),
    };
    for (query, points_xy) in query_frames.iter().zip(query_points) {
        if *query >= s {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "VGGT track query frame {query} is out of range for {s} prepared frames"
                ),
            });
        }
        if !points_xy[0].is_finite() || !points_xy[1].is_finite() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "VGGT track query point ({}, {}) is not finite",
                    points_xy[0], points_xy[1]
                ),
            });
        }
        // Official `query_points_scaled.round().long()`; the query points here
        // are already in the point-map resolution.
        let x = points_xy[0].round().clamp(0.0, (w - 1) as f32) as usize;
        let y = points_xy[1].round().clamp(0.0, (h - 1) as f32) as usize;
        let pixel = query * h * w + y * w + x;
        seeds
            .points3d
            .extend_from_slice(&points[pixel * 3..pixel * 3 + 3]);
        for channel in 0..3 {
            seeds
                .points_rgb
                .push(rgb_u8(rgb[(query * 3 + channel) * h * w + y * w + x]));
        }
    }
    Ok(seeds)
}

/// Write the COLMAP sparse model + `points.ply` for the track-based path,
/// faithful to `batch_np_matrix_to_pycolmap` (`np_to_pycolmap.py:12-145`)
/// without the bundle-adjustment step:
///
/// * visibility mask `visibility > vis_thresh`, reprojection mask
///   `||K [R|t] X / z - track|| < max_reproj_error`; both must hold. The
///   official `projected_points_2d[z <= 0] = 1e6` line runs after the mask is
///   computed and does not affect it (`np_to_pycolmap.py:58-60`), so the mask
///   depends only on the projected difference.
/// * per-frame inlier gate `min_inlier_per_frame` (the official builder
///   returns no reconstruction when a frame has fewer inliers, `:71-73`); this
///   function errors instead of writing a partial model.
/// * per-track gate `inlier_num >= 2` (`:78-88`).
/// * per-observation `max_points3d_val` gate `xyz < max_points3d_val`
///   (`:120`), applied per axis with strict `<`.
/// * 1-based camera/image/point ids, `image_{i+1}` naming fallback,
///   `SIMPLE_PINHOLE` cameras (`_build_pycolmap_intri`, `:310-312`) with
///   `f = (fx + fy) / 2`, rescaled to the original resolution like the
///   feedforward writer.
///
/// In `square` preprocess mode the model-space intrinsics and tracks are
/// multiplied by `load_size / model_size` (the official
/// `intrinsic[:, :2, :] *= 1024 / 518`) before the reprojection check, and the
/// reconstruction uses the loaded-square image size. Crop/pad frames keep
/// model-input pixels (no official COLMAP counterpart).
///
/// `points.ply` contains every input track seed/color (official line 249 uses
/// the unfiltered `points_3d`/`points_rgb` of the track branch).
pub fn write_track_colmap(
    dir: &Path,
    output: &VggtInferenceOutput,
    prepared: &VggtPreparedInputs,
    image_names: &[String],
    pack: &VggtTrackPack<'_>,
    options: &VggtTrackColmapOptions,
) -> Result<VggtTrackColmapStats> {
    let (s, h, w) = resolve_grid_dims(output)?;
    let coords = &prepared.original_coords;
    if coords.len() != s {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT track COLMAP export needs {s} per-frame original coords, found {}",
                coords.len()
            ),
        });
    }
    let model_size = f64::from(prepared.target_size.width.max(prepared.target_size.height));
    if model_size <= 0.0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT prepared target size must be non-zero",
        ));
    }

    // Flat track payload validation. `N` is derived from the `[S, N, 2]`
    // tracks and every other array is checked against it.
    let n = if s == 0 {
        0
    } else {
        pack.tracks.len() / (s * 2)
    };
    let expected_tracks = s * n * 2;
    let expected_visibility = s * n;
    if pack.tracks.len() != expected_tracks
        || pack.visibility.len() != expected_visibility
        || pack.points3d.len() != n * 3
        || pack.points_rgb.len() != n * 3
    {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT track payload shapes do not match: tracks={} (want {}), visibility={} (want {}), points3d={} (want {}), points_rgb={} (want {}) for {s} frames",
                pack.tracks.len(),
                expected_tracks,
                pack.visibility.len(),
                expected_visibility,
                pack.points3d.len(),
                n * 3,
                pack.points_rgb.len(),
                n * 3
            ),
        });
    }

    // Reconstruction pixel space: the official loaded 1024 square for square
    // preprocess (`intrinsic[:, :2, :] *= 1024 / 518`), model input otherwise.
    let space = track_recon_space(coords, model_size, w, h)?;

    let intrinsics =
        tensor_to_cpu_vec::<f64>(&squeeze_leading_batch_3d(&output.intrinsics)?, "intrinsics")?;
    if intrinsics.len() != s * 9 {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT intrinsics carry {} values, expected {} for {s} frames",
                intrinsics.len(),
                s * 9
            ),
        });
    }
    let extrinsics =
        tensor_to_cpu_vec::<f64>(&squeeze_leading_batch_3d(&output.extrinsics)?, "extrinsics")?;
    if extrinsics.len() != s * 12 {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT extrinsics carry {} values, expected {} for {s} frames",
                extrinsics.len(),
                s * 12
            ),
        });
    }

    // Per-frame intrinsics in reconstruction space (rows 0/1 scaled in square
    // mode) for the reprojection check.
    let mut k_recon = Vec::with_capacity(s);
    for frame in 0..s {
        let values = &intrinsics[frame * 9..frame * 9 + 9];
        let mut k = [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
            [values[6], values[7], values[8]],
        ];
        if space.loaded {
            for row in k.iter_mut().take(2) {
                for value in row.iter_mut() {
                    *value *= space.scale;
                }
            }
        }
        k_recon.push(k);
    }

    // Official masks: `vis > vis_thresh`, then AND the reprojection mask.
    let mut masks = vec![false; s * n];
    let mut inliers = vec![0_usize; n];
    let max_reproj_error = f64::from(options.max_reproj_error);
    for frame in 0..s {
        let k = &k_recon[frame];
        let extrinsic = &extrinsics[frame * 12..frame * 12 + 12];
        for point in 0..n {
            let flat = frame * n + point;
            let visible = pack.visibility[flat] > options.vis_thresh;
            let xyz = [
                f64::from(pack.points3d[point * 3]),
                f64::from(pack.points3d[point * 3 + 1]),
                f64::from(pack.points3d[point * 3 + 2]),
            ];
            let (projected_x, projected_y) = project_point(k, extrinsic, &xyz);
            let track_x = f64::from(pack.tracks[flat * 2]) * space.scale;
            let track_y = f64::from(pack.tracks[flat * 2 + 1]) * space.scale;
            let diff = ((projected_x - track_x).powi(2) + (projected_y - track_y).powi(2)).sqrt();
            if visible && diff < max_reproj_error {
                masks[flat] = true;
                inliers[point] += 1;
            }
        }
    }

    // Official per-frame inlier gate (`np_to_pycolmap.py:71-73`): the builder
    // refuses to build a reconstruction when any frame has fewer than
    // `min_inlier_per_frame` observations.
    let min_inliers_per_frame = (0..s)
        .map(|frame| {
            masks[frame * n..(frame + 1) * n]
                .iter()
                .filter(|&&keep| keep)
                .count()
        })
        .min()
        .unwrap_or(0);
    if min_inliers_per_frame < options.min_inlier_per_frame {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT track COLMAP export has a frame with only {min_inliers_per_frame} inliers (min_inlier_per_frame = {}); the official builder returns no reconstruction",
                options.min_inlier_per_frame
            ),
        });
    }

    // `valid_mask = masks.sum(0) >= 2`.
    let valid_idx: Vec<usize> = (0..n).filter(|&point| inliers[point] >= 2).collect();

    struct TrackPoint3D {
        point3d_id: u64,
        xyz: [f64; 3],
        rgb: [u8; 3],
        track: Vec<(u32, u32)>,
    }
    let mut points3d: Vec<TrackPoint3D> = valid_idx
        .iter()
        .enumerate()
        .map(|(rank, &point)| TrackPoint3D {
            point3d_id: rank as u64 + 1,
            xyz: [
                f64::from(pack.points3d[point * 3]),
                f64::from(pack.points3d[point * 3 + 1]),
                f64::from(pack.points3d[point * 3 + 2]),
            ],
            rgb: [
                pack.points_rgb[point * 3],
                pack.points_rgb[point * 3 + 1],
                pack.points_rgb[point * 3 + 2],
            ],
            track: Vec::new(),
        })
        .collect();

    // Image 2D associations + track elements, in official order: frames
    // ascending, then 1-based point3D ids ascending.
    let mut per_frame_points2d: Vec<Vec<(u64, f64, f64)>> = vec![Vec::new(); s];
    for frame in 0..s {
        for (rank, &point) in valid_idx.iter().enumerate() {
            let exported = &mut points3d[rank];
            // Official `max_points3D_val` observation gate (`:120`).
            if !exported
                .xyz
                .iter()
                .all(|value| *value < options.max_points3d_val)
            {
                continue;
            }
            if !masks[frame * n + point] {
                continue;
            }
            let flat = frame * n + point;
            let x_recon = f64::from(pack.tracks[flat * 2]) * space.scale;
            let y_recon = f64::from(pack.tracks[flat * 2 + 1]) * space.scale;
            let (x, y) = point2d_to_original(x_recon, y_recon, &coords[frame], model_size);
            let point2d_idx = per_frame_points2d[frame].len() as u32;
            per_frame_points2d[frame].push((exported.point3d_id, x, y));
            exported.track.push(((frame + 1) as u32, point2d_idx));
        }
    }
    let point2d_count: usize = per_frame_points2d.iter().map(Vec::len).sum();

    // Final camera parameters: `_build_pycolmap_intri` SIMPLE_PINHOLE on the
    // official rescaled intrinsics, then the upstream rename/rescale to the
    // original resolution (principal point = image center, `demo_colmap.py:
    // 254-292`).
    let intrinsics_rescaled =
        rescale_intrinsics_to_original(&output.intrinsics, coords, model_size)?;
    let camera_params: Vec<[f64; 3]> = (0..s)
        .map(|frame| {
            let k = &intrinsics_rescaled[frame];
            [(k[0][0] + k[1][1]) / 2.0, k[0][2], k[1][2]]
        })
        .collect();

    // cameras.bin: u64 count; per camera: u32 camera_id, i32 model_id,
    // u64 width, u64 height, 3 x f64 params (f, cx, cy) for SIMPLE_PINHOLE.
    let camera_frames: Vec<usize> = if s == 0 {
        Vec::new()
    } else if options.shared_camera {
        vec![0]
    } else {
        (0..s).collect()
    };
    let mut cameras_bin = Vec::new();
    put_u64(&mut cameras_bin, camera_frames.len() as u64);
    for (camera_index, &frame) in camera_frames.iter().enumerate() {
        put_u32(&mut cameras_bin, (camera_index + 1) as u32);
        put_i32(&mut cameras_bin, COLMAP_CAMERA_MODEL_SIMPLE_PINHOLE);
        put_u64(&mut cameras_bin, u64::from(coords[frame].width));
        put_u64(&mut cameras_bin, u64::from(coords[frame].height));
        for value in &camera_params[frame] {
            put_f64(&mut cameras_bin, *value);
        }
    }

    // images.bin: same layout as the feedforward writer; `camera_id` is 1 for
    // every image in shared-camera mode (`np_to_pycolmap.py:93-110`).
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
        put_u32(
            &mut images_bin,
            if options.shared_camera {
                1
            } else {
                frame as u32 + 1
            },
        );
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
            put_u64(&mut images_bin, *point3d_id);
        }
    }

    // points3D.bin: 1-based ids for the `valid_mask` points, with the official
    // per-track observation lists (possibly empty under the
    // `max_points3D_val` gate).
    let mut points3d_bin = Vec::new();
    put_u64(&mut points3d_bin, points3d.len() as u64);
    for point in &points3d {
        put_u64(&mut points3d_bin, point.point3d_id);
        for value in &point.xyz {
            put_f64(&mut points3d_bin, *value);
        }
        for channel in &point.rgb {
            put_u8(&mut points3d_bin, *channel);
        }
        put_f64(&mut points3d_bin, 0.0);
        put_u64(&mut points3d_bin, point.track.len() as u64);
        for (image_id, point2d_idx) in &point.track {
            put_u32(&mut points3d_bin, *image_id);
            put_u32(&mut points3d_bin, *point2d_idx);
        }
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
    // Official `trimesh.PointCloud(points_3d, colors=points_rgb)` next to the
    // sparse model (`demo_colmap.py:249`); the track branch passes the
    // unfiltered predicted track points.
    let ply_points: Vec<f32> = pack.points3d.to_vec();
    let ply_colors: Vec<u8> = pack.points_rgb.to_vec();
    let ply_path = dir.join("points.ply");
    let ply_view = PointCloudView {
        points: &ply_points,
        colors: &ply_colors,
    };
    write_ply(&ply_view, &ply_path)?;

    Ok(VggtTrackColmapStats {
        camera_count: camera_frames.len(),
        image_count: s,
        point3d_count: points3d.len(),
        point2d_count,
        track_count: n,
        min_inliers_per_frame,
    })
}

/// Convenience wrapper for the `--track-auto` pipeline: samples track seeds
/// from the prediction point map ([`sample_track_seeds`]) and exports the
/// [`VggtAutoTrackResult`] tracks.
pub fn write_track_colmap_from_auto(
    dir: &Path,
    output: &VggtInferenceOutput,
    prepared: &VggtPreparedInputs,
    image_names: &[String],
    auto: &VggtAutoTrackResult,
    point_source: VggtColmapPointSource,
    options: &VggtTrackColmapOptions,
) -> Result<VggtTrackColmapStats> {
    let seeds = sample_track_seeds(
        output,
        prepared,
        &auto.query_frames,
        &auto.query_points,
        point_source,
    )?;
    let tracks = tensor_to_cpu_vec::<f32>(&auto.track.coords, "track coords")?;
    let visibility = tensor_to_cpu_vec::<f32>(&auto.track.visibility, "track visibility")?;
    let pack = VggtTrackPack {
        points3d: &seeds.points3d,
        points_rgb: &seeds.points_rgb,
        tracks: &tracks,
        visibility: &visibility,
    };
    write_track_colmap(dir, output, prepared, image_names, &pack, options)
}

/// Reconstruction pixel space for the track path.
struct TrackReconSpace {
    /// Model-input pixel -> reconstruction pixel factor (loaded-square factor
    /// in square mode, 1 otherwise). Applied to intrinsics rows 0/1 and to the
    /// track coordinates.
    scale: f64,
    /// Reconstruction images are `load_size` square (loaded 1024 space).
    loaded: bool,
}

/// Square mode reproduces the official `intrinsic[:, :2, :] *= 1024 / 518`
/// step (`demo_colmap.py:168-169`); crop/pad frames stay in model pixels.
fn track_recon_space(
    coords: &[VggtOriginalCoords],
    model_size: f64,
    width: usize,
    height: usize,
) -> Result<TrackReconSpace> {
    let square_count = coords
        .iter()
        .filter(|coords| coords.mode == VggtPreprocessMode::Square)
        .count();
    if square_count == 0 {
        if width == 0 || height == 0 {
            return Err(Lux3dError::InvalidInput(
                "VGGT track COLMAP export needs a non-empty model grid",
            ));
        }
        return Ok(TrackReconSpace {
            scale: 1.0,
            loaded: false,
        });
    }
    if square_count != coords.len() {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: "VGGT track COLMAP export cannot mix square and crop/pad frames".to_string(),
        });
    }
    let first = coords[0]
        .square
        .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: "square preprocess mode requires the official square coordinate payload"
                .to_string(),
        })?;
    let scale = first.model_to_loaded_scale(model_size);
    if !(scale.is_finite() && scale > 0.0) {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "official square intrinsic scale load_size/model_size is not positive ({scale})"
            ),
        });
    }
    for coords in coords {
        let square = coords
            .square
            .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                message: "square preprocess mode requires the official square coordinate payload"
                    .to_string(),
            })?;
        if (square.model_to_loaded_scale(model_size) - scale).abs() > f64::EPSILON {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: "per-frame square preprocess payloads disagree on load_size/model_size"
                    .to_string(),
            });
        }
    }
    Ok(TrackReconSpace {
        scale,
        loaded: true,
    })
}

/// Project one world point through `[R|t]` and `K` with the official
/// `project_3D_points_np` + `img_from_cam_np` math
/// (`vggt/dependency/projection.py:12-102`): perspective divide, then the
/// full `K` product, with NaNs replaced by 0.0 (the official `nan_to_num`).
fn project_point(k: &[[f64; 3]; 3], extrinsic: &[f64], xyz: &[f64; 3]) -> (f64, f64) {
    let x = extrinsic[0] * xyz[0] + extrinsic[1] * xyz[1] + extrinsic[2] * xyz[2] + extrinsic[3];
    let y = extrinsic[4] * xyz[0] + extrinsic[5] * xyz[1] + extrinsic[6] * xyz[2] + extrinsic[7];
    let z = extrinsic[8] * xyz[0] + extrinsic[9] * xyz[1] + extrinsic[10] * xyz[2] + extrinsic[11];
    let u = x / z;
    let v = y / z;
    let mut projected_x = k[0][0] * u + k[0][1] * v + k[0][2];
    let mut projected_y = k[1][0] * u + k[1][1] * v + k[1][2];
    if projected_x.is_nan() {
        projected_x = 0.0;
    }
    if projected_y.is_nan() {
        projected_y = 0.0;
    }
    (projected_x, projected_y)
}

/// Map a reconstruction-space track pixel to the original image.
///
/// Square mode stores tracks in the loaded 1024 space, so the official
/// `(xy - top_left) * resize_ratio` shift applies directly
/// (`demo_colmap.py:280-285`); crop/pad use the stock model-space helper.
fn point2d_to_original(x: f64, y: f64, coords: &VggtOriginalCoords, model_size: f64) -> (f64, f64) {
    match (coords.mode, coords.square) {
        (VggtPreprocessMode::Square, Some(square)) => {
            let ratio = square.resize_ratio();
            ((x - square.x1) * ratio, (y - square.y1) * ratio)
        }
        _ => shift_point2d_to_original(x, y, coords, model_size),
    }
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

        #[derive(Debug, Clone, PartialEq)]
        pub struct TestTrackCamera {
            pub camera_id: u32,
            pub model_id: i32,
            pub width: u64,
            pub height: u64,
            /// SIMPLE_PINHOLE params: f, cx, cy.
            pub params: Vec<f64>,
        }

        /// Track-mode camera reader (`SIMPLE_PINHOLE` stores 3 params).
        pub fn parse_track_cameras(data: &[u8]) -> Vec<TestTrackCamera> {
            let mut cursor = Cursor::new(data);
            let count = cursor.u64() as usize;
            let cameras = (0..count)
                .map(|_| TestTrackCamera {
                    camera_id: cursor.u32(),
                    model_id: cursor.i32(),
                    width: cursor.u64(),
                    height: cursor.u64(),
                    params: cursor.f64_array(3),
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
                    crop_start_y: 0,
                    resized_height: 4,
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
                    crop_start_y: 0,
                    resized_height: 4,
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
        let (scale_x0, scale_y0) = fixture.prepared.original_coords[0].axis_scales_to_original();
        let (scale_x1, scale_y1) = fixture.prepared.original_coords[1].axis_scales_to_original();
        assert_eq!(
            cameras[0],
            reader::TestCamera {
                camera_id: 1,
                model_id: 1,
                width: 1024,
                height: 768,
                params: vec![50.0 * scale_x0, 60.0 * scale_y0, 512.0, 384.0],
            }
        );
        assert_eq!(
            cameras[1],
            reader::TestCamera {
                camera_id: 2,
                model_id: 1,
                width: 500,
                height: 500,
                params: vec![50.0 * scale_x1, 60.0 * scale_y1, 250.0, 250.0],
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

    // -----------------------------------------------------------------------
    // Track-mode export (`batch_np_matrix_to_pycolmap` semantics, no BA)
    // -----------------------------------------------------------------------

    struct TrackFixture {
        output: VggtInferenceOutput,
        prepared: VggtPreparedInputs,
    }

    /// 3-frame, 4x4 synthetic reconstruction: identity extrinsics and
    /// `fx = fy = 10, cx = cy = 2` intrinsics. Per-pixel world points are
    /// `[3 * pixel, 3 * pixel + 1, 3 * pixel + 2]`; RGB channel `c` at local
    /// pixel `i` is `(c * 100 + i % 100) / 255`.
    fn track_fixture() -> TrackFixture {
        let device = Device::Cpu;
        let (s, h, w) = (3usize, 4usize, 4usize);
        let pixels = s * h * w;
        let world_points = Tensor::from_vec(
            (0..pixels * 3).map(|i| i as f32).collect(),
            (1, s, h, w, 3),
            &device,
        )
        .expect("world points");
        let world_points_conf =
            Tensor::ones((1, s, h, w), DType::F32, &device).expect("world points conf");
        let depth = Tensor::ones((1, s, h, w, 1), DType::F32, &device).expect("depth");
        let depth_conf = Tensor::ones((1, s, h, w), DType::F32, &device).expect("depth conf");
        let pose_enc = Tensor::zeros((1, s, 9), DType::F32, &device).expect("pose enc");
        let mut extrinsic_values: Vec<f32> = Vec::with_capacity(s * 12);
        for _ in 0..s {
            extrinsic_values
                .extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        }
        let extrinsics =
            Tensor::from_vec(extrinsic_values, (1, s, 3, 4), &device).expect("extrinsics");
        let mut intrinsic_values: Vec<f32> = Vec::with_capacity(s * 9);
        for _ in 0..s {
            intrinsic_values.extend_from_slice(&[10.0, 0.0, 2.0, 0.0, 10.0, 2.0, 0.0, 0.0, 1.0]);
        }
        let intrinsics =
            Tensor::from_vec(intrinsic_values, (1, s, 3, 3), &device).expect("intrinsics");
        let rgb_frames = Tensor::from_vec(
            (0..s * 3 * h * w)
                .map(|i| {
                    let channel = (i / (h * w)) % 3;
                    let pixel = i % (h * w);
                    (channel * 100 + pixel % 100) as f32 / 255.0
                })
                .collect(),
            (s, 3, h, w),
            &device,
        )
        .expect("rgb frames");
        let normalized_frames =
            Tensor::zeros((1, s, 3, h, w), DType::F32, &device).expect("normalized frames");
        let export_mask = Tensor::ones((1, s, h, w), DType::U8, &device).expect("mask");

        let output = VggtInferenceOutput {
            world_points,
            world_points_conf,
            depth,
            depth_conf,
            pose_enc,
            pose_enc_list: vec![Tensor::zeros((1, s, 9), DType::F32, &device).expect("pose")],
            extrinsics,
            intrinsics,
            rgb_frames: rgb_frames.clone(),
            export_mask,
        };
        let prepared = VggtPreparedInputs {
            interval: 1,
            original_size: crate::contracts::SpatialSize::new(4, 4),
            target_size: crate::contracts::SpatialSize::new(4, 4),
            preprocess_mode: VggtPreprocessMode::Crop,
            rgb_frames,
            normalized_frames,
            original_coords: (0..s)
                .map(|_| VggtOriginalCoords {
                    width: 4,
                    height: 4,
                    offset_x: 0,
                    offset_y: 0,
                    content_width: 4,
                    content_height: 4,
                    crop_start_y: 0,
                    resized_height: 4,
                    mode: VggtPreprocessMode::Crop,
                    square: None,
                })
                .collect(),
        };
        TrackFixture { output, prepared }
    }

    /// Five synthetic tracks with the fixture's (2,2)/(3,2)/(2,3)/(2,2)/(4,4)
    /// projections:
    /// * t0 visible in all 3 frames (exact projection);
    /// * t1 visible in frames 0/1 with a 3-pixel offset (inside the 8.0 gate),
    ///   hidden in frame 2;
    /// * t2 visible only in frame 0 (one inlier -> invalid track);
    /// * t3 visible in all frames, 20-pixel offset in frame 2 (outside the
    ///   8.0 gate);
    /// * t4 invisible everywhere.
    fn track_payload() -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<f32>) {
        let (s, n) = (3usize, 5usize);
        let points3d = vec![
            0.0, 0.0, 1.0, // t0 -> (2, 2)
            0.1, 0.0, 1.0, // t1 -> (3, 2)
            0.0, 0.1, 1.0, // t2 -> (2, 3)
            0.0, 0.0, 2.0, // t3 -> (2, 2)
            0.2, 0.2, 1.0, // t4 -> (4, 4)
        ];
        let points_rgb: Vec<u8> = (0..n * 3).map(|i| (i * 7 % 251) as u8).collect();
        let base = [
            (2.0f32, 2.0f32),
            (3.0, 2.0),
            (2.0, 3.0),
            (2.0, 2.0),
            (4.0, 4.0),
        ];
        let mut tracks = vec![0.0f32; s * n * 2];
        let mut visibility = vec![0.0f32; s * n];
        for frame in 0..s {
            for point in 0..n {
                let (mut x, y) = base[point];
                if point == 1 {
                    x += 3.0;
                }
                if point == 3 && frame == 2 {
                    x += 20.0;
                }
                tracks[(frame * n + point) * 2] = x;
                tracks[(frame * n + point) * 2 + 1] = y;
                visibility[frame * n + point] = match point {
                    0 => 1.0,
                    1 => {
                        if frame == 2 {
                            0.0
                        } else {
                            1.0
                        }
                    }
                    2 => {
                        if frame == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    3 => 1.0,
                    _ => 0.0,
                };
            }
        }
        (points3d, points_rgb, tracks, visibility)
    }

    fn write_track_fixture(
        tag: &str,
        options: &VggtTrackColmapOptions,
        payload: &(Vec<f32>, Vec<u8>, Vec<f32>, Vec<f32>),
    ) -> (std::path::PathBuf, VggtTrackColmapStats) {
        let TrackFixture { output, prepared } = track_fixture();
        let dir = temp_dir(tag);
        let pack = VggtTrackPack {
            points3d: &payload.0,
            points_rgb: &payload.1,
            tracks: &payload.2,
            visibility: &payload.3,
        };
        let stats = write_track_colmap(&dir, &output, &prepared, &[], &pack, options)
            .expect("write track colmap");
        (dir, stats)
    }

    fn square_coords(
        x1: f64,
        y1: f64,
        width: u32,
        height: u32,
        load_size: u32,
        model_size: u32,
    ) -> VggtOriginalCoords {
        VggtOriginalCoords {
            width,
            height,
            offset_x: 0,
            offset_y: 0,
            content_width: width,
            content_height: height,
            crop_start_y: 0,
            resized_height: height,
            mode: VggtPreprocessMode::Square,
            square: Some(crate::preprocess::VggtSquareOriginalCoords {
                x1,
                y1,
                x2: x1 + f64::from(load_size),
                y2: y1 + f64::from(load_size),
                width,
                height,
                load_size,
                model_size,
            }),
        }
    }

    #[test]
    fn vggt_track_colmap_observations_and_reprojection_mask() {
        let payload = track_payload();
        let options = VggtTrackColmapOptions {
            vis_thresh: 0.2,
            max_reproj_error: 8.0,
            shared_camera: false,
            min_inlier_per_frame: 1,
            max_points3d_val: 3000.0,
        };
        let (dir, stats) = write_track_fixture("track-obs", &options, &payload);

        assert_eq!(stats.camera_count, 3);
        assert_eq!(stats.image_count, 3);
        assert_eq!(stats.track_count, 5);
        // t2 has a single inlier (< 2) and t4 none: two tracks are dropped.
        assert_eq!(stats.point3d_count, 3);
        // frame 0: t0,t1,t3; frame 1: t0,t1,t3; frame 2: t0 only (t3's 20 px
        // reprojection error fails the 8.0 gate, t1 is hidden).
        assert_eq!(stats.point2d_count, 7);
        assert_eq!(stats.min_inliers_per_frame, 1);

        let cameras =
            reader::parse_track_cameras(&fs::read(dir.join("cameras.bin")).expect("cameras.bin"));
        assert_eq!(cameras.len(), 3);
        for (frame, camera) in cameras.iter().enumerate() {
            assert_eq!(camera.camera_id, frame as u32 + 1);
            assert_eq!(
                camera.model_id, COLMAP_CAMERA_MODEL_SIMPLE_PINHOLE,
                "SIMPLE_PINHOLE model id"
            );
            assert_eq!(camera.width, 4);
            assert_eq!(camera.height, 4);
            assert_eq!(camera.params, vec![10.0, 2.0, 2.0]);
        }

        let images = reader::parse_images(&fs::read(dir.join("images.bin")).expect("images.bin"));
        assert_eq!(images.len(), 3);
        assert_eq!(images[0].points2d.len(), 3);
        assert_eq!(images[1].points2d.len(), 3);
        assert_eq!(images[2].points2d.len(), 1);
        assert_eq!(
            images[0]
                .points2d
                .iter()
                .map(|point| point.point3d_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            images[1]
                .points2d
                .iter()
                .map(|point| point.point3d_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(images[2].points2d[0].point3d_id, 1);
        // Masked-out tracks never appear in images.bin.
        assert!(images.iter().all(|image| {
            image
                .points2d
                .iter()
                .all(|point| point.point3d_id != 4 && point.point3d_id != 5)
        }));
        assert_eq!(images[0].name, "image_1");
        assert_eq!(images[0].camera_id, 1);

        let points =
            reader::parse_points3d(&fs::read(dir.join("points3D.bin")).expect("points3D.bin"));
        assert_eq!(points.len(), 3);
        let expected_xyz = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.0, 2.0]];
        let expected_tracks = [vec![1u32, 2, 3], vec![1, 2], vec![1, 2]];
        for (rank, point) in points.iter().enumerate() {
            assert_eq!(point.point3d_id, rank as u64 + 1);
            for axis in 0..3 {
                // f32 seed values widen into the f64 COLMAP storage.
                assert!((point.xyz[axis] - expected_xyz[rank][axis]).abs() < 1e-6);
            }
            assert_eq!(point.error, 0.0);
            let track_images: Vec<u32> =
                point.track.iter().map(|element| element.image_id).collect();
            assert_eq!(track_images, expected_tracks[rank], "point {rank} track");
            for element in &point.track {
                let image = &images[element.image_id as usize - 1];
                assert!(
                    element.point2d_idx < image.points2d.len() as u32,
                    "track points into its image's point2D list"
                );
                assert_eq!(
                    image.points2d[element.point2d_idx as usize].point3d_id, point.point3d_id,
                    "track element and image point2D agree"
                );
            }
            assert_eq!(
                point.rgb,
                [
                    payload.1[[0usize, 1, 3][rank] * 3],
                    payload.1[[0usize, 1, 3][rank] * 3 + 1],
                    payload.1[[0usize, 1, 3][rank] * 3 + 2],
                ],
                "colors come from the corresponding track seed"
            );
        }

        // Track seeds are all in points.ply (official line 249 uses the
        // unfiltered points_3d/points_rgb of the track branch).
        let ply = fs::read_to_string(dir.join("points.ply")).expect("points.ply");
        assert!(
            ply.contains("element vertex 5"),
            "all 5 seeds in points.ply"
        );

        // Raising the reprojection threshold admits t3's frame-2 observation.
        let (dir_loose, loose) = write_track_fixture(
            "track-loose",
            &VggtTrackColmapOptions {
                max_reproj_error: 25.0,
                ..options.clone()
            },
            &payload,
        );
        assert_eq!(loose.point3d_count, 3);
        assert_eq!(loose.point2d_count, 8);
        assert_eq!(loose.min_inliers_per_frame, 2);
        let loose_images =
            reader::parse_images(&fs::read(dir_loose.join("images.bin")).expect("images.bin"));
        assert_eq!(loose_images[2].points2d.len(), 2);
        let loose_points =
            reader::parse_points3d(&fs::read(dir_loose.join("points3D.bin")).expect("points"));
        assert_eq!(loose_points[2].track.len(), 3);

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&dir_loose);
    }

    #[test]
    fn vggt_track_colmap_official_gates() {
        let payload = track_payload();

        // Official `min_inlier_per_frame` gate: frame 2 has a single inlier.
        let TrackFixture { output, prepared } = track_fixture();
        let dir = temp_dir("track-min-gate");
        let pack = VggtTrackPack {
            points3d: &payload.0,
            points_rgb: &payload.1,
            tracks: &payload.2,
            visibility: &payload.3,
        };
        let error = write_track_colmap(
            &dir,
            &output,
            &prepared,
            &[],
            &pack,
            &VggtTrackColmapOptions {
                min_inlier_per_frame: 2,
                ..VggtTrackColmapOptions::default()
            },
        )
        .expect_err("official per-frame inlier gate must reject");
        let message = format!("{error}");
        assert!(
            message.contains("only 1 inliers") && message.contains("min_inlier_per_frame = 2"),
            "unexpected gate message: {message}"
        );
        let _ = fs::remove_dir_all(&dir);

        // Official `max_points3D_val` gate: a coordinate equal to 3000 is not
        // `< 3000`, so the point keeps its reconstruction entry but loses every
        // observation. Its tracks are moved to the (30002, 2) projection so
        // the reprojection mask itself still passes.
        let mut gated_payload = payload.clone();
        gated_payload.0[0] = 3000.0;
        for frame in 0..3 {
            gated_payload.2[frame * 5 * 2] = 30002.0;
            gated_payload.2[frame * 5 * 2 + 1] = 2.0;
        }
        let (dir, stats) = write_track_fixture(
            "track-xyz-gate",
            &VggtTrackColmapOptions {
                min_inlier_per_frame: 1,
                ..VggtTrackColmapOptions::default()
            },
            &gated_payload,
        );
        assert_eq!(stats.point3d_count, 3, "point stays in the reconstruction");
        assert_eq!(stats.point2d_count, 4, "all t0 observations are gated out");
        let points =
            reader::parse_points3d(&fs::read(dir.join("points3D.bin")).expect("points3D.bin"));
        assert!(points[0].track.is_empty(), "gated point has an empty track");
        let images = reader::parse_images(&fs::read(dir.join("images.bin")).expect("images.bin"));
        assert!(
            images
                .iter()
                .all(|image| image.points2d.iter().all(|point| point.point3d_id != 1))
        );
        let _ = fs::remove_dir_all(&dir);

        // Track-mode defaults match the official demo/builder arguments.
        let defaults = VggtTrackColmapOptions::default();
        assert_eq!(defaults.vis_thresh, 0.2);
        assert_eq!(defaults.max_reproj_error, 8.0);
        assert!(!defaults.shared_camera);
        assert_eq!(defaults.min_inlier_per_frame, 64);
        assert_eq!(defaults.max_points3d_val, 3000.0);
        assert_eq!(VggtColmapMode::default(), VggtColmapMode::Feedforward);
    }

    #[test]
    fn vggt_track_colmap_shared_camera() {
        let payload = track_payload();
        let (dir, stats) = write_track_fixture(
            "track-shared",
            &VggtTrackColmapOptions {
                shared_camera: true,
                min_inlier_per_frame: 1,
                ..VggtTrackColmapOptions::default()
            },
            &payload,
        );
        assert_eq!(stats.camera_count, 1);
        let cameras =
            reader::parse_track_cameras(&fs::read(dir.join("cameras.bin")).expect("cameras.bin"));
        assert_eq!(cameras.len(), 1);
        assert_eq!(cameras[0].camera_id, 1);
        assert_eq!(cameras[0].model_id, COLMAP_CAMERA_MODEL_SIMPLE_PINHOLE);
        let images = reader::parse_images(&fs::read(dir.join("images.bin")).expect("images.bin"));
        assert_eq!(images.len(), 3);
        assert!(images.iter().all(|image| image.camera_id == 1));
        let _ = fs::remove_dir_all(&dir);
    }

    /// Square preprocess mode reproduces the official
    /// `intrinsic[:, :2, :] *= 1024 / 518` rescale (here 1024 / 4 = 256 for
    /// the 4-pixel synthetic model grid) for both the reprojection check and
    /// the written SIMPLE_PINHOLE camera, and the loaded-space point2D shift.
    #[test]
    fn vggt_track_colmap_square_intrinsics_rescale() {
        let TrackFixture {
            output,
            mut prepared,
        } = track_fixture();
        prepared.preprocess_mode = VggtPreprocessMode::Square;
        prepared.original_coords = (0..3)
            .map(|_| square_coords(0.0, 0.0, 1024, 1024, 1024, 4))
            .collect();

        // Two tracks with exact model-space projections: t0 -> (2,2),
        // t1 -> (3,2). The writer scales them to the loaded 1024 space.
        let points3d = vec![0.0f32, 0.0, 1.0, 0.1, 0.0, 1.0];
        let points_rgb = vec![1u8, 2, 3, 4, 5, 6];
        let mut tracks = vec![0.0f32; 3 * 2 * 2];
        let visibility = vec![1.0f32; 3 * 2];
        for frame in 0..3 {
            tracks[(frame * 2) * 2] = 2.0;
            tracks[(frame * 2) * 2 + 1] = 2.0;
            tracks[(frame * 2 + 1) * 2] = 3.0;
            tracks[(frame * 2 + 1) * 2 + 1] = 2.0;
        }
        let pack = VggtTrackPack {
            points3d: &points3d,
            points_rgb: &points_rgb,
            tracks: &tracks,
            visibility: &visibility,
        };
        let dir = temp_dir("track-square");
        let stats = write_track_colmap(
            &dir,
            &output,
            &prepared,
            &[],
            &pack,
            &VggtTrackColmapOptions {
                min_inlier_per_frame: 1,
                ..VggtTrackColmapOptions::default()
            },
        )
        .expect("write square track colmap");
        assert_eq!(stats.point3d_count, 2);
        assert_eq!(stats.point2d_count, 6);

        let cameras =
            reader::parse_track_cameras(&fs::read(dir.join("cameras.bin")).expect("cameras.bin"));
        assert_eq!(cameras.len(), 3);
        assert_eq!(cameras[0].width, 1024);
        assert_eq!(cameras[0].height, 1024);
        // 10 * (1024 / 4) = 2560; principal point = original center.
        assert_eq!(cameras[0].params, vec![2560.0, 512.0, 512.0]);

        let images = reader::parse_images(&fs::read(dir.join("images.bin")).expect("images.bin"));
        assert_eq!(images[0].points2d[0].x, 512.0);
        assert_eq!(images[0].points2d[0].y, 512.0);
        assert_eq!(images[0].points2d[1].x, 768.0);
        assert_eq!(images[0].points2d[1].y, 512.0);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Crop-mode portrait track export: the synthetic 4x4 grid stands in for a
    /// 518-pixel model window over a 768x1024 original with a 1-row crop start
    /// (pre-crop resized height 6). Model pixel (2, 2) must map to original
    /// (384, 512) with the corrected inverse
    /// `x * 768/4`, `(y - 0 + 1) * 1024/6`; the old uniform-ratio /
    /// subtract-the-crop formula gives a different, wrong point. The
    /// SIMPLE_PINHOLE parameters use the same per-axis scales.
    #[test]
    fn vggt_track_colmap_crop_portrait_maps_to_original_resolution() {
        let TrackFixture {
            output,
            mut prepared,
        } = track_fixture();
        prepared.original_coords = (0..3)
            .map(|_| VggtOriginalCoords {
                width: 768,
                height: 1024,
                offset_x: 0,
                offset_y: 0,
                content_width: 4,
                content_height: 4,
                crop_start_y: 1,
                resized_height: 6,
                mode: VggtPreprocessMode::Crop,
                square: None,
            })
            .collect();

        let points3d = vec![0.0f32, 0.0, 1.0, 0.1, 0.0, 1.0];
        let points_rgb = vec![1u8, 2, 3, 4, 5, 6];
        let mut tracks = vec![0.0f32; 3 * 2 * 2];
        let visibility = vec![1.0f32; 3 * 2];
        for frame in 0..3 {
            tracks[(frame * 2) * 2] = 2.0;
            tracks[(frame * 2) * 2 + 1] = 2.0;
            tracks[(frame * 2 + 1) * 2] = 3.0;
            tracks[(frame * 2 + 1) * 2 + 1] = 2.0;
        }
        let pack = VggtTrackPack {
            points3d: &points3d,
            points_rgb: &points_rgb,
            tracks: &tracks,
            visibility: &visibility,
        };
        let dir = temp_dir("track-crop-portrait");
        let stats = write_track_colmap(
            &dir,
            &output,
            &prepared,
            &[],
            &pack,
            &VggtTrackColmapOptions {
                min_inlier_per_frame: 1,
                ..VggtTrackColmapOptions::default()
            },
        )
        .expect("write crop portrait track colmap");
        assert_eq!(stats.point3d_count, 2);
        assert_eq!(stats.point2d_count, 6);

        let cameras =
            reader::parse_track_cameras(&fs::read(dir.join("cameras.bin")).expect("cameras.bin"));
        assert_eq!(cameras.len(), 3);
        // fx = 10 * 768/4 = 1920, fy = 10 * 1024/6, f = (fx + fy) / 2.
        let fx = 10.0 * 768.0 / 4.0;
        let fy = 10.0 * 1024.0 / 6.0;
        assert!((cameras[0].params[0] - (fx + fy) / 2.0).abs() < 1e-9);
        assert!((cameras[0].params[1] - 384.0).abs() < 1e-9);
        assert!((cameras[0].params[2] - 512.0).abs() < 1e-9);
        assert_eq!(cameras[0].width, 768);
        assert_eq!(cameras[0].height, 1024);

        let images = reader::parse_images(&fs::read(dir.join("images.bin")).expect("images.bin"));
        assert_eq!(images[0].points2d[0].x, 384.0);
        assert_eq!(images[0].points2d[0].y, 512.0);
        assert_eq!(images[0].points2d[1].x, 576.0);
        assert_eq!(images[0].points2d[1].y, 512.0);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Official `track_predict.py:171-179`: per-track 3D points/colors are
    /// sampled from the prediction point map and RGB frames at the rounded
    /// query pixel.
    #[test]
    fn vggt_track_seeds_sample_query_pixels() {
        let TrackFixture { output, prepared } = track_fixture();
        let auto = VggtAutoTrackResult {
            track: crate::runtime::VggtTrackResult {
                coords: Tensor::zeros((3, 3, 2), DType::F32, &Device::Cpu).expect("coords"),
                visibility: Tensor::ones((3, 3), DType::F32, &Device::Cpu).expect("visibility"),
                confidence: None,
            },
            query_frames: vec![0, 1, 2],
            query_points: vec![[1.4, 2.0], [0.0, 0.0], [3.0, 3.0]],
            ranked_query_frames: vec![0, 1, 2],
            queries_per_frame: vec![1, 1, 1],
            pruned_per_frame: vec![0, 0, 0],
            reorder_indices: vec![vec![0, 1, 2]; 3],
        };
        let seeds = sample_track_seeds(
            &output,
            &prepared,
            &auto.query_frames,
            &auto.query_points,
            VggtColmapPointSource::Pointmap,
        )
        .expect("sample seeds");
        // pixel = 3 * (frame * h * w + y * w + x)
        assert_eq!(
            seeds.points3d,
            vec![27.0, 28.0, 29.0, 48.0, 49.0, 50.0, 141.0, 142.0, 143.0]
        );
        // rgb channel c at local pixel i is (c * 100 + i % 100) / 255.
        assert_eq!(
            seeds.points_rgb,
            vec![9, 109, 209, 0, 100, 200, 15, 115, 215]
        );

        // Depth-source seeds follow `world_points_from_depth` (identity
        // extrinsics, depth 1.0, fx=fy=10, cx=cy=2): the fixture's
        // `world_points` differ from that map, so the two sources must differ.
        let depth_seeds = sample_track_seeds(
            &output,
            &prepared,
            &auto.query_frames,
            &auto.query_points,
            VggtColmapPointSource::Depth,
        )
        .expect("sample depth seeds");
        assert_ne!(depth_seeds.points3d, seeds.points3d);
    }

    /// End-to-end core run with the local VGGT-1B canonical weights:
    /// `run_with_auto_tracks` (stage-A queries + track head) then
    /// `write_track_colmap_from_auto`. This is the exact pipeline the CLI
    /// `--track-auto --colmap-mode track` wiring calls. The source defaults to
    /// the official reference `examples/llff_fern/images` multi-view sequence
    /// when present (the repository `test-assets/pi3-frames` fixture is three
    /// flat-color frames with no parallax), overridable with
    /// `LUX3D_VG5_E2E_SOURCE`. Ignored by default because it loads ~1.2 GB of
    /// weights; run with `cargo test -p lux3d-core [--features cuda] --lib
    /// vggt_track_colmap_e2e -- --ignored --nocapture`.
    #[test]
    #[ignore = "loads VGGT-1B weights; run explicitly with --ignored"]
    fn vggt_track_colmap_e2e_local_weights() {
        use crate::{
            runtime::{VggtAutoQueryOptions, VggtPipeline, VggtRunOptions},
            weights::ModelAssetOptions,
        };

        let model_dir = std::env::var_os("LUX3D_VGGT_MODEL_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("G:/jobs/candle-based-projects/3d/canonical-weights/vggt")
            });
        if !model_dir.join("model.safetensors").is_file() {
            eprintln!(
                "skipping VG5 e2e: no VGGT canonical weights at {}",
                model_dir.display()
            );
            return;
        }
        // `cargo test` runs with the package root as the working directory.
        // The repository fixture has only three flat-color frames with no
        // geometric parallax (every track is inlier only on its own query
        // frame), so the official reference multi-view examples are preferred
        // when they sit beside the workspace; `LUX3D_VG5_E2E_SOURCE` overrides
        // both.
        let source = std::env::var_os("LUX3D_VG5_E2E_SOURCE")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                [
                    "../../../candle-3d-refs/vggt/examples/llff_fern/images",
                    "test-assets/pi3-frames",
                    "../../test-assets/pi3-frames",
                ]
                .into_iter()
                .map(std::path::PathBuf::from)
                .find(|path| path.is_dir())
            })
            .unwrap_or_else(|| std::path::PathBuf::from("test-assets/pi3-frames"));
        if !source.is_dir() {
            eprintln!("skipping VG5 e2e: fixture {} is missing", source.display());
            return;
        }
        fn e2e_device() -> Device {
            #[cfg(feature = "cuda")]
            {
                if let Ok(device) = Device::new_cuda(0) {
                    return device;
                }
            }
            Device::Cpu
        }

        let device = e2e_device();
        eprintln!("[vg5-e2e] device={device:?} model={}", model_dir.display());
        let pipeline = VggtPipeline::load_with_options(
            ModelAssetOptions {
                canonical_dir: Some(model_dir),
                cache_dir: None,
            },
            VggtRunOptions::default(),
        )
        .expect("load VGGT pipeline");
        let prepared = pipeline
            .prepare_inputs_from_path_with_mode(&source, None, VggtPreprocessMode::Square, &device)
            .expect("prepare square inputs");
        let (inference, auto) = pipeline
            .run_with_auto_tracks(
                &prepared,
                &VggtAutoQueryOptions {
                    query_frame_num: 7,
                    max_query_pts: 96,
                    track_iters: 4,
                    dino_weights: None,
                },
            )
            .expect("auto tracking");
        eprintln!(
            "[vg5-e2e] query frames={:?} points={:?} pruned={:?}",
            auto.ranked_query_frames, auto.queries_per_frame, auto.pruned_per_frame
        );
        let dir = temp_dir("track-e2e");
        // Diagnostics: per-frame visibility / reprojection-mask counts and
        // error quantiles, using the same math as `write_track_colmap`.
        {
            let seeds = sample_track_seeds(
                &inference,
                &prepared,
                &auto.query_frames,
                &auto.query_points,
                VggtColmapPointSource::Depth,
            )
            .expect("sample e2e seeds");
            let coords_flat =
                tensor_to_cpu_vec::<f32>(&auto.track.coords, "track coords").expect("coords");
            let vis_flat = tensor_to_cpu_vec::<f32>(&auto.track.visibility, "track visibility")
                .expect("visibility");
            let (frames, height, width) = resolve_grid_dims(&inference).expect("grid dims");
            let tracks = auto.query_points.len();
            let model_size = f64::from(prepared.target_size.width.max(prepared.target_size.height));
            let space = track_recon_space(&prepared.original_coords, model_size, width, height)
                .expect("recon space");
            let intri = tensor_to_cpu_vec::<f64>(
                &squeeze_leading_batch_3d(&inference.intrinsics).expect("squeeze intrinsics"),
                "intrinsics",
            )
            .expect("intrinsics");
            let extri = tensor_to_cpu_vec::<f64>(
                &squeeze_leading_batch_3d(&inference.extrinsics).expect("squeeze extrinsics"),
                "extrinsics",
            )
            .expect("extrinsics");
            for frame in 0..frames {
                let mut k = [
                    [intri[frame * 9], intri[frame * 9 + 1], intri[frame * 9 + 2]],
                    [
                        intri[frame * 9 + 3],
                        intri[frame * 9 + 4],
                        intri[frame * 9 + 5],
                    ],
                    [
                        intri[frame * 9 + 6],
                        intri[frame * 9 + 7],
                        intri[frame * 9 + 8],
                    ],
                ];
                if space.loaded {
                    for row in k.iter_mut().take(2) {
                        for value in row.iter_mut() {
                            *value *= space.scale;
                        }
                    }
                }
                let mut visible = 0;
                let mut reprojected = 0;
                let mut both = 0;
                let mut errors = Vec::with_capacity(tracks);
                for point in 0..tracks {
                    let flat = frame * tracks + point;
                    let xyz = [
                        f64::from(seeds.points3d[point * 3]),
                        f64::from(seeds.points3d[point * 3 + 1]),
                        f64::from(seeds.points3d[point * 3 + 2]),
                    ];
                    let (px, py) = project_point(&k, &extri[frame * 12..frame * 12 + 12], &xyz);
                    let tx = f64::from(coords_flat[flat * 2]) * space.scale;
                    let ty = f64::from(coords_flat[flat * 2 + 1]) * space.scale;
                    let error = ((px - tx).powi(2) + (py - ty).powi(2)).sqrt();
                    let is_visible = vis_flat[flat] > 0.2;
                    let in_reproj = error < 8.0;
                    visible += usize::from(is_visible);
                    reprojected += usize::from(in_reproj);
                    both += usize::from(is_visible && in_reproj);
                    if error.is_finite() {
                        errors.push(error);
                    }
                }
                errors.sort_by(|a, b| a.partial_cmp(b).expect("finite errors"));
                let quantile = |q: f64| {
                    errors
                        .get(((errors.len().saturating_sub(1)) as f64 * q) as usize)
                        .copied()
                        .unwrap_or(f64::NAN)
                };
                eprintln!(
                    "[vg5-e2e] frame {frame}: visible={visible} reproj<8={reprojected} both={both} err p50={:.3} p90={:.3} p99={:.3}",
                    quantile(0.5),
                    quantile(0.9),
                    quantile(0.99)
                );
            }
        }
        let stats = write_track_colmap_from_auto(
            &dir,
            &inference,
            &prepared,
            &[],
            &auto,
            VggtColmapPointSource::Depth,
            &VggtTrackColmapOptions {
                // The stage-A tracker without stage B/C fine tracking is a
                // coarse matcher, so this structural e2e lowers the official
                // per-frame gate (default 64, enforced by
                // `VggtTrackColmapOptions::default` and unit tested); the
                // per-track `>= 2 inliers` gate still applies.
                min_inlier_per_frame: 1,
                ..VggtTrackColmapOptions::default()
            },
        )
        .expect("export track colmap");
        eprintln!(
            "[vg5-e2e] tracks={} cameras={} images={} points3d={} point2d={} min_inliers={}",
            stats.track_count,
            stats.camera_count,
            stats.image_count,
            stats.point3d_count,
            stats.point2d_count,
            stats.min_inliers_per_frame
        );
        let cameras =
            reader::parse_track_cameras(&fs::read(dir.join("cameras.bin")).expect("cameras.bin"));
        let images = reader::parse_images(&fs::read(dir.join("images.bin")).expect("images.bin"));
        let points =
            reader::parse_points3d(&fs::read(dir.join("points3D.bin")).expect("points3D.bin"));
        assert_eq!(cameras.len(), prepared.original_coords.len());
        assert_eq!(images.len(), prepared.original_coords.len());
        assert_eq!(points.len(), stats.point3d_count);
        assert!(points.iter().any(|point| point.track.len() >= 2));
        assert!(
            cameras
                .iter()
                .all(|camera| camera.model_id == COLMAP_CAMERA_MODEL_SIMPLE_PINHOLE)
        );
        eprintln!("[vg5-e2e] sparse model at {}", dir.display());
        // Leave the artifact in place for inspection; it is under the OS temp
        // directory, not the repository.
    }
}
