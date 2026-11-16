use candle_core::{D, DType, Device, IndexOp, Tensor};

use crate::{
    Result,
    contracts::{GeometryStage, ModelFamily, VggtPointCloud, VggtSceneCode},
    error::Lux3dError,
    preprocess::{VggtOriginalCoords, VggtPreprocessMode},
    runtime::VggtInferenceOutput,
};

#[derive(Debug, Clone, PartialEq)]
pub struct VggtPointCloudCpu {
    pub points: Vec<f32>,
    pub colors: Vec<u8>,
    pub vertex_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VggtGeometryStage;

impl GeometryStage<VggtSceneCode, VggtPointCloud> for VggtGeometryStage {
    fn assemble(&self, scene: &VggtSceneCode) -> Result<VggtPointCloud> {
        let dims = &scene.world_points.dims;
        let (last, leading) = dims.split_last().ok_or_else(|| {
            Lux3dError::InvalidInput(
                "VGGT world_points contract must declare at least one dimension",
            )
        })?;
        if *last != 3 {
            return Err(Lux3dError::InvalidInput(
                "VGGT world_points contract must end with an xyz dimension of 3",
            ));
        }
        let vertex_count = leading.iter().product::<usize>();

        Ok(VggtPointCloud {
            family: ModelFamily::Vggt,
            points: crate::contracts::TensorContract::new("vggt.points", vec![vertex_count, 3]),
            confidence: scene.world_points_conf.clone(),
            assembly_path: "world_points = point_head(aggregated_tokens)",
            supports_depth_unprojection: true,
        })
    }
}

impl VggtGeometryStage {
    /// Filter `world_points` by the strict `world_points_conf > conf_threshold`
    /// mask and attach colors from the preprocessed RGB frames, matching the
    /// VGGT demo export (`world_points_conf > 1.0` by default).
    pub fn assemble_cpu(
        &self,
        output: &VggtInferenceOutput,
        conf_threshold: f32,
    ) -> Result<VggtPointCloudCpu> {
        // Strict `>`: a confidence exactly equal to the threshold is excluded.
        let mask = output
            .world_points_conf
            .gt(conf_threshold)
            .map_err(|_| Lux3dError::InvalidInput("failed to threshold VGGT world_points_conf"))?
            .to_dtype(DType::U8)
            .map_err(|_| Lux3dError::InvalidInput("failed to cast VGGT export mask to u8"))?
            .to_device(&Device::Cpu)
            .map_err(|_| Lux3dError::InvalidInput("failed to move VGGT export mask to cpu"))?
            .flatten_all()
            .map_err(|_| Lux3dError::InvalidInput("failed to flatten VGGT export mask"))?
            .to_vec1::<u8>()
            .map_err(|_| Lux3dError::InvalidInput("failed to read VGGT export mask values"))?;
        let points = output
            .world_points
            .to_device(&Device::Cpu)
            .map_err(|_| Lux3dError::InvalidInput("failed to move VGGT world points to cpu"))?
            .flatten_all()
            .map_err(|_| Lux3dError::InvalidInput("failed to flatten VGGT world points"))?
            .to_vec1::<f32>()
            .map_err(|_| Lux3dError::InvalidInput("failed to read VGGT world point values"))?;
        let rgb = output
            .rgb_frames
            .to_device(&Device::Cpu)
            .map_err(|_| Lux3dError::InvalidInput("failed to move VGGT rgb frames to cpu"))?
            .permute((0, 2, 3, 1))
            .map_err(|_| Lux3dError::InvalidInput("failed to permute VGGT rgb frames"))?
            .flatten_all()
            .map_err(|_| Lux3dError::InvalidInput("failed to flatten VGGT rgb frames"))?
            .to_vec1::<f32>()
            .map_err(|_| Lux3dError::InvalidInput("failed to read VGGT rgb values"))?;

        // `[1,S,H,W,3]` and `[S,H,W,3]` share the same row-major flat order.
        if points.len() != mask.len() * 3 || rgb.len() != mask.len() * 3 {
            return Err(Lux3dError::InvalidInput(
                "VGGT world points / rgb frames do not match export mask element count",
            ));
        }

        let vertex_count = mask.iter().filter(|&&value| value != 0).count();
        let mut filtered_points = Vec::with_capacity(vertex_count * 3);
        let mut filtered_colors = Vec::with_capacity(vertex_count * 3);
        for (idx, &keep) in mask.iter().enumerate() {
            if keep == 0 {
                continue;
            }
            let offset = idx * 3;
            filtered_points.extend_from_slice(&points[offset..offset + 3]);
            filtered_colors.push((rgb[offset].clamp(0.0, 1.0) * 255.0).round() as u8);
            filtered_colors.push((rgb[offset + 1].clamp(0.0, 1.0) * 255.0).round() as u8);
            filtered_colors.push((rgb[offset + 2].clamp(0.0, 1.0) * 255.0).round() as u8);
        }

        Ok(VggtPointCloudCpu {
            points: filtered_points,
            colors: filtered_colors,
            vertex_count,
        })
    }
}

fn tensor_err(context: &str, source: candle_core::Error) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: format!("{context}: {source}"),
    }
}

/// Drop a leading singleton batch dim (`[1, S, ...]` -> `[S, ...]`).
fn squeeze_leading_batch(tensor: &Tensor) -> candle_core::Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() >= 4 && dims[0] == 1 {
        tensor.squeeze(0)
    } else {
        Ok(tensor.clone())
    }
}

/// Reference `rename_colmap_recons_and_rescale_camera`: rescale per-frame
/// intrinsics from the reconstruction (prepared-frame) pixel space back to the
/// original image resolution.
///
/// * Crop mode (upstream, bit-identical): `resize_ratio =
///   max(real_w, real_h) / img_size`, all nine intrinsic entries are
///   multiplied by the ratio, the principal point is replaced by the original
///   image centre `(real_w / 2, real_h / 2)` and the homogeneous row is
///   restored to `[0, 0, 1]`. `img_size` is the reconstruction resolution (the
///   maximum prepared width/height).
/// * Pad mode (exact per-axis): row 0 is scaled by `real_w / content_w` and
///   row 1 by `real_h / content_h` (the actual resize factors — pad mode
///   preserves the aspect ratio, so they differ only by the `/14` rounding of
///   the shorter side); the principal point and homogeneous row follow the
///   reference replacement.
pub fn rescale_intrinsics_to_original(
    intrinsics: &Tensor,
    original_coords: &[VggtOriginalCoords],
    img_size: f64,
) -> Result<Vec<[[f64; 3]; 3]>> {
    if img_size <= 0.0 || !img_size.is_finite() {
        return Err(Lux3dError::InvalidInput(
            "reconstruction image size must be positive and finite",
        ));
    }
    rescale_intrinsics_to_original_inner(intrinsics, original_coords, img_size)
        .map_err(|source| tensor_err("VGGT intrinsics rescale failed", source))
}

fn rescale_intrinsics_to_original_inner(
    intrinsics: &Tensor,
    original_coords: &[VggtOriginalCoords],
    img_size: f64,
) -> candle_core::Result<Vec<[[f64; 3]; 3]>> {
    let intri = squeeze_leading_batch(intrinsics)?;
    let dims = intri.dims();
    if dims.len() != 3 || dims[1] != 3 || dims[2] != 3 || dims[0] != original_coords.len() {
        candle_core::bail!(
            "VGGT intrinsics shape {dims:?} does not match {} per-frame original coords as [S,3,3]",
            original_coords.len()
        );
    }

    let flat = intri
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F64)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f64>()?;

    Ok(flat
        .chunks_exact(9)
        .zip(original_coords)
        .map(|(values, coords)| {
            let mut k = match coords.mode {
                VggtPreprocessMode::Crop => {
                    let ratio = coords.resize_ratio(img_size);
                    [
                        [values[0] * ratio, values[1] * ratio, values[2] * ratio],
                        [values[3] * ratio, values[4] * ratio, values[5] * ratio],
                        [values[6] * ratio, values[7] * ratio, values[8] * ratio],
                    ]
                }
                // Pad mode: exact per-axis scales. Row 0 is the u axis, row 1
                // the v axis; the principal point is replaced below anyway.
                VggtPreprocessMode::Pad => {
                    let (scale_x, scale_y) = coords.axis_scales_to_original();
                    [
                        [
                            values[0] * scale_x,
                            values[1] * scale_x,
                            values[2] * scale_x,
                        ],
                        [
                            values[3] * scale_y,
                            values[4] * scale_y,
                            values[5] * scale_y,
                        ],
                        [values[6], values[7], values[8]],
                    ]
                }
            };
            let (cx, cy) = coords.original_principal_point();
            k[0][2] = cx;
            k[1][2] = cy;
            k[2] = [0.0, 0.0, 1.0];
            k
        })
        .collect())
}

/// Reference `shift_point2d_to_original_res`: shift a 2D pixel coordinate by
/// the content top-left offset and scale it back to the original image,
/// mapping a prepared-frame pixel to the original image.
///
/// * Crop mode (upstream, bit-identical): uniform ratio
///   `max(real_w, real_h) / img_size` on both axes.
/// * Pad mode (exact per-axis): `(x - offset_x) * real_w / content_w`,
///   `(y - offset_y) * real_h / content_h`.
pub fn shift_point2d_to_original(
    x: f64,
    y: f64,
    coords: &VggtOriginalCoords,
    img_size: f64,
) -> (f64, f64) {
    let (top_left_x, top_left_y) = coords.top_left();
    match coords.mode {
        VggtPreprocessMode::Crop => {
            let ratio = coords.resize_ratio(img_size);
            ((x - top_left_x) * ratio, (y - top_left_y) * ratio)
        }
        VggtPreprocessMode::Pad => {
            let (scale_x, scale_y) = coords.axis_scales_to_original();
            ((x - top_left_x) * scale_x, (y - top_left_y) * scale_y)
        }
    }
}

/// Reference `vggt.utils.geometry.unproject_depth_map_to_point_map` /
/// `depth_to_world_coords_points`: unproject a batch of depth maps to world
/// coordinates.
///
/// * `depth`: `[S, H, W, 1]`, `[S, H, W]` or `[1, S, H, W, 1]`.
/// * `extrinsics`: `[S, 3, 4]` (or `[1, S, 3, 4]`) OpenCV camera-from-world.
/// * `intrinsics`: `[S, 3, 3]` (or `[1, S, 3, 3]`).
///
/// Pixels with `depth > 1e-8` are the valid mask; the unprojection follows
/// `x_cam = (u - cu) * z / fu`, `y_cam = (v - cv) * z / fv`, `z_cam = z`, then
/// transforms by the closed-form inverse SE3 (`R^T`, `-R^T t`) to world
/// coordinates. Pixels with `depth <= 1e-8` are zeroed: upstream computes the
/// same valid mask but discards it, which would leave those pixels collapsed
/// onto the camera centre.
///
/// Returns `[S, H, W, 3]` world points.
pub fn unproject_depth_to_world(
    depth: &Tensor,
    extrinsics: &Tensor,
    intrinsics: &Tensor,
) -> Result<Tensor> {
    unproject_depth_to_world_inner(depth, extrinsics, intrinsics)
        .map_err(|source| tensor_err("VGGT depth unprojection failed", source))
}

fn unproject_depth_to_world_inner(
    depth: &Tensor,
    extrinsics: &Tensor,
    intrinsics: &Tensor,
) -> candle_core::Result<Tensor> {
    const DEPTH_EPS: f32 = 1e-8;

    let depth_sq = normalize_depth_grid(depth)?;
    let (s, h, w) = {
        let dims = depth_sq.dims();
        (dims[0], dims[1], dims[2])
    };
    let n = h * w;
    let extri = squeeze_leading_batch(extrinsics)?;
    let intri = squeeze_leading_batch(intrinsics)?;
    if extri.dims() != [s, 3, 4] {
        candle_core::bail!(
            "VGGT extrinsics shape {:?} does not match depth grid {s}x{h}x{w} as [S,3,4]",
            extri.dims()
        );
    }
    if intri.dims() != [s, 3, 3] {
        candle_core::bail!(
            "VGGT intrinsics shape {:?} does not match depth grid {s}x{h}x{w} as [S,3,3]",
            intri.dims()
        );
    }

    let device = depth_sq.device();

    // Pixel coordinate grids, row-major like `np.meshgrid(arange(W), arange(H))`.
    let u = Tensor::arange(0u32, w as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((1, w))?
        .broadcast_as((h, w))?
        .contiguous()?
        .reshape((1, n))?
        .broadcast_as((s, n))?
        .contiguous()?;
    let v = Tensor::arange(0u32, h as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((h, 1))?
        .broadcast_as((h, w))?
        .contiguous()?
        .reshape((1, n))?
        .broadcast_as((s, n))?
        .contiguous()?;

    let depth_flat = depth_sq.reshape((s, n))?;
    let cu = intri.i((.., 0, 2))?.reshape((s, 1))?;
    let cv = intri.i((.., 1, 2))?.reshape((s, 1))?;
    let fu = intri.i((.., 0, 0))?.reshape((s, 1))?;
    let fv = intri.i((.., 1, 1))?.reshape((s, 1))?;

    // x_cam = (u - cu) * z / fu, y_cam = (v - cv) * z / fv, z_cam = z.
    let x_cam = u
        .broadcast_sub(&cu)?
        .broadcast_mul(&depth_flat)?
        .broadcast_div(&fu)?;
    let y_cam = v
        .broadcast_sub(&cv)?
        .broadcast_mul(&depth_flat)?
        .broadcast_div(&fv)?;
    let cam_coords = Tensor::stack(&[&x_cam, &y_cam, &depth_flat], 2)?; // [S, N, 3]

    // World = R^T p - R^T t. Row-vector form: world^T = p^T R - t^T R, so the
    // batched matmuls use the original camera-from-world R directly.
    let r = extri.narrow(D::Minus1, 0, 3)?; // [S,3,3]
    let t = extri.narrow(D::Minus1, 3, 1)?; // [S,3,1]
    let t_inv = t
        .transpose(1, 2)? // [S,1,3]
        .matmul(&r)?
        .affine(-1.0, 0.0)?; // [S,1,3] = -R^T t
    let world = cam_coords
        .matmul(&r)? // [S,N,3] = p^T R = R^T p
        .broadcast_add(&t_inv)?;

    // Valid mask (depth > eps): zero degenerate pixels (see doc comment).
    let valid = depth_flat
        .gt(DEPTH_EPS)?
        .to_dtype(DType::F32)?
        .unsqueeze(2)?; // [S,N,1]
    let world = world.broadcast_mul(&valid)?;

    world.reshape((s, h, w, 3))
}

/// Normalize a depth tensor to a `[S, H, W]` grid, accepting the public
/// `[1, S, H, W, 1]`, `[S, H, W, 1]` and `[S, H, W]` layouts.
fn normalize_depth_grid(depth: &Tensor) -> candle_core::Result<Tensor> {
    let dims = depth.dims();
    match dims.len() {
        5 if dims[0] == 1 && dims[4] == 1 => depth.squeeze(0)?.squeeze(D::Minus1),
        4 if dims[3] == 1 => depth.squeeze(D::Minus1),
        3 => Ok(depth.clone()),
        _ => candle_core::bail!(
            "VGGT depth shape {dims:?} must be [1,S,H,W,1], [S,H,W,1] or [S,H,W]"
        ),
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    /// Synthetic pinhole roundtrip: build a depth map from known 3D world
    /// points, unproject, and expect the original world points back (<1e-5).
    #[test]
    fn unproject_depth_to_world_matches_synthetic_pinhole() {
        let device = Device::Cpu;
        let (h, w) = (6usize, 8usize);

        // Camera-from-world: 90-degree rotation about z, plus translation.
        let r = [[0.0f64, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let t = [0.5f64, -0.3, 2.0];
        // [R|t] row-major: each rotation row is followed by its translation.
        let mut extrinsics = Vec::with_capacity(12);
        for row in 0..3 {
            for value in &r[row] {
                extrinsics.push(*value as f32);
            }
            extrinsics.push(t[row] as f32);
        }
        let extrinsics = Tensor::from_vec(extrinsics, (1, 1, 3, 4), &device).expect("extrinsics");

        let (fx, fy, cx, cy) = (50.0f64, 40.0, w as f64 / 2.0, h as f64 / 2.0);
        let intrinsics = Tensor::from_vec(
            vec![
                fx as f32, 0.0, cx as f32, //
                0.0, fy as f32, cy as f32, //
                0.0, 0.0, 1.0,
            ],
            (1, 1, 3, 3),
            &device,
        )
        .expect("intrinsics");

        // Depth map: known positive depths, one degenerate zero-depth pixel.
        let zero_pixel = (3usize, 5usize);
        let mut depth_values = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                if (y, x) == zero_pixel {
                    depth_values.push(0.0f32);
                } else {
                    depth_values.push(1.0 + 0.25 * ((x + 2 * y) % 7) as f32);
                }
            }
        }
        let depth = Tensor::from_vec(
            depth_values.clone(),
            (1, 1, h as usize, w as usize, 1),
            &device,
        )
        .expect("depth");

        // Expected world points from the reference math (f64).
        let r_t = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ];
        let mut expected = Vec::with_capacity(h * w * 3);
        for y in 0..h {
            for x in 0..w {
                let z = f64::from(depth_values[y * w + x]);
                if z <= 1e-8 {
                    expected.extend_from_slice(&[0.0; 3]);
                    continue;
                }
                let pc = [(x as f64 - cx) * z / fx, (y as f64 - cy) * z / fy, z];
                for j in 0..3 {
                    let value = r_t[j][0] * (pc[0] - t[0])
                        + r_t[j][1] * (pc[1] - t[1])
                        + r_t[j][2] * (pc[2] - t[2]);
                    expected.push(value as f32);
                }
            }
        }

        let world = unproject_depth_to_world(&depth, &extrinsics, &intrinsics).expect("unproject");
        assert_eq!(world.dims(), &[1, h, w, 3]);
        let actual = world
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("world points vec");

        let mut max_error = 0.0f32;
        for (expected_value, actual_value) in expected.iter().zip(&actual) {
            max_error = max_error.max((expected_value - actual_value).abs());
        }
        assert!(max_error < 1e-5, "unprojection error {max_error}");
    }

    /// The `[S, H, W]` and `[S, 3, 4]`/`[S, 3, 3]` (no batch dim) layouts give
    /// the same result as the `[1, S, ...]` layouts.
    #[test]
    fn unproject_depth_accepts_squeezed_layouts() {
        let device = Device::Cpu;
        let depth = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0], (2, 1, 3, 1), &device)
            .expect("depth");
        let extrinsics = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, -2.0, //
                0.0, 0.0, 1.0, 3.0, //
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, -2.0, //
                0.0, 0.0, 1.0, 3.0,
            ],
            (2, 3, 4),
            &device,
        )
        .expect("extrinsics");
        let intrinsics = Tensor::from_vec(
            vec![
                10.0f32, 0.0, 0.5, //
                0.0, 10.0, 0.5, //
                0.0, 0.0, 1.0, //
                10.0, 0.0, 0.5, //
                0.0, 10.0, 0.5, //
                0.0, 0.0, 1.0,
            ],
            (2, 3, 3),
            &device,
        )
        .expect("intrinsics");

        let world = unproject_depth_to_world(&depth, &extrinsics, &intrinsics).expect("unproject");
        assert_eq!(world.dims(), &[2, 1, 3, 3]);
        let actual = world
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("vec");
        // Frame 0, pixel (0, 0): p_cam = ((0-0.5)*2/10, (0-0.5)*2/10, 2) =
        // (-0.1, -0.1, 2); world = p_cam - t = (-1.1, 1.9, -1.0) (R = I).
        assert!((actual[0] - (-1.1)).abs() < 1e-6);
        assert!((actual[1] - 1.9).abs() < 1e-6);
        assert!((actual[2] - (-1.0)).abs() < 1e-6);
    }

    /// Task-parity unit test: 518x518 prepared frame, original 1024x768
    /// (ratio 1024/518 analog via img_size), non-zero crop offset.
    #[test]
    fn rescale_intrinsics_and_shift_point2d_follow_reference_formula() {
        let device = Device::Cpu;
        let img_size = 518.0f64;
        let coords = VggtOriginalCoords {
            width: 1024,
            height: 768,
            offset_x: 12,
            offset_y: 63,
            content_width: 518,
            content_height: 392,
            mode: VggtPreprocessMode::Crop,
        };
        let intrinsics = Tensor::from_vec(
            vec![
                361.0f32, 0.0, 259.0, //
                0.0, 361.0, 259.0, //
                0.0, 0.0, 1.0,
            ],
            (1, 1, 3, 3),
            &device,
        )
        .expect("intrinsics");

        let rescaled =
            rescale_intrinsics_to_original(&intrinsics, &[coords], img_size).expect("rescale");
        assert_eq!(rescaled.len(), 1);
        let ratio = 1024.0f64 / 518.0;
        let k = rescaled[0];
        assert!((k[0][0] - 361.0 * ratio).abs() < 1e-9);
        assert!((k[1][1] - 361.0 * ratio).abs() < 1e-9);
        assert!((k[0][2] - 512.0).abs() < 1e-9, "cx must be real_w / 2");
        assert!((k[1][2] - 384.0).abs() < 1e-9, "cy must be real_h / 2");
        assert_eq!(k[2], [0.0, 0.0, 1.0]);

        // Point2D shift: (xy - top_left) * resize_ratio.
        let (sx, sy) = shift_point2d_to_original(270.0, 300.0, &coords, img_size);
        assert!((sx - (270.0 - 12.0) * ratio).abs() < 1e-9);
        assert!((sy - (300.0 - 63.0) * ratio).abs() < 1e-9);
    }

    /// Pad mode uses the exact per-axis scales (`real / content`) instead of
    /// the crop-mode uniform ratio; the principal point still moves to the
    /// original image centre.
    #[test]
    fn rescale_intrinsics_and_shift_point2d_use_per_axis_scales_in_pad_mode() {
        let device = Device::Cpu;
        // 1024x768 in pad mode: content 518x392, offsets (0, 63).
        let coords = VggtOriginalCoords {
            width: 1024,
            height: 768,
            offset_x: 0,
            offset_y: 63,
            content_width: 518,
            content_height: 392,
            mode: VggtPreprocessMode::Pad,
        };
        let intrinsics = Tensor::from_vec(
            vec![
                361.0f32, 0.0, 259.0, //
                0.0, 361.0, 259.0, //
                0.0, 0.0, 1.0,
            ],
            (1, 1, 3, 3),
            &device,
        )
        .expect("intrinsics");

        let rescaled =
            rescale_intrinsics_to_original(&intrinsics, &[coords], 518.0).expect("rescale");
        let scale_x = 1024.0f64 / 518.0;
        let scale_y = 768.0f64 / 392.0;
        let k = rescaled[0];
        assert!(
            (k[0][0] - 361.0 * scale_x).abs() < 1e-9,
            "fx scaled per-axis"
        );
        assert!(
            (k[1][1] - 361.0 * scale_y).abs() < 1e-9,
            "fy scaled per-axis"
        );
        assert!((k[0][2] - 512.0).abs() < 1e-9, "cx must be real_w / 2");
        assert!((k[1][2] - 384.0).abs() < 1e-9, "cy must be real_h / 2");
        assert_eq!(k[2], [0.0, 0.0, 1.0]);

        // Point2D shift: (xy - top_left) * per-axis scale.
        let (sx, sy) = shift_point2d_to_original(270.0, 300.0, &coords, 518.0);
        assert!((sx - 270.0 * scale_x).abs() < 1e-9);
        assert!((sy - (300.0 - 63.0) * scale_y).abs() < 1e-9);
    }
}
