//! SuperPoint detector/descriptor (LightGlue port) for VGGT tracking stage A.
//!
//! Reference: `lightglue/superpoint.py` at
//! `G:/jobs/candle-based-projects/candle_refs/LightGlue`, official checkpoint
//! `superpoint_v1.pth`, descriptor width 256, NMS radius 4, remove-border 4.
//!
//! The port follows the reference line by line:
//!
//! - shared encoder `conv1a..conv4b` with ReLU and 2x2 stride-2 max pools
//!   (`superpoint.py:158-169`);
//! - dense score head `convPa/convPb`, softmax over the 65 sub-pixel channels
//!   with the dustbin channel dropped, then the 8x8 channel-to-pixel
//!   rearrangement (`superpoint.py:171-177`);
//! - `simple_nms` with radius 4, border removal and `scores > threshold`
//!   (`superpoint.py:52-68,178-190`);
//! - top-k with the reference ordering: raster order while `k` covers every
//!   candidate, score-descending once the top-k actually truncates
//!   (`superpoint.py:71-75,198-207`);
//! - descriptor head `convDa/convDb`, channel-wise L2 normalization and
//!   bilinear `grid_sample` at the keypoints with `s=8`
//!   (`superpoint.py:78-95,212-221`);
//! - grayscale conversion matching kornia `rgb_to_grayscale`
//!   (BT.601 `0.299 R + 0.587 G + 0.114 B`).
//!
//! Coordinates are pixel centers of the input image. When
//! [`KeypointOptions::resize_long_side`] resizes the image first, the returned
//! coordinates are mapped back to the original input with the reference
//! `Extractor.scale_keypoints` center-origin rule
//! (`utils.py:144-150`): `(kp + 0.5) / scale - 0.5`.
//!
//! Deviation: the reference preprocessor resizes through
//! `kornia.geometry.transform.resize(..., antialias=True)`, which prefilters
//! with a Gaussian when downscaling. This port uses candle's bilinear resize
//! (`align_corners=false`, the same sampling grid), so upscaling and
//! same-size inputs match the reference; images larger than the requested
//! long side see the reference's extra anti-alias blur that this port does not
//! reproduce.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d};

use super::keypoints::{KeypointOptions, KeypointSet};
use crate::Result;
use crate::error::Lux3dError;

/// Reference descriptor width (`descriptor_dim` in `superpoint.py:108`).
const DESCRIPTOR_DIM: usize = 256;
/// `nms_radius` (`superpoint.py:109`).
const NMS_RADIUS: usize = 4;
/// `remove_borders` (`superpoint.py:112`).
const REMOVE_BORDERS: usize = 4;
/// Sub-pixel cell size: the dense score map is 1/8 of the image and each cell
/// carries the 64 non-dustbin channels (`s=8` in `sample_descriptors`).
const CELL_SIZE: usize = 8;
/// `nn.functional.normalize(..., p=2, dim=1)` default epsilon.
const NORMALIZE_EPS: f32 = 1e-12;

/// SuperPoint detector/descriptor network loaded from the official
/// `superpoint_v1.safetensors` state dict (original tensor names).
#[derive(Debug)]
pub struct SuperPointDetector {
    conv1a: Conv2d,
    conv1b: Conv2d,
    conv2a: Conv2d,
    conv2b: Conv2d,
    conv3a: Conv2d,
    conv3b: Conv2d,
    conv4a: Conv2d,
    conv4b: Conv2d,
    conv_pa: Conv2d,
    conv_pb: Conv2d,
    conv_da: Conv2d,
    conv_db: Conv2d,
    device: Device,
    dtype: DType,
}

impl SuperPointDetector {
    /// Loads the official SuperPoint checkpoint.
    ///
    /// `weights` must be the converted safetensors file with the reference
    /// tensor names (`conv1a.weight`, ..., `convDb.bias`). A missing file is
    /// reported as [`Lux3dError::MissingWeightFile`]; unreadable or malformed
    /// tensors are reported as [`Lux3dError::CanonicalWeightsValidation`].
    pub fn load(weights: &Path, device: &Device, dtype: DType) -> Result<Self> {
        if !weights.is_file() {
            return Err(Lux3dError::MissingWeightFile {
                path: weights.to_path_buf(),
            });
        }
        // SAFETY: the mmap-backed builder's tensors are owned by the returned
        // detector; the checkpoint file is read-only and outlives the load.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), dtype, device)
        }
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "failed to open SuperPoint weights `{}`: {source}",
                weights.display()
            ),
        })?;

        let conv = |name: &str,
                    in_channels: usize,
                    out_channels: usize,
                    kernel_size: usize,
                    padding: usize|
         -> Result<Conv2d> {
            conv2d(
                in_channels,
                out_channels,
                kernel_size,
                Conv2dConfig {
                    padding,
                    ..Default::default()
                },
                vb.pp(name),
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load SuperPoint `{name}`: {source}"),
            })
        };

        Ok(Self {
            conv1a: conv("conv1a", 1, 64, 3, 1)?,
            conv1b: conv("conv1b", 64, 64, 3, 1)?,
            conv2a: conv("conv2a", 64, 64, 3, 1)?,
            conv2b: conv("conv2b", 64, 64, 3, 1)?,
            conv3a: conv("conv3a", 64, 128, 3, 1)?,
            conv3b: conv("conv3b", 128, 128, 3, 1)?,
            conv4a: conv("conv4a", 128, 128, 3, 1)?,
            conv4b: conv("conv4b", 128, 128, 3, 1)?,
            conv_pa: conv("convPa", 128, 256, 3, 1)?,
            conv_pb: conv("convPb", 256, 65, 1, 0)?,
            conv_da: conv("convDa", 128, 256, 3, 1)?,
            conv_db: conv("convDb", 256, DESCRIPTOR_DIM, 1, 0)?,
            device: device.clone(),
            dtype,
        })
    }

    /// Runs detection and description on an RGB image in `[0, 1]`.
    ///
    /// `rgb` is `[3, H, W]` (a leading batch dim of 1 is also accepted), f32
    /// by convention. The returned [`KeypointSet`] holds pixel-center
    /// keypoints in the original input coordinate system, their scores and
    /// row-major L2-normalized descriptors of width [`DESCRIPTOR_DIM`].
    /// `max_keypoints == 0` yields an empty set; the reference rejects a
    /// non-positive `max_num_keypoints` when the extractor is constructed.
    pub fn detect(&self, rgb: &Tensor, options: &KeypointOptions) -> Result<KeypointSet> {
        let prepared = self.prepare_input(rgb, options)?;
        let (coords, scores, descriptors) =
            self.forward(&prepared.gray, options).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("SuperPoint forward failed: {source}"),
                }
            })?;

        // `Extractor.scale_keypoints` with origin="center" (`utils.py:150`).
        let keypoints = coords
            .iter()
            .map(|&[x, y]| {
                [
                    (x + 0.5) / prepared.scale_x - 0.5,
                    (y + 0.5) / prepared.scale_y - 0.5,
                ]
            })
            .collect();

        Ok(KeypointSet {
            keypoints,
            scores,
            descriptors,
            descriptor_dim: DESCRIPTOR_DIM,
        })
    }

    /// Grayscale conversion + optional long-side resize.
    fn prepare_input(&self, rgb: &Tensor, options: &KeypointOptions) -> Result<PreparedInput> {
        let rgb = rgb
            .to_device(&self.device)
            .and_then(|tensor| tensor.to_dtype(DType::F32))
            .map_err(|source| forward_err("input transfer", source))?;

        let (channels, height, width) = match rgb.rank() {
            3 => rgb
                .dims3()
                .map_err(|source| forward_err("input shape", source))?,
            4 => {
                let (batch, channels, height, width) = rgb
                    .dims4()
                    .map_err(|source| forward_err("input shape", source))?;
                if batch != 1 {
                    return Err(Lux3dError::InvalidInput(
                        "SuperPoint input must contain exactly one image (batch == 1)",
                    ));
                }
                (channels, height, width)
            }
            _ => {
                return Err(Lux3dError::InvalidInput(
                    "SuperPoint input must be a [3, H, W] RGB tensor",
                ));
            }
        };
        if channels != 3 && channels != 1 {
            return Err(Lux3dError::InvalidInput(
                "SuperPoint input must have 3 RGB channels (or 1 grayscale channel)",
            ));
        }
        if height < CELL_SIZE || width < CELL_SIZE {
            return Err(Lux3dError::InvalidInput(
                "SuperPoint input must be at least 8x8 pixels",
            ));
        }

        let mut image = if rgb.rank() == 4 {
            rgb
        } else {
            rgb.unsqueeze(0)
                .map_err(|source| forward_err("input shape", source))?
        };

        // `ImagePreprocessor`: resize the long side and keep the aspect ratio.
        let (resized_height, resized_width) = match options.resize_long_side {
            None => (height, width),
            Some(long_side) => {
                if long_side == 0 {
                    return Err(Lux3dError::InvalidInput(
                        "SuperPoint resize_long_side must be positive",
                    ));
                }
                // `_side_to_image_size(size, aspect_ratio, side="long")`
                // (`kornia.geometry.transform.resize`): the long side becomes
                // `long_side`, the other side is truncated to an integer.
                let aspect_ratio = width as f64 / height as f64;
                let (resized_height, resized_width) = if aspect_ratio < 1.0 {
                    (
                        long_side,
                        ((long_side as f64) * aspect_ratio).floor() as usize,
                    )
                } else {
                    (
                        ((long_side as f64) / aspect_ratio).floor() as usize,
                        long_side,
                    )
                };
                let resized_height = resized_height.max(1);
                let resized_width = resized_width.max(1);
                if resized_height < CELL_SIZE || resized_width < CELL_SIZE {
                    return Err(Lux3dError::InvalidInput(
                        "SuperPoint resize target must be at least 8x8 pixels",
                    ));
                }
                image = image
                    .upsample_bilinear2d(resized_height, resized_width, false)
                    .map_err(|source| forward_err("input resize", source))?;
                (resized_height, resized_width)
            }
        };

        // `rgb_to_grayscale`: BT.601 luma with the reference weight order.
        let gray = if channels == 3 {
            let weights =
                Tensor::from_vec(vec![0.299_f32, 0.587, 0.114], (1, 3, 1, 1), &self.device)
                    .map_err(|source| forward_err("grayscale weights", source))?;
            image
                .broadcast_mul(&weights)
                .and_then(|image| image.sum_keepdim(1))
                .map_err(|source| forward_err("grayscale", source))?
        } else {
            image
        };
        let gray = gray
            .to_dtype(self.dtype)
            .map_err(|source| forward_err("input dtype", source))?;

        // `Extractor.extract` turns the resized image size into a per-axis
        // scale and `scale_keypoints` maps back with origin="center".
        Ok(PreparedInput {
            gray,
            scale_x: resized_width as f32 / width as f32,
            scale_y: resized_height as f32 / height as f32,
        })
    }

    /// Network forward up to pre-scaling keypoints, scores and descriptors.
    fn forward(
        &self,
        gray: &Tensor,
        options: &KeypointOptions,
    ) -> candle_core::Result<(Vec<[f32; 2]>, Vec<f32>, Vec<f32>)> {
        // Shared encoder (`superpoint.py:158-169`).
        let x = self.conv1a.forward(gray)?.relu()?;
        let x = self.conv1b.forward(&x)?.relu()?;
        let x = x.max_pool2d_with_stride(2, 2)?;
        let x = self.conv2a.forward(&x)?.relu()?;
        let x = self.conv2b.forward(&x)?.relu()?;
        let x = x.max_pool2d_with_stride(2, 2)?;
        let x = self.conv3a.forward(&x)?.relu()?;
        let x = self.conv3b.forward(&x)?.relu()?;
        let x = x.max_pool2d_with_stride(2, 2)?;
        let x = self.conv4a.forward(&x)?.relu()?;
        let x = self.conv4b.forward(&x)?.relu()?;

        // Dense score map: softmax over the 65 sub-pixel channels and drop the
        // dustbin (`superpoint.py:172-174`).
        let c_pa = self.conv_pa.forward(&x)?.relu()?;
        let logits = self.conv_pb.forward(&c_pa)?;
        let probabilities = candle_nn::ops::softmax(&logits, 1)?;
        let probabilities = probabilities.narrow(1, 0, CELL_SIZE * CELL_SIZE)?;

        // Channel `c = row * 8 + col` maps to sub-pixel `(row, col)` of the
        // coarse cell (`superpoint.py:175-177`).
        let (cell_h, cell_w) = (probabilities.dim(2)?, probabilities.dim(3)?);
        let score_map = probabilities
            .squeeze(0)?
            .permute([1, 2, 0])?
            .contiguous()?
            .reshape((cell_h, cell_w, CELL_SIZE, CELL_SIZE))?
            .permute([0, 2, 1, 3])?
            .contiguous()?
            .reshape((cell_h * CELL_SIZE, cell_w * CELL_SIZE))?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let score_height = score_map.dim(0)?;
        let score_width = score_map.dim(1)?;
        let scores = score_map.flatten_all()?.to_vec1::<f32>()?;

        // `simple_nms`, border removal, threshold and top-k on the host.
        let suppressed = simple_nms(&scores, score_height, score_width, NMS_RADIUS);
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for y in 0..score_height {
            if y < REMOVE_BORDERS || y + REMOVE_BORDERS >= score_height {
                continue;
            }
            for x in 0..score_width {
                if x < REMOVE_BORDERS || x + REMOVE_BORDERS >= score_width {
                    continue;
                }
                let index = y * score_width + x;
                let score = suppressed[index];
                if score > options.detection_threshold {
                    candidates.push((index, score));
                }
            }
        }

        // `top_k_keypoints`: raster order when the cap covers every candidate,
        // otherwise score-descending (ties by raster index).
        if candidates.len() > options.max_keypoints {
            candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            candidates.truncate(options.max_keypoints);
        }

        // Descriptor head, normalized per spatial location before sampling
        // (`superpoint.py:213-215`).
        let c_da = self.conv_da.forward(&x)?.relu()?;
        let descriptors = self.conv_db.forward(&c_da)?;
        let norm = descriptors
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(NORMALIZE_EPS, f32::MAX)?;
        let descriptors = descriptors.broadcast_div(&norm)?;
        let descriptors = descriptors
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let descriptor_height = descriptors.dim(1)?;
        let descriptor_width = descriptors.dim(2)?;
        let descriptor_plane = descriptor_height * descriptor_width;
        let dense = descriptors.flatten_all()?.to_vec1::<f32>()?;

        // `sample_descriptors` with s=8 (`superpoint.py:78-95`): the normalized
        // grid coordinate resolves to pixel `(kp - s/2 + 0.5) / (w*s - s/2 -
        // 0.5) * 2 - 1`, and `grid_sample` with `align_corners=True` maps that
        // back to `(kp - 3.5) * (w - 1) / (w*8 - 4.5)` on the descriptor map.
        let cell_w = descriptor_width;
        let cell_h = descriptor_height;
        let denominator_x = (cell_w * CELL_SIZE) as f32 - 4.5;
        let denominator_y = (cell_h * CELL_SIZE) as f32 - 4.5;
        let max_x = cell_w as f32 - 1.0;
        let max_y = cell_h as f32 - 1.0;

        let mut keypoints = Vec::with_capacity(candidates.len());
        let mut keypoint_scores = Vec::with_capacity(candidates.len());
        let mut sampled = Vec::with_capacity(candidates.len() * DESCRIPTOR_DIM);
        let mut descriptor = [0.0_f32; DESCRIPTOR_DIM];
        for (index, score) in candidates {
            let pixel_y = index / score_width;
            let pixel_x = index % score_width;
            let sample_x = (pixel_x as f32 - 3.5) * max_x / denominator_x;
            let sample_y = (pixel_y as f32 - 3.5) * max_y / denominator_y;

            sample_descriptor(
                &dense,
                descriptor_height,
                descriptor_width,
                descriptor_plane,
                sample_x,
                sample_y,
                &mut descriptor,
            );
            normalize_descriptor(&mut descriptor);

            sampled.extend_from_slice(&descriptor);
            keypoints.push([pixel_x as f32, pixel_y as f32]);
            keypoint_scores.push(score);
        }

        Ok((keypoints, keypoint_scores, sampled))
    }
}

struct PreparedInput {
    gray: Tensor,
    scale_x: f32,
    scale_y: f32,
}

fn forward_err(context: &str, source: candle_core::Error) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: format!("SuperPoint {context} failed: {source}"),
    }
}

/// `simple_nms` (`superpoint.py:52-68`) on a host-side score map.
fn simple_nms(scores: &[f32], height: usize, width: usize, radius: usize) -> Vec<f32> {
    debug_assert_eq!(scores.len(), height * width);
    let pooled = max_filter(scores, height, width, radius);
    let mut mask: Vec<f32> = scores
        .iter()
        .zip(pooled.iter())
        .map(|(score, maximum)| if score == maximum { 1.0 } else { 0.0 })
        .collect();

    for _ in 0..2 {
        let support = max_filter(&mask, height, width, radius);
        let support_mask: Vec<f32> = support
            .iter()
            .map(|value| if *value > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let support_scores: Vec<f32> = scores
            .iter()
            .zip(support_mask.iter())
            .map(
                |(score, suppressed)| {
                    if *suppressed > 0.0 { 0.0 } else { *score }
                },
            )
            .collect();
        let pooled = max_filter(&support_scores, height, width, radius);
        for index in 0..mask.len() {
            if support_mask[index] == 0.0 && support_scores[index] == pooled[index] {
                mask[index] = 1.0;
            }
        }
    }

    scores
        .iter()
        .zip(mask.iter())
        .map(|(score, keep)| if *keep > 0.0 { *score } else { 0.0 })
        .collect()
}

/// `max_pool2d(x, kernel=2*radius+1, stride=1, padding=radius)` as used by
/// `simple_nms`.
///
/// The square window maximum is separable, so this runs a horizontal pass then
/// a vertical pass. Torch pads the window with `-inf` in float mode; every
/// input here is non-negative, so clipping the window at the border gives the
/// same maximum. Range clamping is exact: for `radius=0` this copies the input.
fn max_filter(input: &[f32], height: usize, width: usize, radius: usize) -> Vec<f32> {
    if radius == 0 {
        return input.to_vec();
    }
    let mut horizontal = vec![0.0_f32; height * width];
    for y in 0..height {
        let row = &input[y * width..(y + 1) * width];
        for x in 0..width {
            let start = x.saturating_sub(radius);
            let end = (x + radius + 1).min(width);
            let mut maximum = f32::NEG_INFINITY;
            for &value in &row[start..end] {
                maximum = maximum.max(value);
            }
            horizontal[y * width + x] = maximum;
        }
    }

    let mut output = vec![0.0_f32; height * width];
    for y in 0..height {
        let start = y.saturating_sub(radius);
        let end = (y + radius + 1).min(height);
        for x in 0..width {
            let mut maximum = f32::NEG_INFINITY;
            for yy in start..end {
                maximum = maximum.max(horizontal[yy * width + x]);
            }
            output[y * width + x] = maximum;
        }
    }
    output
}

/// Bilinear `grid_sample(mode="bilinear", padding_mode="zeros",
/// align_corners=True)` of one keypoint from the channel-first descriptor map.
#[allow(clippy::too_many_arguments)]
fn sample_descriptor(
    dense: &[f32],
    height: usize,
    width: usize,
    plane: usize,
    x: f32,
    y: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(dense.len(), plane * DESCRIPTOR_DIM);
    debug_assert!(out.len() == DESCRIPTOR_DIM);
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let x0 = x0 as isize;
    let y0 = y0 as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let in_x0 = x0 >= 0 && (x0 as usize) < width;
    let in_x1 = x1 >= 0 && (x1 as usize) < width;
    let in_y0 = y0 >= 0 && (y0 as usize) < height;
    let in_y1 = y1 >= 0 && (y1 as usize) < height;
    let (x0, x1) = (x0 as usize, x1 as usize);
    let (y0, y1) = (y0 as usize, y1 as usize);

    for (channel, value) in out.iter_mut().enumerate() {
        let base = channel * plane;
        let top_left = if in_x0 && in_y0 {
            dense[base + y0 * width + x0]
        } else {
            0.0
        };
        let top_right = if in_x1 && in_y0 {
            dense[base + y0 * width + x1]
        } else {
            0.0
        };
        let bottom_left = if in_x0 && in_y1 {
            dense[base + y1 * width + x0]
        } else {
            0.0
        };
        let bottom_right = if in_x1 && in_y1 {
            dense[base + y1 * width + x1]
        } else {
            0.0
        };
        let top = (1.0 - fx) * top_left + fx * top_right;
        let bottom = (1.0 - fx) * bottom_left + fx * bottom_right;
        *value = (1.0 - fy) * top + fy * bottom;
    }
}

/// `nn.functional.normalize(p=2, dim=1)` after sampling.
fn normalize_descriptor(descriptor: &mut [f32]) {
    let norm = descriptor
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(NORMALIZE_EPS);
    for value in descriptor.iter_mut() {
        *value /= norm;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::Path;

    use candle_core::{DType, Device, Tensor};
    use ndarray_npy::NpzReader;

    use super::*;

    const WEIGHTS: &str = "../../models/aliked-superpoint/superpoint_v1.safetensors";
    const GOLDEN: &str = "tests/fixtures/keypoints/superpoint_golden_256.npz";
    const COORDINATE_TOLERANCE: f32 = 1e-3;
    const SCORE_TOLERANCE: f32 = 1e-4;
    const COSINE_TOLERANCE: f32 = 1e-3;

    fn u8_tensor(array: ndarray::ArrayD<u8>, device: &Device) -> Tensor {
        let shape = array.shape().to_vec();
        assert_eq!(shape.len(), 3, "expected CHW image, got {shape:?}");
        let (data, _offset) = array.into_raw_vec_and_offset();
        let values: Vec<f32> = data.iter().map(|value| f32::from(*value) / 255.0).collect();
        Tensor::from_vec(values, shape.as_slice(), device).expect("image tensor")
    }

    fn f32_array(npz: &mut NpzReader<File>, name: &str) -> ndarray::ArrayD<f32> {
        npz.by_name(name)
            .unwrap_or_else(|error| panic!("missing golden `{name}`: {error}"))
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }

    fn raster_key(x: f32, y: f32) -> (i64, i64) {
        (y.round() as i64, x.round() as i64)
    }

    fn assert_golden_match(
        got: &KeypointSet,
        expected_keypoints: &[[f32; 2]],
        expected_scores: &[f32],
        expected_descriptors: Option<&[f32]>,
        threshold: f32,
    ) {
        assert_eq!(got.descriptor_dim, DESCRIPTOR_DIM);
        if let Some(descriptors) = expected_descriptors {
            assert_eq!(descriptors.len(), expected_keypoints.len() * DESCRIPTOR_DIM);
        }

        // At resize=None keypoints are integer pixel centers; index both sets
        // by coordinate so a borderline threshold difference cannot shift the
        // whole comparison.
        let mut got_by_coordinate: HashMap<(i64, i64), usize> = HashMap::new();
        for (index, keypoint) in got.keypoints.iter().enumerate() {
            assert!(
                (keypoint[0] - keypoint[0].round()).abs() <= COORDINATE_TOLERANCE
                    && (keypoint[1] - keypoint[1].round()).abs() <= COORDINATE_TOLERANCE,
                "resize=None keypoints must be integer pixel centers, got {keypoint:?}"
            );
            let key = raster_key(keypoint[0], keypoint[1]);
            assert!(
                got_by_coordinate.insert(key, index).is_none(),
                "duplicate detected keypoint {key:?}"
            );
        }

        let mut expected_by_coordinate: HashMap<(i64, i64), usize> = HashMap::new();
        for (index, keypoint) in expected_keypoints.iter().enumerate() {
            expected_by_coordinate.insert(raster_key(keypoint[0], keypoint[1]), index);
        }

        let mut matched = 0usize;
        let mut boundary_only = 0usize;
        let mut max_score_error = 0.0_f32;
        let mut min_cosine = 1.0_f32;
        for (coordinate, &got_index) in got_by_coordinate.iter() {
            match expected_by_coordinate.get(coordinate) {
                Some(&expected_index) => {
                    matched += 1;
                    let got_score = got.scores[got_index];
                    let expected_score = expected_scores[expected_index];
                    max_score_error = max_score_error.max((got_score - expected_score).abs());
                    assert!(
                        (got_score - expected_score).abs() <= SCORE_TOLERANCE,
                        "score mismatch at {coordinate:?}: got {got_score}, want {expected_score}"
                    );
                    if let Some(expected) = expected_descriptors {
                        let start = expected_index * DESCRIPTOR_DIM;
                        let expected_descriptor = &expected[start..start + DESCRIPTOR_DIM];
                        let got_descriptor = &got.descriptors
                            [got_index * DESCRIPTOR_DIM..(got_index + 1) * DESCRIPTOR_DIM];
                        let cosine = cosine_similarity(got_descriptor, expected_descriptor);
                        min_cosine = min_cosine.min(cosine);
                        assert!(
                            cosine >= 1.0 - COSINE_TOLERANCE,
                            "descriptor mismatch at {coordinate:?}: cosine {cosine}"
                        );
                    }
                }
                None => {
                    // Only points the official run also sees within one score
                    // tolerance of the threshold may differ.
                    assert!(
                        got.scores[got_index] <= threshold + SCORE_TOLERANCE,
                        "unexpected extra keypoint {coordinate:?} with score {}",
                        got.scores[got_index]
                    );
                    boundary_only += 1;
                }
            }
        }

        let mut missing = 0usize;
        for (coordinate, &expected_index) in expected_by_coordinate.iter() {
            if !got_by_coordinate.contains_key(coordinate) {
                assert!(
                    expected_scores[expected_index] <= threshold + SCORE_TOLERANCE,
                    "missing keypoint {coordinate:?} with score {}",
                    expected_scores[expected_index]
                );
                missing += 1;
            }
        }
        assert!(
            boundary_only <= 4 && missing <= 4,
            "too many threshold-boundary differences: {boundary_only} extra, {missing} missing, {matched} matched"
        );
        assert!(
            matched >= expected_keypoints.len().saturating_sub(4),
            "matched only {matched} of {} official keypoints",
            expected_keypoints.len()
        );
        eprintln!(
            "SuperPoint golden: matched {matched}/{} (extra {boundary_only}, missing {missing}), \
             max score error {max_score_error:e}, min descriptor cosine {min_cosine:e}",
            expected_keypoints.len()
        );

        // Ordering: the reference keeps raster order while the top-k cap is
        // not binding, which is the case on the golden image.
        if boundary_only == 0 && missing == 0 {
            let got_order: Vec<(i64, i64)> = got
                .keypoints
                .iter()
                .map(|keypoint| raster_key(keypoint[0], keypoint[1]))
                .collect();
            let expected_order: Vec<(i64, i64)> = expected_keypoints
                .iter()
                .map(|keypoint| raster_key(keypoint[0], keypoint[1]))
                .collect();
            assert_eq!(
                got_order, expected_order,
                "keypoint order must match the reference raster order"
            );
        }
    }

    fn sorted_pairs(keypoints: &[[f32; 2]], scores: &[f32]) -> Vec<(f32, f32, f32)> {
        let mut pairs: Vec<(f32, f32, f32)> = keypoints
            .iter()
            .zip(scores.iter())
            .map(|(keypoint, score)| (keypoint[1], keypoint[0], *score))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        pairs
    }

    fn assert_resized_golden_match(
        got: &KeypointSet,
        expected_keypoints: &[[f32; 2]],
        expected_scores: &[f32],
        threshold: f32,
    ) {
        let got_pairs = sorted_pairs(&got.keypoints, &got.scores);
        let expected_pairs = sorted_pairs(expected_keypoints, expected_scores);

        let mut got_index = 0usize;
        let mut expected_index = 0usize;
        let mut matched = 0usize;
        let mut skipped_got = 0usize;
        let mut skipped_expected = 0usize;
        while got_index < got_pairs.len() && expected_index < expected_pairs.len() {
            let got_point = got_pairs[got_index];
            let expected_point = expected_pairs[expected_index];
            let dx = (got_point.1 - expected_point.1).abs();
            let dy = (got_point.0 - expected_point.0).abs();
            if dx <= COORDINATE_TOLERANCE && dy <= COORDINATE_TOLERANCE {
                assert!(
                    (got_point.2 - expected_point.2).abs() <= SCORE_TOLERANCE,
                    "score mismatch at ({}, {}): got {}, want {}",
                    got_point.1,
                    got_point.0,
                    got_point.2,
                    expected_point.2
                );
                matched += 1;
                got_index += 1;
                expected_index += 1;
            } else if got_point.2 <= threshold + SCORE_TOLERANCE {
                skipped_got += 1;
                got_index += 1;
            } else if expected_point.2 <= threshold + SCORE_TOLERANCE {
                skipped_expected += 1;
                expected_index += 1;
            } else {
                panic!(
                    "resized keypoint mismatch: got ({}, {}) want ({}, {})",
                    got_point.1, got_point.0, expected_point.1, expected_point.0
                );
            }
        }
        skipped_got += got_pairs.len() - got_index;
        skipped_expected += expected_pairs.len() - expected_index;
        assert!(
            skipped_got <= 16 && skipped_expected <= 16,
            "too many threshold-boundary differences: {skipped_got} extra, {skipped_expected} missing, {matched} matched"
        );
        assert!(
            matched >= expected_pairs.len().saturating_sub(16),
            "matched only {matched} of {} official resized keypoints",
            expected_pairs.len()
        );
        eprintln!(
            "SuperPoint resize golden: matched {matched}/{} (extra {skipped_got}, missing {skipped_expected})",
            expected_pairs.len()
        );
    }

    #[test]
    fn superpoint_matches_official_golden() {
        let weights = Path::new(WEIGHTS);
        let fixture = Path::new(GOLDEN);
        if !weights.is_file() || !fixture.is_file() {
            eprintln!(
                "skipping SuperPoint golden test: `{}` or `{}` is missing (run scripts/gen_superpoint_golden.py)",
                weights.display(),
                fixture.display()
            );
            return;
        }

        let detector =
            SuperPointDetector::load(weights, &Device::Cpu, DType::F32).expect("load SuperPoint");
        let mut npz =
            NpzReader::new(File::open(fixture).expect("open golden")).expect("read golden npz");

        let image = u8_tensor(npz.by_name("image").expect("golden image"), &Device::Cpu);
        let expected_keypoints = f32_array(&mut npz, "keypoints");
        let expected_scores = f32_array(&mut npz, "scores");
        let expected_descriptors = f32_array(&mut npz, "descriptors");

        let keypoint_rows = expected_keypoints.shape()[0];
        let expected_keypoints: Vec<[f32; 2]> = expected_keypoints
            .into_raw_vec_and_offset()
            .0
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect();
        let expected_scores: Vec<f32> = expected_scores.into_raw_vec_and_offset().0;
        let expected_descriptors: Vec<f32> = expected_descriptors.into_raw_vec_and_offset().0;
        assert_eq!(keypoint_rows, expected_scores.len());

        let options = KeypointOptions {
            max_keypoints: 4096,
            detection_threshold: 0.005,
            resize_long_side: None,
        };
        let got = detector
            .detect(&image, &options)
            .expect("SuperPoint detection");
        assert_eq!(got.descriptors.len(), got.len() * DESCRIPTOR_DIM);

        assert_golden_match(
            &got,
            &expected_keypoints,
            &expected_scores,
            Some(&expected_descriptors),
            0.005,
        );

        // Top-k truncation: the reference switches from raster order to
        // score-descending order once `max_num_keypoints` binds. The five
        // highest-scoring official keypoints are the expected selection.
        let mut order: Vec<usize> = (0..expected_scores.len()).collect();
        order.sort_by(|&a, &b| {
            expected_scores[b]
                .total_cmp(&expected_scores[a])
                .then_with(|| a.cmp(&b))
        });
        let expected_top = &order[..5.min(order.len())];
        let capped = detector
            .detect(
                &image,
                &KeypointOptions {
                    max_keypoints: expected_top.len(),
                    ..options
                },
            )
            .expect("SuperPoint top-k detection");
        assert_eq!(capped.len(), expected_top.len());
        for (got_keypoint, &expected_index) in capped.keypoints.iter().zip(expected_top.iter()) {
            let expected_keypoint = expected_keypoints[expected_index];
            assert!(
                (got_keypoint[0] - expected_keypoint[0]).abs() <= COORDINATE_TOLERANCE
                    && (got_keypoint[1] - expected_keypoint[1]).abs() <= COORDINATE_TOLERANCE,
                "top-k keypoint mismatch: got {got_keypoint:?}, want {expected_keypoint:?}"
            );
        }

        // The fixture also stores keypoints/scores for the long-side resize
        // path (`extract(image, resize=512)` -> original-input coordinates).
        let resized_keypoints = f32_array(&mut npz, "keypoints_resize512");
        let resized_scores = f32_array(&mut npz, "scores_resize512");
        let resized_keypoints: Vec<[f32; 2]> = resized_keypoints
            .into_raw_vec_and_offset()
            .0
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect();
        let resized_scores: Vec<f32> = resized_scores.into_raw_vec_and_offset().0;
        let resized_options = KeypointOptions {
            max_keypoints: 4096,
            detection_threshold: 0.005,
            resize_long_side: Some(512),
        };
        let got = detector
            .detect(&image, &resized_options)
            .expect("SuperPoint resized detection");
        assert_resized_golden_match(&got, &resized_keypoints, &resized_scores, 0.005);
    }

    #[test]
    fn superpoint_real_weights_128_cpu() {
        let weights = Path::new(WEIGHTS);
        if !weights.is_file() {
            eprintln!(
                "skipping SuperPoint real-weights test: `{}` is missing",
                weights.display()
            );
            return;
        }
        let detector =
            SuperPointDetector::load(weights, &Device::Cpu, DType::F32).expect("load SuperPoint");

        let rgb = synthetic_rgb(128, 128);
        let options = KeypointOptions {
            max_keypoints: 4096,
            detection_threshold: 0.005,
            resize_long_side: None,
        };
        let first = detector.detect(&rgb, &options).expect("detect");
        assert!(
            !first.is_empty(),
            "expected detections on the synthetic image"
        );
        assert_eq!(first.descriptor_dim, DESCRIPTOR_DIM);
        assert_eq!(first.scores.len(), first.len());
        assert_eq!(first.descriptors.len(), first.len() * DESCRIPTOR_DIM);
        for &[x, y] in &first.keypoints {
            assert!(x >= 0.0 && x <= 127.0, "x out of range: {x}");
            assert!(y >= 0.0 && y <= 127.0, "y out of range: {y}");
        }
        for (index, &score) in first.scores.iter().enumerate() {
            assert!(
                score > options.detection_threshold,
                "score {index} is below the threshold: {score}"
            );
        }
        for descriptor in first.descriptors.chunks_exact(DESCRIPTOR_DIM) {
            let norm: f32 = descriptor
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() <= 1e-3,
                "descriptor is not L2-normalized: {norm}"
            );
        }

        let second = detector.detect(&rgb, &options).expect("detect");
        assert_eq!(first, second, "detection must be deterministic");

        let capped = detector
            .detect(
                &rgb,
                &KeypointOptions {
                    max_keypoints: 5,
                    ..options
                },
            )
            .expect("detect with top-k cap");
        assert_eq!(capped.len(), 5, "top-k must truncate to max_keypoints");
        assert_eq!(capped.descriptors.len(), 5 * DESCRIPTOR_DIM);
    }

    #[test]
    fn superpoint_reports_missing_and_corrupt_weights() {
        let missing = Path::new("definitely-not-a-superpoint-checkpoint.safetensors");
        match SuperPointDetector::load(missing, &Device::Cpu, DType::F32) {
            Err(Lux3dError::MissingWeightFile { path }) => {
                assert_eq!(path, missing);
            }
            other => panic!("expected MissingWeightFile, got {other:?}"),
        }

        let corrupt = std::env::temp_dir().join(format!(
            "lux3d-superpoint-corrupt-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&corrupt, b"not a safetensors checkpoint").expect("write corrupt file");
        let result = SuperPointDetector::load(&corrupt, &Device::Cpu, DType::F32);
        let _ = std::fs::remove_file(&corrupt);
        match result {
            Err(Lux3dError::CanonicalWeightsValidation { message }) => {
                assert!(
                    message.contains("failed to open SuperPoint weights"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected CanonicalWeightsValidation, got {other:?}"),
        }
    }

    #[test]
    fn superpoint_simple_nms_keeps_local_maxima() {
        // Expected output from the reference `simple_nms` on CPU with
        // radius 1; the suppression chain runs exactly two passes.
        let scores: Vec<f32> = [
            0.10_f32, 0.05, 0.02, 0.05, 0.10, //
            0.05, 0.20, 0.03, 0.02, 0.05, //
            0.02, 0.03, 0.90, 0.30, 0.02, //
            0.05, 0.02, 0.30, 0.40, 0.05, //
            0.10, 0.05, 0.02, 0.05, 0.10,
        ]
        .to_vec();
        let expected: Vec<f32> = [
            0.10_f32, 0.0, 0.02, 0.0, 0.10, //
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.02, 0.0, 0.90, 0.0, 0.02, //
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.10, 0.0, 0.02, 0.0, 0.10,
        ]
        .to_vec();
        assert_eq!(simple_nms(&scores, 5, 5, 1), expected);
    }

    fn synthetic_rgb(width: usize, height: usize) -> Tensor {
        let mut values = Vec::with_capacity(3 * width * height);
        for channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let xf = x as f32;
                    let yf = y as f32;
                    let value = match channel {
                        0 => ((xf / 7.0 + yf / 11.0).sin() * 0.5 + 0.5) * 200.0 + 30.0,
                        1 => (((x + 2 * y) % 64) as f32) * 4.0,
                        _ => (((xf - 64.0).powi(2) + (yf - 64.0).powi(2)).sqrt() * 2.0).min(255.0),
                    };
                    values.push((value.clamp(0.0, 255.0).round()) / 255.0);
                }
            }
        }
        Tensor::from_vec(values, (3, height, width), &Device::Cpu).expect("synthetic image")
    }
}
