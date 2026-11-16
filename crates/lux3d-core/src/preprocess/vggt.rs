//! VGGT input preprocessing, mirroring `vggt.utils.load_fn` from the upstream
//! reference.
//!
//! # Preprocess modes ([`VggtPreprocessMode`])
//!
//! * [`VggtPreprocessMode::Crop`] (default) — the reference
//!   `load_and_preprocess_images(mode="crop")` recipe: resize so the width is
//!   518 (height keeps the aspect ratio and is rounded to a multiple of 14),
//!   then center-crop rows when the resized height exceeds 518, then
//!   center-pad to the batch maximum with white (1.0).
//! * [`VggtPreprocessMode::Pad`] — the reference `load_and_preprocess_images(mode="pad")`
//!   recipe: resize so the largest dimension is 518 while the other dimension
//!   keeps the aspect ratio rounded to a multiple of 14, then center-pad to a
//!   518x518 square with white (1.0). The padding happens *after* the `[0,1]`
//!   tensor conversion, exactly like upstream (`F.pad(..., value=1.0)`, with
//!   `pad_top = h_padding // 2`, `pad_left = w_padding // 2`). No pixels are
//!   discarded in this mode.
//!
//! Both modes composite RGBA input over opaque white before the RGB conversion
//! and normalize with the ImageNet mean/std afterwards.
//!
//! # Original-coordinate mapping
//!
//! Upstream `mode="pad"` does not return original coordinates, so this module
//! defines the mapping itself (see [`VggtOriginalCoords`]): the resized
//! original content occupies the centered `content_width x content_height`
//! rectangle inside the prepared 518x518 frame, and per-axis scales follow the
//! actual resize factors. Crop mode keeps the upstream uniform
//! `max(w, h) / img_size` ratio in [`crate::geometry`] helpers for bit-parity.
//!
//! # Sources
//!
//! [`VggtPreprocessStage::prepare_inputs_from_path`] accepts a directory of
//! frames, a video file (`.mp4`/`.mov`/`.avi`/`.mkv`) or a single image file
//! (`.png`/`.jpg`/`.jpeg`/`.webp`/`.bmp`/`.tif`/`.tiff`); a single image is a
//! one-frame source.
//!
//! Frame batching (chunked multi-forward inference) lives in
//! [`crate::runtime::VggtPipeline`] (`VggtRunOptions::batch_size`); the
//! preprocessing itself always materializes the full frame list.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{Device, Tensor};
use image::{Rgb, RgbImage};

use crate::{
    Result,
    contracts::{
        ModelFamily, PreprocessStage, RgbRange, SpatialSize, VggtPreparedBatch, VggtRawInput,
    },
    error::Lux3dError,
};

const TARGET_WIDTH: u32 = 518;
const PATCH_MULTIPLE: u32 = 14;

/// VGGT input preprocessing mode (upstream `load_and_preprocess_images`
/// `mode="crop"` / `mode="pad"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VggtPreprocessMode {
    /// Resize to width 518, center-crop rows taller than 518, then center-pad
    /// to the batch maximum with white (upstream default).
    #[default]
    Crop,
    /// Resize so the largest dimension is 518 (aspect preserved, shorter side
    /// rounded to a multiple of 14) and center-pad to 518x518 with white. No
    /// pixels are discarded.
    Pad,
}

/// Per-frame mapping from the prepared (padded) VGGT frame back to the original
/// image, for both preprocess modes (see [`VggtPreprocessMode`]).
///
/// The resized original content occupies the rectangle
/// `[offset_x, offset_x + content_width) x [offset_y, offset_y + content_height)`
/// inside the prepared frame. A prepared-frame pixel `(x_p, y_p)` inside the
/// content rectangle maps back to the original pixel `(u, v)` with
/// `u = (x_p - offset_x) * width / content_width` and
/// `v = (y_p - offset_y) * height / content_height` (exact for
/// [`VggtPreprocessMode::Pad`], and for crop frames whose height was not
/// center-cropped).
///
/// The reference COLMAP rescale instead uses the uniform ratio
/// `max(width, height) / reconstruction_size` (see
/// `rename_colmap_recons_and_rescale_camera`); both mappings are exposed —
/// [`VggtOriginalCoords::resize_ratio`] stays the crop-formula compatibility
/// helper, while [`VggtOriginalCoords::axis_scales_to_original`] gives the
/// exact per-axis factors and matches what the crop recipe actually does per
/// axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VggtOriginalCoords {
    /// Original image width in pixels.
    pub width: u32,
    /// Original image height in pixels.
    pub height: u32,
    /// Left edge of the resized original content inside the prepared frame.
    /// Non-zero only when white padding was applied (pad mode, or crop mode
    /// batch padding for narrower frames).
    pub offset_x: u32,
    /// Top edge of the resized original content inside the prepared frame.
    /// Combines the height center-crop offset (`start_y`) and the white pad.
    pub offset_y: u32,
    /// Width of the resized original content in prepared-frame pixels
    /// (`target_width` in crop mode; the `/14`-rounded resized width in pad
    /// mode).
    pub content_width: u32,
    /// Height of the resized original content in prepared-frame pixels (the
    /// crop-mode frame height before batch padding; the `/14`-rounded resized
    /// height in pad mode).
    pub content_height: u32,
    /// Preprocess mode that produced this frame. The upstream pad recipe does
    /// not report original coordinates, so this discriminator lets the
    /// geometry helpers pick the exact per-axis mapping (pad) or the reference
    /// uniform ratio (crop).
    pub mode: VggtPreprocessMode,
}

impl VggtOriginalCoords {
    /// Reference `rename_colmap_recons_and_rescale_camera` ratio:
    /// `max(real_width, real_height) / img_size`.
    ///
    /// This is the *crop-formula* compatibility helper: upstream applies it to
    /// both axes even in crop mode where the vertical scale actually differs.
    /// For the exact per-axis factors use [`VggtOriginalCoords::axis_scales_to_original`].
    pub fn resize_ratio(&self, img_size: f64) -> f64 {
        if img_size <= 0.0 {
            return 0.0;
        }
        f64::from(self.width.max(self.height)) / img_size
    }

    /// Exact per-axis prepared→original scale factors
    /// `(width / content_width, height / content_height)`.
    ///
    /// Exact for pad mode (no cropping ever happens) and for crop frames whose
    /// resized height was not center-cropped; crop frames with a height crop
    /// keep the uniform reference ratio instead (see [`Self::resize_ratio`]).
    /// A degenerate zero content dimension yields `0.0` on that axis rather
    /// than a division by zero.
    pub fn axis_scales_to_original(&self) -> (f64, f64) {
        let scale_x = if self.content_width == 0 {
            0.0
        } else {
            f64::from(self.width) / f64::from(self.content_width)
        };
        let scale_y = if self.content_height == 0 {
            0.0
        } else {
            f64::from(self.height) / f64::from(self.content_height)
        };
        (scale_x, scale_y)
    }

    /// Top-left corner of the original content inside the prepared frame.
    pub fn top_left(&self) -> (f64, f64) {
        (f64::from(self.offset_x), f64::from(self.offset_y))
    }

    /// Principal point of the original image: `(real_width / 2, real_height / 2)`.
    pub fn original_principal_point(&self) -> (f64, f64) {
        (f64::from(self.width) / 2.0, f64::from(self.height) / 2.0)
    }
}

#[derive(Debug, Clone)]
pub struct VggtPreparedInputs {
    pub interval: usize,
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    /// Preprocess mode the frames were prepared with (mirrored in every
    /// [`VggtOriginalCoords::mode`]).
    pub preprocess_mode: VggtPreprocessMode,
    /// `[S, 3, H, W]` f32, range `[0, 1]`.
    pub rgb_frames: Tensor,
    /// `[1, S, 3, H, W]` f32, ImageNet-normalized.
    pub normalized_frames: Tensor,
    /// One entry per frame: original resolution plus the crop/pad offset that
    /// locates the resized original content inside the prepared frame.
    pub original_coords: Vec<VggtOriginalCoords>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VggtPreprocessStage {
    target_width: u32,
    patch_multiple: u32,
}

impl Default for VggtPreprocessStage {
    fn default() -> Self {
        Self {
            target_width: TARGET_WIDTH,
            patch_multiple: PATCH_MULTIPLE,
        }
    }
}

impl VggtPreprocessStage {
    /// Load a directory of RGB frames, a video file (`.mp4`/`.mov`/`.avi`/
    /// `.mkv`) or a single image file, then apply the VGGT `mode="crop"`
    /// preprocessing recipe from `vggt.utils.load_fn`.
    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<VggtPreparedInputs> {
        self.prepare_inputs_from_path_with_mode(
            source,
            interval,
            VggtPreprocessMode::default(),
            device,
        )
    }

    /// [`Self::prepare_inputs_from_path`] with an explicit preprocess mode
    /// (see [`VggtPreprocessMode`] and the module docs for the exact recipes).
    pub fn prepare_inputs_from_path_with_mode(
        &self,
        source: &Path,
        interval: Option<usize>,
        mode: VggtPreprocessMode,
        device: &Device,
    ) -> Result<VggtPreparedInputs> {
        let interval = interval.unwrap_or_else(|| default_interval_for(source));
        if interval == 0 {
            return Err(Lux3dError::InvalidInput("interval must be at least 1"));
        }

        let sampled = load_sampled_frames(source, interval)?;
        let sampled_paths = &sampled.paths;

        let first = load_rgb_frame(&sampled_paths[0])?;
        let original_size = SpatialSize::new(first.width(), first.height());

        let mut shapes = Vec::with_capacity(sampled_paths.len());
        let mut frames = Vec::with_capacity(sampled_paths.len());
        let mut original_dims = Vec::with_capacity(sampled_paths.len());
        // Content top-left inside the per-frame tensor *before* the batch
        // maximum white padding: `(0, crop_start_y)` in crop mode, the
        // centered white-pad offsets in pad mode.
        let mut content_offsets = Vec::with_capacity(sampled_paths.len());
        // Size of the resized original content (equals `shape` in crop mode;
        // smaller than the 518x518 frame in pad mode).
        let mut content_sizes = Vec::with_capacity(sampled_paths.len());
        for path in sampled_paths {
            let image = load_rgb_frame(path)?;
            original_dims.push(image.dimensions());
            let (shape, offset, content, chw) = match mode {
                VggtPreprocessMode::Crop => {
                    let (shape, crop_start_y, chw) = self.resize_and_crop_frame(&image)?;
                    (shape, (0, crop_start_y), shape, chw)
                }
                VggtPreprocessMode::Pad => self.resize_and_pad_frame(&image)?,
            };
            shapes.push(shape);
            content_offsets.push(offset);
            content_sizes.push(content);
            frames.push(chw);
        }

        let max_height = shapes.iter().map(|shape| shape.height).max().unwrap_or(0);
        let max_width = shapes.iter().map(|shape| shape.width).max().unwrap_or(0);
        if max_height == 0 || max_width == 0 {
            return Err(Lux3dError::InvalidInput(
                "VGGT preprocessing produced an empty frame",
            ));
        }

        // Per-frame top-left of the resized original content inside the padded
        // frame: the per-mode content offset plus the batch white pad.
        let original_coords = shapes
            .iter()
            .zip(&original_dims)
            .zip(&content_offsets)
            .zip(&content_sizes)
            .map(
                |(((shape, (width, height)), &(offset_x, offset_y)), content)| VggtOriginalCoords {
                    width: *width,
                    height: *height,
                    offset_x: offset_x + (max_width - shape.width) / 2,
                    offset_y: offset_y + (max_height - shape.height) / 2,
                    content_width: content.width,
                    content_height: content.height,
                    mode,
                },
            )
            .collect();

        let plane = (max_height as usize) * (max_width as usize);
        let mut flattened = Vec::with_capacity(frames.len() * 3 * plane);
        for (frame, shape) in frames.into_iter().zip(shapes) {
            flattened.extend_from_slice(&pad_chw_to_max(&frame, shape, max_height, max_width));
        }

        let frame_count = sampled_paths.len();
        let rgb_frames = Tensor::from_vec(
            flattened,
            (frame_count, 3, max_height as usize, max_width as usize),
            device,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to materialize VGGT RGB tensor: {source}"),
        })?;

        let mean = Tensor::from_vec(vec![0.485_f32, 0.456, 0.406], (1, 1, 3, 1, 1), device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize VGGT mean tensor: {source}"),
            })?;
        let std = Tensor::from_vec(vec![0.229_f32, 0.224, 0.225], (1, 1, 3, 1, 1), device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize VGGT std tensor: {source}"),
            })?;
        let normalized_frames = rgb_frames
            .unsqueeze(0)
            .and_then(|tensor| tensor.broadcast_sub(&mean))
            .and_then(|tensor| tensor.broadcast_div(&std))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to normalise VGGT RGB frames: {source}"),
            })?;

        Ok(VggtPreparedInputs {
            interval,
            original_size,
            target_size: SpatialSize::new(max_width, max_height),
            preprocess_mode: mode,
            rgb_frames,
            normalized_frames,
            original_coords,
        })
    }

    /// Reference crop-mode sizing *before* the center crop: width is fixed to
    /// `target_width`, height keeps the aspect ratio and is rounded to
    /// `patch_multiple` with Python's ties-to-even rounding. The prepared frame
    /// is then center-cropped to `target_width` rows, so its final size never
    /// exceeds `target_width` x `target_width`.
    pub fn target_size_for(&self, original: SpatialSize, target_width: u32) -> Result<SpatialSize> {
        scaled_crop_size(
            original.width,
            original.height,
            target_width,
            self.patch_multiple,
        )
    }

    /// Prepared frame size for `mode` (see [`Self::target_size_for`] for the
    /// crop formula). Pad mode returns the `/14`-rounded resized content size;
    /// the frame is additionally white-padded to a `target_width` square
    /// afterwards.
    pub fn target_size_for_mode(
        &self,
        original: SpatialSize,
        target_width: u32,
        mode: VggtPreprocessMode,
    ) -> Result<SpatialSize> {
        match mode {
            VggtPreprocessMode::Crop => self.target_size_for(original, target_width),
            VggtPreprocessMode::Pad => {
                let (width, height) = scaled_pad_size(
                    original.width,
                    original.height,
                    target_width,
                    self.patch_multiple,
                )?;
                Ok(SpatialSize::new(width, height))
            }
        }
    }

    /// Returns `(frame size, crop offset, CHW pixels)`. The crop offset is the
    /// row index inside the resized image from which the kept rows start
    /// (non-zero only when the resized height exceeds `target_width`); it is
    /// needed to map prepared pixels back to the original image.
    fn resize_and_crop_frame(&self, image: &RgbImage) -> Result<(SpatialSize, u32, Vec<f32>)> {
        let (width, height) = image.dimensions();
        let target = scaled_crop_size(width, height, self.target_width, self.patch_multiple)?;
        let resized = resize_bicubic_pillow(image, target.width, target.height);

        // Crop mode keeps the middle `target_width` rows when the resized
        // height exceeds the target width; width already equals target_width.
        let out_height = resized.height().min(self.target_width);
        let start_y = (resized.height() - out_height) / 2;
        let plane = (out_height as usize) * (self.target_width as usize);
        let mut chw = vec![0.0_f32; 3 * plane];

        for y in 0..out_height {
            let source_y = start_y + y;
            for x in 0..self.target_width {
                let pixel = resized.get_pixel(x, source_y);
                for channel in 0..3 {
                    chw[channel * plane
                        + (y as usize) * (self.target_width as usize)
                        + x as usize] = f32::from(pixel[channel]) / 255.0;
                }
            }
        }

        Ok((
            SpatialSize::new(self.target_width, out_height),
            start_y,
            chw,
        ))
    }

    /// Returns `(frame size, content top-left, content size, CHW pixels)` for
    /// pad mode. Mirrors `load_and_preprocess_images(mode="pad")`: resize so
    /// the largest dimension is `target_width` (the other dimension keeps the
    /// aspect ratio, rounded to `patch_multiple`), convert to `[0,1]`, *then*
    /// center-pad to a `target_width` square with white (1.0) —
    /// `pad_top = h_padding // 2`, `pad_left = w_padding // 2`, remainder on
    /// the bottom/right. No rows or columns are cropped.
    fn resize_and_pad_frame(
        &self,
        image: &RgbImage,
    ) -> Result<(SpatialSize, (u32, u32), SpatialSize, Vec<f32>)> {
        let (width, height) = image.dimensions();
        let (new_width, new_height) =
            scaled_pad_size(width, height, self.target_width, self.patch_multiple)?;
        let resized = resize_bicubic_pillow(image, new_width, new_height);

        let content = SpatialSize::new(new_width, new_height);
        let plane = (new_height as usize) * (new_width as usize);
        let mut chw = vec![0.0_f32; 3 * plane];
        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = resized.get_pixel(x, y);
                for channel in 0..3 {
                    chw[channel * plane + (y as usize) * (new_width as usize) + x as usize] =
                        f32::from(pixel[channel]) / 255.0;
                }
            }
        }

        // White-pad the [0,1] tensor to the square target, centered exactly
        // like the reference `F.pad` call.
        let padded = pad_chw_to_max(&chw, content, self.target_width, self.target_width);
        let offset_x = (self.target_width - new_width) / 2;
        let offset_y = (self.target_width - new_height) / 2;
        Ok((
            SpatialSize::new(self.target_width, self.target_width),
            (offset_x, offset_y),
            content,
            padded,
        ))
    }
}

impl PreprocessStage<VggtRawInput, VggtPreparedBatch> for VggtPreprocessStage {
    /// Contract-level stage descriptor. `VggtRawInput` carries no preprocess
    /// mode, so this follows the upstream default (crop formula); the
    /// mode-aware run surface is
    /// [`Self::prepare_inputs_from_path_with_mode`].
    fn preprocess(&self, input: VggtRawInput) -> Result<VggtPreparedBatch> {
        if input.interval == 0 {
            return Err(Lux3dError::InvalidInput("interval must be at least 1"));
        }
        let target_size = self.target_size_for(input.original_size, self.target_width)?;

        Ok(VggtPreparedBatch {
            family: ModelFamily::Vggt,
            source: input.source,
            sampled_frames: input.frame_count.div_ceil(input.interval),
            original_size: input.original_size,
            target_size,
            interval: input.interval,
            pixel_range: RgbRange::ZeroToOne,
        })
    }
}

fn scaled_crop_size(
    width: u32,
    height: u32,
    target_width: u32,
    patch_multiple: u32,
) -> Result<SpatialSize> {
    if width == 0 || height == 0 {
        return Err(Lux3dError::InvalidInput("original size must be non-zero"));
    }
    if target_width == 0 {
        return Err(Lux3dError::InvalidInput("target width must be non-zero"));
    }
    if patch_multiple == 0 {
        return Err(Lux3dError::InvalidInput("patch multiple must be non-zero"));
    }

    let scale = f64::from(target_width) / f64::from(width);
    let multiple = f64::from(patch_multiple);
    let steps = (f64::from(height) * scale / multiple).round_ties_even();
    if steps < 1.0 {
        return Err(Lux3dError::InvalidInput(
            "scaled target height must be non-zero",
        ));
    }

    Ok(SpatialSize::new(
        target_width,
        (steps as u32) * patch_multiple,
    ))
}

/// Reference pad-mode sizing from `load_and_preprocess_images(mode="pad")`:
/// the largest dimension becomes `target`, the other keeps the aspect ratio
/// and is rounded to `patch_multiple` with Python's ties-to-even rounding.
/// Returns `(new_width, new_height)`.
fn scaled_pad_size(
    width: u32,
    height: u32,
    target: u32,
    patch_multiple: u32,
) -> Result<(u32, u32)> {
    if width == 0 || height == 0 {
        return Err(Lux3dError::InvalidInput("original size must be non-zero"));
    }
    if target == 0 {
        return Err(Lux3dError::InvalidInput("target size must be non-zero"));
    }
    if patch_multiple == 0 {
        return Err(Lux3dError::InvalidInput("patch multiple must be non-zero"));
    }

    // `width >= height` matches the reference branch order exactly.
    let (long, short) = if width >= height {
        (width, height)
    } else {
        (height, width)
    };
    let scale = f64::from(target) / f64::from(long);
    let multiple = f64::from(patch_multiple);
    let steps = (f64::from(short) * scale / multiple).round_ties_even();
    if steps < 1.0 {
        return Err(Lux3dError::InvalidInput(
            "scaled target size must be non-zero",
        ));
    }
    let new_short = (steps as u32) * patch_multiple;

    if width >= height {
        Ok((target, new_short))
    } else {
        Ok((new_short, target))
    }
}

/// Pillow's 8-bit resampler uses 22 fixed-point bits:
/// `PRECISION_BITS = 32 - 8 - 2` (`src/libImaging/Resample.c`).
const PILLOW_PRECISION_BITS: i64 = 22;
const PILLOW_PRECISION_SCALE: f64 = (1u64 << PILLOW_PRECISION_BITS) as f64;

/// Pillow's BICUBIC kernel (`a = -0.5` Catmull-Rom). Algebraically identical
/// to `image::imageops::FilterType::CatmullRom`, but Pillow evaluates it in
/// f64 and quantizes the coefficients to 22-bit fixed point.
fn pillow_bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct PillowCoefficients {
    starts: Vec<usize>,
    counts: Vec<usize>,
    weights: Vec<i32>,
    stride: usize,
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc` from Pillow's `Resample.c`.
fn pillow_coefficients(input_size: usize, output_size: usize) -> PillowCoefficients {
    let scale = input_size as f64 / output_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let stride = (support.ceil() as usize) * 2 + 1;
    let inverse_filter_scale = 1.0 / filter_scale;

    let mut starts = Vec::with_capacity(output_size);
    let mut counts = Vec::with_capacity(output_size);
    let mut weights = vec![0_i32; output_size * stride];
    let mut scratch = vec![0.0_f64; stride];

    for out in 0..output_size {
        let center = (out as f64 + 0.5) * scale;
        let mut start = (center - support + 0.5) as i64;
        if start < 0 {
            start = 0;
        }
        let mut end = (center + support + 0.5) as i64;
        if end > input_size as i64 {
            end = input_size as i64;
        }
        let count = (end - start) as usize;
        debug_assert!(count <= stride);

        let mut sum = 0.0_f64;
        for x in 0..count {
            let weight =
                pillow_bicubic((x as f64 + start as f64 - center + 0.5) * inverse_filter_scale);
            scratch[x] = weight;
            sum += weight;
        }
        if sum != 0.0 {
            for value in scratch.iter_mut().take(count) {
                *value /= sum;
            }
        }
        for x in 0..count {
            let scaled = scratch[x] * PILLOW_PRECISION_SCALE;
            weights[out * stride + x] = if scratch[x] < 0.0 {
                (scaled - 0.5) as i32
            } else {
                (scaled + 0.5) as i32
            };
        }

        starts.push(start as usize);
        counts.push(count);
    }

    PillowCoefficients {
        starts,
        counts,
        weights,
        stride,
    }
}

#[inline]
fn pillow_clip8(accumulator: i64) -> u8 {
    (accumulator >> PILLOW_PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow-compatible 8-bit BICUBIC resize: horizontal pass into an 8-bit
/// intermediate, then a vertical pass, exactly like `ImagingResampleInner`.
///
/// The `image` crate's `CatmullRom` filter uses the same kernel but a
/// single-rounding f32 pipeline, which differs from Pillow by up to one 8-bit
/// level (3.9e-3 in `[0, 1]`). That exceeds the VGGT preprocess parity budget
/// (1e-5), so this port reproduces Pillow's integer arithmetic instead.
fn resize_bicubic_pillow(image: &RgbImage, out_width: u32, out_height: u32) -> RgbImage {
    let (in_width, in_height) = image.dimensions();
    if in_width == out_width && in_height == out_height {
        return image.clone();
    }

    let source = image.as_raw();
    let out_width_usize = out_width as usize;
    let out_height_usize = out_height as usize;

    // Horizontal pass: every input row, output width.
    let horizontal = pillow_coefficients(in_width as usize, out_width_usize);
    let mut intermediate = vec![0_u8; (in_height as usize) * out_width_usize * 3];
    for y in 0..in_height as usize {
        let row = y * in_width as usize * 3;
        for x_out in 0..out_width_usize {
            let start = horizontal.starts[x_out];
            let count = horizontal.counts[x_out];
            let mut accumulator = [1_i64 << (PILLOW_PRECISION_BITS - 1); 3];
            for x in 0..count {
                let weight = i64::from(horizontal.weights[x_out * horizontal.stride + x]);
                let index = row + (start + x) * 3;
                for channel in 0..3 {
                    accumulator[channel] += i64::from(source[index + channel]) * weight;
                }
            }
            let target = (y * out_width_usize + x_out) * 3;
            for channel in 0..3 {
                intermediate[target + channel] = pillow_clip8(accumulator[channel]);
            }
        }
    }

    // Vertical pass: output width from the intermediate image.
    let vertical = pillow_coefficients(in_height as usize, out_height_usize);
    let mut output = vec![0_u8; out_height_usize * out_width_usize * 3];
    for y_out in 0..out_height_usize {
        let start = vertical.starts[y_out];
        let count = vertical.counts[y_out];
        for x in 0..out_width_usize {
            let mut accumulator = [1_i64 << (PILLOW_PRECISION_BITS - 1); 3];
            for y in 0..count {
                let weight = i64::from(vertical.weights[y_out * vertical.stride + y]);
                let index = ((start + y) * out_width_usize + x) * 3;
                for channel in 0..3 {
                    accumulator[channel] += i64::from(intermediate[index + channel]) * weight;
                }
            }
            let target = (y_out * out_width_usize + x) * 3;
            for channel in 0..3 {
                output[target + channel] = pillow_clip8(accumulator[channel]);
            }
        }
    }

    RgbImage::from_raw(out_width, out_height, output).expect("resized buffer dimensions")
}

/// PIL parity: an RGBA image is alpha-composited over opaque white before the
/// RGB conversion. Frames without alpha are converted directly.
fn load_rgb_frame(path: &Path) -> Result<RgbImage> {
    let dynamic = image::open(path)
        .map_err(|_| Lux3dError::InvalidInput("failed to open VGGT input image"))?;

    if !dynamic.color().has_alpha() {
        return Ok(dynamic.to_rgb8());
    }

    let rgba = dynamic.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut composited = RgbImage::new(width, height);
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let alpha = f32::from(pixel[3]) / 255.0;
        let inverse = 1.0 - alpha;
        let blend =
            |value: u8| -> u8 { (f32::from(value) * alpha + 255.0 * inverse).round() as u8 };
        composited.put_pixel(
            x,
            y,
            Rgb([blend(pixel[0]), blend(pixel[1]), blend(pixel[2])]),
        );
    }
    Ok(composited)
}

/// Center-pad a CHW frame to `max_height` x `max_width` with white (1.0),
/// matching the reference `pad_top = pad_h / 2` split.
fn pad_chw_to_max(frame: &[f32], shape: SpatialSize, max_height: u32, max_width: u32) -> Vec<f32> {
    let height = shape.height;
    let width = shape.width;
    if height == max_height && width == max_width {
        return frame.to_vec();
    }

    let pad_top = (max_height - height) / 2;
    let pad_left = (max_width - width) / 2;
    let max_plane = (max_height as usize) * (max_width as usize);
    let plane = (height as usize) * (width as usize);
    let mut padded = vec![1.0_f32; 3 * max_plane];

    for channel in 0..3 {
        for y in 0..height as usize {
            let source_start = channel * plane + y * (width as usize);
            let destination_start = channel * max_plane
                + (pad_top as usize + y) * (max_width as usize)
                + pad_left as usize;
            padded[destination_start..destination_start + width as usize]
                .copy_from_slice(&frame[source_start..source_start + width as usize]);
        }
    }

    padded
}

fn is_video_extension(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "mp4" | "mov" | "avi" | "mkv"
    )
}

/// Extensions accepted for a *direct single-image* VGGT source. Directory
/// listing keeps its own (narrower) filter so existing behavior is unchanged.
fn is_image_extension(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff"
    )
}

fn default_interval_for(source: &Path) -> usize {
    match source.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if is_video_extension(ext) => 10,
        _ => 1,
    }
}

#[derive(Debug)]
struct TempFrameDir {
    path: PathBuf,
}

impl TempFrameDir {
    fn new() -> Result<Self> {
        let mut path = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!("lux3d-vggt-frames-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.clone(),
            source,
        })?;
        Ok(Self { path })
    }
}

impl Drop for TempFrameDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[derive(Debug)]
struct SampledVggtFrames {
    paths: Vec<PathBuf>,
    _temp_dir: Option<TempFrameDir>,
}

fn load_sampled_frames(source: &Path, interval: usize) -> Result<SampledVggtFrames> {
    if source.is_dir() {
        let entries = collect_rgb_entries(source)?;
        if entries.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "VGGT input directory did not contain any RGB frames",
            ));
        }
        return Ok(SampledVggtFrames {
            paths: entries.into_iter().step_by(interval).collect(),
            _temp_dir: None,
        });
    }

    let extension = source
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);

    if extension.as_deref().is_some_and(is_video_extension) {
        return extract_video_frames(source, interval);
    }

    // A single image file is a one-frame source; `interval` does not apply.
    if source.is_file() && extension.as_deref().is_some_and(is_image_extension) {
        return Ok(SampledVggtFrames {
            paths: vec![source.to_path_buf()],
            _temp_dir: None,
        });
    }

    Err(Lux3dError::InvalidInput(
        "VGGT input source must be a directory, a video file (.mp4/.mov/.avi/.mkv) or an image file (.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff)",
    ))
}

fn collect_rgb_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = fs::read_dir(dir)
        .map_err(|source| Lux3dError::CanonicalManifestIo {
            path: dir.to_path_buf(),
            source,
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    // Mirror the reference's Python `sorted(...)`: lexicographic by lowercased
    // file name, then by full path.
    entries.sort_by(|lhs, rhs| {
        let lhs_name = lhs
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| lhs.to_string_lossy().to_ascii_lowercase());
        let rhs_name = rhs
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| rhs.to_string_lossy().to_ascii_lowercase());
        lhs_name.cmp(&rhs_name).then_with(|| lhs.cmp(rhs))
    });
    Ok(entries)
}

fn extract_video_frames(source: &Path, interval: usize) -> Result<SampledVggtFrames> {
    if !source.is_file() {
        return Err(Lux3dError::InvalidInput(
            "VGGT video input file was not found",
        ));
    }

    let temp_dir = TempFrameDir::new()?;
    let output_pattern = temp_dir.path.join("%06d.png");
    let filter = format!("select=not(mod(n\\,{interval}))");
    let output = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(source)
        .arg("-vf")
        .arg(filter)
        .arg("-vsync")
        .arg("vfr")
        .arg(output_pattern.as_os_str())
        .output()
        .map_err(|_| Lux3dError::InvalidInput("failed to start ffmpeg for VGGT video input"))?;

    if !output.status.success() {
        return Err(Lux3dError::InvalidInput(
            "failed to extract VGGT video frames with ffmpeg",
        ));
    }

    let paths = collect_rgb_entries(&temp_dir.path)?;
    if paths.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "VGGT video input did not yield any RGB frames",
        ));
    }

    Ok(SampledVggtFrames {
        paths,
        _temp_dir: Some(temp_dir),
    })
}

#[cfg(test)]
mod tests {
    use image::{Rgba, RgbaImage};

    use super::*;

    #[test]
    fn composites_rgba_over_white() {
        let mut rgba = RgbaImage::new(2, 1);
        rgba.put_pixel(0, 0, Rgba([0, 0, 0, 128]));
        rgba.put_pixel(1, 0, Rgba([10, 20, 30, 255]));

        let mut path = std::env::temp_dir();
        path.push(format!(
            "lux3d-vggt-rgba-{}-{}.png",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        rgba.save(&path).expect("save rgba fixture");

        let rgb = load_rgb_frame(&path).expect("composite rgba");
        let _ = fs::remove_file(&path);

        // 0 * (128/255) + 255 * (1 - 128/255) == 127.
        assert_eq!(*rgb.get_pixel(0, 0), Rgb([127, 127, 127]));
        assert_eq!(*rgb.get_pixel(1, 0), Rgb([10, 20, 30]));
    }

    #[test]
    fn recognizes_all_server_video_extensions() {
        for extension in ["mp4", "MOV", "avi", "MkV"] {
            assert!(is_video_extension(extension), "expected video: {extension}");
            let path = PathBuf::from(format!("clip.{extension}"));
            assert_eq!(10, default_interval_for(&path));
        }
        assert!(!is_video_extension("png"));
        assert!(!is_video_extension("zip"));
        assert_eq!(1, default_interval_for(Path::new("frames")));
    }

    /// Per-frame original coords must record the center-crop offset (frame
    /// taller than 518) and the white-pad offset (frame shorter than 518).
    #[test]
    fn original_coords_record_crop_and_pad_offsets() {
        let dir = std::env::temp_dir().join(format!(
            "lux3d-vggt-coords-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create frame dir");

        // 1024x768 -> resized 518x392 (no crop), then padded to 518 rows.
        RgbImage::from_fn(1024, 768, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 7u8])
        })
        .save(dir.join("a.png"))
        .expect("save frame a");
        // 1024x2048 -> resized 518x1036, center-cropped to 518 rows
        // (start_y = (1036 - 518) / 2 = 259).
        RgbImage::from_fn(1024, 2048, |x, y| {
            Rgb([3u8, (x % 256) as u8, (y % 256) as u8])
        })
        .save(dir.join("b.png"))
        .expect("save frame b");

        let stage = VggtPreprocessStage::default();
        let prepared = stage
            .prepare_inputs_from_path(&dir, Some(1), &Device::Cpu)
            .expect("prepare vggt inputs");
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(prepared.original_size, SpatialSize::new(1024, 768));
        assert_eq!(prepared.target_size, SpatialSize::new(518, 518));
        assert_eq!(prepared.preprocess_mode, VggtPreprocessMode::Crop);
        assert_eq!(prepared.rgb_frames.dims(), &[2, 3, 518, 518]);
        assert_eq!(prepared.original_coords.len(), 2);
        assert_eq!(
            prepared.original_coords[0],
            VggtOriginalCoords {
                width: 1024,
                height: 768,
                offset_x: 0,
                offset_y: 63, // (518 - 392) / 2 white-pad rows
                content_width: 518,
                content_height: 392,
                mode: VggtPreprocessMode::Crop,
            }
        );
        assert_eq!(
            prepared.original_coords[1],
            VggtOriginalCoords {
                width: 1024,
                height: 2048,
                offset_x: 0,
                offset_y: 259, // height center-crop start row
                content_width: 518,
                content_height: 518,
                mode: VggtPreprocessMode::Crop,
            }
        );

        // Mapping prepared pixels back: the frame-a content rectangle starts
        // at row 63, so prepared row 63 is original row 0 and prepared row 454
        // is original row 391 * 768 / 392.
        let coords = &prepared.original_coords[0];
        let v = (454.0 - f64::from(coords.offset_y)) * f64::from(coords.height)
            / f64::from(coords.content_height);
        assert!((v - 391.0 * 768.0 / 392.0).abs() < 1e-9);
    }

    #[test]
    fn scaled_pad_size_matches_reference_formula() {
        // 1024x768: long side 518, short side round(768 * 518/1024 / 14) * 14
        // == round(27.75) * 14 == 392.
        assert_eq!(
            scaled_pad_size(1024, 768, 518, 14).expect("landscape"),
            (518, 392)
        );
        assert_eq!(
            scaled_pad_size(768, 1024, 518, 14).expect("portrait"),
            (392, 518)
        );
        assert_eq!(
            scaled_pad_size(512, 512, 518, 14).expect("square"),
            (518, 518)
        );
        // Ties-to-even: 4 * (518/8) / 14 == 18.5 -> 18 -> 252.
        assert_eq!(scaled_pad_size(8, 4, 518, 14).expect("wide"), (518, 252));
        assert_eq!(scaled_pad_size(4, 8, 518, 14).expect("tall"), (252, 518));
        // `width == height` takes the width-first reference branch.
        assert_eq!(scaled_pad_size(4, 4, 518, 14).expect("equal"), (518, 518));

        assert!(scaled_pad_size(0, 8, 518, 14).is_err());
        assert!(scaled_pad_size(8, 0, 518, 14).is_err());
        assert!(scaled_pad_size(8, 8, 0, 14).is_err());
    }

    #[test]
    fn target_size_for_mode_matches_crop_and_pad_formulas() {
        let stage = VggtPreprocessStage::default();
        let landscape = SpatialSize::new(1024, 768);
        assert_eq!(
            stage
                .target_size_for_mode(landscape, 518, VggtPreprocessMode::Crop)
                .expect("crop target"),
            SpatialSize::new(518, 392)
        );
        assert_eq!(
            stage
                .target_size_for_mode(landscape, 518, VggtPreprocessMode::Pad)
                .expect("pad target"),
            SpatialSize::new(518, 392)
        );
        assert_eq!(
            stage
                .target_size_for_mode(SpatialSize::new(480, 640), 518, VggtPreprocessMode::Pad)
                .expect("pad portrait target"),
            SpatialSize::new(392, 518)
        );
    }

    /// Pad-mode golden values for 1024x768: content 518x392, centered with 63
    /// white rows top/bottom, no white columns.
    #[test]
    fn pad_mode_1024x768_centers_content_in_white_frame() {
        let dir = std::env::temp_dir().join(format!(
            "lux3d-vggt-pad-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create frame dir");
        RgbImage::from_fn(1024, 768, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 7u8])
        })
        .save(dir.join("a.png"))
        .expect("save frame");

        let stage = VggtPreprocessStage::default();
        let prepared = stage
            .prepare_inputs_from_path_with_mode(
                &dir,
                Some(1),
                VggtPreprocessMode::Pad,
                &Device::Cpu,
            )
            .expect("prepare pad inputs");
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(prepared.original_size, SpatialSize::new(1024, 768));
        assert_eq!(prepared.preprocess_mode, VggtPreprocessMode::Pad);
        assert_eq!(prepared.target_size, SpatialSize::new(518, 518));
        assert_eq!(prepared.rgb_frames.dims(), &[1, 3, 518, 518]);
        assert_eq!(prepared.normalized_frames.dims(), &[1, 1, 3, 518, 518]);
        assert_eq!(
            prepared.original_coords[0],
            VggtOriginalCoords {
                width: 1024,
                height: 768,
                offset_x: 0,
                offset_y: 63,
                content_width: 518,
                content_height: 392,
                mode: VggtPreprocessMode::Pad,
            }
        );

        // White rows exactly on [0, 62] and [455, 517]; content rows in between.
        let channel0 = prepared
            .rgb_frames
            .get(0)
            .expect("frame")
            .get(0)
            .expect("c0");
        for y in [0usize, 62, 455, 517] {
            let row = channel0.get(y).expect("row").to_vec1::<f32>().expect("vec");
            assert!(
                row.iter().all(|value| (*value - 1.0).abs() <= 1e-6),
                "pad row {y} is not white"
            );
        }
        let content_row = channel0
            .get(63)
            .expect("content first row")
            .to_vec1::<f32>()
            .expect("vec");
        assert!(
            content_row.iter().any(|value| (*value - 1.0).abs() > 1e-6),
            "first content row must not be white"
        );

        // Per-axis mapping: prepared row 454 (last content row) maps back to
        // original row 391 * 768 / 392.
        let coords = &prepared.original_coords[0];
        let (scale_x, scale_y) = coords.axis_scales_to_original();
        assert!((scale_x - 1024.0 / 518.0).abs() < 1e-12);
        assert!((scale_y - 768.0 / 392.0).abs() < 1e-12);
        let v = (454.0 - f64::from(coords.offset_y)) * scale_y;
        assert!((v - 391.0 * 768.0 / 392.0).abs() < 1e-9);
    }

    /// Pad-mode portrait: content 392x518 centered horizontally, full height.
    #[test]
    fn pad_mode_portrait_pads_left_and_right() {
        let dir = std::env::temp_dir().join(format!(
            "lux3d-vggt-pad-portrait-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create frame dir");
        RgbImage::from_pixel(768, 1024, Rgb([30, 140, 220]))
            .save(dir.join("a.png"))
            .expect("save frame");

        let stage = VggtPreprocessStage::default();
        let prepared = stage
            .prepare_inputs_from_path_with_mode(
                &dir,
                Some(1),
                VggtPreprocessMode::Pad,
                &Device::Cpu,
            )
            .expect("prepare pad inputs");
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(
            prepared.original_coords[0],
            VggtOriginalCoords {
                width: 768,
                height: 1024,
                offset_x: 63,
                offset_y: 0,
                content_width: 392,
                content_height: 518,
                mode: VggtPreprocessMode::Pad,
            }
        );

        // A constant image stays constant through the bicubic resize, so the
        // white pad columns are exactly 1.0 and the content exactly 30/255.
        let channel0 = prepared
            .rgb_frames
            .get(0)
            .expect("frame")
            .get(0)
            .expect("c0");
        for x in [0usize, 62, 455, 517] {
            let column_ok = (0..518usize).all(|y| {
                (channel0
                    .get(y)
                    .expect("row")
                    .get(x)
                    .expect("col")
                    .to_scalar::<f32>()
                    .expect("scalar")
                    - 1.0)
                    .abs()
                    <= 1e-6
            });
            assert!(column_ok, "pad column {x} is not white");
        }
        for x in [63usize, 454] {
            let column_ok = (0..518usize).all(|y| {
                (channel0
                    .get(y)
                    .expect("row")
                    .get(x)
                    .expect("col")
                    .to_scalar::<f32>()
                    .expect("scalar")
                    - 30.0 / 255.0)
                    .abs()
                    <= 1e-6
            });
            assert!(column_ok, "content column {x} changed under resize");
        }
    }

    /// Single image files are accepted directly as one-frame sources.
    #[test]
    fn single_image_file_is_a_one_frame_source() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "lux3d-vggt-single-{}-{}.png",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        RgbImage::from_pixel(320, 240, Rgb([90, 60, 30]))
            .save(&path)
            .expect("save image");

        let stage = VggtPreprocessStage::default();
        let prepared = stage
            .prepare_inputs_from_path(&path, None, &Device::Cpu)
            .expect("prepare single image");
        let _ = fs::remove_file(&path);

        assert_eq!(prepared.interval, 1);
        assert_eq!(prepared.original_size, SpatialSize::new(320, 240));
        // Crop mode: single landscape frame keeps its 518x392 resized size.
        assert_eq!(prepared.target_size, SpatialSize::new(518, 392));
        assert_eq!(prepared.rgb_frames.dims(), &[1, 3, 392, 518]);
        assert_eq!(prepared.original_coords.len(), 1);
        assert_eq!(
            prepared.original_coords[0],
            VggtOriginalCoords {
                width: 320,
                height: 240,
                offset_x: 0,
                offset_y: 0,
                content_width: 518,
                content_height: 392,
                mode: VggtPreprocessMode::Crop,
            }
        );
    }

    #[test]
    fn recognizes_all_single_image_extensions() {
        for extension in ["png", "JPG", "jpeg", "WebP", "bmp", "tif", "TIFF"] {
            assert!(is_image_extension(extension), "expected image: {extension}");
        }
        assert!(!is_image_extension("mp4"));
        assert!(!is_image_extension("txt"));
        assert!(!is_image_extension(""));
    }
}
