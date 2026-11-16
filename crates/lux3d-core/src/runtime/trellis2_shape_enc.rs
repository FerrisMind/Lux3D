//! TRELLIS.2 shape SLAT encoder (`FlexiDualGridVaeEncoder` /
//! `SparseUnetVaeEncoder`, checkpoint `shape_enc_next_dc_f16c32_fp16`, 284
//! stage tensors in the raw package under `ckpts/`).
//!
//! Reference: `trellis2/models/sc_vaes/fdg_vae.py` +
//! `sparse_unet_vae.py:297` (`SparseUnetVaeEncoder`); pipeline entry
//! `trellis2/pipelines/trellis2_texturing.py::encode_shape_slat`.
//!
//! The encoder is the missing half of the shape VAE: it compresses the
//! flexible dual grid of an existing mesh into the shape SLAT that conditions
//! the texture SLAT flow (`sample_tex_slat` normalizes the raw mean with
//! `shape_slat_normalization` afterwards — this module returns the raw mean).
//!
//! Input semantics (exact, verified against the checkpoint):
//!
//! - `vertices.feats = dual_vertices * resolution - voxel_indices` — the
//!   dual-vertex offsets in voxel units, roughly `[-0.5, 1.5]`, where
//!   `dual_vertices` are the normalized `[0, 1]` grid positions returned by
//!   the voxelization and `voxel_indices` the integer parent voxels.
//! - `coords = [0, voxel_indices]` (batch column dropped here).
//! - `FlexiDualGridVaeEncoder.forward` builds
//!   `x = cat([vertices.feats - 0.5, intersected.feats.float() - 0.5], dim=1)`
//!   giving `[N, 6]`: 3 offset channels minus 0.5, then 3 binary
//!   intersection flags per axis minus 0.5. `in_channels = 6` is confirmed by
//!   `input_layer.weight` `[64, 6]` in the checkpoint (not 9).
//!
//! Architecture (config `shape_enc_next_dc_f16c32_fp16.json`):
//!
//! - `input_layer` SparseLinear(6, 64) (+ bias).
//! - Stages 0..4 (`num_blocks [0, 4, 8, 16, 4]`): `num_blocks[s]` x
//!   `SparseConvNeXtBlock3d` (shared implementation with the decoder) then,
//!   for s < 4, one `SparseResBlockS2C3d` down block halving the grid and
//!   doubling the channels 64 -> 128 -> 256 -> 512 -> 1024.
//! - `SparseResBlockS2C3d` (`sparse_unet_vae.py:179`): `h = silu(norm1(x))`
//!   (norm1 affine, eps 1e-6) -> `conv1(C -> Cout/8)` on the parent coords ->
//!   `S2C(h)` (`SparseSpatial2Channel(2)`, parents sorted row-major, channel
//!   layout `[parent, child, channel]`, zero-filled missing children);
//!   `x = S2C(x)`; `h = silu(norm2(h))` (norm2 NON-affine, eps 1e-6) ->
//!   `conv2(Cout -> Cout)`; skip =
//!   `S2C(x).feats.reshape(Np, Cout, C*8/Cout).mean(-1)`; `h += skip`. The
//!   block carries no `skip_connection`/`to_subdiv` weights.
//! - After 4 downsamples the latent lives on the `resolution / 16` grid;
//!   output coordinates are the row-major sorted unique `voxel_index / 16`.
//! - Final (upstream `forward` epilogue): cast back to the input dtype, then
//!   NON-affine `F.layer_norm(feats, eps 1e-5)`, then `to_latent`
//!   SparseLinear(1024, 2 * 32); `mean, logvar = chunk(2, dim=-1)`. With
//!   `sample_posterior = false` the returned latent is the mean.
//!
//! Dtype: all compute is F32 (locked decision 3, matching
//! [`super::trellis2_vae`]). The checkpoint stores F16 and the VarBuilder
//! casts on access. Upstream `use_fp16 = true` casts the block torso to F16
//! (`h.type(fp16)` in / `h.type(x.dtype)` out, final norm + `to_latent` in
//! the input dtype); with F32 everywhere the casts are trivially preserved
//! and the measured F16-vs-F32 reference delta on the mean latent is ~1.3e-2
//! max (fixture manifest), which is why the golden comparison runs the F32
//! reference.
//!
//! Sparse conv and S2C come from [`super::trellis2_sparse`]; the ConvNeXt
//! block and the affine channel LayerNorm are shared with
//! [`super::trellis2_vae`] (no duplication).

use std::path::Path;

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use super::nn_blocks::{Linear, linear};
use super::trellis2_blocks::layer_norm_no_affine;
use super::trellis2_sparse::{SparseConv3d, SparseConvIndices, spatial_to_channel};
use super::trellis2_vae::{ConvNeXtBlock, channel_layer_norm_affine};
use crate::weights::VarBuilder;

/// Shape SLAT encoder architecture parameters
/// (`shape_enc_next_dc_f16c32_fp16.json` args).
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2ShapeEncoderConfig {
    /// Encoder input channels (6: 3 dual-vertex offsets + 3 intersection
    /// flags, each minus 0.5).
    pub in_channels: usize,
    /// Channel pyramid in encoder order ([64, 128, 256, 512, 1024]).
    pub model_channels: Vec<usize>,
    /// Latent output channels per half (32; `to_latent` emits 2 * 32 = 64).
    pub latent_channels: usize,
    /// ConvNeXt blocks per stage ([0, 4, 8, 16, 4]; an S2C down block follows
    /// each stage except the last).
    pub num_blocks: Vec<usize>,
    /// ConvNeXt MLP ratio (4.0).
    pub mlp_ratio: f64,
}

impl Trellis2ShapeEncoderConfig {
    /// `FlexiDualGridVaeEncoder` shape checkpoint args.
    pub fn flexi_dual_shape() -> Self {
        Self {
            in_channels: 6,
            model_channels: vec![64, 128, 256, 512, 1024],
            latent_channels: 32,
            num_blocks: vec![0, 4, 8, 16, 4],
            mlp_ratio: 4.0,
        }
    }

    /// The exact tensor names this encoder consumes below its prefix (284 for
    /// the shape checkpoint: 2 + 2 top-level plus
    /// `sum(num_blocks[s] * 8)` ConvNeXt tensors and 6 per down block
    /// (`conv1/conv2` weight+bias, `norm1` weight+bias; `norm2` is
    /// non-affine and stores no tensors)).
    pub fn expected_tensor_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        names.push("input_layer.weight".to_string());
        names.push("input_layer.bias".to_string());
        names.push("to_latent.weight".to_string());
        names.push("to_latent.bias".to_string());
        let stage_count = self.num_blocks.len();
        for (stage, &count) in self.num_blocks.iter().enumerate() {
            let channels = self.model_channels[stage];
            for block in 0..count {
                let base = format!("blocks.{stage}.{block}.");
                names.push(format!("{base}conv.weight"));
                names.push(format!("{base}conv.bias"));
                names.push(format!("{base}norm.weight"));
                names.push(format!("{base}norm.bias"));
                names.push(format!("{base}mlp.0.weight"));
                names.push(format!("{base}mlp.0.bias"));
                names.push(format!("{base}mlp.2.weight"));
                names.push(format!("{base}mlp.2.bias"));
            }
            if stage + 1 < stage_count {
                let out_channels = self.model_channels[stage + 1];
                debug_assert_eq!(out_channels % 8, 0, "S2C down block out channels");
                debug_assert_eq!(channels * 8 % out_channels, 0, "S2C skip grouping");
                let base = format!("blocks.{stage}.{count}.");
                names.push(format!("{base}conv1.weight"));
                names.push(format!("{base}conv1.bias"));
                names.push(format!("{base}conv2.weight"));
                names.push(format!("{base}conv2.bias"));
                names.push(format!("{base}norm1.weight"));
                names.push(format!("{base}norm1.bias"));
            }
        }
        names
    }
}

/// Shape SLAT produced by the encoder: the mean latent on the 4x-downsampled
/// (resolution / 16) grid. Not normalized — `sample_tex_slat` applies
/// `shape_slat_normalization` downstream.
#[derive(Debug, Clone)]
pub struct Trellis2ShapeSlat {
    /// Latent grid coordinates (`voxel_index / 16` after four S2C
    /// downsamples), row-major sorted ascending, unique.
    pub coords: Vec<[i32; 3]>,
    /// `[N, latent_channels]` F32 mean latent, rows aligned with `coords`.
    pub feats: Tensor,
}

/// `SparseResBlockS2C3d` (`sparse_unet_vae.py:179`): the S2C down block.
#[derive(Debug)]
struct S2CDownBlock {
    norm1_weight: Tensor,
    norm1_bias: Tensor,
    conv1: SparseConv3d,
    conv2: SparseConv3d,
    channels: usize,
    out_channels: usize,
}

impl S2CDownBlock {
    fn load(vb: VarBuilder, channels: usize, out_channels: usize) -> CandleResult<Self> {
        if out_channels % 8 != 0 {
            candle_core::bail!(
                "S2C down block out_channels must be divisible by 8, got {out_channels}"
            );
        }
        if channels * 8 % out_channels != 0 {
            candle_core::bail!(
                "S2C skip grouping requires channels * 8 divisible by out_channels, \
                 got {channels} -> {out_channels}"
            );
        }
        Ok(Self {
            norm1_weight: vb.get(channels, "norm1.weight")?.to_dtype(DType::F32)?,
            norm1_bias: vb.get(channels, "norm1.bias")?.to_dtype(DType::F32)?,
            conv1: SparseConv3d::load(vb.pp("conv1"), channels, out_channels / 8, true)?,
            conv2: SparseConv3d::load(vb.pp("conv2"), out_channels, out_channels, true)?,
            channels,
            out_channels,
        })
    }

    /// `coords`/`feats`: `[N, channels]` in; returns the downsampled coords
    /// (row-major sorted unique parents) and `[Np, out_channels]` features.
    /// Reference `_forward` order: `h = silu(norm1(x))` -> `conv1` (parent
    /// coords) -> `updown` (S2C); `x = updown(x)`; `h = silu(norm2(h))`
    /// (non-affine) -> `conv2`; `h += mean-reshaped S2C(x)`.
    fn forward_with_indices(
        &self,
        coords: &[[i32; 3]],
        feats: &Tensor,
        parent_indices: &SparseConvIndices,
    ) -> CandleResult<(Vec<[i32; 3]>, Tensor)> {
        let rows = feats.dim(0)?;
        if rows != coords.len() {
            candle_core::bail!(
                "S2C down block coords/feats mismatch: {} coords vs {rows} rows",
                coords.len()
            );
        }
        let h = channel_layer_norm_affine(feats, &self.norm1_weight, &self.norm1_bias, 1e-6)?;
        let h = h.silu()?;
        let h = self.conv1.forward_with_indices(&h, parent_indices)?; // [N, out/8]
        let h_s2c = spatial_to_channel(coords, &h)?; // [Np, out]
        let x_s2c = spatial_to_channel(coords, feats)?; // [Np, in * 8]
        let h = layer_norm_no_affine(&h_s2c.feats, 1e-6)?.silu()?;
        let child_indices = SparseConvIndices::new(&h_s2c.coords, h.device())?;
        let h = self.conv2.forward_with_indices(&h, &child_indices)?; // [Np, out]
        let parents = h_s2c.coords.len();
        let groups = self.channels * 8 / self.out_channels;
        let skip = x_s2c
            .feats
            .reshape((parents, self.out_channels, groups))?
            .mean(candle_core::D::Minus1)?;
        let out = h.add(&skip)?;
        Ok((h_s2c.coords, out))
    }
}

#[derive(Debug)]
enum EncoderStageBlock {
    ConvNeXt(ConvNeXtBlock),
    Down(S2CDownBlock),
}

/// Sparse shape SLAT encoder (`FlexiDualGridVaeEncoder`). Loads the 284
/// `shape_enc` tensors from a VarBuilder rooted at the stage.
#[derive(Debug)]
pub struct Trellis2ShapeEncoder {
    config: Trellis2ShapeEncoderConfig,
    input_layer: Linear,
    to_latent: Linear,
    stages: Vec<Vec<EncoderStageBlock>>,
}

impl Trellis2ShapeEncoder {
    /// Load from a VarBuilder rooted at the encoder tensors (raw stage file:
    /// the root; canonical package: `vb.pp("shape_enc")` when the stage is
    /// present). Fails if any expected tensor is missing.
    pub fn load(vb: VarBuilder, config: &Trellis2ShapeEncoderConfig) -> CandleResult<Self> {
        if config.model_channels.len() != config.num_blocks.len() {
            candle_core::bail!(
                "encoder config model_channels ({}) and num_blocks ({}) lengths differ",
                config.model_channels.len(),
                config.num_blocks.len()
            );
        }
        let first = config.model_channels[0];
        let last = *config.model_channels.last().expect("non-empty channels");
        let input_layer = linear(vb.pp("input_layer"), config.in_channels, first, true)?;
        let to_latent = linear(vb.pp("to_latent"), last, config.latent_channels * 2, true)?;
        let mut stages = Vec::with_capacity(config.num_blocks.len());
        for (stage, &count) in config.num_blocks.iter().enumerate() {
            let channels = config.model_channels[stage];
            let mut blocks = Vec::with_capacity(count + 1);
            for block in 0..count {
                blocks.push(EncoderStageBlock::ConvNeXt(ConvNeXtBlock::load(
                    vb.pp("blocks").pp(stage.to_string()).pp(block.to_string()),
                    channels,
                    config.mlp_ratio,
                )?));
            }
            if stage + 1 < config.num_blocks.len() {
                let out_channels = config.model_channels[stage + 1];
                blocks.push(EncoderStageBlock::Down(S2CDownBlock::load(
                    vb.pp("blocks").pp(stage.to_string()).pp(count.to_string()),
                    channels,
                    out_channels,
                )?));
            }
            super::drain_upload_staging(vb.device())?;
            stages.push(blocks);
        }
        Ok(Self {
            config: config.clone(),
            input_layer,
            to_latent,
            stages,
        })
    }

    pub fn config(&self) -> &Trellis2ShapeEncoderConfig {
        &self.config
    }

    /// Raw encoder body (`SparseUnetVaeEncoder.forward` with
    /// `sample_posterior = false, return_raw = false`).
    ///
    /// `coords`: `[N]` voxel indices on the resolution grid; `feats`:
    /// `[N, 6]` F32, the FINAL encoder input `x = cat(vertices.feats - 0.5,
    /// intersected.feats.float() - 0.5)` (build it with
    /// [`concat_dual_grid_input`] or use [`Self::encode_shape_slat`]).
    /// Returns the mean latent `[N/16, latent_channels]` on the downsampled
    /// coordinates (row-major sorted unique `coord / 16`).
    pub fn forward(
        &self,
        coords: &[[i32; 3]],
        feats: &Tensor,
        device: &Device,
    ) -> CandleResult<Trellis2ShapeSlat> {
        let rows = feats.dim(0)?;
        if rows != coords.len() {
            candle_core::bail!(
                "shape encoder coords/feats mismatch: {} coords vs {rows} rows",
                coords.len()
            );
        }
        if feats.dim(1)? != self.config.in_channels {
            candle_core::bail!(
                "shape encoder expects {} input channels, got {}",
                self.config.in_channels,
                feats.dim(1)?
            );
        }
        let feats = if feats.device().same_device(device) {
            feats.clone()
        } else {
            feats.to_device(device)?
        };
        let mut h = self.input_layer.forward(&feats)?;
        let mut coords = coords.to_vec();
        for blocks in &self.stages {
            let stage_indices = SparseConvIndices::new(&coords, h.device())?;
            for block in blocks {
                match block {
                    EncoderStageBlock::ConvNeXt(conv_next) => {
                        h = conv_next.forward_with_indices(&h, &stage_indices)?;
                    }
                    EncoderStageBlock::Down(down) => {
                        let (parent_coords, parent_feats) =
                            down.forward_with_indices(&coords, &h, &stage_indices)?;
                        coords = parent_coords;
                        h = parent_feats;
                    }
                }
            }
        }
        // Final non-affine F.layer_norm (PyTorch default eps 1e-5), then
        // to_latent and the mean half of the 2 * latent_channels output.
        let normed = layer_norm_no_affine(&h, 1e-5)?;
        let out = self.to_latent.forward(&normed)?; // [Np, 2 * latent]
        let mean = out
            .narrow(1, 0, self.config.latent_channels)?
            .contiguous()?;
        Ok(Trellis2ShapeSlat {
            coords,
            feats: mean,
        })
    }

    /// Pipeline `encode_shape_slat` (`trellis2_texturing.py:183`) +
    /// `FlexiDualGridVaeEncoder.forward`: build the 6-channel input from the
    /// voxelization outputs and run the network.
    ///
    /// - `coords`: `[N]` integer voxel indices on the `resolution` grid
    ///   (upstream `voxel_indices`).
    /// - `dual_vertices`: `[N, 3]` F32 normalized `[0, 1]` grid positions
    ///   (upstream `dual_vertices`; `feats = dual_vertices * resolution -
    ///   voxel_indices` recovers the voxel-unit offsets).
    /// - `intersected`: `[N, 3]` per-axis intersection flags (any dtype;
    ///   nonzero = true, matching upstream `.float()` of a bool tensor).
    ///
    /// Returns the raw mean latent (no normalization; downstream
    /// `sample_tex_slat` applies `shape_slat_normalization`).
    pub fn encode_shape_slat(
        &self,
        coords: &[[i32; 3]],
        dual_vertices: &Tensor,
        intersected: &Tensor,
        resolution: usize,
        device: &Device,
    ) -> CandleResult<Trellis2ShapeSlat> {
        let feats = concat_dual_grid_input(coords, dual_vertices, intersected, resolution, device)?;
        self.forward(coords, &feats, device)
    }
}

/// Build the encoder input `x` (`[N, 6]` F32): the voxel-unit dual-vertex
/// offsets minus 0.5 followed by the per-axis intersection flags minus 0.5
/// (`FlexiDualGridVaeEncoder.forward`).
pub fn concat_dual_grid_input(
    coords: &[[i32; 3]],
    dual_vertices: &Tensor,
    intersected: &Tensor,
    resolution: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    let rows = dual_vertices.dim(0)?;
    if rows != coords.len() || intersected.dim(0)? != rows {
        candle_core::bail!(
            "dual grid input mismatch: {} coords, {} dual vertices, {} intersected rows",
            coords.len(),
            rows,
            intersected.dim(0)?
        );
    }
    if dual_vertices.dim(1)? != 3 || intersected.dim(1)? != 3 {
        candle_core::bail!(
            "dual vertices and intersected must be [N, 3], got {:?} / {:?}",
            dual_vertices.shape(),
            intersected.shape()
        );
    }
    let dual = dual_vertices.to_device(device)?.to_dtype(DType::F32)?;
    let flags = intersected.to_device(device)?.to_dtype(DType::F32)?;
    let indices = Tensor::new(
        coords
            .iter()
            .flat_map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .collect::<Vec<_>>(),
        device,
    )?
    .reshape((coords.len(), 3))?;
    let offsets = dual
        .affine((resolution as f64).into(), 0.0)? // dual_vertices * resolution
        .broadcast_sub(&indices)?
        .affine(1.0.into(), (-0.5).into())?; // vertices.feats - 0.5
    let flags = flags.affine(1.0, -0.5)?; // intersected.feats.float() - 0.5
    Tensor::cat(&[offsets, flags], 1)
}

/// Load the shape encoder from the standalone stage checkpoint
/// (`ckpts/shape_enc_next_dc_f16c32_fp16.safetensors`, unprefixed keys) with
/// the canonical flexi-dual config.
///
/// # Safety
///
/// The mmap-backed builder keeps the mapping alive for every tensor created
/// from it; the file is not mutated.
pub unsafe fn load_trellis2_shape_encoder_from_stage_file(
    path: &Path,
    device: &Device,
) -> CandleResult<Trellis2ShapeEncoder> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), DType::F32, device)
    }?;
    Trellis2ShapeEncoder::load(vb, &Trellis2ShapeEncoderConfig::flexi_dual_shape())
}
