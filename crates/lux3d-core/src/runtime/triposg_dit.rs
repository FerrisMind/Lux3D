//! TripoSG DiT transformer (HunyuanDiT-style UViT), the rectified-flow Euler
//! scheduler, and the classifier-free-guidance denoise loop.
//!
//! Reference semantics (`triposg/models/transformers/triposg_transformer.py`,
//! `triposg/models/attention_processor.py`,
//! `triposg/schedulers/scheduling_rectified_flow.py`,
//! `triposg/pipelines/pipeline_triposg.py`):
//!
//! - Sequence = timestep token PREPENDED to `proj_in(latents)` (2049 tokens
//!   for 2048 latent tokens); no adaLN, no RoPE, no LayerScale.
//! - Timestep embedding: sinusoid `half = width/2`,
//!   `exp(-ln(10000) * i / half)`, `cat([sin, cos])` (sin FIRST), then MLP
//!   `width -> 4*width -> width` with exact GELU.
//! - 21 blocks: long skip on blocks 11..20 only (`cat([skip, x])` skip FIRST,
//!   then `skip_linear` then `skip_norm` — the reference's `skip_norm_last`
//!   quirk, replicated verbatim), self-attn, cross-attn, exact-GELU FFN, all
//!   pre-normed with LayerNorm(eps 1e-5, affine).
//! - Attention re-splits the projections EXACTLY like
//!   `TripoSGAttnProcessor2_0`: self-attn concatenates `[q, k, v]` on the
//!   channel dim then re-splits per head (head h takes columns
//!   `[384h : 384h+128)` of each of q/k/v); cross-attn uses a standard
//!   head-major query and re-splits `cat([k, v])` with stride 256. Q and K
//!   get RMSNorm (eps 1e-6, per-head weight `[head_dim]`); SDPA scale is the
//!   default `1/sqrt(head_dim)`.
//! - Final `norm_out` over all 2049 tokens, slice the LAST 2048, `proj_out`
//!   `width -> in_channels` -> rectified-flow velocity `[B, L, 64]`.
//!
//! Everything stays F32 end to end (the canonical weights are F32 and the
//! parity baseline was generated in F32). All ops are backend-generic Candle
//! primitives; self-attention scores are computed in query chunks
//! (`LUX3D_TRIPOSG_ATTN_CHUNK`, default 1024) to bound peak memory on 12 GiB
//! GPUs.
//!
//! ## Custom timesteps
//!
//! [`RectifiedFlowSchedule::from_timesteps`] implements the explicit
//! timestep override (`retrieve_timesteps(scheduler, ..., timesteps)`):
//! `sigmas = timesteps / 1000` + shift transform + terminal 0, and the denoise
//! loop runs exactly `len(timesteps)` steps. See the fn docs for the shipped
//! `RectifiedFlowScheduler` caveat (upstream's `set_timesteps` has no
//! `timesteps` parameter and would reject the override).

use std::sync::OnceLock;

use super::nn_blocks::{linear_dense as linear, linear_no_bias_dense as linear_no_bias, Linear};
use crate::weights::VarBuilder;
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Module};

use super::attention_math::{exact_query_chunked_sdpa, exact_sdpa_heads};

/// Self-attention query chunk (`LUX3D_TRIPOSG_ATTN_CHUNK`, default 1024).
fn attn_chunk_size() -> usize {
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("LUX3D_TRIPOSG_ATTN_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1024)
    })
}

/// Parsed `transformer` section of the canonical `resolved_config.json`.
///
/// Defaults replicate `TripoSGDiTModel.__init__` (`num_attention_heads=16`,
/// `width=2048`, `in_channels=64`, `num_layers=21`,
/// `cross_attention_dim=1024`, `use_cross_attention_2=false`) plus the
/// hardcoded `mlp_ratio=4.0`. The scribble checkpoint sets
/// `use_cross_attention_2=true` with `cross_attention_dim=768` (CLIP text) and
/// `cross_attention_2_dim=1024` (DINOv2 image); see
/// `DiTBlock.forward` in `triposg/models/transformers/triposg_transformer.py`.
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgTransformerConfig {
    pub in_channels: usize,
    pub width: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub cross_attention_dim: usize,
    /// Second cross-attention branch (`attn2_2` in the reference block).
    pub use_cross_attention_2: bool,
    /// Encoder width of the second cross-attention branch.
    pub cross_attention_2_dim: Option<usize>,
    pub mlp_ratio: f64,
}

impl Default for TripoSgTransformerConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            width: 2048,
            num_attention_heads: 16,
            num_layers: 21,
            cross_attention_dim: 1024,
            use_cross_attention_2: false,
            cross_attention_2_dim: None,
            mlp_ratio: 4.0,
        }
    }
}

impl TripoSgTransformerConfig {
    /// Parse from the canonical `resolved_config.json` document (or directly
    /// from its `transformer` subsection). Unknown/missing fields fall back to
    /// the reference defaults.
    pub fn from_resolved_config(value: &serde_json::Value) -> CandleResult<Self> {
        let source = value.get("transformer").unwrap_or(value);
        let get_usize = |key: &str, default: usize| -> usize {
            source
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let cfg = Self {
            in_channels: get_usize("in_channels", 64),
            width: get_usize("width", 2048),
            num_attention_heads: get_usize("num_attention_heads", 16),
            num_layers: get_usize("num_layers", 21),
            cross_attention_dim: get_usize("cross_attention_dim", 1024),
            use_cross_attention_2: source
                .get("use_cross_attention_2")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            cross_attention_2_dim: source
                .get("cross_attention_2_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            mlp_ratio: source
                .get("mlp_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> CandleResult<()> {
        if self.in_channels == 0 || self.width == 0 || self.cross_attention_dim == 0 {
            candle_core::bail!(
                "TripoSG DiT config has a zero dimension: in_channels {}, width {}, \
                 cross_attention_dim {}",
                self.in_channels,
                self.width,
                self.cross_attention_dim
            );
        }
        if self.use_cross_attention_2 {
            match self.cross_attention_2_dim {
                Some(dim) if dim > 0 => {}
                Some(dim) => candle_core::bail!(
                    "TripoSG DiT use_cross_attention_2 is set with \
                     cross_attention_2_dim={dim}; it must be positive"
                ),
                None => candle_core::bail!(
                    "TripoSG DiT use_cross_attention_2 is set but the config has no \
                     cross_attention_2_dim"
                ),
            }
        }
        if self.num_attention_heads == 0 || self.width % self.num_attention_heads != 0 {
            candle_core::bail!(
                "TripoSG DiT width {} is not divisible by num_attention_heads {}",
                self.width,
                self.num_attention_heads
            );
        }
        if self.width % 2 != 0 {
            candle_core::bail!("TripoSG DiT width {} must be even", self.width);
        }
        if self.mlp_ratio <= 0.0 || (self.width as f64 * self.mlp_ratio).fract() != 0.0 {
            candle_core::bail!(
                "TripoSG DiT mlp_ratio {} must be positive and yield an integer FFN width",
                self.mlp_ratio
            );
        }
        // Skip topology: blocks 0..(num_layers/2 - 1) push, blocks
        // (num_layers/2 + 1)..pop; both halves must be non-empty.
        if self.num_layers < 3 {
            candle_core::bail!(
                "TripoSG DiT needs at least 3 layers for the UViT skip topology, got {}",
                self.num_layers
            );
        }
        Ok(())
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.width / self.num_attention_heads
    }

    pub(crate) fn ff_inner_dim(&self) -> usize {
        (self.width as f64 * self.mlp_ratio) as usize
    }

    /// `time_embed_dim = 4 * width` (diffusers `_set_time_proj("positional")`
    /// with `time_embedding_dim=None`).
    pub(crate) fn time_embed_dim(&self) -> usize {
        self.width * 4
    }
}

/// TripoSG 21-block DiT. Public for the parity test harness; runtime-internal
/// surface (the pipeline drives it through `denoise`).
#[derive(Debug)]
pub struct TripoSgDiT {
    cfg: TripoSgTransformerConfig,
    proj_in: Linear,
    time_linear_1: Linear,
    time_linear_2: Linear,
    blocks: Vec<DiTBlock>,
    norm_out: LayerNorm,
    proj_out: Linear,
    attn_chunk: usize,
}

impl TripoSgDiT {
    /// Load from a VarBuilder rooted at the canonical `transformer.` prefix
    /// (i.e. pass `vb.pp("transformer")` from the canonical weight set).
    pub fn load(vb: VarBuilder, cfg: &TripoSgTransformerConfig) -> CandleResult<Self> {
        let proj_in = linear(cfg.in_channels, cfg.width, vb.pp("proj_in"))?;
        let time_linear_1 = linear(
            cfg.width,
            cfg.time_embed_dim(),
            vb.pp("time_proj").pp("linear_1"),
        )?;
        let time_linear_2 = linear(
            cfg.time_embed_dim(),
            cfg.width,
            vb.pp("time_proj").pp("linear_2"),
        )?;
        let blocks = (0..cfg.num_layers)
            .map(|index| {
                let block = DiTBlock::load(vb.pp("blocks").pp(index.to_string()), cfg, index)?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let norm_out = layer_norm(cfg.width, 1e-5, vb.pp("norm_out"))?;
        let proj_out = linear(cfg.width, cfg.in_channels, vb.pp("proj_out"))?;

        Ok(Self {
            cfg: cfg.clone(),
            proj_in,
            time_linear_1,
            time_linear_2,
            blocks,
            norm_out,
            proj_out,
            attn_chunk: attn_chunk_size(),
        })
    }

    /// `latents [B, L, in_channels] F32`, `timestep` = the scheduler timestep
    /// (e.g. 1000.0), `cond [B, S, cross_attention_dim]` (for CFG the batch
    /// already carries `[uncond, cond]`) -> velocity `[B, L, in_channels]`.
    pub fn forward(&self, latents: &Tensor, timestep: f32, cond: &Tensor) -> CandleResult<Tensor> {
        self.forward_with_conditioning(latents, timestep, cond, None, 1.0, 1.0)
    }

    /// Forward with the optional SECOND cross-attention conditioning branch
    /// and the per-branch output scales of the reference scribble pipeline.
    ///
    /// `encoder_hidden_states_2 = image_cond` feeds `attn2_2`;
    /// `cross_attention_scale`/`cross_attention_2_scale` multiply the two
    /// cross-attention outputs before they are added to the residual
    /// (`pipeline_triposg_scribble.py:271-278` passes
    /// `attention_kwargs={"cross_attention_scale": prompt_confidence,
    /// "cross_attention_2_scale": scribble_confidence}`, consumed by
    /// `DiTBlock.forward` at `triposg_transformer.py:300-301, 333-349`).
    ///
    /// The image (single-conditioning) path is `cond2 = None` with both
    /// scales `1.0`, i.e. exactly the pre-existing behavior; the `1.0`
    /// multiply is skipped so no extra kernel runs on that path.
    pub fn forward_with_conditioning(
        &self,
        latents: &Tensor,
        timestep: f32,
        cond: &Tensor,
        cond2: Option<&Tensor>,
        cross_attention_scale: f32,
        cross_attention_2_scale: f32,
    ) -> CandleResult<Tensor> {
        let (batch, num_tokens, channels) = latents.dims3()?;
        if channels != self.cfg.in_channels {
            candle_core::bail!(
                "TripoSG DiT expects {} latent channels, got {channels}",
                self.cfg.in_channels
            );
        }
        let (cond_batch, _cond_seq, cond_dim) = cond.dims3()?;
        if cond_batch != batch {
            candle_core::bail!(
                "TripoSG DiT condition batch {} does not match latent batch {batch}",
                cond_batch
            );
        }
        if cond_dim != self.cfg.cross_attention_dim {
            candle_core::bail!(
                "TripoSG DiT expects cross-attention dim {}, got {cond_dim}",
                self.cfg.cross_attention_dim
            );
        }
        match (self.cfg.use_cross_attention_2, cond2) {
            (true, Some(cond2)) => {
                let (cond2_batch, _cond2_seq, cond2_dim) = cond2.dims3()?;
                if cond2_batch != batch {
                    candle_core::bail!(
                        "TripoSG DiT second condition batch {cond2_batch} does not match \
                         latent batch {batch}"
                    );
                }
                let expected = self
                    .cfg
                    .cross_attention_2_dim
                    .expect("validated use_cross_attention_2 config");
                if cond2_dim != expected {
                    candle_core::bail!(
                        "TripoSG DiT expects cross-attention_2 dim {expected}, got {cond2_dim}"
                    );
                }
            }
            (true, None) => candle_core::bail!(
                "TripoSG DiT config has use_cross_attention_2 but no encoder_hidden_states_2 \
                 was supplied"
            ),
            (false, Some(_)) => candle_core::bail!(
                "TripoSG DiT received encoder_hidden_states_2 but the config does not use \
                 cross_attention_2"
            ),
            (false, None) => {}
        }
        if !cross_attention_scale.is_finite() || !cross_attention_2_scale.is_finite() {
            candle_core::bail!(
                "TripoSG DiT cross-attention scales must be finite, got \
                 ({cross_attention_scale}, {cross_attention_2_scale})"
            );
        }

        // Timestep token: [B, 1, width].
        let temb = self.timestep_embedding(timestep, latents.device(), latents.dtype())?;
        let temb = temb
            .reshape((1, 1, self.cfg.width))?
            .broadcast_as((batch, 1, self.cfg.width))?
            .contiguous()?;

        // Sequence = [temb, proj_in(latents)] -> [B, L + 1, width].
        let projected = self.proj_in.forward(latents)?;
        let mut hidden = Tensor::cat(&[&temb, &projected], 1)?;

        // 21 blocks with the UViT long-skip LIFO stack: blocks 0..(mid-1)
        // push their output, block `mid` neither pushes nor pops, blocks
        // (mid+1).. pop (block mid+1 consumes block mid-1's output).
        let mid = self.cfg.num_layers / 2;
        let mut skips: Vec<Tensor> = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            let skip = if index <= mid {
                None
            } else {
                Some(
                    skips
                        .pop()
                        .expect("TripoSG DiT skip stack underflow is a config bug"),
                )
            };
            hidden = block.forward(
                &hidden,
                cond,
                cond2,
                skip.as_ref(),
                self.attn_chunk,
                cross_attention_scale,
                cross_attention_2_scale,
            )?;
            if index < mid {
                skips.push(hidden.clone());
            }
        }

        // Final: LayerNorm over ALL tokens, slice the LAST num_tokens, proj_out.
        let normalized = self.norm_out.forward(&hidden)?;
        let seq = normalized.dim(1)?;
        let tail = normalized.narrow(1, seq - num_tokens, num_tokens)?;
        self.proj_out.forward(&tail)
    }

    /// diffusers `Timesteps(width, flip_sin_to_cos=False, shift=0)` ->
    /// `TimestepEmbedding` MLP: `[width]` vector, sin FIRST concat.
    fn timestep_embedding(
        &self,
        timestep: f32,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Tensor> {
        let half = self.cfg.width / 2;
        let mut emb = Vec::with_capacity(half);
        let log_max_period = -(10000.0f32).ln();
        for i in 0..half {
            let freq = (log_max_period * i as f32 / half as f32).exp();
            emb.push(timestep * freq);
        }
        let emb = Tensor::from_vec(emb, (1, half), device)?;
        let emb = emb.to_dtype(dtype)?;
        let sin = emb.sin()?;
        let cos = emb.cos()?;
        let projected = Tensor::cat(&[&sin, &cos], D::Minus1)?; // [1, width]
        let projected = self.time_linear_1.forward(&projected)?.gelu_erf()?;
        self.time_linear_2.forward(&projected)
    }
}

/// One HunyuanDiT-style block: (optional long skip) -> self-attn ->
/// cross-attn(s) -> FFN, all pre-normed, all residual.
///
/// With `use_cross_attention_2` the block carries the reference `attn2_2`
/// second cross-attention branch. Both cross-attentions read the SAME
/// pre-cross-attention hidden state (upstream expression
/// `hidden + attn2(norm2(hidden)) * s1 + attn2_2(norm2_2(hidden)) * s2`,
/// `triposg_transformer.py:333-349`), and their outputs are scaled
/// independently before the sum.
#[derive(Debug)]
struct DiTBlock {
    skip_norm: Option<LayerNorm>,
    skip_linear: Option<Linear>,
    norm1: LayerNorm,
    attn1: DiTAttention,
    norm2: LayerNorm,
    attn2: DiTAttention,
    norm2_2: Option<LayerNorm>,
    attn2_2: Option<DiTAttention>,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl DiTBlock {
    fn load(vb: VarBuilder, cfg: &TripoSgTransformerConfig, index: usize) -> CandleResult<Self> {
        let has_skip = index > cfg.num_layers / 2;
        let (skip_norm, skip_linear) = if has_skip {
            (
                Some(layer_norm(cfg.width, 1e-5, vb.pp("skip_norm"))?),
                Some(linear(2 * cfg.width, cfg.width, vb.pp("skip_linear"))?),
            )
        } else {
            (None, None)
        };
        let (norm2_2, attn2_2) = if cfg.use_cross_attention_2 {
            let dim = cfg
                .cross_attention_2_dim
                .expect("validated use_cross_attention_2 config");
            (
                Some(layer_norm(cfg.width, 1e-5, vb.pp("norm2_2"))?),
                Some(DiTAttention::load(vb.pp("attn2_2"), cfg, Some(dim))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            skip_norm,
            skip_linear,
            norm1: layer_norm(cfg.width, 1e-5, vb.pp("norm1"))?,
            attn1: DiTAttention::load(vb.pp("attn1"), cfg, None)?,
            norm2: layer_norm(cfg.width, 1e-5, vb.pp("norm2"))?,
            attn2: DiTAttention::load(vb.pp("attn2"), cfg, Some(cfg.cross_attention_dim))?,
            norm2_2,
            attn2_2,
            norm3: layer_norm(cfg.width, 1e-5, vb.pp("norm3"))?,
            ff_in: linear(
                cfg.width,
                cfg.ff_inner_dim(),
                vb.pp("ff").pp("net").pp("0").pp("proj"),
            )?,
            ff_out: linear(cfg.ff_inner_dim(), cfg.width, vb.pp("ff").pp("net").pp("2"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        cond: &Tensor,
        cond2: Option<&Tensor>,
        skip: Option<&Tensor>,
        attn_chunk: usize,
        cross_attention_scale: f32,
        cross_attention_2_scale: f32,
    ) -> CandleResult<Tensor> {
        let mut hidden = match (self.skip_linear.as_ref(), self.skip_norm.as_ref(), skip) {
            (Some(skip_linear), Some(skip_norm), Some(skip)) => {
                // skip FIRST (skip_concat_front=True), then linear THEN norm
                // (skip_norm_last=True — the reference's documented quirk).
                let cat = Tensor::cat(&[skip, xs], D::Minus1)?;
                skip_norm.forward(&skip_linear.forward(&cat)?)?
            }
            _ => xs.clone(),
        };

        // Self-attention (no rotary embedding in the image pipeline).
        let attn_out = self
            .attn1
            .forward(&self.norm1.forward(&hidden)?, None, attn_chunk)?;
        hidden = (hidden + attn_out)?;

        // Cross-attention on the image embeds, reference scale 1.0.
        let cross_residual = hidden.clone();
        let cross_out = self.attn2.forward(
            &self.norm2.forward(&cross_residual)?,
            Some(cond),
            attn_chunk,
        )?;
        let cross_out = if cross_attention_scale == 1.0 {
            cross_out
        } else {
            cross_out.affine(f64::from(cross_attention_scale), 0.0)?
        };
        hidden = (&cross_residual + cross_out)?;
        if let (Some(norm2_2), Some(attn2_2), Some(cond2)) =
            (self.norm2_2.as_ref(), self.attn2_2.as_ref(), cond2)
        {
            // The second branch reads the SAME residual as the first one
            // (`triposg_transformer.py:343-348` applies `norm2_2` to the
            // un-updated `hidden_states`).
            let cross_out_2 =
                attn2_2.forward(&norm2_2.forward(&cross_residual)?, Some(cond2), attn_chunk)?;
            let cross_out_2 = if cross_attention_2_scale == 1.0 {
                cross_out_2
            } else {
                cross_out_2.affine(f64::from(cross_attention_2_scale), 0.0)?
            };
            hidden = (hidden + cross_out_2)?;
        }

        // FFN: exact GELU between the two projections.
        let ff = self.ff_out.forward(
            &self
                .ff_in
                .forward(&self.norm3.forward(&hidden)?)?
                .gelu_erf()?,
        )?;
        hidden + ff
    }
}

/// diffusers `Attention` with `qk_norm="rms_norm"`, `qkv_bias=False`,
/// `out_bias=True`, loaded with the checkpoint's unusual projection layout.
#[derive(Debug)]
struct DiTAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q_weight: Tensor,
    norm_k_weight: Tensor,
    heads: usize,
    head_dim: usize,
}

impl DiTAttention {
    /// `kv_input_dim` is `None` for self-attention (k/v see the hidden width)
    /// and `cross_attention_dim` for cross-attention.
    fn load(
        vb: VarBuilder,
        cfg: &TripoSgTransformerConfig,
        kv_input_dim: Option<usize>,
    ) -> CandleResult<Self> {
        let kv_in = kv_input_dim.unwrap_or(cfg.width);
        let head_dim = cfg.head_dim();
        Ok(Self {
            to_q: linear_no_bias(cfg.width, cfg.width, vb.pp("to_q"))?,
            to_k: linear_no_bias(kv_in, cfg.width, vb.pp("to_k"))?,
            to_v: linear_no_bias(kv_in, cfg.width, vb.pp("to_v"))?,
            to_out: linear(cfg.width, cfg.width, vb.pp("to_out").pp("0"))?,
            norm_q_weight: vb.get((head_dim,), "norm_q.weight")?,
            norm_k_weight: vb.get((head_dim,), "norm_k.weight")?,
            heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        encoder: Option<&Tensor>,
        attn_chunk: usize,
    ) -> CandleResult<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        match encoder {
            None => {
                // Self-attention. Replicate TripoSGAttnProcessor2_0 EXACTLY:
                // cat([q,k,v], -1) -> (B, L, heads, 3*head_dim) -> split the
                // last dim into thirds per head (NOT head-major thirds of the
                // fused matrix), then transpose to heads-first.
                let q = self.to_q.forward(xs)?;
                let k = self.to_k.forward(xs)?;
                let v = self.to_v.forward(xs)?;
                let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?.reshape((
                    batch,
                    seq,
                    self.heads,
                    3 * self.head_dim,
                ))?;
                let parts = qkv.chunk(3, D::Minus1)?;
                let q = parts[0]
                    .transpose(1, 2)?
                    .contiguous()?
                    .rms_norm_heads(&self.norm_q_weight)?;
                let k = parts[1]
                    .transpose(1, 2)?
                    .contiguous()?
                    .rms_norm_heads(&self.norm_k_weight)?;
                let v = parts[2].transpose(1, 2)?.contiguous()?;
                let out = exact_query_chunked_sdpa(&q, &k, &v, scale, attn_chunk)?;
                let out = out
                    .transpose(1, 2)?
                    .reshape((batch, seq, self.heads * self.head_dim))?;
                self.to_out.forward(&out)
            }
            Some(encoder_states) => {
                // Cross-attention: query is a STANDARD head-major split of
                // to_q(x); k/v come from the encoder states and are re-split
                // from cat([k, v], -1) with per-head stride 2*head_dim.
                let (enc_batch, enc_seq, enc_dim) = encoder_states.dims3()?;
                if enc_batch != batch {
                    candle_core::bail!(
                        "TripoSG cross-attention batch mismatch: {enc_batch} vs {batch}"
                    );
                }
                if enc_dim != self.to_k.in_dim() {
                    candle_core::bail!(
                        "TripoSG cross-attention input dim {} does not match to_k weight {}",
                        enc_dim,
                        self.to_k.in_dim()
                    );
                }
                let q = self
                    .to_q
                    .forward(xs)?
                    .reshape((batch, seq, self.heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?
                    .rms_norm_heads(&self.norm_q_weight)?;
                let k = self.to_k.forward(encoder_states)?;
                let v = self.to_v.forward(encoder_states)?;
                let kv = Tensor::cat(&[&k, &v], D::Minus1)?.reshape((
                    enc_batch,
                    enc_seq,
                    self.heads,
                    2 * self.head_dim,
                ))?;
                let kv_parts = kv.chunk(2, D::Minus1)?;
                let k = kv_parts[0]
                    .transpose(1, 2)?
                    .contiguous()?
                    .rms_norm_heads(&self.norm_k_weight)?;
                let v = kv_parts[1].transpose(1, 2)?.contiguous()?;
                let out = exact_sdpa_heads(&q, &k, &v, scale)?;
                let out = out
                    .transpose(1, 2)?
                    .reshape((batch, seq, self.heads * self.head_dim))?;
                self.to_out.forward(&out)
            }
        }
    }
}

/// RMSNorm over the last (head) dim with a per-head `[head_dim]` weight:
/// `x * rsqrt(mean(x^2, last) + eps) * weight` (diffusers `RMSNorm`, F32).
trait RmsNormHeads {
    fn rms_norm_heads(&self, weight: &Tensor) -> CandleResult<Tensor>;
}

impl RmsNormHeads for Tensor {
    fn rms_norm_heads(&self, weight: &Tensor) -> CandleResult<Tensor> {
        let eps = 1e-6_f64;
        let variance = self.sqr()?.mean_keepdim(D::Minus1)?; // [.., 1]
        let normed = self.broadcast_div(&(variance + eps)?.sqrt()?)?;
        normed.broadcast_mul(weight)
    }
}

/// Rectified-flow Euler schedule (`RectifiedFlowScheduler.set_timesteps`).
///
/// `sigmas[i] = shift*s_i / (1 + (shift-1)*s_i)` with
/// `s_i = (1 - i/steps)`, plus a terminal 0.0; `timesteps[i] = sigmas[i] *
/// 1000` (F32, matching the torch `sigmas * num_train_timesteps`). `shift=1`
/// is the identity (the shipped TripoSG config).
#[derive(Debug, Clone)]
pub struct RectifiedFlowSchedule {
    /// Length `steps + 1`; last entry is the terminal 0.0.
    pub sigmas: Vec<f32>,
}

/// `shift` of the shipped TripoSG scheduler config
/// (`scheduler_config.shift == 1` in the canonical `resolved_config.json`).
/// `shift = 1` is the identity transform; the schedule helpers keep it as an
/// explicit parameter so a config change is a one-line diff.
pub const SCHEDULER_SHIFT: f32 = 1.0;

/// `num_train_timesteps` of the shipped scheduler config (1000). Custom
/// timesteps are converted to sigmas as `timestep / SCHEDULER_NUM_TRAIN_TIMESTEPS`,
/// matching both the reference `RectifiedFlowScheduler` and diffusers'
/// `FlowMatchEulerDiscreteScheduler`.
pub const SCHEDULER_NUM_TRAIN_TIMESTEPS: f64 = 1000.0;

impl RectifiedFlowSchedule {
    pub fn new(steps: usize, shift: f32) -> Self {
        let mut sigmas = Vec::with_capacity(steps + 1);
        for i in 0..steps {
            let timestep = (1.0 - i as f64 / steps as f64) * 1000.0;
            let sigma = timestep / 1000.0;
            let sigma = shift as f64 * sigma / (1.0 + (shift as f64 - 1.0) * sigma);
            sigmas.push(sigma as f32);
        }
        sigmas.push(0.0);
        Self { sigmas }
    }

    /// Explicit timestep schedule (custom `timesteps` override).
    ///
    /// Mirrors the custom-timesteps semantics of diffusers'
    /// `FlowMatchEulerDiscreteScheduler.set_timesteps(timesteps=...)` — the
    /// scheduler class the reference `retrieve_timesteps` helper is written
    /// for: `sigmas = timesteps / num_train_timesteps`, the configured
    /// `shift` transform applied, and a terminal 0.0 appended. The denoise
    /// loop then runs EXACTLY `timesteps.len()` steps (`num_inference_steps`
    /// is derived from the list length, as in `retrieve_timesteps`).
    ///
    /// Upstream caveat (documented deviation): the SHIPPED TripoSG scheduler
    /// is `triposg.schedulers.scheduling_rectified_flow.RectifiedFlowScheduler`
    /// (see the canonical `resolved_config.json` `scheduler_config`), whose
    /// `set_timesteps` has no `timesteps` parameter — upstream,
    /// `retrieve_timesteps` would raise
    /// `"The current scheduler class ...'s set_timesteps does not support
    /// custom timestep schedules"` for this configuration. Lux3D implements
    /// the override with the FlowMatch semantics above so an explicit
    /// schedule is usable; with `shift = 1` the sigma transform is the
    /// identity. The list is used AS GIVEN (no monotonicity requirement,
    /// matching upstream): a non-descending schedule produces negative
    /// Euler deltas `sigma_i - sigma_{i+1}` there too. Callers should pass
    /// descending values in `(0, 1000]`.
    pub fn from_timesteps(timesteps: &[f32], shift: f32) -> Self {
        let mut sigmas = Vec::with_capacity(timesteps.len() + 1);
        for &timestep in timesteps {
            let sigma = f64::from(timestep) / SCHEDULER_NUM_TRAIN_TIMESTEPS;
            let sigma = shift as f64 * sigma / (1.0 + (shift as f64 - 1.0) * sigma);
            sigmas.push(sigma as f32);
        }
        sigmas.push(0.0);
        Self { sigmas }
    }

    /// The per-step timesteps fed to the DiT (`sigmas[i] * 1000`, F32).
    pub fn timesteps(&self) -> Vec<f32> {
        self.sigmas[..self.sigmas.len() - 1]
            .iter()
            .map(|sigma| sigma * 1000.0)
            .collect()
    }

    pub fn timestep(&self, index: usize) -> f32 {
        self.sigmas[index] * 1000.0
    }

    /// Euler update `x + (sigma_i - sigma_{i+1}) * v` in F32
    /// (`RectifiedFlowScheduler.step` with the sample upcast to F32).
    pub fn step(
        &self,
        index: usize,
        model_output: &Tensor,
        sample: &Tensor,
    ) -> CandleResult<Tensor> {
        if index + 1 >= self.sigmas.len() {
            candle_core::bail!(
                "RectifiedFlowSchedule step index {index} out of range ({} sigmas)",
                self.sigmas.len()
            );
        }
        let sigma = self.sigmas[index];
        let sigma_next = self.sigmas[index + 1];
        let delta = sigma - sigma_next;
        let sample = sample.to_dtype(DType::F32)?;
        let update = model_output
            .to_dtype(DType::F32)?
            .affine(delta as f64, 0.0)?;
        sample + update
    }
}

/// CFG denoise loop (`TripoSGPipeline.__call__` steps 4-6, denoise part only;
/// no RNG — the initial latents are supplied by the caller).
///
/// `latents_init [B=1, L, 64]`, `cond [B=1, 257, 1024]` conditional image
/// embeds (the uncond branch is `zeros_like(cond)`, batch order
/// `[uncond, cond]`). Public for the parity test harness.
pub fn denoise(
    dit: &TripoSgDiT,
    latents_init: &Tensor,
    cond: &Tensor,
    steps: usize,
    guidance: f32,
) -> CandleResult<Tensor> {
    denoise_with_dtype(dit, latents_init, cond, steps, guidance, DType::F32)
}

/// [`denoise`] with an explicit compute dtype. The network runs in `dtype`;
/// the Euler update stays in F32 (`RectifiedFlowScheduler.step` contract) and
/// is cast back for the next network pass. The returned latents are F32.
pub fn denoise_with_dtype(
    dit: &TripoSgDiT,
    latents_init: &Tensor,
    cond: &Tensor,
    steps: usize,
    guidance: f32,
    dtype: DType,
) -> CandleResult<Tensor> {
    let scheduler = RectifiedFlowSchedule::new(steps, SCHEDULER_SHIFT);
    denoise_with_schedule(dit, latents_init, cond, &scheduler, guidance, dtype)
}

/// [`denoise_with_dtype`] with a caller-built schedule — the custom
/// `timesteps` override path (`RectifiedFlowSchedule::from_timesteps`). The
/// loop runs `schedule.sigmas.len() - 1` steps, i.e. exactly the number of
/// timesteps the schedule was built from.
pub fn denoise_with_schedule(
    dit: &TripoSgDiT,
    latents_init: &Tensor,
    cond: &Tensor,
    scheduler: &RectifiedFlowSchedule,
    guidance: f32,
    dtype: DType,
) -> CandleResult<Tensor> {
    let steps = scheduler.sigmas.len().saturating_sub(1);
    let latents_init = latents_init.to_dtype(dtype)?;
    let cond = cond.to_dtype(dtype)?;
    let uncond_cond = Tensor::zeros_like(&cond)?;
    let mut latents = latents_init;
    for index in 0..steps {
        let batch = Tensor::cat(&[&latents, &latents], 0)?;
        let cond_batch = Tensor::cat(&[&uncond_cond, &cond], 0)?;
        let velocity = dit.forward(&batch, scheduler.timestep(index), &cond_batch)?;
        let parts = velocity.chunk(2, 0)?;
        // v = v_uncond + guidance * (v_cond - v_uncond), batch 0 = uncond.
        let combined = (&parts[0] + (&parts[1] - &parts[0])?.affine(guidance as f64, 0.0)?)?;
        latents = scheduler
            .step(index, &combined, &latents)?
            .to_dtype(dtype)?;
    }
    latents.to_dtype(DType::F32)
}

/// Dual-controller denoise loop of the reference scribble pipeline
/// (`TripoSGScribblePipeline.__call__`, `pipeline_triposg_scribble.py:258-291`).
///
/// Text (`encoder_hidden_states`) and scribble-image
/// (`encoder_hidden_states_2`) conditioning feed the two cross-attention
/// branches with independent output scales. Classifier-free guidance follows
/// the reference `do_classifier_free_guidance = guidance_scale > 1` rule; the
/// shipped scribble script passes `guidance_scale=0`, so the default path is a
/// single batch with no uncond pass. The negative branches, when guidance is
/// enabled, are `zeros_like` of the conditioning tensors exactly like
/// `encode_text`/`encode_image` (`:146, :158`).
///
/// `latents_init [B, L, 64]`, `text_cond [B, 77, 768]`,
/// `image_cond [B, 257, 1024]`; returns F32 latents.
#[allow(clippy::too_many_arguments)]
pub fn denoise_dual_with_schedule(
    dit: &TripoSgDiT,
    latents_init: &Tensor,
    text_cond: &Tensor,
    image_cond: &Tensor,
    scheduler: &RectifiedFlowSchedule,
    guidance: f32,
    cross_attention_scale: f32,
    cross_attention_2_scale: f32,
    dtype: DType,
) -> CandleResult<Tensor> {
    if !guidance.is_finite() {
        candle_core::bail!("TripoSG scribble guidance scale must be finite, got {guidance}");
    }
    let steps = scheduler.sigmas.len().saturating_sub(1);
    let latents_init = latents_init.to_dtype(dtype)?;
    let text_cond = text_cond.to_dtype(dtype)?;
    let image_cond = image_cond.to_dtype(dtype)?;
    let do_classifier_free_guidance = guidance > 1.0;

    let (text_cond, image_cond) = if do_classifier_free_guidance {
        (
            Tensor::cat(&[&Tensor::zeros_like(&text_cond)?, &text_cond], 0)?,
            Tensor::cat(&[&Tensor::zeros_like(&image_cond)?, &image_cond], 0)?,
        )
    } else {
        (text_cond, image_cond)
    };

    let mut latents = latents_init;
    for index in 0..steps {
        let latent_model_input = if do_classifier_free_guidance {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        let velocity = dit.forward_with_conditioning(
            &latent_model_input,
            scheduler.timestep(index),
            &text_cond,
            Some(&image_cond),
            cross_attention_scale,
            cross_attention_2_scale,
        )?;
        let velocity = if do_classifier_free_guidance {
            let parts = velocity.chunk(2, 0)?;
            // v = v_uncond + guidance * (v_cond - v_uncond), batch 0 = uncond.
            (&parts[0] + (&parts[1] - &parts[0])?.affine(guidance as f64, 0.0)?)?
        } else {
            velocity
        };
        latents = scheduler
            .step(index, &velocity, &latents)?
            .to_dtype(dtype)?;
    }
    latents.to_dtype(DType::F32)
}

#[cfg(test)]
mod dual_cross_attention_tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic pseudo-random weights in `[-0.5, 0.5)` for structural
    /// forward tests (no checkpoint needed).
    fn next_weight(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (*state >> 40) as f32 / (1u32 << 24) as f32;
        unit - 0.5
    }

    fn tensor(shape: &[usize], state: &mut u64, device: &Device) -> Tensor {
        let count: usize = shape.iter().product();
        let values = (0..count).map(|_| next_weight(state)).collect::<Vec<_>>();
        Tensor::from_vec(values, shape.to_vec(), device).expect("random tensor")
    }

    fn norm_tensor(size: usize, state: &mut u64, device: &Device) -> Tensor {
        let values = (0..size)
            .map(|_| 1.0 + next_weight(state) * 0.1)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, size, device).expect("norm weight")
    }

    fn zeros(size: usize, device: &Device) -> Tensor {
        Tensor::zeros(size, DType::F32, device).expect("zeros")
    }

    fn zero_tensor(shape: &[usize], device: &Device) -> Tensor {
        Tensor::zeros(shape.to_vec(), DType::F32, device).expect("zero tensor")
    }

    fn insert_attention(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        width: usize,
        kv_dim: usize,
        head_dim: usize,
        state: &mut u64,
        device: &Device,
    ) {
        map.insert(
            format!("{prefix}.to_q.weight"),
            tensor(&[width, width], state, device),
        );
        map.insert(
            format!("{prefix}.to_k.weight"),
            tensor(&[width, kv_dim], state, device),
        );
        map.insert(
            format!("{prefix}.to_v.weight"),
            tensor(&[width, kv_dim], state, device),
        );
        map.insert(
            format!("{prefix}.norm_q.weight"),
            norm_tensor(head_dim, state, device),
        );
        map.insert(
            format!("{prefix}.norm_k.weight"),
            norm_tensor(head_dim, state, device),
        );
        map.insert(
            format!("{prefix}.to_out.0.weight"),
            tensor(&[width, width], state, device),
        );
        map.insert(format!("{prefix}.to_out.0.bias"), zeros(width, device));
    }

    fn insert_block(
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        cfg: &TripoSgTransformerConfig,
        state: &mut u64,
        device: &Device,
        zero_ff: bool,
    ) {
        let width = cfg.width;
        let head_dim = cfg.head_dim();
        for norm in ["norm1", "norm2", "norm3"] {
            map.insert(
                format!("{prefix}.{norm}.weight"),
                norm_tensor(width, state, device),
            );
            map.insert(format!("{prefix}.{norm}.bias"), zeros(width, device));
        }
        insert_attention(
            map,
            &format!("{prefix}.attn1"),
            width,
            width,
            head_dim,
            state,
            device,
        );
        insert_attention(
            map,
            &format!("{prefix}.attn2"),
            width,
            cfg.cross_attention_dim,
            head_dim,
            state,
            device,
        );
        if cfg.use_cross_attention_2 {
            map.insert(
                format!("{prefix}.norm2_2.weight"),
                norm_tensor(width, state, device),
            );
            map.insert(format!("{prefix}.norm2_2.bias"), zeros(width, device));
            insert_attention(
                map,
                &format!("{prefix}.attn2_2"),
                width,
                cfg.cross_attention_2_dim.expect("dual config dim"),
                head_dim,
                state,
                device,
            );
        }
        let ff_inner = cfg.ff_inner_dim();
        if zero_ff {
            map.insert(
                format!("{prefix}.ff.net.0.proj.weight"),
                zero_tensor(&[ff_inner, width], device),
            );
            map.insert(
                format!("{prefix}.ff.net.0.proj.bias"),
                zeros(ff_inner, device),
            );
            map.insert(
                format!("{prefix}.ff.net.2.weight"),
                zero_tensor(&[width, ff_inner], device),
            );
            map.insert(format!("{prefix}.ff.net.2.bias"), zeros(width, device));
        } else {
            map.insert(
                format!("{prefix}.ff.net.0.proj.weight"),
                tensor(&[ff_inner, width], state, device),
            );
            map.insert(
                format!("{prefix}.ff.net.0.proj.bias"),
                zeros(ff_inner, device),
            );
            map.insert(
                format!("{prefix}.ff.net.2.weight"),
                tensor(&[width, ff_inner], state, device),
            );
            map.insert(format!("{prefix}.ff.net.2.bias"), zeros(width, device));
        }
    }

    fn tiny_config(dual: bool) -> TripoSgTransformerConfig {
        TripoSgTransformerConfig {
            in_channels: 4,
            width: 16,
            num_attention_heads: 2,
            num_layers: 3,
            cross_attention_dim: 8,
            use_cross_attention_2: dual,
            cross_attention_2_dim: dual.then_some(12),
            mlp_ratio: 4.0,
        }
    }

    /// Build a one-block VarBuilder with the FFN zeroed so the block output is
    /// exactly `hidden + cross_1 * s1 + cross_2 * s2` (no nonlinearity after
    /// the scaled sum).
    fn single_block_vars(
        cfg: &TripoSgTransformerConfig,
        device: &Device,
    ) -> (VarBuilder<'static>, Tensor, Tensor, Tensor) {
        let mut state = 42_u64;
        let mut map = HashMap::new();
        insert_block(&mut map, "blocks.0", cfg, &mut state, device, true);
        let vb = VarBuilder::from_tensors(map, DType::F32, device);
        let latents = tensor(&[1, 5, cfg.width], &mut state, device);
        let text = tensor(&[1, 3, cfg.cross_attention_dim], &mut state, device);
        let image = tensor(
            &[1, 4, cfg.cross_attention_2_dim.expect("dual dim")],
            &mut state,
            device,
        );
        (vb, latents, text, image)
    }

    #[test]
    fn per_branch_scales_are_applied_independently() {
        let device = Device::Cpu;
        let cfg = tiny_config(true);
        let (vb, latents, text, image) = single_block_vars(&cfg, &device);
        let block = DiTBlock::load(vb.pp("blocks").pp("0"), &cfg, 0).expect("dual block load");
        let run = |scale1: f32, scale2: f32| {
            block
                .forward(&latents, &text, Some(&image), None, 16, scale1, scale2)
                .expect("block forward")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("vec")
        };

        let base = run(0.0, 0.0);
        let branch1 = run(1.0, 0.0);
        let branch2 = run(0.0, 1.0);
        let half = run(0.5, 0.5);

        let delta1 = branch1
            .iter()
            .zip(&base)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let delta2 = branch2
            .iter()
            .zip(&base)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        assert!(
            delta1.iter().any(|d| d.abs() > 1e-6),
            "text cross-attention branch must influence the block output"
        );
        assert!(
            delta2.iter().any(|d| d.abs() > 1e-6),
            "second cross-attention branch must influence the block output"
        );

        // One block, zeroed FFN: out(s1, s2) = base + s1*delta1 + s2*delta2
        // exactly (both branches read the same pre-cross-attention hidden
        // state, matching triposg_transformer.py:333-349).
        let mut max_error = 0.0_f32;
        for index in 0..base.len() {
            let expected = base[index] + 0.5 * delta1[index] + 0.5 * delta2[index];
            max_error = max_error.max((half[index] - expected).abs());
        }
        assert!(
            max_error <= 2e-5,
            "per-branch scale linearity violated: max abs error {max_error:.3e}"
        );
    }

    #[test]
    fn single_cross_attention_path_is_unchanged() {
        let device = Device::Cpu;
        let single_cfg = tiny_config(false);
        let mut state = 7_u64;
        let mut map = HashMap::new();
        insert_block(&mut map, "blocks.0", &single_cfg, &mut state, &device, true);
        let vb = VarBuilder::from_tensors(map, DType::F32, &device);
        let block =
            DiTBlock::load(vb.pp("blocks").pp("0"), &single_cfg, 0).expect("single block load");
        let latents = tensor(&[1, 5, single_cfg.width], &mut state, &device);
        let text = tensor(&[1, 3, single_cfg.cross_attention_dim], &mut state, &device);
        let output = block
            .forward(&latents, &text, None, None, 16, 1.0, 1.0)
            .expect("single forward");
        assert_eq!(output.dims(), [1, 5, single_cfg.width]);
        assert!(
            output
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("vec")
                .iter()
                .all(|value| value.is_finite()),
            "single-attention block output must be finite"
        );
    }

    /// A full tiny DiT (3 blocks, skip on the last block) with deterministic
    /// random weights.
    fn build_tiny_dit(cfg: &TripoSgTransformerConfig, seed: u64) -> TripoSgDiT {
        let device = Device::Cpu;
        let mut state = seed;
        let mut map = HashMap::new();
        map.insert(
            "proj_in.weight".to_string(),
            tensor(&[cfg.width, cfg.in_channels], &mut state, &device),
        );
        map.insert("proj_in.bias".to_string(), zeros(cfg.width, &device));
        map.insert(
            "time_proj.linear_1.weight".to_string(),
            tensor(&[cfg.time_embed_dim(), cfg.width], &mut state, &device),
        );
        map.insert(
            "time_proj.linear_1.bias".to_string(),
            zeros(cfg.time_embed_dim(), &device),
        );
        map.insert(
            "time_proj.linear_2.weight".to_string(),
            tensor(&[cfg.width, cfg.time_embed_dim()], &mut state, &device),
        );
        map.insert(
            "time_proj.linear_2.bias".to_string(),
            zeros(cfg.width, &device),
        );
        for index in 0..cfg.num_layers {
            let prefix = format!("blocks.{index}");
            insert_block(&mut map, &prefix, cfg, &mut state, &device, false);
            if index > cfg.num_layers / 2 {
                map.insert(
                    format!("{prefix}.skip_norm.weight"),
                    norm_tensor(cfg.width, &mut state, &device),
                );
                map.insert(
                    format!("{prefix}.skip_norm.bias"),
                    zeros(cfg.width, &device),
                );
                map.insert(
                    format!("{prefix}.skip_linear.weight"),
                    tensor(&[cfg.width, 2 * cfg.width], &mut state, &device),
                );
                map.insert(
                    format!("{prefix}.skip_linear.bias"),
                    zeros(cfg.width, &device),
                );
            }
        }
        map.insert(
            "norm_out.weight".to_string(),
            norm_tensor(cfg.width, &mut state, &device),
        );
        map.insert("norm_out.bias".to_string(), zeros(cfg.width, &device));
        map.insert(
            "proj_out.weight".to_string(),
            tensor(&[cfg.in_channels, cfg.width], &mut state, &device),
        );
        map.insert("proj_out.bias".to_string(), zeros(cfg.in_channels, &device));
        TripoSgDiT::load(VarBuilder::from_tensors(map, DType::F32, &device), cfg)
            .expect("tiny DiT load")
    }

    /// Full 3-block DiT forward with random weights: dual conditioning
    /// produces the latent shape, the image path still runs, and mismatched
    /// conditioning is rejected.
    #[test]
    fn full_dit_dual_forward_shapes_and_guards() {
        let device = Device::Cpu;
        let cfg = tiny_config(true);
        let dit = build_tiny_dit(&cfg, 99);
        let mut state = 1234_u64;
        let latents = tensor(&[1, 5, cfg.in_channels], &mut state, &device);
        let text = tensor(&[1, 3, cfg.cross_attention_dim], &mut state, &device);
        let image = tensor(
            &[1, 4, cfg.cross_attention_2_dim.expect("dual dim")],
            &mut state,
            &device,
        );

        let dual = dit
            .forward_with_conditioning(&latents, 500.0, &text, Some(&image), 0.4, 1.0)
            .expect("dual forward");
        assert_eq!(dual.dims(), [1, 5, cfg.in_channels]);
        assert!(dual
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("vec")
            .iter()
            .all(|value| value.is_finite()));

        let missing_second = dit.forward(&latents, 500.0, &text).unwrap_err();
        assert!(
            missing_second
                .to_string()
                .contains("no encoder_hidden_states_2"),
            "missing second conditioning must be reported, got {missing_second}"
        );

        let single_cfg = tiny_config(false);
        let single = build_tiny_dit(&single_cfg, 5);
        let single_text = tensor(&[1, 3, single_cfg.cross_attention_dim], &mut state, &device);
        let single_latents = tensor(&[1, 5, single_cfg.in_channels], &mut state, &device);
        let output = single
            .forward_with_conditioning(&single_latents, 500.0, &single_text, None, 1.0, 1.0)
            .expect("single forward");
        assert_eq!(output.dims(), [1, 5, single_cfg.in_channels]);

        let rejected = single
            .forward_with_conditioning(&single_latents, 500.0, &single_text, Some(&image), 1.0, 1.0)
            .unwrap_err();
        assert!(
            rejected.to_string().contains("does not use"),
            "unexpected second conditioning must be rejected, got {rejected}"
        );
    }
}
