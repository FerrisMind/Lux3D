//! TripoSG neural stage: the static latent-scene contract for the DINOv2 +
//! DiT rectified-flow stack, plus a thin helper that drives the pipeline for
//! image embeds -> seeded latents -> denoise (no math is duplicated here —
//! everything delegates to [`TripoSgPipeline`]).

use candle_core::Tensor;

use crate::{
    contracts::{ModelFamily, NeuralStage, TensorContract, TripoSgPreparedImage, TripoSgSceneCode},
    error::Lux3dError,
    runtime::{
        repeat_interleave_batch, TripoSgInferenceOutput, TripoSgPipeline, TripoSgRunOptions,
        TRIPOSG_LATENT_CHANNELS,
    },
    Result,
};

/// Static description of the TripoSG latent scene produced by the DiT
/// rectified-flow denoise loop. Defaults match the reference pipeline call
/// (`num_inference_steps=50`, `guidance_scale=7.0`) and the shipped configs
/// (`dinov2-large` conditioning, 21 DiT blocks of width 2048,
/// cross-attention dim 1024, `[1, 2048, 64]` latents).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TripoSgNeuralStage;

impl NeuralStage<TripoSgPreparedImage, TripoSgSceneCode> for TripoSgNeuralStage {
    fn infer(&self, _input: &TripoSgPreparedImage) -> Result<TripoSgSceneCode> {
        Ok(TripoSgSceneCode {
            family: ModelFamily::TripoSg,
            image_encoder: "dinov2-large",
            dit_layers: 21,
            dit_width: 2048,
            cross_attention_dim: 1024,
            latent_tokens: 2048,
            latent_channels: 64,
            guidance_scale: 7.0,
            num_inference_steps: 50,
            latents: TensorContract::new("triposg.latents", vec![1, 2048, 64]),
        })
    }
}

impl TripoSgNeuralStage {
    /// Conditioning -> latents for a single image and a single shape (the
    /// historical entry point). Delegates to
    /// [`Self::infer_latents_batched`] with `num_shapes = 1`, which is the
    /// identity expansion and keeps the seeded draw byte-identical.
    pub fn infer_latents(
        &self,
        pipeline: &TripoSgPipeline,
        pixel_values: &Tensor,
        options: &TripoSgRunOptions,
    ) -> Result<TripoSgInferenceOutput> {
        self.infer_latents_batched(pipeline, pixel_values, 1, options)
    }

    /// Batched conditioning -> latents: DINOv2 embeds for the `N` images in
    /// `pixel_values [N, 3, 224, 224]`, expanded with
    /// `repeat_interleave(num_shapes)` exactly like the reference
    /// `encode_image` (`pipeline_triposg.py:138-149`), the Lux3D-seeded
    /// `[N * num_shapes, num_tokens, 64]` init latents drawn as ONE tensor
    /// (`prepare_latents`, `:151-174`), then the batched CFG denoise loop
    /// (`:243-275`). The official RNG deviation stays documented: the draw
    /// STRUCTURE matches (one `[B, ...]` tensor), not the torch values.
    pub fn infer_latents_batched(
        &self,
        pipeline: &TripoSgPipeline,
        pixel_values: &Tensor,
        num_shapes: usize,
        options: &TripoSgRunOptions,
    ) -> Result<TripoSgInferenceOutput> {
        if num_shapes == 0 {
            return Err(Lux3dError::InvalidInput(
                "TripoSG num_shapes must be at least 1, got 0",
            ));
        }
        let image_embeds = pipeline.image_embeds(pixel_values)?;
        crate::runtime::log_wgpu_alloc(pixel_values.device(), "wgpu-dino");
        // Reference `image_embeds.repeat_interleave(num_images_per_prompt,
        // dim=0)` (`pipeline_triposg.py:146`): image-major order, each image
        // repeated `num_shapes` times.
        let image_embeds = repeat_interleave_batch(&image_embeds, num_shapes)?;
        let batch =
            image_embeds
                .dim(0)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("TripoSG conditioning batch dim unavailable: {source}"),
                })?;
        let latents_init = TripoSgPipeline::sample_latents_batched(
            options.seed,
            batch,
            options.num_tokens,
            TRIPOSG_LATENT_CHANNELS,
            pixel_values.device(),
        )?;
        let latents = pipeline.infer_with_latents(&image_embeds, &latents_init, options)?;
        crate::runtime::log_wgpu_alloc(pixel_values.device(), "wgpu-denoise");
        Ok(TripoSgInferenceOutput {
            image_embeds,
            latents_init,
            latents,
        })
    }
}
