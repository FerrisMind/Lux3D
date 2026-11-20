//! Offline safetensors -> GGUF quantization for canonical packages.
//!
//! Quantizes rank-2 `*.weight` tensors (the linear layers) and stores
//! everything else — norms, biases, embeddings, convolutions — in the
//! fallback dtype. The resulting `model.gguf` sits next to the canonical
//! `model.safetensors`; [`super::CanonicalWeightSetPaths::var_builder`] picks
//! it up automatically.
//!
//! Candle's GGUF reader caps tensor rank at four, so tensors with a higher
//! rank (3D convolution kernels) are stored as flat vectors; the runtime
//! loaders reshape them back after dequantization
//! ([`crate::runtime::nn_blocks::get_weight_shaped`]).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{Device, Tensor};

use crate::{
    Result,
    contracts::ModelFamily,
    error::Lux3dError,
    weights::{
        GGUF_FAMILY_KEY, GGUF_RESOLVED_CONFIG_KEY, GGUF_VARIANT_KEY, RESOLVED_CONFIG_FILENAME,
        TRIPOSG_SCRIBBLE_VARIANT,
    },
};

/// Quantization policy for [`quantize_checkpoint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizeOptions {
    /// Dtype used for rank-2 `*.weight` tensors whose row length is a multiple
    /// of the dtype block size.
    pub dtype: GgmlDType,
    /// Dtype for everything else (norms, biases, embeddings, convolutions,
    /// and weights whose row length does not divide into blocks).
    pub fallback: GgmlDType,
    /// Quantize every rank-2 tensor instead of only `*.weight`.
    pub all_2d: bool,
    /// Optional name prefixes: when non-empty, only matching tensors are
    /// quantized. Useful for isolating a sub-model (`transformer.`,
    /// `vae.`, `image_encoder.`) during accuracy bisection.
    pub prefixes: Vec<String>,
}

impl Default for QuantizeOptions {
    fn default() -> Self {
        Self {
            dtype: GgmlDType::Q8_0,
            fallback: GgmlDType::F16,
            all_2d: false,
            prefixes: Vec::new(),
        }
    }
}

/// An additional safetensors file merged into a standalone GGUF.
///
/// Used for stage checkpoints that live outside the canonical package, such
/// as the TRELLIS.2 `shape_enc_next_dc_f16c32_fp16` encoder. Every tensor is
/// written as `<prefix>.<original name>`; the merged tensors follow the
/// canonical file's tensors in the output order, so the GGUF layout stays
/// deterministic. The merge file must not reuse a name from the canonical
/// package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizeMergeSource {
    /// Safetensors file to merge.
    pub path: PathBuf,
    /// Tensor-name prefix (for example `shape_enc`), with surrounding dots
    /// trimmed.
    pub prefix: String,
}

impl QuantizeMergeSource {
    pub fn new(path: impl AsRef<Path>, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into().trim_matches('.').to_string();
        Self {
            path: path.as_ref().to_path_buf(),
            prefix,
        }
    }
}

/// Summary of a completed quantization run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizeReport {
    pub output: PathBuf,
    pub dtype: GgmlDType,
    pub fallback: GgmlDType,
    pub tensors: usize,
    pub quantized: usize,
    pub fallback_tensors: usize,
    /// Rank-0 tensors (e.g. `num_batches_tracked`) cannot be represented in
    /// GGML and are omitted from the GGUF.
    pub skipped_tensors: usize,
    pub source_bytes: u64,
    pub output_bytes: u64,
    /// Whether `resolved_config.json` was embedded for standalone loading.
    pub embedded_resolved_config: bool,
}

impl QuantizeReport {
    /// Human-readable one-line summary for the CLI.
    pub fn render(&self) -> String {
        let skipped = if self.skipped_tensors == 0 {
            String::new()
        } else {
            format!(", {} scalars skipped", self.skipped_tensors)
        };
        format!(
            "{}: {} tensors ({} {}, {} {}){skipped}, {:.2} GiB -> {:.2} GiB{}",
            self.output.display(),
            self.tensors,
            self.quantized,
            dtype_name(self.dtype),
            self.fallback_tensors,
            dtype_name(self.fallback),
            self.source_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.output_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            if self.embedded_resolved_config {
                ", resolved config embedded"
            } else {
                ""
            },
        )
    }
}

/// Quantize a canonical `model.safetensors` into `output` (`.gguf`).
pub fn quantize_checkpoint(
    source: &Path,
    output: &Path,
    family: ModelFamily,
    options: QuantizeOptions,
) -> Result<QuantizeReport> {
    quantize_checkpoint_with_family(source, output, family.as_str(), options)
}

/// Quantize a canonical `model.safetensors` and merge extra safetensors
/// stages into the same standalone GGUF, prefixing each stage's tensor names
/// (for example `shape_enc.input_layer.weight`).
pub fn quantize_checkpoint_with_merges(
    source: &Path,
    output: &Path,
    family: ModelFamily,
    options: QuantizeOptions,
    merge_sources: &[QuantizeMergeSource],
) -> Result<QuantizeReport> {
    quantize_checkpoint_with_family_and_merges(
        source,
        output,
        family.as_str(),
        options,
        merge_sources,
    )
}

/// Resolve the TripoSG-scribble DiT safetensors from a raw
/// `VAST-AI/TripoSG-scribble` checkpoint directory or a direct shard path.
///
/// Only the DiT is quantized (the reference quantization policy); the text
/// encoder and VAE keep their raw float safetensors.
pub fn resolve_scribble_dit_source(source: &Path) -> Result<PathBuf> {
    if source.is_file() {
        return Ok(source.to_path_buf());
    }
    if source.is_dir() {
        let candidate = source
            .join("transformer")
            .join("diffusion_pytorch_model.safetensors");
        if candidate.is_file() {
            return Ok(candidate);
        }
        return Err(Lux3dError::MissingCanonicalArtifact { path: candidate });
    }
    Err(Lux3dError::MissingCanonicalArtifact {
        path: source.to_path_buf(),
    })
}

/// Components merged into one standalone TripoSG-scribble GGUF.
///
/// The order is fixed so the output tensor order is deterministic. Only the
/// `transformer.` prefix is quantized by default; the other components stay in
/// the fallback dtype, matching the evaluated DiT-only quantization policy.
const SCRIBBLE_STANDALONE_COMPONENTS: [(&str, &str); 4] = [
    (
        "transformer",
        "transformer/diffusion_pytorch_model.safetensors",
    ),
    ("vae", "vae/diffusion_pytorch_model.safetensors"),
    ("image_encoder", "image_encoder_dinov2/model.safetensors"),
    ("text_encoder", "text_encoder/model.safetensors"),
];

/// Build one self-contained GGUF from a raw `VAST-AI/TripoSG-scribble`
/// checkpoint.
///
/// The output contains all four network components under `transformer.`,
/// `vae.`, `image_encoder.` and `text_encoder.` prefixes plus the embedded
/// resolved config and CLIP tokenizer documents. The quantizer keeps the
/// reference policy: only rank-2 `.weight` tensors inside `transformer.` are
/// quantized; every other tensor (including the whole VAE, DINOv2 and CLIP
/// text encoder) is stored in the fallback dtype.
pub fn quantize_scribble_standalone(
    source: &Path,
    output: &Path,
    options: QuantizeOptions,
) -> Result<QuantizeReport> {
    if !source.is_dir() {
        return Err(Lux3dError::MissingCanonicalArtifact {
            path: source.to_path_buf(),
        });
    }
    if matches!(options.dtype, GgmlDType::Q8K | GgmlDType::Q8_1) {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "{} is an activation-side GGML type, not a weight quantization; use q8_0",
                dtype_name(options.dtype)
            ),
        });
    }

    let mut options = options;
    if options.prefixes.is_empty() {
        options.prefixes.push("transformer.".to_string());
    }

    let mut tensors = Vec::new();
    let mut quantized = 0usize;
    let mut fallback = 0usize;
    let mut skipped = 0usize;
    let mut source_bytes = 0u64;
    for (prefix, relative) in SCRIBBLE_STANDALONE_COMPONENTS {
        let path = source.join(relative);
        if !path.is_file() {
            return Err(Lux3dError::MissingCanonicalArtifact { path });
        }
        source_bytes += fs::metadata(&path)
            .map_err(|source| Lux3dError::ModelAssetIo {
                path: path.clone(),
                source,
            })?
            .len();
        // SAFETY: the raw checkpoint is read-only for the duration of this
        // function and the mapping is dropped before returning.
        let mmap = unsafe { MmapedSafetensors::new(&path) }.map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to open {}: {source}", path.display()),
            }
        })?;
        for (name, _) in mmap.tensors() {
            let tensor = mmap.load(&name, &Device::Cpu).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to load tensor `{name}` from {}: {source}",
                        path.display()
                    ),
                }
            })?;
            if tensor.rank() == 0 {
                skipped += 1;
                continue;
            }
            let full_name = format!("{prefix}.{name}");
            let use_quant = should_quantize(&full_name, &tensor, &options);
            let stored = if tensor.rank() > 4 {
                tensor
                    .flatten_all()
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to flatten tensor `{full_name}`: {source}"),
                    })?
            } else {
                tensor
            };
            let dtype = if use_quant {
                quantized += 1;
                options.dtype
            } else {
                fallback += 1;
                options.fallback
            };
            let qtensor = QTensor::quantize(&stored, dtype).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to quantize tensor `{full_name}` {:?} to {dtype:?}: {source}",
                        stored.shape()
                    ),
                }
            })?;
            tensors.push((full_name, qtensor));
        }
    }
    tensors.sort_by(|left, right| left.0.cmp(&right.0));

    let resolved_config = build_scribble_standalone_config(source)?;
    let serialized = serde_json::to_string(&resolved_config).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to serialize TripoSG-scribble resolved config: {source}"),
        }
    })?;

    write_quantized_package(
        output,
        "triposg",
        "triposg",
        &options,
        &tensors,
        quantized,
        fallback,
        skipped,
        source_bytes,
        Some(serialized),
        &[
            (GGUF_VARIANT_KEY, TRIPOSG_SCRIBBLE_VARIANT.to_string()),
            (
                "lux3d.components",
                "image_encoder,transformer,vae,text_encoder".to_string(),
            ),
        ],
    )
}

fn build_scribble_standalone_config(source: &Path) -> Result<serde_json::Value> {
    let model_index = read_scribble_json(source, "model_index.json")?;
    let preprocessor =
        read_scribble_json(source, "feature_extractor_dinov2/preprocessor_config.json")?;
    let image_encoder = read_scribble_json(source, "image_encoder_dinov2/config.json")?;
    let transformer = read_scribble_json(source, "transformer/config.json")?;
    let vae = read_scribble_json(source, "vae/config.json")?;
    let text_encoder = read_scribble_json(source, "text_encoder/config.json")?;
    let scheduler_config = read_scribble_json(source, "scheduler/scheduler_config.json")?;
    let tokenizer_config = read_scribble_json(source, "tokenizer/tokenizer_config.json")?;
    let special_tokens_map = read_scribble_json(source, "tokenizer/special_tokens_map.json")?;
    let vocab_json = read_scribble_text(source, "tokenizer/vocab.json")?;
    let merges_txt = read_scribble_text(source, "tokenizer/merges.txt")?;

    Ok(serde_json::json!({
        "family": "triposg",
        "variant": TRIPOSG_SCRIBBLE_VARIANT,
        "model_index": model_index,
        "preprocessor": preprocessor,
        "image_encoder": image_encoder,
        "transformer": transformer,
        "vae": vae,
        "text_encoder": text_encoder,
        "scheduler_config": scheduler_config,
        "pipeline_defaults": {
            "num_tokens": 512,
            "num_inference_steps": 16,
            "guidance_scale": 0.0,
            "scribble_confidence": 0.4,
            "prompt_confidence": 1.0,
            "bounds": [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005],
            "dense_octree_depth": 8,
            "hierarchical_octree_depth": 8,
            "sdf_level": 0.0
        },
        "tokenizer": {
            "vocab_json": vocab_json,
            "merges_txt": merges_txt,
            "config": tokenizer_config,
            "special_tokens_map": special_tokens_map
        }
    }))
}

fn read_scribble_json(source: &Path, relative: &str) -> Result<serde_json::Value> {
    let path = source.join(relative);
    let body = fs::read_to_string(&path).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.clone(),
        source,
    })?;
    serde_json::from_str(&body).map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("failed to parse `{}`: {source}", path.display()),
    })
}

fn read_scribble_text(source: &Path, relative: &str) -> Result<String> {
    let path = source.join(relative);
    fs::read_to_string(&path).map_err(|source| Lux3dError::ModelAssetIo { path, source })
}

/// Quantize an auxiliary package (for example `rmbg`) whose family is not a
/// [`ModelFamily`]. The resulting GGUF is standalone when the source package
/// has a sibling `resolved_config.json`.
pub fn quantize_auxiliary_checkpoint(
    source: &Path,
    output: &Path,
    family: &str,
    options: QuantizeOptions,
) -> Result<QuantizeReport> {
    if family.trim().is_empty() {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: "auxiliary family label must not be empty".to_string(),
        });
    }
    quantize_checkpoint_with_family(source, output, family.trim(), options)
}

fn quantize_checkpoint_with_family(
    source: &Path,
    output: &Path,
    family: &str,
    options: QuantizeOptions,
) -> Result<QuantizeReport> {
    quantize_checkpoint_with_family_and_merges(source, output, family, options, &[])
}

/// Load one safetensors source, prefix its tensor names, and append the
/// quantized tensors to the package. `source_bytes` accumulates the input
/// sizes so the report covers the canonical file plus every merge source.
#[allow(clippy::too_many_arguments)]
fn append_source_tensors(
    source: &Path,
    prefix: &str,
    options: &QuantizeOptions,
    names: &mut HashSet<String>,
    tensors: &mut Vec<(String, QTensor)>,
    quantized: &mut usize,
    fallback: &mut usize,
    skipped: &mut usize,
    source_bytes: &mut u64,
) -> Result<()> {
    if !source.is_file() {
        return Err(Lux3dError::MissingCanonicalArtifact {
            path: source.to_path_buf(),
        });
    }
    *source_bytes += fs::metadata(source)
        .map_err(|err| Lux3dError::ModelAssetIo {
            path: source.to_path_buf(),
            source: err,
        })?
        .len();
    // SAFETY: the file is read-only for the duration of this function and the
    // mapping is dropped before returning.
    let mmap =
        unsafe { MmapedSafetensors::multi(std::slice::from_ref(&source)) }.map_err(|err| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to open {}: {err}", source.display()),
            }
        })?;
    tensors.reserve(mmap.tensors().len());
    for (name, view) in mmap.tensors() {
        let output_name = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}.{name}")
        };
        if !names.insert(output_name.clone()) {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "duplicate tensor `{output_name}` while merging `{}`",
                    source.display()
                ),
            });
        }
        let tensor = mmap.load(&name, &Device::Cpu).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load tensor `{name}`: {source}"),
            }
        })?;
        // GGML has no scalar dtype representation; training-only buffers such
        // as BatchNorm `num_batches_tracked` are skipped.
        if tensor.rank() == 0 {
            *skipped += 1;
            continue;
        }
        let use_quant = should_quantize(&output_name, &tensor, options);
        // Candle's GGUF reader rejects tensors with more than four
        // dimensions. The only rank>4 tensors in canonical packages are 3D
        // convolution kernels; store them as flat vectors (never quantized:
        // the rank check in `should_quantize` already saw the original rank)
        // and let the runtime loaders reshape after dequantization.
        let stored = if tensor.rank() > 4 {
            tensor
                .flatten_all()
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to flatten tensor `{output_name}`: {source}"),
                })?
        } else {
            tensor
        };
        let dtype = if use_quant {
            *quantized += 1;
            options.dtype
        } else {
            *fallback += 1;
            options.fallback
        };
        let qtensor = QTensor::quantize(&stored, dtype).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to quantize tensor `{output_name}` {:?} to {dtype:?}: {source}",
                    view.shape()
                ),
            }
        })?;
        tensors.push((output_name, qtensor));
    }
    Ok(())
}

fn quantize_checkpoint_with_family_and_merges(
    source: &Path,
    output: &Path,
    family: &str,
    options: QuantizeOptions,
    merge_sources: &[QuantizeMergeSource],
) -> Result<QuantizeReport> {
    if !source.is_file() {
        return Err(Lux3dError::MissingCanonicalArtifact {
            path: source.to_path_buf(),
        });
    }
    if matches!(options.dtype, GgmlDType::Q8K | GgmlDType::Q8_1) {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "{} is an activation-side GGML type, not a weight quantization; use q8_0",
                dtype_name(options.dtype)
            ),
        });
    }
    let mut source_bytes = 0u64;
    let mut tensors = Vec::new();
    let mut names = HashSet::new();
    let mut quantized = 0usize;
    let mut fallback = 0usize;
    let mut skipped = 0usize;
    append_source_tensors(
        source,
        "",
        &options,
        &mut names,
        &mut tensors,
        &mut quantized,
        &mut fallback,
        &mut skipped,
        &mut source_bytes,
    )?;
    for merge in merge_sources {
        if merge.prefix.is_empty() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "merge source `{}` has an empty tensor prefix",
                    merge.path.display()
                ),
            });
        }
        append_source_tensors(
            &merge.path,
            &merge.prefix,
            &options,
            &mut names,
            &mut tensors,
            &mut quantized,
            &mut fallback,
            &mut skipped,
            &mut source_bytes,
        )?;
    }

    // Embed the sibling resolved config so the GGUF can be used on its own,
    // without `model.safetensors` / manifest / checksums. `general.architecture`
    // mirrors the config's top-level `architecture`, then `family`, then the
    // requested model family.
    let resolved_config_path = source.with_file_name(RESOLVED_CONFIG_FILENAME);
    let (embedded_resolved_config, architecture) = if resolved_config_path.is_file() {
        let body = fs::read_to_string(&resolved_config_path).map_err(|source_err| {
            Lux3dError::ModelAssetIo {
                path: resolved_config_path.clone(),
                source: source_err,
            }
        })?;
        let parsed: serde_json::Value = serde_json::from_str(&body).map_err(|source_err| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to parse `{}`: {source_err}",
                    resolved_config_path.display()
                ),
            }
        })?;
        let architecture = resolve_architecture(Some(&parsed), family);
        let serialized = serde_json::to_string(&parsed).map_err(|source_err| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to re-serialize `{}`: {source_err}",
                    resolved_config_path.display()
                ),
            }
        })?;
        (Some(serialized), architecture)
    } else {
        (None, resolve_architecture(None, family))
    };

    write_quantized_package(
        output,
        family,
        &architecture,
        &options,
        &tensors,
        quantized,
        fallback,
        skipped,
        source_bytes,
        embedded_resolved_config,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn write_quantized_package(
    output: &Path,
    family: &str,
    architecture: &str,
    options: &QuantizeOptions,
    tensors: &[(String, QTensor)],
    quantized: usize,
    fallback_tensors: usize,
    skipped_tensors: usize,
    source_bytes: u64,
    embedded_resolved_config: Option<String>,
    extra_metadata: &[(&str, String)],
) -> Result<QuantizeReport> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|source| Lux3dError::ModelAssetIo {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut file = fs::File::create(output).map_err(|source| Lux3dError::ModelAssetIo {
        path: output.to_path_buf(),
        source,
    })?;
    let mut metadata = vec![
        (
            "general.architecture",
            gguf_file::Value::String(architecture.to_string()),
        ),
        (
            GGUF_FAMILY_KEY,
            gguf_file::Value::String(family.to_string()),
        ),
        (
            "lux3d.quantization",
            gguf_file::Value::String(dtype_name(options.dtype).into()),
        ),
        (
            "lux3d.fallback_dtype",
            gguf_file::Value::String(dtype_name(options.fallback).into()),
        ),
        (
            "lux3d.tensor_count",
            gguf_file::Value::U32(tensors.len().try_into().unwrap_or(u32::MAX)),
        ),
    ];
    if let Some(config) = embedded_resolved_config.as_ref() {
        metadata.push((
            GGUF_RESOLVED_CONFIG_KEY,
            gguf_file::Value::String(config.clone()),
        ));
    }
    for (key, value) in extra_metadata {
        metadata.push((*key, gguf_file::Value::String(value.clone())));
    }
    let metadata_refs: Vec<(&str, &gguf_file::Value)> = metadata
        .iter()
        .map(|(name, value)| (*name, value))
        .collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    gguf_file::write(&mut file, &metadata_refs, &tensor_refs).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to write {}: {source}", output.display()),
        }
    })?;
    drop(file);

    let output_bytes = fs::metadata(output)
        .map_err(|source| Lux3dError::ModelAssetIo {
            path: output.to_path_buf(),
            source,
        })?
        .len();
    Ok(QuantizeReport {
        output: output.to_path_buf(),
        dtype: options.dtype,
        fallback: options.fallback,
        tensors: tensors.len(),
        quantized,
        fallback_tensors,
        skipped_tensors,
        source_bytes,
        output_bytes,
        embedded_resolved_config: embedded_resolved_config.is_some(),
    })
}

fn should_quantize(name: &str, tensor: &Tensor, options: &QuantizeOptions) -> bool {
    if tensor.rank() != 2 {
        return false;
    }
    if !options.all_2d && !name.ends_with(".weight") {
        return false;
    }
    if !options.prefixes.is_empty()
        && !options
            .prefixes
            .iter()
            .any(|prefix| name.starts_with(prefix))
    {
        return false;
    }
    match tensor.dim(1) {
        Ok(row) => row % options.dtype.block_size() == 0,
        Err(_) => false,
    }
}

/// `general.architecture` value: top-level `architecture`, then `family`,
/// then the caller-provided model name.
fn resolve_architecture(config: Option<&serde_json::Value>, fallback: &str) -> String {
    config
        .and_then(|config| {
            config
                .get("architecture")
                .and_then(serde_json::Value::as_str)
                .or_else(|| config.get("family").and_then(serde_json::Value::as_str))
        })
        .unwrap_or(fallback)
        .to_string()
}

/// GGUF dtype name used in metadata and CLI parsing.
pub const fn dtype_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::BF16 => "bf16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        GgmlDType::Q8K => "q8_k",
    }
}

/// Parse a GGUF dtype name; `None` for unknown names.
pub fn dtype_from_name(name: &str) -> Option<GgmlDType> {
    Some(match name.to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "q4_0" => GgmlDType::Q4_0,
        "q4_1" => GgmlDType::Q4_1,
        "q5_0" => GgmlDType::Q5_0,
        "q5_1" => GgmlDType::Q5_1,
        "q8_0" => GgmlDType::Q8_0,
        "q8_1" => GgmlDType::Q8_1,
        "q2_k" => GgmlDType::Q2K,
        "q3_k" => GgmlDType::Q3K,
        "q4_k" => GgmlDType::Q4K,
        "q5_k" => GgmlDType::Q5K,
        "q6_k" => GgmlDType::Q6K,
        "q8_k" => GgmlDType::Q8K,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        QuantizeMergeSource, QuantizeOptions, dtype_from_name, dtype_name,
        quantize_auxiliary_checkpoint, quantize_checkpoint, quantize_checkpoint_with_merges,
        quantize_scribble_standalone, resolve_architecture, resolve_scribble_dit_source,
        should_quantize,
    };
    use crate::contracts::ModelFamily;
    use candle_core::quantized::{GgmlDType, gguf_file};
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn architecture_prefers_config_architecture_then_family() {
        let explicit = serde_json::json!({"architecture": "BriaRMBG", "family": "rmbg"});
        assert_eq!(
            resolve_architecture(Some(&explicit), "fallback"),
            "BriaRMBG"
        );
        let family = serde_json::json!({"family": "triposg"});
        assert_eq!(resolve_architecture(Some(&family), "fallback"), "triposg");
        let neither = serde_json::json!({"hidden_size": 64});
        assert_eq!(resolve_architecture(Some(&neither), "vggt"), "vggt");
        assert_eq!(resolve_architecture(None, "rmbg"), "rmbg");
    }

    #[test]
    fn dtype_names_round_trip() {
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q4_0,
            GgmlDType::Q5_0,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8K,
        ] {
            assert_eq!(dtype_from_name(dtype_name(dtype)), Some(dtype));
        }
        assert_eq!(dtype_from_name("nope"), None);
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "lux3d-quantize-{tag}-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_checkpoint(dir: &std::path::Path, with_config: bool) -> PathBuf {
        let cpu = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "transformer.blocks.0.attn.to_q.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        tensors.insert(
            "transformer.blocks.0.norm.weight".to_string(),
            Tensor::rand(0.5f32, 1.5f32, (64,), &cpu).unwrap(),
        );
        // Row length 10 is not a multiple of the Q8_0 block size.
        tensors.insert(
            "transformer.blocks.0.proj.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 10), &cpu).unwrap(),
        );
        // Rank-2 but no `.weight` suffix: only quantized with `all_2d`.
        tensors.insert(
            "transformer.blocks.0.table".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        // Rank-0 training-only buffer: cannot be represented in GGML.
        tensors.insert(
            "transformer.blocks.0.num_batches_tracked".to_string(),
            Tensor::zeros((), candle_core::DType::F32, &cpu).unwrap(),
        );
        if with_config {
            std::fs::write(
                dir.join("resolved_config.json"),
                r#"{"hidden_size":64,"num_layers":1}"#,
            )
            .unwrap();
        }
        let source = dir.join("model.safetensors");
        candle_core::safetensors::save(&tensors, &source).unwrap();
        source
    }

    fn tiny_scribble_checkpoint(dir: &std::path::Path) -> PathBuf {
        let cpu = Device::Cpu;
        let mut transformer = HashMap::new();
        transformer.insert(
            "blocks.0.attn.to_q.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        transformer.insert(
            "blocks.0.norm.weight".to_string(),
            Tensor::rand(0.5f32, 1.5f32, (64,), &cpu).unwrap(),
        );
        transformer.insert(
            "blocks.0.num_batches_tracked".to_string(),
            Tensor::zeros((), candle_core::DType::F32, &cpu).unwrap(),
        );
        let mut vae = HashMap::new();
        vae.insert(
            "decoder.conv.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        let mut image_encoder = HashMap::new();
        image_encoder.insert(
            "embeddings.patch_embeddings.projection.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        let mut text_encoder = HashMap::new();
        text_encoder.insert(
            "text_model.encoder.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );

        for (relative, tensors) in [
            (
                "transformer/diffusion_pytorch_model.safetensors",
                &transformer,
            ),
            ("vae/diffusion_pytorch_model.safetensors", &vae),
            ("image_encoder_dinov2/model.safetensors", &image_encoder),
            ("text_encoder/model.safetensors", &text_encoder),
        ] {
            let path = dir.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            candle_core::safetensors::save(tensors, &path).unwrap();
        }
        for (relative, body) in [
            ("model_index.json", "{}"),
            ("feature_extractor_dinov2/preprocessor_config.json", "{}"),
            (
                "image_encoder_dinov2/config.json",
                r#"{"hidden_size":1024,"num_hidden_layers":24}"#,
            ),
            (
                "transformer/config.json",
                r#"{"use_cross_attention_2":true}"#,
            ),
            ("vae/config.json", "{}"),
            ("text_encoder/config.json", "{}"),
            ("scheduler/scheduler_config.json", "{}"),
            (
                "tokenizer/tokenizer_config.json",
                r#"{"model_max_length":77}"#,
            ),
            ("tokenizer/special_tokens_map.json", "{}"),
            (
                "tokenizer/vocab.json",
                r#"{"<|startoftext|>":49406,"<|endoftext|>":49407}"#,
            ),
            ("tokenizer/merges.txt", "#version: 0.2\n<|startoftext|> a\n"),
        ] {
            let path = dir.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        }
        dir.to_path_buf()
    }

    fn read_gguf(path: &std::path::Path) -> gguf_file::Content {
        let mut file = std::fs::File::open(path).unwrap();
        gguf_file::Content::read(&mut file).unwrap()
    }

    fn gguf_architecture(content: &gguf_file::Content) -> &str {
        match content.metadata.get("general.architecture") {
            Some(gguf_file::Value::String(value)) => value,
            other => panic!("missing general.architecture: {other:?}"),
        }
    }

    #[test]
    fn quantize_selects_linear_weights_and_keeps_fallback() {
        let dir = scratch_dir("policy");
        let source = tiny_checkpoint(&dir, false);
        let output = dir.join("model.gguf");
        let report = quantize_checkpoint(
            &source,
            &output,
            ModelFamily::Vggt,
            QuantizeOptions::default(),
        )
        .unwrap();
        assert_eq!(report.tensors, 4);
        assert_eq!(report.quantized, 1);
        assert_eq!(report.fallback_tensors, 3);
        assert_eq!(report.skipped_tensors, 1);
        assert!(output.is_file());

        let content = read_gguf(&output);
        assert_eq!(gguf_architecture(&content), "vggt");
        assert_eq!(content.tensor_infos.len(), 4);
        assert!(
            !content
                .tensor_infos
                .contains_key("transformer.blocks.0.num_batches_tracked")
        );
        assert_eq!(
            content.tensor_infos["transformer.blocks.0.attn.to_q.weight"].ggml_dtype,
            GgmlDType::Q8_0
        );
        assert_eq!(
            content.tensor_infos["transformer.blocks.0.norm.weight"].ggml_dtype,
            GgmlDType::F16
        );
        assert_eq!(
            content.tensor_infos["transformer.blocks.0.proj.weight"].ggml_dtype,
            GgmlDType::F16
        );
        assert_eq!(
            content.tensor_infos["transformer.blocks.0.table"].ggml_dtype,
            GgmlDType::F16
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantize_merges_extra_stage_with_prefix() {
        let dir = scratch_dir("merge");
        let source = tiny_checkpoint(&dir, false);
        let extra_dir = dir.join("shape_enc_stage");
        std::fs::create_dir_all(&extra_dir).unwrap();
        let cpu = Device::Cpu;
        let mut extra = HashMap::new();
        extra.insert(
            "input_layer.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        // Rank-5 sparse-convolution kernel: stored flattened and F16.
        extra.insert(
            "blocks.0.conv.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 3, 3, 3, 64), &cpu).unwrap(),
        );
        let extra_path = extra_dir.join("shape_enc.safetensors");
        candle_core::safetensors::save(&extra, &extra_path).unwrap();

        let output = dir.join("model.gguf");
        let report = quantize_checkpoint_with_merges(
            &source,
            &output,
            ModelFamily::Trellis2,
            QuantizeOptions::default(),
            &[QuantizeMergeSource::new(&extra_path, ".shape_enc.")],
        )
        .unwrap();
        assert_eq!(report.tensors, 6);
        assert_eq!(report.quantized, 2);
        assert_eq!(report.fallback_tensors, 4);

        let content = read_gguf(&output);
        assert_eq!(
            content.tensor_infos["shape_enc.input_layer.weight"].ggml_dtype,
            GgmlDType::Q8_0
        );
        let conv = &content.tensor_infos["shape_enc.blocks.0.conv.weight"];
        assert_eq!(conv.ggml_dtype, GgmlDType::F16);
        assert_eq!(conv.shape.dims(), &[64 * 27 * 64]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantize_merge_rejects_duplicate_tensor_names() {
        let dir = scratch_dir("merge-duplicate");
        let source = tiny_checkpoint(&dir, false);
        let extra_dir = dir.join("duplicate");
        std::fs::create_dir_all(&extra_dir).unwrap();
        let cpu = Device::Cpu;
        let mut extra = HashMap::new();
        extra.insert(
            "to_q.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        let extra_path = extra_dir.join("duplicate.safetensors");
        candle_core::safetensors::save(&extra, &extra_path).unwrap();
        let error = quantize_checkpoint_with_merges(
            &source,
            &dir.join("model.gguf"),
            ModelFamily::Trellis2,
            QuantizeOptions::default(),
            &[QuantizeMergeSource::new(&extra_path, "")],
        )
        .expect_err("empty prefix rejected before opening the merge source");
        assert!(error.to_string().contains("empty tensor prefix"));

        let error = quantize_checkpoint_with_merges(
            &source,
            &dir.join("model.gguf"),
            ModelFamily::Trellis2,
            QuantizeOptions::default(),
            &[QuantizeMergeSource::new(
                &extra_path,
                "transformer.blocks.0.attn",
            )],
        )
        .expect_err("duplicate merged name rejected");
        assert!(error.to_string().contains("duplicate tensor"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantize_honours_all_2d_and_prefixes() {
        let dir = scratch_dir("options");
        let source = tiny_checkpoint(&dir, false);
        let all_2d = QuantizeOptions {
            all_2d: true,
            ..QuantizeOptions::default()
        };
        let report =
            quantize_checkpoint(&source, &dir.join("all_2d.gguf"), ModelFamily::Vggt, all_2d)
                .unwrap();
        assert_eq!(report.quantized, 2);

        let prefixed = QuantizeOptions {
            prefixes: vec!["vae.".to_string()],
            ..QuantizeOptions::default()
        };
        let report = quantize_checkpoint(
            &source,
            &dir.join("prefixed.gguf"),
            ModelFamily::Vggt,
            prefixed,
        )
        .unwrap();
        assert_eq!(report.quantized, 0);
        assert_eq!(report.fallback_tensors, 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantize_rejects_activation_dtypes() {
        let dir = scratch_dir("dtype");
        let source = tiny_checkpoint(&dir, false);
        let options = QuantizeOptions {
            dtype: GgmlDType::Q8K,
            ..QuantizeOptions::default()
        };
        let err = quantize_checkpoint(&source, &dir.join("bad.gguf"), ModelFamily::Vggt, options)
            .unwrap_err();
        assert!(err.to_string().contains("activation-side"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scribble_source_resolves_the_transformer_shard() {
        let dir = scratch_dir("scribble-source");
        let transformer = dir.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        let shard = transformer.join("diffusion_pytorch_model.safetensors");
        std::fs::write(&shard, b"placeholder").unwrap();

        assert_eq!(resolve_scribble_dit_source(&dir).unwrap(), shard);
        assert_eq!(resolve_scribble_dit_source(&shard).unwrap(), shard);

        let missing = dir.join("missing");
        let err = resolve_scribble_dit_source(&missing).unwrap_err();
        assert!(err.to_string().contains("missing"));
        let empty = scratch_dir("scribble-source-empty");
        let err = resolve_scribble_dit_source(&empty).unwrap_err();
        assert!(
            err.to_string()
                .contains("transformer/diffusion_pytorch_model.safetensors")
                || err
                    .to_string()
                    .contains("diffusion_pytorch_model.safetensors"),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&empty);
    }

    #[test]
    fn standalone_scribble_merges_components_and_embeds_tokenizer() {
        let dir = scratch_dir("scribble-standalone");
        let source = tiny_scribble_checkpoint(&dir);
        let output = dir.join("triposg-scribble-q8_0.gguf");
        let report =
            quantize_scribble_standalone(&source, &output, QuantizeOptions::default()).unwrap();
        assert!(report.embedded_resolved_config);
        assert_eq!(report.tensors, 5);
        assert_eq!(report.quantized, 1);
        assert_eq!(report.fallback_tensors, 4);
        assert_eq!(report.skipped_tensors, 1);

        let content = read_gguf(&output);
        assert_eq!(gguf_architecture(&content), "triposg");
        assert_eq!(content.tensor_infos.len(), 5);
        for prefix in ["transformer.", "vae.", "image_encoder.", "text_encoder."] {
            assert!(
                content
                    .tensor_infos
                    .keys()
                    .any(|name| name.starts_with(prefix)),
                "standalone GGUF must contain `{prefix}` tensors"
            );
        }
        match content.metadata.get("lux3d.variant") {
            Some(gguf_file::Value::String(value)) => assert_eq!(value, "triposg-scribble"),
            other => panic!("missing lux3d.variant metadata: {other:?}"),
        }
        match content.metadata.get("lux3d.components") {
            Some(gguf_file::Value::String(value)) => {
                assert_eq!(value, "image_encoder,transformer,vae,text_encoder")
            }
            other => panic!("missing lux3d.components metadata: {other:?}"),
        }

        // Only the transformer linear is quantized; the other components keep
        // the fallback dtype even though their tensors are rank-2 `.weight`.
        assert_eq!(
            content.tensor_infos["transformer.blocks.0.attn.to_q.weight"].ggml_dtype,
            GgmlDType::Q8_0
        );
        for name in [
            "transformer.blocks.0.norm.weight",
            "vae.decoder.conv.weight",
            "image_encoder.embeddings.patch_embeddings.projection.weight",
            "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.weight",
        ] {
            assert_eq!(
                content.tensor_infos[name].ggml_dtype,
                GgmlDType::F16,
                "`{name}` must stay in the fallback dtype"
            );
        }

        let embedded = match content.metadata.get("lux3d.resolved_config") {
            Some(gguf_file::Value::String(value)) => value,
            other => panic!("missing embedded resolved config: {other:?}"),
        };
        let config: serde_json::Value = serde_json::from_str(embedded).unwrap();
        assert_eq!(config["variant"], "triposg-scribble");
        assert_eq!(config["transformer"]["use_cross_attention_2"], true);
        assert_eq!(config["image_encoder"]["hidden_size"], 1024);
        assert_eq!(config["pipeline_defaults"]["num_inference_steps"], 16);
        assert_eq!(
            config["tokenizer"]["vocab_json"],
            r#"{"<|startoftext|>":49406,"<|endoftext|>":49407}"#
        );
        assert!(
            config["tokenizer"]["merges_txt"]
                .as_str()
                .unwrap()
                .contains("<|startoftext|> a")
        );

        let options = QuantizeOptions {
            dtype: GgmlDType::Q8K,
            ..QuantizeOptions::default()
        };
        let error =
            quantize_scribble_standalone(&source, &dir.join("bad.gguf"), options).unwrap_err();
        assert!(error.to_string().contains("activation-side"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn should_quantize_checks_shape_and_suffix() {
        let cpu = Device::Cpu;
        let weight = Tensor::rand(0f32, 1f32, (64, 64), &cpu).unwrap();
        let options = QuantizeOptions::default();
        assert!(should_quantize("block.weight", &weight, &options));
        assert!(!should_quantize("block.table", &weight, &options));
        let narrow = Tensor::rand(0f32, 1f32, (64, 10), &cpu).unwrap();
        assert!(!should_quantize("block.weight", &narrow, &options));
        let bias = Tensor::rand(0f32, 1f32, (64,), &cpu).unwrap();
        assert!(!should_quantize("block.weight", &bias, &options));
    }

    #[test]
    fn standalone_gguf_loads_without_safetensors() {
        let dir = scratch_dir("standalone");
        let source = tiny_checkpoint(&dir, true);
        let output = dir.join("model.gguf");
        let report = quantize_checkpoint(
            &source,
            &output,
            ModelFamily::Vggt,
            QuantizeOptions::default(),
        )
        .unwrap();
        assert!(report.embedded_resolved_config);
        std::fs::remove_file(&source).unwrap();
        std::fs::remove_file(dir.join("resolved_config.json")).unwrap();

        let paths = crate::weights::load_canonical_weights(
            ModelFamily::Vggt,
            crate::weights::ModelAssetOptions {
                canonical_dir: Some(dir.clone()),
                cache_dir: None,
            },
        )
        .unwrap();
        assert_eq!(paths.canonical_file, output);

        // The same GGUF can be passed directly (flat distribution layout).
        let file_paths = crate::weights::load_canonical_weights(
            ModelFamily::Vggt,
            crate::weights::ModelAssetOptions {
                canonical_dir: Some(output.clone()),
                cache_dir: None,
            },
        )
        .unwrap();
        assert_eq!(
            file_paths.read_resolved_config_json().unwrap()["hidden_size"],
            64
        );
        let config = paths.read_resolved_config_json().unwrap();
        assert_eq!(config["hidden_size"].as_u64(), Some(64));
        let vb = unsafe {
            paths
                .var_builder(candle_core::DType::F32, &Device::Cpu)
                .unwrap()
        };
        let weight = vb
            .get((64, 64), "transformer.blocks.0.attn.to_q.weight")
            .unwrap();
        assert_eq!(weight.dims(), &[64, 64]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Candle's GGUF reader caps tensor rank at four. Rank>4 tensors (3D
    /// convolution kernels) must be stored flattened and reshaped back by the
    /// runtime loader (`crate::runtime::get_weight_shaped`).
    #[test]
    fn rank_above_four_tensors_are_flattened_and_reshaped_back() {
        let dir = scratch_dir("rank5");
        let cpu = Device::Cpu;
        let kernel = Tensor::rand(-0.05f32, 0.05f32, (2, 3, 3, 3, 4), &cpu).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("conv3d.weight".to_string(), kernel.clone());
        tensors.insert(
            "linear.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        std::fs::write(dir.join("resolved_config.json"), r#"{"hidden_size":64}"#).unwrap();
        let source = dir.join("model.safetensors");
        candle_core::safetensors::save(&tensors, &source).unwrap();

        let output = dir.join("model.gguf");
        let report = quantize_checkpoint(
            &source,
            &output,
            ModelFamily::Vggt,
            QuantizeOptions::default(),
        )
        .unwrap();
        // The rank-5 kernel is never quantized (rank != 2); the linear is.
        assert_eq!(report.quantized, 1);
        assert_eq!(report.fallback_tensors, 1);

        let content = read_gguf(&output);
        let info = &content.tensor_infos["conv3d.weight"];
        assert_eq!(info.ggml_dtype, GgmlDType::F16);
        assert_eq!(info.shape.dims(), &[2 * 3 * 3 * 3 * 4]);
        assert_eq!(
            content.tensor_infos["linear.weight"].shape.dims(),
            &[64, 64]
        );

        // The runtime fetch path reshapes the flat vector back to the kernel.
        let vb = candle_nn::VarBuilder::from_gguf(&output, candle_core::DType::F32, &cpu).unwrap();
        let fetched =
            crate::runtime::get_weight_shaped(&vb, (2, 3, 3, 3, 4), "conv3d.weight").unwrap();
        assert_eq!(fetched.dims(), &[2, 3, 3, 3, 4]);
        let diff = fetched
            .sub(&kernel.to_dtype(candle_core::DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "reshaped kernel diverged by {diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn standalone_gguf_without_embedded_config_is_rejected() {
        let dir = scratch_dir("standalone-no-config");
        let source = tiny_checkpoint(&dir, false);
        let output = dir.join("model.gguf");
        let report = quantize_checkpoint(
            &source,
            &output,
            ModelFamily::Vggt,
            QuantizeOptions::default(),
        )
        .unwrap();
        assert!(!report.embedded_resolved_config);
        std::fs::remove_file(&source).unwrap();

        let err = crate::weights::load_canonical_weights(
            ModelFamily::Vggt,
            crate::weights::ModelAssetOptions {
                canonical_dir: Some(dir.clone()),
                cache_dir: None,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("no embedded"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn auxiliary_standalone_gguf_loads_in_f16() {
        let dir = scratch_dir("rmbg");
        let source = tiny_checkpoint(&dir, true);
        let output = dir.join("model.gguf");
        let options = QuantizeOptions {
            dtype: GgmlDType::F16,
            fallback: GgmlDType::F16,
            ..QuantizeOptions::default()
        };
        let report = quantize_auxiliary_checkpoint(&source, &output, "rmbg", options).unwrap();
        assert!(report.embedded_resolved_config);
        assert_eq!(report.skipped_tensors, 1);
        let content = read_gguf(&output);
        assert_eq!(gguf_architecture(&content), "rmbg");
        for info in content.tensor_infos.values() {
            assert_eq!(info.ggml_dtype, GgmlDType::F16);
        }
        std::fs::remove_file(&source).unwrap();
        std::fs::remove_file(dir.join("resolved_config.json")).unwrap();

        let paths = crate::weights::load_auxiliary_canonical_weights(&dir, "rmbg").unwrap();
        assert_eq!(paths.canonical_file, output);
        let file_paths = crate::weights::load_auxiliary_canonical_weights(&output, "rmbg").unwrap();
        assert_eq!(
            file_paths.read_resolved_config_json().unwrap()["hidden_size"],
            64
        );
        let config = paths.read_resolved_config_json().unwrap();
        assert_eq!(config["hidden_size"].as_u64(), Some(64));
        let vb = unsafe {
            paths
                .var_builder(candle_core::DType::F32, &Device::Cpu)
                .unwrap()
        };
        let weight = vb
            .get((64, 64), "transformer.blocks.0.attn.to_q.weight")
            .unwrap();
        assert_eq!(weight.dims(), &[64, 64]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
