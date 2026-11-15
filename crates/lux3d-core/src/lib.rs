pub mod contracts;
pub mod error;
pub mod export;
pub mod geometry;
pub mod neural;
pub mod preprocess;
pub mod runtime;
pub mod test_support;
pub mod weights;

pub use contracts::{
    BaselineGeometrySummary, BaselineManifest, BaselineParityTap, BaselinePreviewArtifact,
    BaselineSampleKind, BaselineTensorArtifact, ContractExclusion, ContractSourceOfTruth,
    ContractStage, ExportPlan, ExportSpec, GeometryBufferContract, GeometrySpec, LicenseEntry,
    LicensePolicy, ModelFamily, ModelSpec, NeuralSpec, Pi3ExportSpec, Pi3GeometrySpec,
    Pi3InputSource, Pi3NeuralSpec, Pi3PreprocessSpec, Pi3xExportSpec, Pi3xGeometrySpec,
    Pi3xNeuralSpec, Pi3xPreprocessSpec, PreprocessSpec, RgbRange, RuntimeArchitecture,
    RuntimeGeometry, RuntimeTensorContract, SourceDisposition, SpatialSize, TensorDType,
    Trellis2ExportSpec, Trellis2GeometrySpec, Trellis2Mesh, Trellis2NeuralSpec,
    Trellis2PipelineType, Trellis2PreparedImage, Trellis2PreprocessSpec, Trellis2RawInput,
    Trellis2RuntimeGeometry, TripoExportSpec, TripoGeometrySpec, TripoNeuralSpec,
    TripoPreprocessSpec, TripoSgExportSpec, TripoSgGeometrySpec, TripoSgMesh, TripoSgNeuralSpec,
    TripoSgPreparedImage, TripoSgPreprocessSpec, TripoSgRawInput, TripoSgRmbgSpec,
    TripoSgRuntimeGeometry, TripoSgSceneCode, VendorSource, VggtExportSpec, VggtGeometrySpec,
    VggtNeuralSpec, VggtPointCloud, VggtPreparedBatch, VggtPreprocessSpec, VggtRawInput,
    VggtRuntimeGeometry, VggtSceneCode,
};
pub use error::{Lux3dError, Result};
pub use runtime::{RmbgConfig, RmbgModel};
pub use weights::{
    AuxiliaryWeightSetPaths, AuxiliaryWeightsManifest, CanonicalChecksumEntry, CanonicalChecksums,
    CanonicalWeightSet, CanonicalWeightSetPaths, CanonicalWeightsManifest, CanonicalizationPlan,
    FutureWeightLoader, ModelAssetOptions, QuantizeOptions, QuantizeReport, RawWeightFormat,
    WeightLocator, dtype_from_name, dtype_name, ensure_canonical_weights,
    load_auxiliary_canonical_weights, load_canonical_weights, quantize_auxiliary_checkpoint,
    quantize_checkpoint,
};

/// Families whose GPU timings and output parity were validated with candle's
/// reduced-precision GEMM opt-ins active (vulkan TF32-class coopmat, wgpu
/// f16-MMA coop): pi3 and pi3x only.
///
/// The opt-ins trade ~10-bit mantissa input precision for 1.9–2.5× on the
/// attention/linear GEMM shapes those pipelines use. TRELLIS.2 must stay on
/// exact fp32: its O-Voxel occupancy is discrete, so the drift flips voxels
/// and changes the exported topology (measured on the 2-step 512 smoke:
/// 563,889 vertices with the opt-in vs 465,520 CPU, 465,514 Vulkan exact-fp32
/// and 465,524 CUDA).
pub const fn reduced_precision_gemm_validated_for(family: ModelFamily) -> bool {
    matches!(family, ModelFamily::Pi3 | ModelFamily::Pi3x)
}

/// Enable candle's reduced-precision GPU GEMM opt-ins for one model family.
///
/// A no-op for families where the precision tradeoff changes model output;
/// see [`reduced_precision_gemm_validated_for`]. Explicit user environment
/// overrides always win.
pub fn enable_reduced_precision_gemm_opt_ins_for(family: ModelFamily) {
    if !reduced_precision_gemm_validated_for(family) {
        return;
    }
    enable_reduced_precision_gemm_opt_ins();
}

/// Unconditionally enable candle's reduced-precision GPU GEMM opt-ins
/// (TF32-class coopmat on vulkan, f16-MMA coop on wgpu).
///
/// Prefer [`enable_reduced_precision_gemm_opt_ins_for`] in product code paths:
/// this only exists for benchmarks and parity tools that deliberately measure
/// the reduced-precision kernels. Candle defaults both paths to OFF so plain
/// `DType::F32` matmul keeps exact-fp32 accumulation. Explicit user
/// environment overrides win.
pub fn enable_reduced_precision_gemm_opt_ins() {
    for (key, value) in [
        ("CANDLE_VULKAN_F32_UNALIGNED_COOPMAT", "1"),
        ("CANDLE_WGPU_COOP_MATMUL", "1"),
    ] {
        if std::env::var_os(key).is_none() {
            // SAFETY: called before any threads are spawned and before any
            // other thread reads the environment; no concurrent access.
            unsafe { std::env::set_var(key, value) };
        }
    }
}
