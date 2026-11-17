//! VGGT tracking stage-A query points from ALIKED + SuperPoint.
//!
//! Official reference: `vggt/dependency/track_predict.py:67-69,172` and
//! `vggt/dependency/vggsfm_utils.py:189-255`. The official tracker builds the
//! query set by running every configured extractor on the query image and
//! concatenating the returned keypoints:
//!
//! ```text
//! extractors = {"aliked": ALIKED(...), "sp": SuperPoint(...)}
//! query_points = cat([extractor.extract(query_image)["keypoints"] ...], dim=1)
//! ```
//!
//! This module owns the same composition and the local weight resolution.
//! Detectors run at the VGGT model-input resolution; the official demo runs
//! them on the 1024 px loaded images because its tracker is separate from the
//! VGGT aggregator, while Lux3D's track head consumes the aggregator features
//! at model resolution. The adapters scale keypoints back to the input image
//! exactly like LightGlue's `Extractor.scale_keypoints`.
//!
//! The `fps` source keeps the previous no-weights behavior and is an explicit
//! opt-out, not a silent fallback: `aliked-sp` is the default and a missing
//! checkpoint is a hard error.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use crate::{Result, error::Lux3dError};

use super::{
    aliked::AlikedDetector,
    keypoints::{KeypointOptions, KeypointSet},
    superpoint::SuperPointDetector,
};

/// Environment override for the ALIKED-n16 safetensors checkpoint.
pub const ALIKED_WEIGHTS_ENV: &str = "LUX3D_ALIKED_WEIGHTS";
/// Environment override for the SuperPoint safetensors checkpoint.
pub const SUPERPOINT_WEIGHTS_ENV: &str = "LUX3D_SUPERPOINT_WEIGHTS";
/// Default directory relative to the workspace root.
pub const KEYPOINT_WEIGHTS_DIR: &str = "models/aliked-superpoint";
/// Default ALIKED file name inside [`KEYPOINT_WEIGHTS_DIR`].
pub const ALIKED_WEIGHTS_FILE: &str = "aliked-n16.safetensors";
/// Default SuperPoint file name inside [`KEYPOINT_WEIGHTS_DIR`].
pub const SUPERPOINT_WEIGHTS_FILE: &str = "superpoint_v1.safetensors";

/// Query-point source for `--track-auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VggtKeypointSource {
    /// Official `keypoint_extractor="aliked+sp"` (the default).
    AlikedSuperPoint,
    /// Previous deterministic FPS grid; requires no external weights.
    Fps,
}

/// Stage-A query-point options.
#[derive(Debug, Clone, PartialEq)]
pub struct VggtKeypointOptions {
    /// Detector composition or the FPS fallback.
    pub source: VggtKeypointSource,
    /// Per-detector keypoint cap (official `max_query_pts`, default 4096).
    pub max_keypoints: usize,
    /// Per-detector detection threshold (official `det_thres=0.005`).
    pub detection_threshold: f32,
    /// Long-side resize before detection (LightGlue `preprocess_conf.resize`,
    /// 1024). `None` disables the resize (tests).
    pub resize_long_side: Option<usize>,
    /// Explicit ALIKED safetensors path; `None` resolves env/default.
    pub aliked_weights: Option<PathBuf>,
    /// Explicit SuperPoint safetensors path; `None` resolves env/default.
    pub superpoint_weights: Option<PathBuf>,
}

impl Default for VggtKeypointOptions {
    fn default() -> Self {
        Self {
            source: VggtKeypointSource::AlikedSuperPoint,
            max_keypoints: 4096,
            detection_threshold: 0.005,
            resize_long_side: Some(1024),
            aliked_weights: None,
            superpoint_weights: None,
        }
    }
}

impl VggtKeypointOptions {
    fn keypoint_options(&self) -> KeypointOptions {
        KeypointOptions {
            max_keypoints: self.max_keypoints,
            detection_threshold: self.detection_threshold,
            resize_long_side: self.resize_long_side,
        }
    }
}

/// Loaded ALIKED + SuperPoint pair, reused across every query frame.
#[derive(Debug)]
pub struct VggtKeypointDetectors {
    aliked: AlikedDetector,
    superpoint: SuperPointDetector,
    options: KeypointOptions,
}

impl VggtKeypointDetectors {
    /// Runs ALIKED then SuperPoint on one `[3, H, W]` `[0, 1]` RGB image and
    /// concatenates their keypoints in that order (`extract_keypoints` uses
    /// the extractor dict insertion order: `aliked`, then `sp`).
    ///
    /// Descriptors from the two detectors have different widths (128 vs 256),
    /// so [`KeypointSet::append`] keeps the keypoints/scores and drops the
    /// mixed descriptor matrix; callers that need descriptors use one
    /// detector directly.
    pub fn detect_query_points(&self, rgb: &Tensor) -> Result<KeypointSet> {
        let mut set = self.aliked.detect(rgb, &self.options)?;
        let superpoint = self.superpoint.detect(rgb, &self.options)?;
        set.append(&superpoint);
        if set.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "VGGT auto query: ALIKED+SuperPoint returned no keypoints; lower \
                 --track-keypoint-threshold or pass --track-keypoints fps",
            ));
        }
        Ok(set)
    }
}

/// Resolve and load the detectors before the expensive VGGT forward.
///
/// `Fps` returns `None` without touching any checkpoint. `AlikedSuperPoint`
/// resolves both safetensors files strictly (an explicit path or env var that
/// is set but unreadable is an error; there is no silent fallback) and loads
/// them onto `device` in F32 for stable detector numerics.
pub fn preflight_keypoint_detectors(
    options: &VggtKeypointOptions,
    device: &Device,
) -> Result<Option<VggtKeypointDetectors>> {
    if options.source == VggtKeypointSource::Fps {
        return Ok(None);
    }
    if options.max_keypoints == 0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT auto query: --max-query-pts must be at least 1",
        ));
    }
    let aliked_path = resolve_aliked_weights_path(options.aliked_weights.as_deref())?;
    let superpoint_path = resolve_superpoint_weights_path(options.superpoint_weights.as_deref())?;
    let aliked = AlikedDetector::load(&aliked_path, device, DType::F32)?;
    let superpoint = SuperPointDetector::load(&superpoint_path, device, DType::F32)?;
    Ok(Some(VggtKeypointDetectors {
        aliked,
        superpoint,
        options: options.keypoint_options(),
    }))
}

/// Resolve the ALIKED safetensors path (explicit, env, cwd, workspace).
pub fn resolve_aliked_weights_path(explicit: Option<&Path>) -> Result<PathBuf> {
    resolve_detector_weights_path(
        explicit,
        ALIKED_WEIGHTS_ENV,
        ALIKED_WEIGHTS_FILE,
        "ALIKED-n16",
        "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth",
    )
}

/// Resolve the SuperPoint safetensors path (explicit, env, cwd, workspace).
pub fn resolve_superpoint_weights_path(explicit: Option<&Path>) -> Result<PathBuf> {
    resolve_detector_weights_path(
        explicit,
        SUPERPOINT_WEIGHTS_ENV,
        SUPERPOINT_WEIGHTS_FILE,
        "SuperPoint",
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth",
    )
}

fn resolve_detector_weights_path(
    explicit: Option<&Path>,
    env_var: &str,
    filename: &str,
    label: &str,
    download_url: &str,
) -> Result<PathBuf> {
    let candidates = |path: PathBuf| -> Vec<PathBuf> {
        if path.is_dir() {
            vec![path.join(filename)]
        } else {
            vec![path]
        }
    };
    if let Some(path) = explicit {
        for candidate in candidates(path.to_path_buf()) {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT auto query: {label} weights not found at {}",
                path.display()
            ),
        });
    }
    if let Some(value) = std::env::var_os(env_var) {
        let path = PathBuf::from(value);
        for candidate in candidates(path.clone()) {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT auto query: {env_var} points to {}, but it contains no readable \
                 `{filename}`; fix the path or unset {env_var} (the resolver does not \
                 silently fall back)",
                path.display()
            ),
        });
    }
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(KEYPOINT_WEIGHTS_DIR).join(filename));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(KEYPOINT_WEIGHTS_DIR)
            .join(filename),
    );
    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(Lux3dError::CanonicalWeightsValidation {
        message: format!(
            "VGGT auto query: {label} weights not found. Download {download_url}, convert \
             it with `python scripts/convert_keypoint_weights.py --input <file.pth> \
             --output {KEYPOINT_WEIGHTS_DIR}/{filename}`, or set {env_var}. To run without \
             the detectors, pass `--track-keypoints fps`."
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_source_needs_no_weights() {
        let options = VggtKeypointOptions {
            source: VggtKeypointSource::Fps,
            aliked_weights: Some(PathBuf::from("definitely/missing.pth")),
            superpoint_weights: Some(PathBuf::from("definitely/missing.pth")),
            ..VggtKeypointOptions::default()
        };
        assert!(
            preflight_keypoint_detectors(&options, &Device::Cpu)
                .expect("fps does not load checkpoints")
                .is_none()
        );
    }

    #[test]
    fn explicit_missing_aliked_weights_is_an_error() {
        let options = VggtKeypointOptions {
            aliked_weights: Some(PathBuf::from("definitely/missing.safetensors")),
            ..VggtKeypointOptions::default()
        };
        let error = preflight_keypoint_detectors(&options, &Device::Cpu)
            .expect_err("missing explicit weights must fail");
        assert!(
            error.to_string().contains("ALIKED-n16 weights not found"),
            "unexpected message: {error}"
        );
    }

    /// Official composition test: ALIKED keypoints first, then SuperPoint,
    /// descriptors dropped because the widths differ (128 vs 256).
    /// Skipped when the converted checkpoints are not present.
    #[test]
    fn aliked_superpoint_composition_matches_individual_detectors() {
        let Ok(aliked_path) = resolve_aliked_weights_path(None) else {
            eprintln!("skipping ALIKED+SuperPoint composition: ALIKED weights missing");
            return;
        };
        let Ok(superpoint_path) = resolve_superpoint_weights_path(None) else {
            eprintln!("skipping ALIKED+SuperPoint composition: SuperPoint weights missing");
            return;
        };
        let options = VggtKeypointOptions {
            max_keypoints: 32,
            resize_long_side: None,
            aliked_weights: Some(aliked_path),
            superpoint_weights: Some(superpoint_path),
            ..VggtKeypointOptions::default()
        };
        let detectors = preflight_keypoint_detectors(&options, &Device::Cpu)
            .expect("load both detectors")
            .expect("detectors are loaded for aliked-sp");

        // Deterministic high-frequency RGB checkerboard so both detectors see
        // strong corners/edges without any external image fixture.
        let side = 64usize;
        let mut pixels = Vec::with_capacity(3 * side * side);
        for channel in 0..3 {
            for y in 0..side {
                for x in 0..side {
                    let light = ((x / 8) + (y / 8)) % 2 == 0;
                    let value = if light { 1.0 } else { 0.0 };
                    // Per-channel offsets keep the pattern colored while
                    // preserving the checkerboard luminance contrast.
                    let offset = 0.05 * channel as f32;
                    pixels.push((value + offset).min(1.0));
                }
            }
        }
        let rgb = Tensor::from_vec(pixels, (3, side, side), &Device::Cpu).expect("image tensor");

        let keypoint_options = options.keypoint_options();
        let aliked = detectors
            .aliked
            .detect(&rgb, &keypoint_options)
            .expect("ALIKED detect");
        let superpoint = detectors
            .superpoint
            .detect(&rgb, &keypoint_options)
            .expect("SuperPoint detect");
        let combined = detectors
            .detect_query_points(&rgb)
            .expect("combined detect");

        assert!(!aliked.is_empty(), "ALIKED must find keypoints");
        assert!(!superpoint.is_empty(), "SuperPoint must find keypoints");
        assert_eq!(aliked.len() + superpoint.len(), combined.len());
        assert_eq!(aliked.keypoints, combined.keypoints[..aliked.len()]);
        assert_eq!(
            superpoint.keypoints,
            combined.keypoints[aliked.len()..],
            "SuperPoint keypoints follow ALIKED in extractor order"
        );
        assert!(combined.descriptors.is_empty());
        assert_eq!(0, combined.descriptor_dim);
        for [x, y] in &combined.keypoints {
            assert!(x.is_finite() && y.is_finite());
            assert!((-0.5..=(side as f32 - 0.5)).contains(x), "x={x}");
            assert!((-0.5..=(side as f32 - 0.5)).contains(y), "y={y}");
        }
    }
}
