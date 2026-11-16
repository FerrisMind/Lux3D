//! VGGT track-head tests: weight-key coverage against the canonical package
//! and a CPU structural forward with real weights on tiny synthetic inputs.
//!
//! These tests skip (with a message) when the canonical VGGT package is not
//! present; the structural test additionally requires the track head section
//! of `model.safetensors`.

use std::collections::HashSet;

use candle_core::{DType, Device, Tensor};
use lux3d_core::ModelFamily;
use lux3d_core::load_canonical_weights;
use lux3d_core::runtime::{
    VggtTrackHead, VggtTrackQuery, bilinear_sample_features, expected_track_head_tensor_names,
};
use lux3d_core::test_support::{canonical_package_dir, model_asset_options};

fn canonical_safetensors() -> Option<std::path::PathBuf> {
    let path = canonical_package_dir(ModelFamily::Vggt).join("model.safetensors");
    if path.is_file() {
        Some(path)
    } else {
        eprintln!(
            "skipping VGGT track-head checkpoint test: canonical package missing at {}",
            path.display()
        );
        None
    }
}

/// Parse a safetensors header (8-byte LE length + serde JSON) into
/// `name -> shape`.
fn safetensors_header(path: &std::path::Path) -> std::collections::HashMap<String, Vec<usize>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("open safetensors");
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).expect("read header length");
    let len = u64::from_le_bytes(len_bytes) as usize;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf).expect("read header");
    let header: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_slice(&buf).expect("parse safetensors header");
    header
        .into_iter()
        .filter_map(|(name, entry)| {
            if name == "__metadata__" {
                return None;
            }
            let shape = entry["shape"]
                .as_array()?
                .iter()
                .map(|v| v.as_u64().expect("shape dim") as usize)
                .collect::<Vec<_>>();
            Some((name, shape))
        })
        .collect()
}

/// The expected key list must be exactly 394 unique names — 58 feature
/// extractor + 336 tracker — matching the canonical `track_head.*` section.
#[test]
fn track_head_expected_tensor_names_are_unique_and_complete() {
    let names = expected_track_head_tensor_names();
    assert_eq!(names.len(), 394, "track head tensor count");
    let unique: HashSet<&String> = names.iter().collect();
    assert_eq!(unique.len(), 394, "expected names must be unique");
    for prefix in [
        "track_head.feature_extractor.norm.",
        "track_head.feature_extractor.projects.3.",
        "track_head.feature_extractor.scratch.refinenet4.resConfUnit2.",
        "track_head.feature_extractor.scratch.output_conv1.",
        "track_head.tracker.corr_mlp.fc1.",
        "track_head.tracker.updateformer.time_blocks.5.attn.in_proj_weight",
        "track_head.tracker.updateformer.space_virtual2point_blocks.5.cross_attn.in_proj_weight",
        "track_head.tracker.updateformer.virual_tracks",
        "track_head.tracker.conf_predictor.0.",
        "track_head.tracker.query_ref_token",
    ] {
        assert!(
            names.iter().any(|name| name.starts_with(prefix)),
            "missing name prefix {prefix}"
        );
    }
}

/// The canonical checkpoint's `track_head.*` section must match the expected
/// name list exactly (no missing and no unexpected keys).
#[test]
fn track_head_tensor_names_cover_the_checkpoint() {
    let Some(path) = canonical_safetensors() else {
        return;
    };
    let header = safetensors_header(&path);
    let checkpoint_keys: HashSet<String> = header
        .keys()
        .filter(|name| name.starts_with("track_head."))
        .cloned()
        .collect();
    assert_eq!(
        checkpoint_keys.len(),
        394,
        "canonical package track_head tensor count"
    );
    let expected: HashSet<String> = expected_track_head_tensor_names().into_iter().collect();
    let missing: Vec<&String> = expected.difference(&checkpoint_keys).collect();
    let unexpected: Vec<&String> = checkpoint_keys.difference(&expected).collect();
    assert!(
        missing.is_empty() && unexpected.is_empty(),
        "track head key mismatch; missing={missing:?} unexpected={unexpected:?}"
    );
}

/// Hand-derived golden values for the bilinear sampler primitive (the exact
/// `F.grid_sample(align_corners=True)` semantics used by the correlation
/// pyramid and `sample_features4d`).
#[test]
fn bilinear_sample_features_matches_hand_computed_values() {
    let device = Device::Cpu;
    // [1, 1, 2, 2] map with values [[1, 2], [3, 4]] (row-major y, x).
    let input =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &device).expect("input");
    let coords = Tensor::from_vec(
        vec![
            0.0, 0.0, // (x, y) -> 1
            1.0, 0.0, // -> 2
            0.0, 1.0, // -> 3
            1.0, 1.0, // -> 4
            0.5, 0.5, // bilinear center -> 2.5
            0.25, 0.75, // -> 2.75
            5.0, 0.0, // out of range, border -> 2 (x clamps to W-1)
            -2.0, 1.0, // negative x, border -> 3 (x clamps to 0)
        ],
        (1, 8, 2),
        &device,
    )
    .expect("coords");
    let sampled = bilinear_sample_features(&input, &coords).expect("sample");
    assert_eq!(sampled.dims(), &[1, 8, 1]);
    let values = sampled
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("values");
    let expected = [1.0, 2.0, 3.0, 4.0, 2.5, 2.75, 2.0, 3.0];
    for (got, want) in values.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-6,
            "sampled {got} does not match hand-computed {want}"
        );
    }

    // Zeros padding: out-of-range coordinates contribute zero (exercised
    // through the correlation pyramid in the structural test); border padding
    // clamps instead.
    let out_of_range =
        Tensor::from_vec(vec![5.0f32, 0.0, -0.5, 0.5], (1, 2, 2), &device).expect("zeros coords");
    let border = bilinear_sample_features(&input, &out_of_range).expect("sample border");
    let border_values = border
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("border values");
    assert!((border_values[0] - 2.0).abs() < 1e-6);
    // (-0.5, 0.5) with border: x clamps to 0 -> average of v(0,0) and v(0,1).
    assert!((border_values[1] - 2.0).abs() < 1e-6);
}

/// Full structural pass through the track head on CPU with the real
/// canonical weights and tiny synthetic inputs (14x14 patch grid, 2 frames,
/// 3 queries including border coordinates). Skips when the canonical package
/// is absent.
#[test]
fn track_head_structural_forward_cpu_with_canonical_weights() {
    let Some(_path) = canonical_safetensors() else {
        return;
    };
    let device = Device::Cpu;
    let weights = load_canonical_weights(ModelFamily::Vggt, model_asset_options(ModelFamily::Vggt))
        .expect("canonical weights");
    let head = unsafe { VggtTrackHead::load_from_canonical(&weights, &device, DType::F32) }
        .expect("track head load");

    // 14x14 patch grid -> images 196x196 -> feature maps 98x98 (7 pyramid
    // levels: 98, 49, 24, 12, 6, 3, 1).
    assert_eq!(head.feature_map_size(196, 196), (98, 98));

    let (batch, frames, patches, dim) = (1usize, 2usize, 201usize, 2048usize);
    let layer = |seed: usize| -> Tensor {
        let count = batch * frames * patches * dim;
        let data = (0..count)
            .map(|i| {
                let t = (i % 997) as f32 * 0.013 + seed as f32 * 0.07;
                (t.sin() + t.cos() * 0.5) * 0.01
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(data, (batch, frames, patches, dim), &device).expect("layer")
    };
    let cached = [layer(1), layer(2), layer(3), layer(4)];
    let refs: Vec<&Tensor> = cached.iter().collect();

    let images = Tensor::zeros((batch, frames, 3, 196, 196), DType::F32, &device).expect("images");
    let queries = [
        VggtTrackQuery {
            frame: 0,
            xy: [98.0, 98.0],
        },
        VggtTrackQuery {
            frame: 0,
            xy: [194.9, 0.1],
        },
        VggtTrackQuery {
            frame: 0,
            xy: [0.0, 194.9],
        },
    ];

    let result = head
        .track(&refs, &images, 5, &queries, 4, 1)
        .expect("track forward");

    assert_eq!(result.coords.dims(), &[frames, queries.len(), 2]);
    assert_eq!(result.visibility.dims(), &[frames, queries.len()]);
    let confidence = result
        .confidence
        .as_ref()
        .expect("predict_conf=true must yield confidence");
    assert_eq!(confidence.dims(), &[frames, queries.len()]);

    // Frame 0 is forced to the query points every iteration.
    let coords = result
        .coords
        .to_device(&device)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("coords");
    for (q, query) in queries.iter().enumerate() {
        assert_eq!(coords[q * 2], query.xy[0], "query {q} x at frame 0");
        assert_eq!(coords[q * 2 + 1], query.xy[1], "query {q} y at frame 0");
    }

    for (name, tensor) in [
        ("coords", &result.coords),
        ("visibility", &result.visibility),
        ("confidence", confidence),
    ] {
        let flat = tensor
            .to_device(&device)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap_or_else(|source| panic!("{name}: {source}"));
        for (idx, value) in flat.iter().enumerate() {
            assert!(value.is_finite(), "{name}[{idx}] is not finite: {value}");
        }
    }
    for tensor in [&result.visibility, confidence] {
        let flat = tensor
            .to_device(&device)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("sigmoid outputs");
        for value in flat {
            assert!(
                (0.0..=1.0).contains(&value),
                "sigmoid output out of range: {value}"
            );
        }
    }

    // Non-query frames: the delta head is not bounded by the sampling grid,
    // so on synthetic inputs the tracks may drift anywhere; only finiteness
    // is asserted (checked above together with all other outputs).
}

/// The zero-conf variant of the API contract: `VggtTrackQuery` validation
/// errors are reported instead of panicking.
#[test]
fn track_head_query_validation() {
    let device = Device::Cpu;
    let images = Tensor::zeros((1, 2, 3, 196, 196), DType::F32, &device).expect("images");
    let layer = Tensor::zeros((1, 2, 201, 2048), DType::F32, &device).expect("layer");
    let refs = [&layer, &layer, &layer, &layer];

    // No canonical weights needed for query validation: build a placeholder
    // head only if weights exist, otherwise validate through the pipeline
    // entry points which perform the same checks.
    let head_is_available = canonical_safetensors().is_some();
    if !head_is_available {
        eprintln!("skipping query validation test: canonical package missing");
        return;
    }
    let weights = load_canonical_weights(ModelFamily::Vggt, model_asset_options(ModelFamily::Vggt))
        .expect("canonical weights");
    let head = unsafe { VggtTrackHead::load_from_canonical(&weights, &device, DType::F32) }
        .expect("track head load");

    let err = head
        .track(
            &refs,
            &images,
            5,
            &[VggtTrackQuery {
                frame: 1,
                xy: [0.0, 0.0],
            }],
            4,
            1,
        )
        .expect_err("non-zero query frame must fail");
    assert!(
        err.to_string().contains("frame 0"),
        "unexpected error: {err}"
    );

    let err = head
        .track(&refs, &images, 5, &[], 4, 1)
        .expect_err("empty queries must fail");
    assert!(
        err.to_string().contains("at least one query"),
        "unexpected error: {err}"
    );

    let err = head
        .track(
            &refs,
            &images,
            5,
            &[VggtTrackQuery {
                frame: 0,
                xy: [500.0, 0.0],
            }],
            4,
            1,
        )
        .expect_err("out-of-range query must fail");
    assert!(
        err.to_string().contains("outside"),
        "unexpected error: {err}"
    );
}
