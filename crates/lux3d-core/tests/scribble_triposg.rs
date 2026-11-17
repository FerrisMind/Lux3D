//! TripoSG-scribble tests: CLIP tokenizer goldens generated with the official
//! `transformers.CLIPTokenizer` from the checkpoint tokenizer files, the CLIP
//! text encoder `last_hidden_state` shape, and an opt-in CUDA end-to-end
//! scribble -> GLB run against the locally downloaded
//! `VAST-AI/TripoSG-scribble` checkpoint.
//!
//! Weights are looked up in `LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR` first, then
//! `<workspace>/candle-3d/models/TripoSG-scribble`. The tokenizer/encoder
//! tests skip when the checkpoint is absent; the e2e test is additionally
//! gated behind `LUX3D_TRIPOSG_SCRIBBLE_E2E=1` because it runs the full
//! 16-step DiT and the dense/hierarchical SDF decode on a GPU.

use std::path::PathBuf;

use candle_core::{DType, Device};
use lux3d_core::runtime::{
    ClipBpeTokenizer, TripoSgClipTextEncoder, TripoSgScribblePipeline, TripoSgScribbleRunOptions,
};
use lux3d_core::test_support::{runtime_root, GpuTestLock};

fn repo_root() -> PathBuf {
    runtime_root().join("candle-3d")
}

fn scribble_model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("models").join("TripoSG-scribble"));
    (dir.join("tokenizer").join("vocab.json").is_file()).then_some(dir)
}

/// Pad a body sequence to the CLIP `model_max_length` with the pad token id.
fn padded(body: &[u32]) -> Vec<u32> {
    let mut ids = body.to_vec();
    ids.resize(77, 49407);
    ids
}

/// Exact `CLIPTokenizer([prompt], max_length=77, padding="max_length",
/// truncation=True)` ids, generated with transformers 5.17 against
/// `models/TripoSG-scribble/tokenizer`.
#[test]
fn tokenizer_matches_python_clip_goldens() {
    let Some(dir) = scribble_model_dir() else {
        eprintln!("skipping tokenizer golden test: TripoSG-scribble checkpoint not found");
        return;
    };
    let tokenizer =
        ClipBpeTokenizer::from_dir(&dir.join("tokenizer")).expect("CLIP tokenizer load");
    assert_eq!(77, tokenizer.model_max_length());
    assert_eq!(49406, tokenizer.bos_token_id());
    assert_eq!(49407, tokenizer.eos_token_id());
    assert_eq!(49407, tokenizer.pad_token_id());

    let cases: [(&str, &[u32]); 5] = [
        (
            "a photo of a cat",
            &[49406, 320, 1125, 539, 320, 2368, 49407],
        ),
        ("Hello, World!", &[49406, 3306, 267, 1002, 256, 49407]),
        (
            "a red sports car 's wheel",
            &[49406, 320, 736, 2054, 1615, 568, 6744, 49407],
        ),
        (
            "low poly 3D model, cute dragon",
            &[49406, 1042, 18367, 274, 323, 2863, 267, 2242, 5471, 49407],
        ),
        (
            "  multiple   spaces\tand\nnewlines  ",
            &[49406, 6470, 9006, 537, 1218, 3418, 49407],
        ),
    ];
    for (prompt, expected) in cases {
        let actual = tokenizer.encode(prompt);
        assert_eq!(
            padded(expected),
            actual,
            "token ids mismatch for prompt `{prompt}`"
        );
        assert_eq!(77, actual.len(), "padding must reach model_max_length");
    }

    // 100 repetitions of `word ` truncate to 75 body tokens plus BOS/EOS:
    // exactly one EOS at the end and no padding.
    let long = tokenizer.encode(&"word ".repeat(100));
    let mut expected = vec![49406u32];
    expected.extend(std::iter::repeat_n(2653u32, 75));
    expected.push(49407);
    assert_eq!(expected, long);
    assert_eq!(1, long.iter().filter(|&&id| id == 49407).count());
}

/// `CLIPTextModelWithProjection(input_ids).last_hidden_state` is the
/// PRE-projection `[1, 77, 768]` sequence (the scribble DiT consumes it as
/// `encoder_hidden_states`).
#[test]
fn clip_text_encoder_returns_pre_projection_last_hidden_state() {
    let Some(dir) = scribble_model_dir() else {
        eprintln!("skipping CLIP text encoder test: TripoSG-scribble checkpoint not found");
        return;
    };
    let weights = dir.join("text_encoder").join("model.safetensors");
    if !weights.is_file() {
        eprintln!(
            "skipping CLIP text encoder test: {} missing",
            weights.display()
        );
        return;
    }
    let device = Device::Cpu;
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("text_encoder").join("config.json"))
            .expect("text encoder config"),
    )
    .expect("parse text encoder config");
    // SAFETY: the checkpoint file is read-only and the mmap is kept alive by
    // the VarBuilder for the duration of this test (same contract as the
    // pipeline loader).
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, &device)
            .expect("open text encoder weights")
    };
    let encoder =
        TripoSgClipTextEncoder::load(vb, &config, DType::F32).expect("CLIP text encoder load");
    assert_eq!(12, encoder.config().num_hidden_layers);
    assert_eq!(768, encoder.config().embed_dim);
    assert_eq!(77, encoder.config().max_position_embeddings);

    let tokenizer =
        ClipBpeTokenizer::from_dir(&dir.join("tokenizer")).expect("CLIP tokenizer load");
    let hidden = encoder
        .encode_prompt(&tokenizer, "a photo of a cat", &device)
        .expect("CLIP text encode");
    assert_eq!(hidden.dims(), [1, 77, 768]);
    let values = hidden
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("values");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "text embeddings must be finite"
    );
    assert!(
        values.iter().any(|value| value.abs() > 1e-6),
        "text embeddings must not be all zero"
    );
}

/// Opt-in CUDA e2e: full scribble inference (16 steps, guidance 0), a second
/// identical-seed run for reproducibility, then dense/hierarchical SDF decode
/// and GLB export with structural mesh assertions.
#[test]
fn scribble_e2e_produces_reproducible_glb() {
    if std::env::var_os("LUX3D_TRIPOSG_SCRIBBLE_E2E").is_none() {
        eprintln!("skipping scribble e2e: set LUX3D_TRIPOSG_SCRIBBLE_E2E=1 to run it");
        return;
    }
    let dir = scribble_model_dir().expect("TripoSG-scribble checkpoint is required for the e2e");
    let source = repo_root().join("test-assets").join("mesh-input.png");
    assert!(source.is_file(), "missing {}", source.display());

    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let device = Device::new_cuda(0).expect("CUDA device 0");
    let load_start = std::time::Instant::now();
    let pipeline = TripoSgScribblePipeline::load_with_dtype(dir, DType::F16)
        .expect("TripoSG-scribble pipeline load");
    eprintln!(
        "[scribble e2e] load: {:.2}s",
        load_start.elapsed().as_secs_f64()
    );

    // Reference defaults: 16 steps, 512 tokens, dense 8 / hierarchical 8,
    // scribble confidence 0.4, prompt confidence 1.0, no CFG.
    let options = TripoSgScribbleRunOptions::default();
    assert_eq!(16, options.steps);
    assert_eq!(512, options.num_tokens);

    let first_start = std::time::Instant::now();
    let first = pipeline
        .infer_from_path(&source, "a photo of a cat", &device, &options)
        .expect("first scribble inference");
    eprintln!(
        "[scribble e2e] infer 1: {:.2}s",
        first_start.elapsed().as_secs_f64()
    );
    assert_eq!(first.text_embeds.dims(), [1, 77, 768]);
    assert_eq!(first.image_embeds.dims(), [1, 257, 1024]);
    assert_eq!(first.latents_init.dims(), [1, 512, 64]);
    assert_eq!(first.latents.dims(), [1, 512, 64]);

    let second_start = std::time::Instant::now();
    let second = pipeline
        .infer_from_path(&source, "a photo of a cat", &device, &options)
        .expect("second scribble inference");
    eprintln!(
        "[scribble e2e] infer 2: {:.2}s",
        second_start.elapsed().as_secs_f64()
    );
    let init_a = first
        .latents_init
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .expect("init latents");
    let init_b = second
        .latents_init
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .expect("init latents");
    assert_eq!(init_a, init_b, "same seed must reproduce the init latents");
    let latents_a = first
        .latents
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .expect("denoised latents");
    let latents_b = second
        .latents
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .expect("denoised latents");
    let mut max_delta = 0.0_f32;
    for (a, b) in latents_a.iter().zip(latents_b.iter()) {
        max_delta = max_delta.max((a - b).abs());
    }
    assert!(
        max_delta <= 1e-4,
        "same seed/input must reproduce the denoised latents, max delta {max_delta}"
    );

    let mesh_start = std::time::Instant::now();
    let mesh = pipeline
        .extract_mesh(&first.latents, &options)
        .expect("scribble mesh extraction");
    eprintln!(
        "[scribble e2e] mesh: {:.2}s",
        mesh_start.elapsed().as_secs_f64()
    );
    let vertex_count = mesh.vertices.dims()[0];
    let face_count = mesh.faces.dims()[0];
    eprintln!("[scribble e2e] {vertex_count} vertices, {face_count} faces");
    assert!(vertex_count > 0, "scribble mesh must have vertices");
    assert!(face_count > 0, "scribble mesh must have faces");
    let vertices = mesh
        .vertices
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .expect("mesh vertices");
    assert!(
        vertices.iter().all(|value| value.is_finite()),
        "mesh vertices must be finite"
    );
    assert!(
        vertices.iter().all(|value| value.abs() <= 1.0051),
        "mesh vertices must stay inside the canonical +/-1.005 decode bounds"
    );
    let faces = mesh
        .faces
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<i64>())
        .expect("mesh faces");
    assert!(
        faces
            .iter()
            .all(|&index| index >= 0 && (index as usize) < vertex_count),
        "mesh face indices must reference existing vertices"
    );

    let output =
        std::env::temp_dir().join(format!("lux3d-triposg-scribble-{}.glb", std::process::id()));
    let export_start = std::time::Instant::now();
    pipeline
        .export_mesh(&mesh, &output)
        .expect("scribble GLB export");
    eprintln!(
        "[scribble e2e] export: {:.2}s",
        export_start.elapsed().as_secs_f64()
    );
    let bytes = std::fs::read(&output).expect("read GLB");
    let _ = std::fs::remove_file(&output);
    assert!(bytes.len() > 12, "GLB must not be empty");
    assert_eq!(&bytes[..4], b"glTF", "export must write a binary glTF");
}
