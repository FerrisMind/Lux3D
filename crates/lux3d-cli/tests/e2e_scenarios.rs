//! Real-weight CLI end-to-end scenarios.
//!
//! The tests in this file exercise the public `lux3d` binary as a subprocess
//! (not `run_model` in-process), using the weight packages that are already on
//! the machine. They are deliberately gated:
//!
//! ```text
//! LUX3D_E2E=1 \
//! LUX3D_E2E_DEVICE=cuda \
//! cargo test --release -p lux3d-cli --features cuda --test e2e_scenarios -- --test-threads=1
//! ```
//!
//! Without `LUX3D_E2E=1` every test prints a `SKIP` line. Set
//! `LUX3D_E2E_REQUIRE=1` to turn missing weights/assets into hard failures.
//! Overrides:
//!
//! - `LUX3D_E2E_WEIGHTS`  (default `<runtime-root>/3d/standalone`)
//! - `LUX3D_E2E_CANONICAL` (default `<runtime-root>/3d/canonical-weights`)
//! - `LUX3D_E2E_ASSETS`   (default `<workspace>/test-assets`)
//! - `LUX3D_E2E_SCRIBBLE_MODEL` (default `<runtime-root>/3d/huggingface/TripoSG-scribble-GGUF/triposg-scribble-q8_0.gguf`)
//! - `LUX3D_E2E_RMBG2`    (default `<workspace>/3d/canonical-weights/rmbg-2.0`)
//! - `LUX3D_E2E_DINOV2_REG` (default `<workspace>/models/dinov2_reg4/model.safetensors`)
//! - `LUX3D_E2E_PYTHON`   (default `python`; only used by the surface-input test)
//!
//! Outputs are kept under `target/e2e-scenarios/<test-name>/` for inspection.

use std::{
    env,
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use lux3d_core::test_support::{GpuTestLock, workspace_root};

const BIN: &str = env!("CARGO_BIN_EXE_lux3d-cli");

fn enabled() -> bool {
    env::var_os("LUX3D_E2E").is_some()
}

fn require_mode() -> bool {
    env::var_os("LUX3D_E2E_REQUIRE").is_some()
}

fn skip_or_fail(name: &str, message: impl AsRef<str>) {
    let message = message.as_ref();
    if require_mode() {
        panic!("{name}: {message} (LUX3D_E2E_REQUIRE=1)");
    }
    eprintln!("SKIP {name}: {message}");
}

/// Serializes real GPU runs across tests and test processes. The lock is
/// deliberately acquired only after the cheap skip checks.
fn setup(name: &str, required: &[(&str, PathBuf)]) -> Option<GpuTestLock> {
    if !enabled() {
        eprintln!("SKIP {name}: set LUX3D_E2E=1 to run real-weight scenarios");
        return None;
    }
    for (label, path) in required {
        if !path.exists() {
            skip_or_fail(name, format!("{label} is missing: {}", path.display()));
            return None;
        }
    }
    Some(GpuTestLock::acquire().expect("GPU test lock"))
}

fn runtime_root() -> PathBuf {
    lux3d_core::test_support::runtime_root()
}

fn weights_root() -> PathBuf {
    env::var_os("LUX3D_E2E_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime_root().join("3d").join("standalone"))
}

fn canonical_root() -> PathBuf {
    env::var_os("LUX3D_E2E_CANONICAL")
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime_root().join("3d").join("canonical-weights"))
}

fn assets_root() -> PathBuf {
    env::var_os("LUX3D_E2E_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("test-assets"))
}

fn standalone(name: &str) -> PathBuf {
    weights_root().join(name)
}

fn canonical(name: &str) -> PathBuf {
    canonical_root().join(name)
}

fn device() -> String {
    env::var("LUX3D_E2E_DEVICE").unwrap_or_else(|_| "cuda".to_owned())
}

fn output_dir(name: &str) -> PathBuf {
    let dir = workspace_root()
        .join("target")
        .join("e2e-scenarios")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create e2e output dir");
    dir
}

fn push_flag(args: &mut Vec<OsString>, flag: &str) {
    args.push(flag.into());
}

fn push_path(args: &mut Vec<OsString>, flag: &str, value: &Path) {
    args.push(flag.into());
    args.push(value.as_os_str().to_owned());
}

fn push_value(args: &mut Vec<OsString>, flag: &str, value: impl AsRef<OsStr>) {
    args.push(flag.into());
    args.push(value.as_ref().to_owned());
}

fn run_cli(name: &str, args: Vec<OsString>) -> std::process::Output {
    run_cli_with_os_env(name, args, &[])
}

fn run_cli_with_env(
    name: &str,
    args: Vec<OsString>,
    envs: &[(&str, &Path)],
) -> std::process::Output {
    let envs = envs
        .iter()
        .map(|(key, value)| (*key, value.as_os_str()))
        .collect::<Vec<_>>();
    run_cli_with_os_env(name, args, &envs)
}

/// TripoSG q8 weights hold ~2 GiB packed. On WGPU the default dense path
/// expands them to ~8 GiB of F32 and OOMs on a 12 GiB card (the whole
/// DiT/encoder bundle stays resident, unlike TRELLIS.2's sequential stages),
/// while dense F16 (`=f16`) corrupts batched DiT denoise: candle's WGPU F16
/// GEMM mishandles batch > 1, producing one noisy and one empty mesh for a
/// two-shape batch. Keep packed Q8 QMatMul (`=0`), which is memory-safe and
/// batch-correct. The VAE decode still runs dense F32 (the TripoSG VAE loader
/// forces F32 on WGPU; this also avoids the 1 MiB hot-ring alias at 512 latent
/// tokens).
fn run_cli_for_triposg(name: &str, args: Vec<OsString>) -> std::process::Output {
    let mut envs: Vec<(&str, &OsStr)> = Vec::new();
    if device() == "wgpu" {
        envs.push(("LUX3D_WGPU_DEQUANT_WEIGHTS", OsStr::new("0")));
    }
    run_cli_with_os_env(name, args, &envs)
}

fn run_cli_with_os_env(
    name: &str,
    args: Vec<OsString>,
    envs: &[(&str, &OsStr)],
) -> std::process::Output {
    let rendered = args
        .iter()
        .map(|arg| format!("{arg:?}"))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[e2e] {name}: {BIN} {rendered}");
    let started = Instant::now();
    let mut command = Command::new(BIN);
    command.current_dir(workspace_root()).args(&args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{name}: failed to launch {BIN}: {error}"));
    eprintln!(
        "[e2e] {name}: exit={} wall={:.2}s",
        output.status,
        started.elapsed().as_secs_f64()
    );
    if !output.status.success() {
        panic!(
            "{name} failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output
}

fn read_ply_header(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert!(
        bytes.starts_with(b"ply"),
        "{} is not a PLY file",
        path.display()
    );
    let end = bytes
        .windows(b"end_header".len())
        .position(|window| window == b"end_header")
        .unwrap_or_else(|| panic!("{} has no end_header", path.display()));
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn assert_ply(path: &Path, min_vertices: usize) {
    assert!(path.is_file(), "missing PLY {}", path.display());
    let header = read_ply_header(path);
    let line = header
        .lines()
        .find(|line| line.starts_with("element vertex "))
        .unwrap_or_else(|| panic!("{} has no vertex element", path.display()));
    let vertices: usize = line
        .split_whitespace()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("bad vertex line in {}: {line}", path.display()));
    assert!(
        vertices >= min_vertices,
        "{} has {vertices} vertices, expected at least {min_vertices}",
        path.display()
    );
}

fn glb_json(path: &Path) -> serde_json::Value {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert!(bytes.len() >= 20, "{} is too small for GLB", path.display());
    assert_eq!(&bytes[..4], b"glTF", "{} has no GLB magic", path.display());
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let length = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    assert_eq!(version, 2, "{} is not glTF 2", path.display());
    assert_eq!(
        length as usize,
        bytes.len(),
        "{} GLB header length mismatch",
        path.display()
    );
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json_type = &bytes[16..20];
    assert_eq!(
        json_type,
        b"JSON",
        "{} first GLB chunk is not JSON",
        path.display()
    );
    assert!(
        20 + json_len <= bytes.len(),
        "{} JSON chunk overruns file",
        path.display()
    );
    serde_json::from_slice(&bytes[20..20 + json_len])
        .unwrap_or_else(|error| panic!("{} has invalid GLB JSON: {error}", path.display()))
}

fn assert_glb_mesh(path: &Path) {
    assert!(path.is_file(), "missing GLB {}", path.display());
    let json = glb_json(path);
    assert!(
        json.get("meshes")
            .and_then(|value| value.as_array())
            .is_some_and(|meshes| !meshes.is_empty()),
        "{} has no non-empty meshes array",
        path.display()
    );
}

fn assert_glb_textured(path: &Path) {
    assert_glb_mesh(path);
    let json = glb_json(path);
    assert!(
        json.get("images")
            .and_then(|value| value.as_array())
            .is_some_and(|images| !images.is_empty()),
        "{} has no embedded texture images",
        path.display()
    );
}

fn write_cube_obj(path: &Path) {
    let body = "\
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 2 3
f 1 3 4
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
";
    fs::write(path, body).unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
}

fn write_surface_npy(name: &str, path: &Path) -> bool {
    let python = env::var_os("LUX3D_E2E_PYTHON").unwrap_or_else(|| "python".into());
    let script = r#"
import sys
import numpy as np
n = 204800
i = np.arange(n, dtype=np.float64)
phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
theta = np.pi * (1.0 + 5.0 ** 0.5) * i
points = np.stack(
    [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
    axis=1,
).astype(np.float32)
# Separate objects: the reference pickle has two independent ndarray payloads,
# and the restricted reader does not resolve a shared pickle-memo array twice.
np.save(sys.argv[1], {"surface_points": points, "surface_normals": points.copy()})
"#;
    let output = Command::new(&python)
        .args(["-I", "-c", script])
        .arg(path)
        .output();
    match output {
        Ok(output) if output.status.success() && path.is_file() => true,
        Ok(output) => {
            skip_or_fail(
                name,
                format!(
                    "surface fixture generation failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                ),
            );
            false
        }
        Err(error) => {
            skip_or_fail(
                name,
                format!("python/numpy unavailable for surface fixture: {error}"),
            );
            false
        }
    }
}

fn checked_workdir(name: &str, required: &[(&str, PathBuf)]) -> Option<(GpuTestLock, PathBuf)> {
    let lock = setup(name, required)?;
    Some((lock, output_dir(name)))
}

// ---------------------------------------------------------------------------
// Pi3 / Pi3X / TripoSR
// ---------------------------------------------------------------------------

#[test]
fn pi3_frames_to_point_cloud() {
    const NAME: &str = "pi3_frames_to_point_cloud";
    let model = standalone("pi3-q8_0");
    let source = assets_root().join("pi3-frames");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[("pi3 model", model.clone()), ("pi3 frames", source.clone())],
    ) else {
        return;
    };
    let output = out_dir.join("cloud.ply");
    let mut args = vec!["run".into(), "pi3".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", &model);
    push_path(&mut args, "--source", &source);
    push_path(&mut args, "--output", &output);
    run_cli(NAME, args);
    assert_ply(&output, 100);
}

#[test]
fn pi3x_frames_to_point_cloud() {
    const NAME: &str = "pi3x_frames_to_point_cloud";
    let model = standalone("pi3x-q8_0");
    let source = assets_root().join("pi3-frames");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("pi3x model", model.clone()),
            ("pi3x frames", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("cloud.ply");
    let mut args = vec!["run".into(), "pi3x".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", &model);
    push_path(&mut args, "--source", &source);
    push_path(&mut args, "--output", &output);
    run_cli(NAME, args);
    assert_ply(&output, 100);
}

#[test]
fn pi3x_vo_inject_to_point_cloud() {
    const NAME: &str = "pi3x_vo_inject_to_point_cloud";
    let model = standalone("pi3x-q8_0");
    let source = assets_root().join("pi3-frames");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("pi3x model", model.clone()),
            ("pi3x frames", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("vo.ply");
    let mut args = vec!["run".into(), "pi3x".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", &model);
    push_path(&mut args, "--source", &source);
    push_path(&mut args, "--output", &output);
    push_flag(&mut args, "--vo");
    push_value(&mut args, "--chunk-size", "2");
    push_value(&mut args, "--overlap", "1");
    push_value(&mut args, "--inject-condition", "pose,depth,ray");
    run_cli(NAME, args);
    assert_ply(&output, 100);
}

#[test]
fn triposr_image_to_mesh() {
    const NAME: &str = "triposr_image_to_mesh";
    let model = standalone("triposr-q8_0");
    let source = assets_root().join("mesh-input.png");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("triposr model", model.clone()),
            ("triposr image", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("mesh.glb");
    let mut args = vec!["run".into(), "triposr".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", &model);
    push_path(&mut args, "--source", &source);
    push_path(&mut args, "--output", &output);
    push_value(&mut args, "--mc-resolution", "128");
    push_value(&mut args, "--mc-threshold", "25.0");
    run_cli(NAME, args);
    assert_glb_mesh(&output);
}

// ---------------------------------------------------------------------------
// TripoSG: image, batch, surface, scribble
// ---------------------------------------------------------------------------

#[test]
fn triposg_image_to_mesh() {
    const NAME: &str = "triposg_image_to_mesh";
    let model = standalone("triposg-q8_0");
    let rmbg = canonical("rmbg-1.4");
    let source = assets_root().join("mesh-input-rgb.png");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("triposg model", model.clone()),
            ("RMBG-1.4 package", rmbg.clone()),
            ("triposg RGB image", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("mesh.glb");
    let mut args = vec!["run".into(), "triposg".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", &model);
    push_path(&mut args, "--rmbg-model-path", &rmbg);
    push_path(&mut args, "--source", &source);
    push_path(&mut args, "--output", &output);
    push_value(&mut args, "--steps", "4");
    push_value(&mut args, "--num-tokens", "1024");
    push_value(&mut args, "--seed", "42");
    run_cli_for_triposg(NAME, args);
    assert_glb_mesh(&output);
}

#[test]
fn triposg_batch_to_meshes() {
    const NAME: &str = "triposg_batch_to_meshes";
    let model = standalone("triposg-q8_0");
    let rmbg = canonical("rmbg-1.4");
    let source_a = assets_root().join("mesh-input.png");
    let source_b = assets_root().join("mesh-input-rgb.png");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("triposg model", model.clone()),
            ("RMBG-1.4 package", rmbg.clone()),
            ("triposg image A", source_a.clone()),
            ("triposg image B", source_b.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("batch.glb");
    let mut args = vec!["run".into(), "triposg".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", &model);
    push_path(&mut args, "--rmbg-model-path", &rmbg);
    push_path(&mut args, "--source", &source_a);
    push_path(&mut args, "--source", &source_b);
    push_path(&mut args, "--output", &output);
    push_value(&mut args, "--num-shapes", "2");
    // Four one-step samples at 512 tokens gave one empty mesh; the reference
    // decoder needs the full latent budget before every batch item is non-empty.
    push_value(&mut args, "--steps", "4");
    push_value(&mut args, "--num-tokens", "1024");
    push_value(&mut args, "--flash-decoder", "false");
    push_value(&mut args, "--dense-octree-depth", "8");
    push_value(&mut args, "--hierarchical-octree-depth", "9");
    run_cli_for_triposg(NAME, args);
    for index in 0..4 {
        assert_glb_mesh(&out_dir.join(format!("batch_{index}.glb")));
    }
}

#[test]
fn triposg_surface_to_mesh() {
    const NAME: &str = "triposg_surface_to_mesh";
    let model = standalone("triposg-q8_0");
    let Some((_lock, out_dir)) = checked_workdir(NAME, &[("triposg model", model.clone())]) else {
        return;
    };
    let surface = out_dir.join("surface.npy");
    if !write_surface_npy(NAME, &surface) {
        return;
    }
    let output = out_dir.join("surface.glb");
    let mut args = vec!["run".into(), "triposg".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", &model);
    push_path(&mut args, "--surface-input", &surface);
    push_path(&mut args, "--output", &output);
    push_value(&mut args, "--num-tokens", "512");
    push_value(&mut args, "--flash-decoder", "false");
    push_value(&mut args, "--dense-octree-depth", "7");
    push_value(&mut args, "--hierarchical-octree-depth", "8");
    run_cli_for_triposg(NAME, args);
    assert_glb_mesh(&output);
}

#[test]
fn triposg_scribble_to_mesh() {
    const NAME: &str = "triposg_scribble_to_mesh";
    let model = env::var_os("LUX3D_E2E_SCRIBBLE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            runtime_root()
                .join("3d")
                .join("huggingface")
                .join("TripoSG-scribble-GGUF")
                .join("triposg-scribble-q8_0.gguf")
        });
    let source = assets_root().join("mesh-input.png");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("scribble model", model.clone()),
            ("scribble image", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("scribble.glb");
    let mut args = vec!["run".into(), "triposg".into()];
    push_value(&mut args, "--device", device());
    push_value(&mut args, "--mode", "scribble");
    push_path(&mut args, "--model-path", &model);
    push_path(&mut args, "--source", &source);
    push_path(&mut args, "--output", &output);
    push_value(&mut args, "--prompt", "a small test object");
    push_value(&mut args, "--steps", "4");
    push_value(&mut args, "--seed", "42");
    run_cli_for_triposg(NAME, args);
    assert_glb_mesh(&output);
}

// ---------------------------------------------------------------------------
// VGGT
// ---------------------------------------------------------------------------

fn vggt_base(source: &Path, model: &Path, output: &Path, batch_size: usize) -> Vec<OsString> {
    let mut args = vec!["run".into(), "vggt".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", model);
    push_path(&mut args, "--source", source);
    push_path(&mut args, "--output", output);
    push_value(&mut args, "--batch-size", batch_size.to_string());
    // The standalone q8 package keeps its non-quantized tensors in F32, so
    // F16 activations would hit a dtype mismatch in the camera head.
    push_value(&mut args, "--vggt-dtype", "f32");
    args
}

#[test]
fn vggt_frames_to_point_cloud() {
    const NAME: &str = "vggt_frames_to_point_cloud";
    let model = standalone("vggt-q8_0");
    let source = assets_root().join("pi3-frames");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("vggt model", model.clone()),
            ("vggt frames", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("cloud.ply");
    run_cli(NAME, vggt_base(&source, &model, &output, 3));
    assert_ply(&output, 100);
}

#[test]
fn vggt_depth_points() {
    const NAME: &str = "vggt_depth_points";
    let model = standalone("vggt-q8_0");
    let source = assets_root().join("pi3-frames");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("vggt model", model.clone()),
            ("vggt frames", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("cloud.ply");
    let depth = out_dir.join("depth.ply");
    let mut args = vggt_base(&source, &model, &output, 3);
    push_path(&mut args, "--depth-points", &depth);
    run_cli(NAME, args);
    assert_ply(&output, 100);
    assert_ply(&depth, 100);
}

#[test]
fn vggt_glb_scene() {
    const NAME: &str = "vggt_glb_scene";
    let model = standalone("vggt-q8_0");
    let source = assets_root().join("pi3-frames");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("vggt model", model.clone()),
            ("vggt frames", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("cloud.ply");
    let glb = out_dir.join("scene.glb");
    let mut args = vggt_base(&source, &model, &output, 3);
    push_path(&mut args, "--glb", &glb);
    push_flag(&mut args, "--glb-show-cam");
    run_cli(NAME, args);
    assert_ply(&output, 100);
    assert_glb_mesh(&glb);
}

#[test]
fn vggt_colmap_feedforward() {
    const NAME: &str = "vggt_colmap_feedforward";
    let model = standalone("vggt-q8_0");
    let source = assets_root().join("pi3-frames");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("vggt model", model.clone()),
            ("vggt frames", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("cloud.ply");
    let colmap = out_dir.join("colmap");
    let mut args = vggt_base(&source, &model, &output, 3);
    push_path(&mut args, "--colmap-dir", &colmap);
    push_value(&mut args, "--colmap-mode", "feedforward");
    push_value(&mut args, "--colmap-max-points", "500");
    run_cli(NAME, args);
    assert_ply(&output, 100);
    for file in ["cameras.bin", "images.bin", "points3D.bin"] {
        let path = colmap.join(file);
        assert!(path.is_file(), "missing COLMAP output {}", path.display());
        assert!(
            fs::metadata(&path).unwrap().len() > 0,
            "empty COLMAP output {}",
            path.display()
        );
    }
}

#[test]
fn vggt_track_queries() {
    const NAME: &str = "vggt_track_queries";
    let model = standalone("vggt-q8_0");
    let source = assets_root().join("pi3-frames");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("vggt model", model.clone()),
            ("vggt frames", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("cloud.ply");
    let tracks = out_dir.join("tracks.json");
    // The track head needs one attention pass over every frame.
    let mut args = vggt_base(&source, &model, &output, 1);
    push_value(
        &mut args,
        "--track-queries",
        r#"[{"frame":0,"x":256.0,"y":256.0}]"#,
    );
    push_path(&mut args, "--track-output", &tracks);
    run_cli(NAME, args);
    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(&tracks).expect("read track JSON"))
            .expect("parse track JSON");
    let coords = json["coords"].as_array().expect("coords array");
    assert_eq!(coords.len(), 3, "expected one track per input frame");
    assert_eq!(
        coords[0].as_array().expect("frame 0 array").len(),
        1,
        "expected one query in frame 0"
    );
}

#[test]
fn vggt_track_auto_fps() {
    const NAME: &str = "vggt_track_auto_fps";
    let model = standalone("vggt-q8_0");
    let source = assets_root().join("pi3-frames");
    let dinov2 = env::var_os("LUX3D_E2E_DINOV2_REG")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            workspace_root()
                .join("models")
                .join("dinov2_reg4")
                .join("model.safetensors")
        });
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("vggt model", model.clone()),
            ("vggt frames", source.clone()),
            ("DINOv2-reg weights", dinov2.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("cloud.ply");
    let tracks = out_dir.join("tracks.json");
    let mut args = vggt_base(&source, &model, &output, 1);
    push_flag(&mut args, "--track-auto");
    push_value(&mut args, "--track-keypoints", "fps");
    push_value(&mut args, "--query-frame-num", "2");
    push_value(&mut args, "--max-query-pts", "32");
    push_path(&mut args, "--track-output", &tracks);
    run_cli_with_env(
        NAME,
        args,
        &[("LUX3D_DINOV2_REG_MODEL_PATH", dinov2.as_path())],
    );
    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(&tracks).expect("read track JSON"))
            .expect("parse track JSON");
    assert!(
        json["coords"]
            .as_array()
            .is_some_and(|coords| coords.len() == 3),
        "automatic track output must cover all frames"
    );
}

// ---------------------------------------------------------------------------
// TRELLIS.2
// ---------------------------------------------------------------------------

fn trellis2_model() -> PathBuf {
    env::var_os("LUX3D_E2E_TRELLIS2_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| standalone("trellis2-q4_k"))
}

fn rmbg2_model() -> PathBuf {
    env::var_os("LUX3D_E2E_RMBG2")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            workspace_root()
                .join("3d")
                .join("canonical-weights")
                .join("rmbg-2.0")
        })
}

/// The standalone GGUF package embeds the `shape_enc` stage, so mesh
/// texturing uses the same package as image-to-3D. The override still accepts
/// the raw `TRELLIS.2-4B` checkpoint tree for comparison runs.
fn trellis2_texture_model() -> PathBuf {
    env::var_os("LUX3D_E2E_TRELLIS2_TEXTURE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(trellis2_model)
}

fn trellis2_base(source: &Path, model: &Path, output: &Path, rmbg2: &Path) -> Vec<OsString> {
    let mut args = vec!["run".into(), "trellis2".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", model);
    push_path(&mut args, "--rmbg-model-path", rmbg2);
    push_path(&mut args, "--source", source);
    push_path(&mut args, "--output", output);
    push_value(&mut args, "--pipeline-type", "512");
    push_value(&mut args, "--steps", "1");
    push_value(&mut args, "--seed", "42");
    args
}

#[test]
fn trellis2_512_geometry() {
    const NAME: &str = "trellis2_512_geometry";
    let model = trellis2_model();
    let rmbg2 = rmbg2_model();
    let source = assets_root().join("mesh-input-rgb.png");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("trellis2 model", model.clone()),
            ("RMBG-2.0 package", rmbg2.clone()),
            ("trellis2 RGB image", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("geometry.glb");
    let mut args = trellis2_base(&source, &model, &output, &rmbg2);
    push_value(&mut args, "--texture", "false");
    push_value(&mut args, "--decimation-target", "50000");
    // The default narrow-band remesh is CPU-bound and exceeded 17 minutes on
    // this host's 1.5M-voxel decode; the cleanup/decimation branch is the
    // practical local path. The remesh algorithm itself has unit tests.
    push_flag(&mut args, "--no-remesh");
    run_cli(NAME, args);
    assert_glb_mesh(&output);
}

#[test]
fn trellis2_512_textured_glb() {
    const NAME: &str = "trellis2_512_textured_glb";
    let model = trellis2_model();
    let rmbg2 = rmbg2_model();
    let source = assets_root().join("mesh-input.png");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("trellis2 model", model.clone()),
            ("RMBG-2.0 package", rmbg2.clone()),
            ("trellis2 image", source.clone()),
        ],
    ) else {
        return;
    };
    let output = out_dir.join("textured.glb");
    let mut args = trellis2_base(&source, &model, &output, &rmbg2);
    push_value(&mut args, "--texture-size", "256");
    push_value(&mut args, "--decimation-target", "50000");
    push_flag(&mut args, "--no-remesh");
    run_cli(NAME, args);
    assert_glb_textured(&output);
}

#[test]
fn trellis2_texture_mesh() {
    const NAME: &str = "trellis2_texture_mesh";
    let model = trellis2_texture_model();
    let rmbg2 = rmbg2_model();
    let source = assets_root().join("mesh-input.png");
    let Some((_lock, out_dir)) = checked_workdir(
        NAME,
        &[
            ("trellis2 model", model.clone()),
            ("RMBG-2.0 package", rmbg2.clone()),
            ("texture image", source.clone()),
        ],
    ) else {
        return;
    };
    let mesh = out_dir.join("cube.obj");
    write_cube_obj(&mesh);
    let output = out_dir.join("cube_textured.glb");
    // `--texture-mesh` rejects image-to-3D-only flags such as `--pipeline-type`,
    // so build this invocation without `trellis2_base`.
    let mut args = vec!["run".into(), "trellis2".into()];
    push_value(&mut args, "--device", device());
    push_path(&mut args, "--model-path", &model);
    push_path(&mut args, "--rmbg-model-path", &rmbg2);
    push_path(&mut args, "--source", &source);
    push_path(&mut args, "--output", &output);
    push_path(&mut args, "--texture-mesh", &mesh);
    push_value(&mut args, "--texture-resolution", "512");
    push_value(&mut args, "--texture-size", "256");
    push_value(&mut args, "--steps", "1");
    push_value(&mut args, "--tex-steps", "1");
    push_value(&mut args, "--seed", "42");
    run_cli(NAME, args);
    assert_glb_textured(&output);
}

// ---------------------------------------------------------------------------
// O-Voxel CLI round-trip (no model weights)
// ---------------------------------------------------------------------------

#[test]
fn ovox_roundtrip() {
    const NAME: &str = "ovox_roundtrip";
    let Some((_lock, out_dir)) = checked_workdir(NAME, &[]) else {
        return;
    };
    let mesh = out_dir.join("cube.obj");
    write_cube_obj(&mesh);
    let vxz = out_dir.join("cube.vxz");
    let ply = out_dir.join("cube.ply");
    let glb = out_dir.join("cube.glb");

    // `mesh2ovox` takes positional input/output, not flags.
    let mut to_vox = vec![
        "ovox".into(),
        "mesh2ovox".into(),
        mesh.as_os_str().to_owned(),
        vxz.as_os_str().to_owned(),
    ];
    push_value(&mut to_vox, "--grid-size", "32");
    run_cli(NAME, to_vox);

    let mut to_ply = vec!["ovox".into(), "ovox2mesh".into()];
    to_ply.push(vxz.as_os_str().to_owned());
    to_ply.push(ply.as_os_str().to_owned());
    push_value(&mut to_ply, "--grid-size", "32");
    run_cli(NAME, to_ply);

    let mut to_glb = vec!["ovox".into(), "ovox2glb".into()];
    to_glb.push(vxz.as_os_str().to_owned());
    to_glb.push(glb.as_os_str().to_owned());
    push_value(&mut to_glb, "--grid-size", "32");
    push_value(&mut to_glb, "--texture-size", "128");
    push_value(&mut to_glb, "--decimation-target", "1000");
    run_cli(NAME, to_glb);

    assert_ply(&ply, 3);
    assert_glb_mesh(&glb);
}
