# Lux3D / candle-3d — Feature Inventory & User Stories

## Legend
| Status | Meaning |
|--------|---------|
| ✅ Done | Implemented & tested |
| 🔍 Review | Code exists, needs testing |
| 🐛 Bug | Issue found |
| ❌ Missing | Not implemented |
| ⏳ Pending | Planned |

---

## 1. CLI — Entry & Dispatch

### US-1.1: CLI parses subcommands
As a user, I can run `lux3d inspect|weights normalize|run <family>` and get correct dispatch.

**Functions:**
- `main.rs:main()` — Entry point ✅
- `cli.rs:Cli` — Clap parser struct ✅
- `cli.rs:Command::Inspect` / `Weights` / `Run` — Subcommand enum ✅
- `cli.rs:Family` — Pi3 / Pi3x / Triposr / Triposg / Vggt / Trellis2 enum ✅
- `cli.rs:InjectCondition` — Pose / Depth / Ray / Intrinsic enum ✅

**Tests:** `tests/run_args.rs` ✅

---

### US-1.2: Inspect contracts
As a user, I can run `lux3d inspect --repo-root <dir> <family>` and get a JSON spec.

**Functions:**
- `cli.rs:inspect_model()` — Reads ModelSpec from repo root ✅

**Tests:** None found 🔍

---

### US-1.3: Normalize weights
As a maintainer, I can run `lux3d weights normalize` to canonicalize raw weights.

**Functions:**
- `cli.rs:normalize_weights()` — Calls Python baseline script ✅
- `cli.rs:python_executable()` — Resolves Python binary ✅
- `lux3d_core::test_support::canonical_package_dir()` — Finds canonical weight package (test-support helper; imported by `cli.rs` tests) ✅

**Tests:** None found 🔍

---

### US-1.4: Run inference (CLI)
As a user, I can run `lux3d run <family> --source <input> --output <file>` to produce 3D output.

**Functions:**
- `cli.rs:run_model()` — Main inference dispatcher ✅
- `cli.rs:RunArgs` — All run parameters ✅
- `main.rs:25` (`Command::Run`) + `cli.rs:run_model()` — Dispatch from the parsed CLI ✅

**Tests:** `tests/run_args.rs` covers model_path & cache_dir ✅

---

## 2. Pi3 Pipeline

### US-2.1: Pi3 loads model weights
As a user, the Pi3 pipeline loads canonical weights from disk or HuggingFace cache.

**Functions:**
- `runtime/mod.rs:Pi3Pipeline::load()` ✅
- `runtime/mod.rs:Pi3ModelBundle::load()` — Loads encoder, decoder, heads ✅
- `runtime/mod.rs:Pi3ModelBundle` — struct with 8 sub-modules ✅
- `runtime/mod.rs:bundle_for()` — DeviceLocalCache wrapper ✅
- `runtime/mod.rs:DeviceLocalCache` — Per-device singleton cache ✅

**Tests:** `runtime/mod.rs:pi3_encoder_reuses_cached_interpolated_positional_embeddings` ✅

---

### US-2.2: Pi3 preprocesses input
As a user, the Pi3 pipeline resizes input frames to a patch-multiple size.

**Functions:**
- `preprocess/pi3.rs:Pi3PreprocessStage::prepare_inputs_from_path()` ✅
- `preprocess/pi3.rs:Pi3PreprocessStage::target_size_for()` — Computes patch-aligned target size ✅
- `preprocess/pi3.rs:Pi3PreprocessStage::rescale_intrinsics()` ✅
- `runtime/mod.rs:prepare_pi3_inputs_with_stage()` ✅
- `runtime/mod.rs:load_sampled_pi3_frames()` — Directory or MP4 input ✅
- `runtime/mod.rs:collect_rgb_entries()` — Filters PNG/JPG/JPEG ✅
- `runtime/mod.rs:extract_video_frames()` — ffmpeg frame extraction ✅

**Tests:** `preprocess/pi3.rs:rescales_intrinsics_using_target_size` ✅

---

### US-2.3: Pi3 encodes frames with DINOv2
As the model, patch tokens are extracted from normalized frames.

**Functions:**
- `runtime/pi3_encoder.rs:Pi3DinoEncoder::forward_patch_tokens()` — Full forward pass ✅
- `runtime/pi3_encoder.rs:Pi3DinoEncoder::prepare_tokens()` — Patch embed + cls + register + pos ✅
- `runtime/pi3_encoder.rs:Pi3DinoEncoder::interpolate_pos_encoding()` — Cached bicubic resize ✅
- `runtime/pi3_encoder.rs:PatchEmbed::forward()` — Conv2d patch embedding ✅
- `runtime/pi3_encoder.rs:Block` — Pre-normalized ViT block with LayerScale ✅
- `runtime/pi3_encoder.rs:Attention` — QKV attention ✅
- `runtime/pi3_encoder.rs:resize_patch_pos_embed()` — AA-cubic interpolation ✅

**Tests:** `cached_pos_encoding_entries()` assertion ✅

---

### US-2.4: Pi3 decodes hidden states
As the model, encoder outputs are decoded into point/confidence/camera representations.

**Functions:**
- `runtime/pi3_decoder.rs:Pi3Decoder::decode()` — 36-block RoPE decoder ✅
- `runtime/pi3_decoder.rs:Pi3Decoder::decoder_positions()` — Grid + special token positions ✅
- `runtime/pi3_decoder.rs:RopeBlock` — Transformer block with RoPE attention ✅
- `runtime/pi3_decoder.rs:RopeAttention` — Q/K norm + RoPE + flash attention ✅
- `runtime/pi3_decoder.rs:Pi3BranchDecoder` — 5-block branch decoder ✅
- `runtime/pi3_decoder.rs:BranchBlock` — Simplified branch block (no LayerScale) ✅

**Tests:** None found 🔍

---

### US-2.5: Pi3 produces point cloud heads
As the model, local points, confidence, and camera poses are predicted.

**Functions:**
- `runtime/pi3_heads.rs:LinearPts3dHead::forward()` — Pixel-shuffle upsampling ✅
- `runtime/pi3_heads.rs:CameraHead::forward()` — Pooled camera prediction ✅
- `runtime/pi3_heads.rs:ResConvBlock` — Residual conv block ✅

**Tests:** None found 🔍

---

### US-2.6: Pi3 assembles geometry
As the model, local points are transformed to world space and filtered.

**Functions:**
- `runtime/mod.rs:Pi3Pipeline::encode_patch_tokens()` ✅
- `runtime/mod.rs:Pi3Pipeline::decode_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::decode_positions_only()` ✅
- `runtime/mod.rs:Pi3Pipeline::point_decoder_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::conf_decoder_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::camera_decoder_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::point_head_output()` ✅
- `runtime/mod.rs:Pi3Pipeline::conf_head_output()` ✅
- `runtime/mod.rs:Pi3Pipeline::camera_poses_from_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::world_points()` ✅
- `runtime/mod.rs:Pi3Pipeline::export_mask()` ✅
- `runtime/mod.rs:cast_tensor_to_f32()` ✅

**Tests:** None found 🔍

---

### US-2.7: Pi3 exports to PLY
As a user, the output point cloud is written to a PLY file.

**Functions:**
- `export/pi3.rs:Pi3ExportStage::write_ply()` — ASCII PLY writer ✅
- `geometry/pi3.rs:Pi3GeometryStage::assemble_cpu()` — Mask-based filtering ✅
- `geometry/pi3.rs:Pi3GeometryStage::assemble()` — Contract assembly ✅

**Tests:** CLI test `run_pi3_smoke_emits_ply_output` (in cli.rs) ✅

---

## 3. Pi3X Pipeline

### US-3.1: Pi3X loads extended weights
As the model, Pi3X loads encoder + depth encoder + ray embed + metric modules.

**Functions:**
- `runtime/pi3x.rs:Pi3xModelBundle::load()` — Loads 15 sub-modules ✅
- `runtime/pi3x.rs:Pi3xPipeline::load()` ✅
- `runtime/pi3x.rs:Pi3xPatchEmbed` — Conv2d patch embedding for rays ✅
- `runtime/pi3x.rs:Pi3xCoreDecoder` — 36-block decoder with projected cross-attention ✅
- `runtime/pi3x.rs:Pi3xCrossBlock` — Block with cross-attn to pose/depth features ✅
- `runtime/pi3x.rs:Pi3xConvHead` — Convolutional output head ✅
- `runtime/pi3x.rs:Pi3xContextOnlyTransformerDecoder` — Metric decoder ✅

**Tests:** None found 🔍

---

### US-3.2: Pi3X processes frames and conditions
As a user, Pi3X accepts RGB frames + optional condition NPZ files.

**Functions:**
- `runtime/pi3x.rs:Pi3xPipeline::infer_prepared()` — Full inference pipeline ✅
- `runtime/pi3x.rs:Pi3xPipeline::prepare_inputs_from_path()` ✅
- `runtime/pi3x.rs:load_sampled_pi3x_frames()` — Directory or MP4 ✅
- `runtime/pi3x.rs:prepare_pi3x_inputs_with_stage()` ✅
- `runtime/pi3x.rs:Pi3xPreparedInputs` struct ✅
- `runtime/pi3x.rs:load_pi3x_conditions()` — NPZ condition loading ✅
- `runtime/pi3x.rs:Pi3xPipeline::cross_blocks_with_inject()` — Condition injection ✅
- `runtime/pi3x.rs:Pi3xPipeline::infer_vo_from_path()` — Video object pipeline ✅
- `runtime/pi3x.rs:Pi3xPipeline::infer_vo()` — Chunked VO inference ✅
- `runtime/pi3x.rs:Pi3xPipeline::infer_vo_chunk()` ✅
- `runtime/pi3x.rs:build_pi3x_prefix_mask()` — Causal masking ✅
- `runtime/pi3x.rs:repeated_identity_poses()` ✅

**Tests:** CLI tests `run_pi3x_smoke_emits_ply_output`, `run_pi3x_vo_inject_smoke_emits_ply_output` ✅

---

### US-3.3: Pi3X ray and depth embedding
As the model, rays and depth maps are embedded into the decoder.

**Functions:**
- `runtime/pi3x.rs:Pi3xPipeline::embed_rays_and_depth()` ✅
- `runtime/pi3x.rs:Pi3xPipeline::build_ray_embeds()` ✅
- `runtime/pi3x.rs:resize_depth_nearest()` ✅

**Tests:** None found 🔍

---

### US-3.4: Pi3X metric head
As the model, a metric value is predicted per scene.

**Functions:**
- `runtime/pi3x.rs:Pi3xPipeline::run_metric_head()` ✅
- `runtime/pi3x.rs:Pi3xMetricCache` struct ✅

**Tests:** None found 🔍

---

## 4. TripoSR Pipeline

### US-4.1: TripoSR loads model
As the model, TripoSR loads image tokenizer, backbone, post-processor, and NeRF decoder.

**Functions:**
- `runtime/triposr.rs:TripoSrModelBundle::load()` ✅
- `runtime/triposr.rs:TripoSrPipeline::load()` ✅
- `runtime/triposr.rs:TripoImageTokenizer` — DINO ViT-based encoder ✅
- `runtime/triposr.rs:Triplane1DTokenizer` — Learned triplane tokens ✅
- `runtime/triposr.rs:Transformer1D` — 16-block transformer backbone ✅
- `runtime/triposr.rs:BasicTransformerBlock1d` — Dual cross-attention block ✅
- `runtime/triposr.rs:CrossAttention` — Q/K/V cross-attn ✅
- `runtime/triposr.rs:TriplaneUpsampleNetwork` — Transposed conv upsampler ✅
- `runtime/triposr.rs:NerfMlp` — 10-layer NeRF MLP ✅

**Tests:** None found 🔍

---

### US-4.2: TripoSR preprocesses image
As a user, an input image is resized, composited over gray, and normalized.

**Functions:**
- `preprocess/triposr.rs:TripoPreprocessStage::prepare_inputs_from_path()` ✅
- `preprocess/triposr.rs:TripoPreprocessStage::composite_over_gray()` ✅
- `preprocess/triposr.rs:TripoPreprocessStage::normalize_rgb()` ✅
- `runtime/triposr.rs:prepare_triposr_inputs_with_stage()` ✅

**Tests:** None found 🔍

---

### US-4.3: TripoSR generates scene codes
As the model, image tokens are transformed into triplane scene codes.

**Functions:**
- `runtime/triposr.rs:TripoSrPipeline::scene_codes()` ✅
- `runtime/triposr.rs:TripoSrPipeline::image_tokens()` ✅
- `runtime/triposr.rs:TripoSrPipeline::seed_tokens()` ✅
- `runtime/triposr.rs:TripoSrPipeline::backbone_forward()` ✅
- `runtime/triposr.rs:TripoSrPipeline::detokenize_triplanes()` ✅
- `runtime/triposr.rs:TripoSrPipeline::query_triplane()` ✅

**Tests:** None found 🔍

---

### US-4.4: TripoSR runs marching cubes
As a user, the scene codes are converted to a mesh.

**Functions:**
- `geometry/triposr.rs:TripoGeometryStage::assemble_cpu()` — Marching cubes with dedup ✅
- `geometry/triposr.rs:TripoGeometryStage::attach_vertex_colors()` ✅
- `geometry/triposr.rs:TripoGeometryStage::materialize_buffers()` ✅
- `geometry/triposr.rs:TripoGeometryStage::cpu_from_buffers()` ✅
- `neural/triposr.rs:TripoNeuralStage::build_density_grid_cpu()` — Chunked density extraction ✅
- `runtime/triposr_field.rs:query_triplane_chunked()` ✅
- `runtime/triposr_field.rs:query_triplane_density_chunked()` ✅
- `runtime/triposr_field.rs:query_triplane_color_chunked()` ✅
- `runtime/triposr_field.rs:TriplaneDecoder` trait ✅

**Tests:** CLI test `run_triposr_smoke_emits_obj_output` ✅
**Tests:** CLI test `run_triposr_rejects_invalid_marching_cubes_resolution` ✅

---

### US-4.5: TripoSR exports to OBJ
As a user, the mesh is written to an OBJ file.

**Functions:**
- `export/triposr.rs:TripoExportStage::write_obj()` — OBJ writer ✅

**Tests:** Via CLI smoke test ✅

---

## 5. Attention & Math Utilities

### US-5.1: Rotary Position Embedding (RoPE)
As the model, 2D RoPE is applied to attention queries and keys.

**Functions:**
- `runtime/attention_math.rs:Rope2d::embeddings()` ✅
- `runtime/attention_math.rs:Rope2d::apply_with_embeddings()` ✅
- `runtime/attention_math.rs:Rope2d::cos_sin()` ✅
- `runtime/attention_math.rs:Rope2d::apply_embedding()` ✅
- `runtime/attention_math.rs:Rope2d::apply_rope_embedded()` ✅
- `runtime/attention_math.rs:RopeEmbeddings` struct ✅
- `runtime/attention_math.rs:position_getter()` ✅
- `runtime/attention_math.rs:rotate_half()` ✅
- `runtime/attention_math.rs:exact_query_chunked_sdpa()` ✅
- `runtime/attention_math.rs:exact_sdpa_heads()` ✅

**Tests:** None found 🔍

---

### US-5.2: Resampling
As the model, bicubic and linear resampling is used for positional encoding.

**Functions:**
- `runtime/resampling.rs:compute_aa_cubic_weights()` ✅
- `runtime/resampling.rs:compute_aa_linear_weights()` ✅
- `runtime/resampling.rs:cubic_weight()` ✅
- `runtime/resampling.rs:clamp_isize()` ✅

**Tests:** None found 🔍

---

### US-5.3: Camera operations
As the model, 3D camera pose is computed from rotation components and translation.

**Functions:**
- `runtime/point_camera_ops.rs:camera_pose_from_components()` ✅
- `runtime/point_camera_ops.rs:world_points_from_local_and_pose()` ✅
- `runtime/point_camera_ops.rs:export_mask_from_local_points_and_confidence()` ✅
- `runtime/point_camera_ops.rs:non_edge_mask_from_local_points()` ✅

**Tests:** None found 🔍

---

## 6. Neural Stage Controllers

### US-6.1: Pi3 neural stage
The neural stage wires Pi3 pipeline outputs to contract types.

**Functions:**
- `neural/pi3.rs:Pi3NeuralStage::infer()` — Contract tensor shapes ✅
- `neural/pi3.rs:Pi3NeuralStage::infer_tensors()` — Full inference pipeline ✅

**Tests:** None found 🔍

---

### US-6.2: Pi3X neural stage
The neural stage wires Pi3X pipeline outputs.

**Functions:**
- `neural/pi3x.rs:Pi3xNeuralStage::infer()` ✅
- `neural/pi3x.rs:Pi3xNeuralStage::infer_tensors()` ✅
- `neural/pi3x.rs:Pi3xNeuralStage::infer_vo_from_path()` ✅

**Tests:** None found 🔍

---

### US-6.3: TripoSR neural stage
The neural stage wires TripoSR pipeline outputs.

**Functions:**
- `neural/triposr.rs:TripoNeuralStage::infer()` ✅
- `neural/triposr.rs:TripoNeuralStage::infer_tensors()` ✅
- `neural/triposr.rs:TripoNeuralStage::build_density_grid_cpu()` ✅

**Tests:** None found 🔍

---

## 7. Weight Management

### US-7.1: Load canonical weights
As the runtime, canonical safetensors weights are loaded from disk or HF cache.

**Functions:**
- `weights/mod.rs:load_canonical_weights()` ✅
- `weights/mod.rs:ensure_canonical_weights()` — HF auto-download ✅
- `weights/mod.rs:CanonicalWeightSetPaths` — File path bundle ✅
- `weights/mod.rs:CanonicalWeightSetPaths::var_builder()` — Unsafe mmap builder ✅
- `weights/mod.rs:ModelAssetOptions` struct ✅
- `weights/mod.rs:WeightLocator` — HF or local resolution ✅

**Tests:** None found 🔍

---

### US-7.2: Validate weight checksums
As a maintainer, canonical package integrity is verified via SHA-256.

**Functions:**
- `weights/mod.rs:verify_checksums()` ✅
- `weights/mod.rs:CanonicalChecksums` — Serialization struct ✅
- `weights/mod.rs:CanonicalWeightsManifest` — Manifest struct ✅

**Tests:** None found 🔍

---

## 8. Contract System

### US-8.1: Model specification contracts
As a maintainer, model specs define the expected architecture and data flow.

**Functions:**
- `contracts/spec.rs:ModelSpec` — Full model specification ✅
- `contracts/spec.rs:ModelSpec::inspect()` — Reads from repo root ✅
- `contracts/family.rs:ModelFamily` — 3-model enum ✅
- `contracts/baseline.rs:BaselineManifest` — Baseline artifact manifest ✅
- `contracts/baseline.rs:BaselineManifest::validate_against_model_spec()` ✅
- `contracts/stages.rs:PreprocessStage` / `NeuralStage` / `GeometryStage` / `ExportStage` traits ✅
- `contracts/common.rs:SpatialSize` / `TensorContract` / `NormalizationStats` ✅
- `contracts/pi3.rs` / `pi3x.rs` / `triposr.rs` — Family-specific contracts ✅

**Tests:** `tests/contracts.rs` ✅, `tests/model_spec.rs` ✅

---

## 9. Natural Sort Utility

### US-9.1: Natural filename sorting
As the runtime, filenames are sorted with natural number ordering (e.g., frame_2 before frame_10).

**Functions:**
- `runtime/path_utils.rs:sort_paths_natural()` ✅
- `runtime/path_utils.rs:compare_natural()` ✅
- `runtime/path_utils.rs:trim_leading_zeros()` ✅

**Tests:** `runtime/path_utils.rs:sorts_numeric_suffixes_in_human_order` ✅

---

## 10. Neural Network Building Blocks

### US-10.1: Shared NN components
Common neural network layers used across all pipelines.

**Functions:**
- `runtime/nn_blocks.rs:linear()` ✅
- `runtime/nn_blocks.rs:LayerScale` ✅
- `runtime/nn_blocks.rs:Mlp` — 4x hidden MLP with GeLU ✅
- `runtime/nn_blocks.rs:GeGlu` — Gated GeLU ✅
- `runtime/nn_blocks.rs:FeedForward` — GeGLU + Linear ✅

**Tests:** None found 🔍

---

## 11. Vision Preprocessing

### US-11.1: Image loading and resize
As the runtime, images are loaded and resized for model input.

**Functions:**
- `runtime/vision_preproc.rs` — Image loading pipeline ✅

**Tests:** None found 🔍

---

## 12. Utility & Error Handling

### US-12.1: Error types
The library defines typed errors for all failure modes.

**Functions:**
- `error.rs:Lux3dError` — 17 error variants ✅
- `error.rs:Result<T>` — Type alias ✅

**Tests:** None needed ✅

---

## 13. VGGT Pipeline

### US-13.1: VGGT-1B loads from canonical weights
As the model, VGGT loads the DINOv2 patch backbone, 24 frame + 24 global aggregator blocks,
the camera head, and the DPT point/depth heads from the canonical package.

**Functions:**
- `runtime/vggt.rs:VggtPipeline::load()` ✅
- `runtime/vggt.rs:VggtAggregator::load()` — DINOv2 ViT-L/14 + camera/register tokens ✅
- `runtime/vggt.rs:VggtCameraHead::load()` — 4 adaLN refinement iterations ✅
- `runtime/vggt_dpt.rs:VggtDptHead::load()` — point (inv_log) and depth (exp) DPT heads ✅
- `weights/mod.rs` — `vggt` family normalization/validation ✅

### US-13.2: VGGT preprocess matches the reference loader
As a user, I get the same crop/resize/pad/normalization as
`vggt.utils.load_fn.load_and_preprocess_images(mode="crop")`, plus the pad recipe and
single-image sources.

**Functions:**
- `preprocess/vggt.rs:VggtPreprocessStage::prepare_inputs_from_path()` ✅
- `preprocess/vggt.rs:VggtPreprocessStage::prepare_inputs_from_path_with_mode()` —
  `--preprocess-mode crop|pad` (pad: largest side 518, white-padded 518x518 square,
  no pixels discarded) ✅
- `preprocess/vggt.rs:VggtPreprocessStage::target_size_for()` ✅
- Single-image file sources (`.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff` — one-frame scene) ✅
- Pillow-compatible integer BICUBIC resampler (0.0 delta vs baseline) ✅

### US-13.3: VGGT inference and export
As a user, I can run multi-frame inference and export a confidence-filtered PLY.

**Functions:**
- `runtime/vggt.rs:VggtPipeline::infer_from_path()` / `infer_tensors()` ✅
- `runtime/vggt.rs:VggtPipeline::infer_from_path_with_options()` /
  `infer_prepared_with_options()` — dtype / preprocess mode / interval / sequential
  `batch_size` chunking (`merge_vggt_inference_outputs`) ✅
- `runtime/vggt.rs:VggtPipeline::load_with_options()` — `--vggt-dtype` load dtype ✅
- `runtime/vggt.rs:VggtPipeline::run_aggregator()` ✅
- `runtime/vggt.rs:decode_pose()` — quaternion XYZW to extrinsics/intrinsics ✅
- `runtime/vggt.rs:VggtInferenceOutput::world_points_from_depth()` — reference
  `unproject_depth_map_to_point_map` depth unprojection ✅
- `geometry/vggt.rs:VggtGeometryStage::assemble_cpu()` — strict `conf > threshold` ✅
- `export/vggt.rs:VggtExportStage::write_ply()` ✅
- `export/colmap.rs:write_feedforward_colmap()` — upstream `demo_colmap.py`
  feedforward mode (`cameras.bin`/`images.bin`/`points3D.bin`, PINHOLE per image,
  depth-confidence gate + point cap, rescaled intrinsics) ✅
- `runtime/vggt.rs:VggtPipeline::export_ply_with_threshold()` ✅
- `runtime/vggt.rs:VggtPipeline::export_inference_npz()` — includes the stacked
  `pose_enc_list` `[K, S, 9]` ✅
- CLI `run vggt` (see US-13.4); server `vggt` point-cloud jobs ✅

**Tests:** `runtime_vggt` (CUDA parity vs Python baseline; requires a CUDA device),
`vggt_preprocess` (CPU golden RGB + normalized parity), `vggt_export` (CPU),
DPT separable-pos-embed/cache unit tests, and `model_spec`/`contracts` VGGT cases ✅

CPU and Vulkan numeric parity was verified manually (PLY vertex count and bbox against the
Python baseline; see `.swarm/results/backends-and-bench.md`). The automated parity test is
CUDA-only; Metal has no test host in this repository.

**Excluded:** track auto-query/keypoint extraction (ALIKED/SP keypoints, DINO
query-frame ranking) — tracking requires explicit queries, see US-13.4.

### US-13.4: VGGT CLI run surface (exports + tracking)
As a user, I can run `lux3d run vggt` with the full VGGT option surface: preprocess mode,
load dtype, batch chunking, COLMAP feedforward export, depth-unprojected point clouds and
explicit-query tracking.

**Functions:**
- `cli.rs:VggtRunArgs` — flattened VGGT-only flag family; every flag is rejected for
  the other families (`reject_vggt_only_options`) ✅
- `--preprocess-mode crop|pad` (default crop) → `VggtRunOptions.preprocess_mode` ✅
- `--vggt-dtype f32|f16|bf16` (default f32) → `VggtPipeline::load_with_options` ✅
- `--batch-size N` (default 1; 0 rejected) → sequential frame chunks, NOT one attention
  pass; documented in the flag help ✅
- `--colmap-dir` + `--colmap-conf-threshold` (default 5.0) + `--colmap-max-points`
  (default 100000) → `write_feedforward_colmap` + `ColmapExportStats` print ✅
- `--depth-points` + `--depth-conf-threshold` (default 1.0) → depth-unprojected
  PLY/XYZ point cloud from `world_points_from_depth()` ✅
- `--track-queries '<json>'` (objects or `[frame,x,y]` arrays; frame must be 0,
  x/y in [0, 518); parse: `parse_vggt_track_queries`) + `--track-output` (default
  `<source-stem>_tracks.json`) + `--track-iters` (default 4) →
  `run_with_tracks` + `{"coords", "visibility", "confidence"}` JSON ✅
- `--track-points-ply` — one colored vertex per query (frame-0 world point at the
  query pixel) ✅
- `--dump-npz` includes `pose_enc_list` and works together with the track options ✅
- Single-image `--source` accepted for `vggt` ✅

**Excluded:** track auto-query/keypoint extraction is not available in the CLI —
`--track-queries` requires explicit query points.

**Tests:** `tests/run_args.rs` (`run_vggt_parses_vggt_flags`,
`run_vggt_flags_default_to_historical_behavior`,
`run_vggt_enum_flags_reject_unknown_values`,
`run_vggt_hyphen_values_parse_for_track_queries_but_not_thresholds`,
`run_model_rejects_vggt_batch_size_zero_at_dispatch_before_weights`,
`run_model_rejects_vggt_only_flag_at_dispatch_before_weights`) and the in-crate
`cli::tests` VGGT tests (option mapping, query JSON formats, track JSON
serialization, family rejection, COLMAP image-name listing, default track path) ✅

---

## 14. TripoSG Pipeline

### US-14.1: TripoSG loads from canonical weights
As the model, TripoSG loads the DINOv2-large image encoder, the 21-block rectified-flow DiT,
and the SDF VAE decoder/encoder from the canonical safetensors files.

**Functions:**
- `runtime/triposg.rs:TripoSgPipeline::load()` ✅
- `runtime/triposg.rs:TripoSgModelBundle::load()` — one `VarBuilder`, prefixes `image_encoder` / `transformer` / `vae` ✅
- `runtime/triposg_dino.rs:TripoSgDinoEncoder::load()` — 24 layers, width 1024, patch 14 ✅
- `runtime/triposg_dit.rs:TripoSgDiT::load()` — 21 blocks, width 2048, UViT long skips ✅
- `runtime/triposg_vae.rs:TripoSgVaeDecoder::load()` — 16 self-attention blocks + cross block ✅
- `runtime/triposg_vae.rs:TripoSgVaeEncoder::load()` — cross block + 8 self blocks + `vae.quant` ✅
- `weights/mod.rs` — `triposg` family normalization/validation (9 raw files, MIT) ✅
- `weights/auxiliary.rs` — separate `rmbg-1.4` canonical package (798 tensors, BRIA license) ✅

### US-14.2: TripoSG preprocess matches the reference image loader
As a user, I get the reference alpha-composite branch, the BriaRMBG 1.4 fallback for
RGB/invalid-alpha inputs, 10% square padding, short-edge-256 / center-crop-224 resize, and
ImageNet normalization.

**Functions:**
- `preprocess/triposg.rs:TripoSgPreprocessStage::prepare_inputs_from_path()` ✅
- RGBA alpha validation (20-bin histogram, ≥1% at both ends) + white composite ✅
- `runtime/rmbg.rs:RmbgModel` — IS-Net/RSU BriaRMBG 1.4, sigmoid(d1) mask ✅
- Exact reference mask postprocess: min-max, Otsu, 8-connected label, `remove_small_objects(200)`, largest contour bbox ✅
- `--rmbg-model-path` / `LUX3D_RMBG_MODEL_PATH`; auto-discovery of a sibling `rmbg-1.4` package ✅
- torchvision-v2 antialiased bicubic (bit-exact vs the installed transformers path) ✅

### US-14.3: TripoSG inference and SDF mesh export
As a user, I can denoise with classifier-free guidance and extract an OBJ mesh through the
reference-default flash decoder/DiffDMC (or the dense/hierarchical marching-cubes fallback).

**Functions:**
- `runtime/triposg.rs:TripoSgPipeline::infer_from_path()` / `infer_with_latents()` ✅
- `runtime/triposg.rs:TripoSgPipeline::sample_latents()` — seeded SplitMix64/Box-Muller (not torch-RNG-identical) ✅
- `runtime/triposg_dit.rs:denoise()` — rectified-flow Euler, CFG 7.0, uncond/cond batch order ✅
- `runtime/triposg_dit.rs:RectifiedFlowSchedule::from_timesteps()` — explicit `timesteps` run-option override (sigmas = t/1000 + shift, terminal 0, step count = list length); shipped `RectifiedFlowScheduler` has no upstream `set_timesteps(timesteps=)` and would reject it (documented deviation) ✅
- `runtime/triposg.rs:TripoSgRunOptions` — upstream `pipeline_triposg.py` call options: steps / guidance / seed / num_tokens (2048) / dense 8 / hierarchical 9 / flash depth 9 / `use_flash_decoder=true` / query chunk, plus `timesteps` override, `bounds` override (canonical ±1.005 default, absolute frame preserved in both decode paths; the CLI also accepts the upstream scalar shorthand `b` = symmetric cube `[-b, b]`) and `faces` (`--faces` merge-close + quadric decimation) ✅
- `geometry/mesh_ops.rs:merge_close_vertices()` — pymeshlab `meshing_merge_close_vertices` equivalent (spatial hash, first-index representative, default threshold 1/10000 of bbox diagonal) ✅
- `runtime/triposg.rs:extract_mesh()` — `--faces` postprocess (`merge_close_vertices` + meshopt quadric `simplify_mesh` to the target face count; no-op at/below target or `faces ≤ 0`) ✅
- `runtime/triposg_flash.rs:flash_sdf_grid()` — reference resolution schedule, mini-grid dense pass (FULL dense query, not narrow-band), near-surface mask/dilation/refinement, NaN sentinel ✅
- `runtime/triposg_flash.rs:flash_dual_marching_cubes()` — clean-room DMC (patch averaging, quads, min-max-cosine triangulation, world mapping) ✅
- `runtime/triposg_field.rs:hierarchical_sdf_grid()` — non-flash dense depth 8 + refinement to depth 9 ✅
- `geometry/triposg.rs:TripoSgGeometryStage::assemble_cpu()` — non-flash level 0, position-bit dedup ✅
- `runtime/triposg.rs:TripoSgPipeline::encode_point_cloud*()` — full VAE encode API (FPS + quant distribution) ✅
- `export/triposg.rs:TripoSgExportStage` — ASCII OBJ ✅
- CLI `run triposg` (steps / `timesteps` schedule override / guidance / seed / num-tokens / depths / flash options / `bounds` window override / `faces` face-count target / `dtype` f32|f16|bf16 load dtype); server `triposg` mesh jobs (`triposg.obj`) ✅

**Tests:** `runtime_triposg` (CUDA parity against the 50-step Python golden, dense depth-8 SDF,
8→9 refinement, CPU 5→6 refinement, smoke), `preprocess_triposg` (bit-exact resize +
normalization), `rmbg_triposg` (sigmoid and mask-preprocess parity), `flash_triposg`
(depth-6/7 CPU grid+mesh parity, depth-9 CUDA grid parity with exact NaN mask, depth-9
reference-SDF DMC/final-trimesh parity, synthetic exact DMC cases, trimesh-style compaction),
`vae_encode_triposg` (FPS, quant parameters, distribution/sample), `geometry_triposg`
(golden depth-9 mesh bbox/volume/face count), `pipeline_triposg` (end-to-end seeded +
baseline-latents parity, run-option validation incl. timesteps/bounds/faces, bounds-
override + `--faces` smoke, F16 load/decode smoke), plus `model_spec` / `contracts` /
`weights` cases ✅

The 50-step CFG ODE is numerically chaotic: steps 1 / 10 / 25 match within 1e-2 / 5e-2 / 5e-2,
while final latents are accepted statistically (mean ≤ 2e-2, relative L2 ≤ 1.2e-1, max ≤ 4.0). A
cross-environment reference control drifts at least as much. VAE/grid/mesh parity remains tight
when decoding the baseline latents (SDF probe max 1.78e-3; mesh bbox 1.7e-4, volume 8.3e-5).

**Provenance/licensing:** RMBG/BriaRMBG background removal is implemented and normalized
into the separate `rmbg-1.4` canonical package because its BRIA license differs from
TripoSG's MIT license. Flash decoder/DiffDMC matches upstream's default
(`use_flash_decoder=True`, depth 9); the DMC kernel is a clean-room Rust implementation of
the documented algorithm, while Diso's CC BY-NC source/tables are not copied, so ambiguous
checkerboard cells may resolve differently. The full VAE encoder + `vae.quant` encode API
(FPS, quant parameters, `TripoSgLatentDistribution`) is implemented for parity/tooling even
though the image-to-3D pipeline itself calls only `vae.decode`. The contract has no
exclusions.

---

## 15. TRELLIS.2 Pipeline — Textured-GLB Export

### US-15.1: TRELLIS.2 UV unwrap + PBR bake + textured GLB
As a user, I can run `lux3d run trellis2` with `--texture` on (the default) and get an
upstream-parity textured `.glb` (embedded baked textures, UVs, normals) plus the lossless
PBR voxel sidecar pair.

**Functions:**
- `runtime/trellis2.rs:Trellis2TextureExportOptions` — `enabled` / `texture_size` (default 2048, multiple of 8 in 8..=8192) / `uv_unwrap`; skips cleanly when disabled, unwrap off, or the mesh carries no 6-channel PBR payload ✅
- `runtime/trellis2.rs:build_trellis2_textured_mesh()` — xatlas UV unwrap over the post-processed mesh, area-weighted vertex normals, sparse-trilinear PBR bake at `texture_size` into `Trellis2Stages::textured`; `geometry/uv_atlas.rs:uv_unwrap()` welds duplicate-position vertices (relative epsilon 1e-6 of the bbox diagonal) before charting so `repair_non_manifold_edges` vertex splits do not turn every split sheet into its own chart, while `vmapping` still points at the original input indices ✅
- `export/trellis2.rs:Trellis2ExportStage::write_textured_glb()` — glTF 2.0 binary with POSITION/NORMAL/TEXCOORD_0 + u32 indices, embedded PNG base-color (RGBA) + metallic-roughness textures, reference material (`metallicFactor = 1`, `roughnessFactor = 1`, `alphaMode = "OPAQUE"`, `doubleSided = false` when the mesh was remeshed and `true` otherwise, `postprocess.py:303`), upstream glTF axis conversion `(x, y, z) -> (x, z, -y)` and UV V flip ✅
- `export/trellis2.rs:write_pbr_vox()` / `write_pbr_json()` — `<stem>.pbrvox.bin` + `<stem>.pbr.json` sidecar pair (sparse PBR voxel grid, 6 channels per voxel) ✅
- `cli.rs:RunArgs` `--texture-size` (CLI range 64..=8192; upstream reference allows 1024..4096 step 1024) / `--no-texture-export` / `--no-uv-unwrap`, wired through `trellis2_run_options_from_args` -> `Trellis2RunOptions.texture_export`; `reject_trellis2_only_options` rejects the flags for every other family ✅
- `cli.rs:run_trellis2` export selection — textured atlas + `.glb` output -> `write_textured_glb` (sidecar written next to it, the plain mesh writer does not overwrite it); `.obj`/`.ply`/`.stl` keep the vertex-color/positions-only mesh; geometry-only or disabled runs keep the pre-textured behavior ✅

**Tests:** `export::trellis2` (`textured_cube_runs_atlas_bake_glb_pipeline` full atlas -> bake ->
GLB parse-back incl. embedded PNG decode, `textured_export_skips_cleanly`,
`texture_export_options_validate_size`), CLI `run_args`/`cli` parse + mapping + foreign-family
rejection cases ✅

**Cost/notes:** the xatlas UV unwrap and the PBR bake run on the CPU after the GPU pipeline —
budget extra wall time that scales with `texture_size` squared (~42 s unwrap + ~9 s bake for
an 80k-triangle sphere at 2048 in this tree).
Uncovered atlas texels are filled with a TELEA-approximation inpaint (iterative nearest-valid
onion-peel with `1/distance^2` weighting), an approximation of the reference
`cv2.inpaint(..., INPAINT_TELEA)`; bake statistics (`charts`/`atlases`/`atlas_size`/
`texture_size`, valid/inpainted texels) are printed per run.

---

## 16. TRELLIS.2 Pipeline Variants, Samplers & Latent Export

### US-16.1: Pipeline variant dispatch
As a user, I can select the reference `512`, `1024`, `1024_cascade` or `1536_cascade` pipeline.

**Functions:**
- `contracts/trellis2.rs:Trellis2PipelineType` — all four reference variants ✅
- `runtime/trellis2.rs:Trellis2Pipeline::resolutions()` — `(ss_resolution, decode_resolution)`: `512` -> `(32, 512)`, `1024` -> `(64, 1024)`, `1024_cascade` -> `(32, 1024)`, `1536_cascade` -> `(32, 1536 target)` ✅
- `runtime/trellis2.rs:quantize_hr_coords()` — HR quantization `((coord + 0.5) / 512 * (hr // 16)).int()`, row-major dedup, token-budget loop (`< max_num_tokens` or `hr == 1024`, else `hr -= 128`); both cascades stop at 1024, so the 1536 run decodes at 1536 only when the budget allows it ✅
- `runtime/trellis2.rs:Trellis2SlatConfig::shape_1536()` / `tex_1536()` — elastic DiT configs for the 96-cell HR grid ✅
- `cli.rs:Trellis2PipelineKind` -> `Trellis2PipelineType`; server `jobs.rs` maps all four names ✅

**Tests:** `trellis2_variants` (`resolutions_map_all_four_variants`,
`quantize_1536_target_maps_and_stops_at_1024`,
`run_with_latents_accepts_all_variants_up_to_weight_load`) and the CLI/server parse tests ✅

**Gap:** no real-weight end-to-end `1536_cascade` parity run in this tree — the gated
variant test only resolves options before weight load.

### US-16.2: Per-stage sampler overrides
As a user, I can override steps, guidance, rescale and guidance interval per stage.

**Functions:**
- `runtime/trellis2.rs:Trellis2SamplerOverrides` — `steps`, `guidance_strength`, `guidance_rescale`, `guidance_interval` (`lo,hi`), `rescale_t` ✅
- `runtime/trellis2.rs:resolve_stage_params()` — pipeline-config defaults <- legacy `--steps` base <- per-stage overrides (override `steps` wins) ✅
- Config defaults: SS 12 steps / 7.5 guidance / 0.7 rescale / `[0.6, 1.0]` / `rescale_t` 5.0; shape 12 / 7.5 / 0.5 / `[0.6, 1.0]` / 3.0; texture 12 / 1.0 / 0.0 / `[0.6, 0.9]` / 3.0 ✅
- CLI `--ss-*` / `--shape-*` / `--tex-*` flags (`steps`, `guidance`, `rescale`, `interval`; `rescale_t` is API-only) wired through `trellis2_run_options_from_args` ✅

**Tests:** `trellis2_variants::resolve_stage_params_merge_precedence`, CLI
`run_trellis2_parses_1536_cascade_and_per_stage_flags` ✅

### US-16.3: Batched multi-sample runs
As a user, I can generate N samples with the upstream joint-draw structure.

**Functions:**
- `runtime/trellis2.rs:Trellis2Pipeline::run_many_from_path()` /
  `run_many_with_latents()` — the upstream `run(num_samples)` draw structure
  (`trellis2_image_to_3d.py:208, 253`): ONE joint sparse-structure noise
  tensor `[N, 8, R, R, R]` (`torch.randn(num_samples, ...)`) and ONE joint
  SLAT noise tensor per stage over all samples' concatenated coords
  (`torch.randn(coords.shape[0], in_channels)`), one shared conditioning pass,
  per-sample SLAT sampling / VAE decode / post-process / export. The Lux3D RNG
  is not bit-identical to `torch.randn`, so the draws are
  distribution-equivalent, not value-identical, to the reference ✅
- `Trellis2RunOptions::num_samples`; `num_samples = 0` rejected; single-run entry points
  reject `num_samples != 1` and point at `run_many_*`; `num_samples = 1` keeps the
  unbatched single pass (same call, same noise consumption order) ✅
- `cli.rs:sample_output_path()` — `_0`, `_1`, ... suffixes on both the mesh output and
  `--return-latent` path ✅

**Tests:** `pipeline_trellis2` (`num_samples > 1` rejection and `num_samples = 0`),
`trellis2::tests::batched_ss_noise_is_one_joint_draw`,
`trellis2::tests::batched_slat_noise_is_one_joint_draw`,
`trellis2::tests::single_sample_noise_matches_the_legacy_single_pass`,
CLI option-mapping tests ✅

### US-16.4: Latent NPZ export
As a user, I can export the final SLAT latents for parity/tooling.

**Functions:**
- `runtime/trellis2.rs:Trellis2Stages::write_latent_npz()` — `shape_latent [N,32]` F32,
  `tex_latent [N,32]` F32 (empty `[0,32]` when texture is off), `coords [N,3]` I32,
  `resolution` scalar, cascade `lr_shape_latent [N_lr,32]` (empty `[0,32]` otherwise),
  UTF-8 JSON `meta` (`pipeline_type`, `resolution`, `num_coords`, `cascade`, `hr_resolution`) ✅
- CLI `--return-latent <path>`; per-sample suffix with `--num-samples > 1` ✅

**Tests:** `trellis2_variants::write_latent_npz_round_trip` (cascade-1536 and 512
variants) ✅

### US-16.5: Mesh post-processing
As a user, I get the upstream `to_glb` mesh remesh/cleanup before export.

**Functions:**
- `runtime/trellis2.rs:Trellis2MeshPostprocessOptions` — `enabled` (true),
  `fill_hole_perimeter` (3e-2), `decimation_target` (1,000,000 triangles),
  `min_component_area` (1e-5), `remesh` (true), `remesh_band` (1), `remesh_project` (0) —
  the defaults match the shipped entry points `example.py:43` / `app.py:503` ✅
- `geometry/trellis2.rs:postprocess_decoded_mesh()` — shared `fill_holes` pre-pass then
  one of two branches (`postprocess.py:108-186`): the narrow-band dual-contouring remesh
  (`geometry/remesh_dc.rs:remesh_narrow_band_dc`) on the canonical `[-0.5, 0.5]^3` frame
  with `resolution = mesh.resolution`, `center = 0`, `scale = (resolution + 3 * band) /
  resolution` (`geometry/remesh_dc.rs:narrow_band_grid_scale`, remesh quads split into
  triangles), followed by a single `simplify(decimation_target)` pass; or
  `geometry/mesh_ops.rs:postprocess_mesh()` — simplify -> duplicate-face / non-manifold
  repair -> small-component removal -> fill holes -> simplify -> cleanup -> fill holes ->
  orientation unification -> vertex compaction ✅
- `geometry/mesh_ops.rs:simplify_mesh()` — deterministic meshopt quadric
  simplifier with `Permissive` collapses (required for the non-manifold
  O-Voxel dual-grid topology; without it meshopt stalls above the target);
  `decimation_target = 0` skips simplification ✅
- CLI `--no-postprocess`, `--no-remesh`, `--remesh-band`, `--remesh-project`,
  `--decimation-target`, `--fill-hole-perimeter`, `--min-component-area`; applied BEFORE
  `attach_texture` so PBR sampling sees the final vertices; server
  `MeshGenerationOptions` maps `postprocess`/`decimation_target`/`fill_hole_perimeter`/
  `min_component_area`/`remesh`/`remesh_band`/`remesh_project` ✅
- `export/glb_textured.rs:TexturedGlbMesh::double_sided` — `false` on remeshed exports,
  `true` on the non-remeshed/pass-through paths (`postprocess.py:303`) ✅

**Tests:** `geometry::trellis2` remesh branch tests (watertight topology rebuild,
remesh decimation target, pre-attached-color rejection, byte-identical `--no-remesh`
cleanup path), `mesh_ops` unit suite (26 tests), `pipeline_trellis2`
(`pipeline_golden_mesh_postprocess_options` covers remesh + cleanup + both decimation
paths), CLI/server postprocess mapping tests ✅

**Caveat:** the remesh is a clean-room narrow-band dual-contouring port (BVH closest-point
distance + ray-parity sign + QEF placement), and cleanup/decimation are clean-room ports
driven by the meshopt simplifier, not cumesh; topology changes can differ from the
reference.

---

## 17. TRELLIS.2 Texturing Pipeline (Existing Meshes)

### US-17.1: `--texture-mesh` end-to-end
As a user, I can texture an existing mesh from one condition image and write a textured GLB.

**Functions:**
- `runtime/trellis2_texture.rs:Trellis2TexturingOptions` — `seed` (42), `resolution`
  (512|1024, default 1024), `tex_sampler` overrides, `texture_size` (2048),
  `preprocess_image` / `preprocess_mesh` ✅
- `runtime/trellis2_texture.rs:preprocess_mesh_for_texturing()` — center, scale by
  `0.99999 / max_extent`, y/z swap `(x, y, z) -> (x, -z, y)`; rejects empty, non-finite,
  degenerate and out-of-range meshes ✅
- `runtime/trellis2_texture.rs:Trellis2Pipeline::texture_mesh()` /
  `texture_mesh_from_path()` — preprocess mesh + image, voxelize, encode shape SLAT,
  sample texture SLAT, guided texture VAE decode, UV unwrap + bake ✅
- `geometry/dual_grid.rs:mesh_to_flexible_dual_grid()` — O-Voxel CPU voxelization
  (`MeshToDualGridOptions::for_shape_slat_encoder`: explicit `[-0.5, 0.5]^3` aabb,
  `boundary_weight = 0.2`, `regularization_weight = 1e-2`) ✅
- `runtime/trellis2_shape_enc.rs:Trellis2ShapeEncoder` — `FlexiDualGridVaeEncoder`
  (`shape_enc_next_dc_f16c32_fp16`, 284 tensors; 6-channel input, four S2C downsamples,
  latent on the `resolution / 16` grid); `concat_dual_grid_input()` builds the
  3 offset + 3 intersection channels ✅
- `runtime/trellis2_texture.rs:synthesize_tex_guide_subs()` — rebuilds the upstream
  `SparseSpatial2Channel` cache as occupancy guide logits from the voxelization chain,
  then validates the guided decode reconstructs the voxel set ✅
- `runtime/trellis2_texture.rs:Trellis2Pipeline::load_shape_encoder()` /
  `load_trellis2_shape_encoder_from_model_assets()` — raw stage file,
  `external/shape_enc/model.safetensors` beside a canonical/GGUF package, or
  `shape_enc.*` keys in a merged canonical package ✅
- `runtime/trellis2.rs:build_trellis2_textured_mesh()` — xatlas unwrap + area-weighted
  normals + PBR bake into `Trellis2Stages::textured` ✅
- `export/trellis2.rs:Trellis2ExportStage::write_textured_glb()` — embedded PNGs, UVs,
  normals, glTF axis conversion + V flip ✅

**Tests:** `trellis2_texture` (12 tests: preprocess goldens, option validation, guide
replay, shape-encoder load/missing-checkpoint errors, gated coord chain, env-gated
`texture_mesh_run_e2e`), `trellis2_shape_enc` (5: 284 tensor names, tiny forward,
determinism, Python-oracle golden), `dual_grid` (9 unit tests) ✅

**Gap:** the production-resolution `texture_mesh_run_e2e` is opt-in
(`LUX3D_TRELLIS2_RUN_TEXTURE_E2E=1`, CPU by default) and no GPU end-to-end run is recorded
in this tree.

### US-17.2: Mesh input formats
As a user, I can texture `.glb`, `.obj`, `.ply` or `.stl` inputs.

**Functions:**
- `export/import.rs:load_mesh()` — extension dispatch ✅
- `read_obj()` — ASCII `v`/`f`, fan-triangulation, `v`, `v/vt`, `v//vn`, `v/vt/vn`,
  1-based positive and negative relative indices; `vt`/`vn`/groups/materials ignored ✅
- `read_ply()` — ASCII and binary little-endian; arbitrary vertex property order;
  `vertex_indices`/`vertex_index` face lists, fan-triangulated; big-endian/compressed
  variants rejected ✅
- `read_glb()` — glTF 2.0 JSON+BIN, active scene or all meshes, static node matrix/TRS
  (unit quaternion), u8/u16/u32 indices, f32 positions; Draco/meshopt/sparse accessors
  rejected; skins ignored (bind pose) ✅
- `read_stl()` — ASCII and binary; exact-bit vertex dedup (no epsilon welding) ✅
- All readers reject empty meshes, non-finite coordinates, out-of-range faces, and drop
  index-degenerate triangles without reordering ✅

**Tests:** `export::import` (19 unit tests: OBJ/PLY/GLB/STL round trips and rejections) ✅

### US-17.3: Texturing CLI surface and limits
As a user, I know exactly which flags apply and what the output contains.

**Functions:**
- CLI `--texture-mesh <file>`, `--texture-resolution 512|1024`,
  `--no-preprocess-mesh`, `--no-preprocess-image`, shared `--seed` / `--texture-size` /
  `--tex-*` overrides; `validate_trellis2_texturing_mode()` rejects `.glb`-incompatible
  output extensions and image-to-3D-only flags ✅
- Output: one textured `.glb`; no `<stem>.pbrvox.bin` / `<stem>.pbr.json` sidecar ✅

**Limits/approximations:**
- Existing input UVs are not preserved; the atlas is always re-unwrapped. Duplicate-position
  vertices (including `repair_non_manifold_edges` splits) are welded before charting, UV seams
  duplicate vertices, and xatlas' non-parameterizable (zero-area) faces are dropped from the
  atlas index buffer ✅
- No cleanup, decimation, component removal or remeshing in texturing mode; input topology
  is kept except for the UV split ✅
- The shape encoder runs F32; upstream `use_fp16 = true` casts the block torso. The
  fixture manifest records ~1.3e-2 max mean-latent delta from the F16 reference ✅
- Guide logits are occupancy constants (`+1`/`-1`), not learned `to_subdiv` outputs; the
  decoder's child row order differs from the upstream cache but downstream sampling is
  order-insensitive ✅
- xatlas approximates the cumesh unwrap: no normal-cone threshold or `smooth_strength`
  mapping, `max_iterations = 1`, two-pass packing for an exact `texture_size` atlas;
  extra atlases are merged vertically into one UV frame ✅
- xatlas + bake are CPU-bound: ~42 s unwrap + ~9 s bake for an 80k-triangle sphere at
  2048 (release build, this tree); cost scales with `texture_size^2` ✅
- Uncovered texels use a TELEA approximation (onion-peel, `1/d^2`), not `cv2.inpaint` ✅

**Weight requirement:** the merged canonical/GGUF file has no `shape_enc` stage. The
encoder loads `<model-dir>/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` (raw
checkpoint layout) or `<model-dir>/external/shape_enc/model.safetensors` (sidecar next
to a canonical `model.gguf`); `shape_enc.*` keys in a merged safetensors package are
also honored. A q8\_0 GGUF + external fp16 encoder + `external/dinov3` package completed
the CUDA texturing smoke (`--texture-resolution 512 --texture-size 512`) on the q8 512
shape output (507,540 vertices / 1,000,200 faces): 3,241 charts, 2 atlases, 110,789 valid
texels, 30 MB PBR GLB with base-color and metallic-roughness PNGs. A geometry-only q8 1024
run (`--no-postprocess`, 12,679,988 faces) unwraps to 164 charts / 2 atlases at 512 in
647 s, confirming chart count tracks connected components (plus normal-based splitting),
not raw face count. Geometry-only q8\_0 512 smoke on the same package also passed on
Vulkan and WGPU with `cpu_fallback=0` and valid `[-0.5, 0.5]` GLBs. Release-CLI wall time
for one sampler step (q8\_0, 512, `--no-postprocess`) is 22.0 s (CUDA), 55.9 s (Vulkan)
and 97.6 s (WGPU), down from 97 / 129 / 303 s. The WGPU gain comes from routing ragged
F16 sparse-conv GEMMs to the register-tiled kernel and unrolling the WGPU GEMM register
tiles (their dynamically indexed accumulator arrays were lowered to local memory); the
Vulkan residual is WDDM submission/drain latency, not GEMM math. The isolated SS 3D
decoder is 0.6–1.8 s after the layout/kernel fixes.

**Tests:** CLI `run_args` texture-mesh mapping/rejection cases; `trellis2_texture`
missing-checkpoint test ✅

---

## 18. TripoSG Reference Run Options

### US-18.1: `--faces` post-process
As a user, I can cap the extracted triangle count.

**Functions:**
- `runtime/triposg.rs:extract_mesh()` — runs the postprocess only when
  `faces > 0` and the mesh has more faces than the target; no-op at/above target ✅
- `geometry/mesh_ops.rs:merge_close_vertices()` — pymeshlab
  `meshing_merge_close_vertices` equivalent (spatial hash, first-index representative,
  threshold 1/10000 of the bbox diagonal) ✅
- `geometry/mesh_ops.rs:simplify_mesh()` — meshopt quadric-style simplifier to the target
  face count ✅

**Tests:** `pipeline_triposg` (`--faces` smoke), `runtime_triposg` option validation,
`run_args` (`--faces 0` accepted, `-1` rejected) ✅

**Caveat:** decimation is meshopt, not pymeshlab's quadric edge collapse; face counts are
honored, but vertex placement/topology can differ from the upstream script.

### US-18.2: `--dtype` load precision
As a user, I can load DINOv2/DiT/VAE in F32 or F16.

**Functions:**
- `runtime/triposg.rs:TripoSgPipeline::load_with_dtype()` /
  `load_with_rmbg_and_dtype()` — pipeline-wide load/compute dtype; public outputs are F32 ✅
- CLI `--dtype f32|f16|bf16` -> `DTypeKind` -> `candle_core::DType` ✅

**Tests:** `pipeline_triposg::pipeline_f16_smoke` (F16 load, DINOv2 forward, VAE decode;
public outputs F32), CLI dtype mapping/unknown-value tests ✅

**Caveats:** F16 DiT forward is not exercised by the CPU smoke (load + DINO + VAE decode
only); BF16 is accepted by the CLI but has no runtime test and is unverified.

### US-18.3: `--bounds` and `--timesteps`
As a user, I can override the SDF query window and the denoise schedule.

**Functions:**
- `runtime/triposg.rs:TripoSgRunOptions::effective_bounds()` — canonical
  `[-1.005, 1.005]^3` default; override is `[min_x, min_y, min_z, max_x, max_y, max_z]`,
  finite with `min < max` per axis; both flash and hierarchical decode paths use it and
  keep the absolute world frame ✅
- `runtime/triposg_dit.rs:RectifiedFlowSchedule::from_timesteps()` — sigmas =
  `t / 1000` with the shift, terminal 0, step count = list length; `--steps` ignored when
  set ✅
- Validation: non-empty finite timesteps; no monotonicity requirement (upstream behavior) ✅

**Tests:** `runtime_triposg` (`rectified_flow_from_timesteps_*` 3 tests),
`pipeline_triposg` (far-window bounds in both decode paths, explicit canonical-bounds
equality), `flash_triposg::flash_dmc_honors_bounds_override`, `run_args` parse/validation ✅

---

## 19. Working-Tree Test Inventory

Counts are `#[test]` declarations; gated suites skip without their weights/fixtures.

| Suite / file | Tests | Scope |
|---|---:|---|
| `crates/lux3d-core/tests/trellis2_texture.rs` | 12 | mesh preprocess goldens, option validation, guide synthesis, shape-encoder load/missing errors, gated coord chain, env-gated e2e |
| `crates/lux3d-core/tests/trellis2_variants.rs` | 5 | four-variant dispatch, sampler precedence, 1536 token budget, latent npz, gated variant smoke |
| `crates/lux3d-core/tests/trellis2_shape_enc.rs` | 5 | 284-tensor name coverage, tiny forward/determinism, Python-oracle golden |
| `crates/lux3d-core/tests/vggt_track.rs` | 5 | track tensor-name coverage, bilinear sampler, CPU structural forward, query validation |
| `crates/lux3d-core/src/export/colmap.rs` (unit) | 2 | COLMAP binary write + parse-back, confidence gate + point cap |
| `crates/lux3d-core/src/geometry/mesh_ops.rs` (unit) | 26 | hole fill, simplify, cleanup, orientation, merge-close, postprocess order |
| `crates/lux3d-core/src/geometry/dual_grid.rs` (unit) | 9 | voxelization invariants, encoder frame, roundtrip extraction, malformed input |
| `crates/lux3d-core/src/geometry/uv_atlas.rs` (unit) | 7 | atlas validity/determinism/seam map, duplicate-position weld (soup + vmapping), area-weighted normals |
| `crates/lux3d-core/src/geometry/pbr_bake.rs` (unit) | 8 | constant/gradient bake, MR packing, inpaint, boundary clamp, determinism |
| `crates/lux3d-core/src/export/glb_textured.rs` (unit) | 3 | embedded PNG/material, alpha modes, axis/V conversion, rejections |
| `crates/lux3d-core/src/export/import.rs` (unit) | 19 | OBJ/PLY/GLB/STL readers, transforms, dispatch, malformed input |
| `crates/lux3d-core/src/export/trellis2.rs` (unit) | 3 | textured cube atlas -> bake -> GLB parse-back, clean skips, size validation |
| `crates/lux3d-core/tests/trellis2_export.rs` | 9 | PBR sidecars, vertex colors, glTF axis conversion |
| `crates/lux3d-core/tests/pipeline_trellis2.rs` | 8 | golden mesh postprocess, num-samples validation, run options |
| `crates/lux3d-core/tests/trellis2_cascade.rs` | 12 | cascade parity/gating |
| `crates/lux3d-core/tests/runtime_triposg.rs` | 14 | TripoSG golden/runtime; 3 new timestep-schedule tests |
| `crates/lux3d-core/tests/pipeline_triposg.rs` | 5 | seeded/golden runs, bounds/faces smoke, F16 smoke |
| `crates/lux3d-core/tests/flash_triposg.rs` | 12 | flash SDF/DMC parity; bounds-override test |
| `crates/lux3d-core/tests/vggt_preprocess.rs` | 8 | crop/pad parity, alpha/rounding, single-image source |
| `crates/lux3d-core/tests/runtime_vggt.rs` | 4 | F32 parity; dtype/default selection, batch-size 0 rejection |
| `crates/lux3d-cli/tests/run_args.rs` | 22 | CLI parse/mapping/rejection for TRELLIS.2, VGGT, TripoSG flags |

Also touched: `crates/lux3d-core/tests/vggt_export.rs` (fixture gains empty `pose_enc_list`),
`crates/lux3d-core/tests/trellis2_cascade.rs`,
`crates/lux3d-core/tests/pipeline_trellis2.rs`,
`crates/lux3d-core/tests/trellis2_export.rs`,
`crates/lux3d-core/tests/runtime_triposg.rs`,
`crates/lux3d-core/tests/pipeline_triposg.rs`,
`crates/lux3d-core/tests/flash_triposg.rs`,
`crates/lux3d-core/tests/vggt_preprocess.rs`.

---

## 20. Known Gaps, Exclusions & Approximations

- **Metal:** CI builds `lux3d-core` with `--features metal` and runs the lib/quant-probe
  checks; no model inference parity is validated in this repository.
- **Laya:** not used by this workspace.
- **TRELLIS.2 texturing weights:** the merged canonical/GGUF weight file lacks
  `shape_enc`; `--texture-mesh` needs the raw stage file or an
  `external/shape_enc/model.safetensors` sidecar (plus `external/dinov3` for the
  quantized canonical-GGUF layout). A package with neither fails with a clear error.
- **TRELLIS.2 `1536_cascade`:** variant dispatch and token-budget math are tested; no
  real-weight end-to-end run is recorded.
- **TRELLIS.2 remesh branch:** implemented as a clean-room narrow-band dual-contouring
  port (`geometry/remesh_dc.rs`) and enabled by default, matching the shipped entry
  points; `--no-remesh` / `remesh = false` restores the previous cleanup chain. The
  default decodes on fresh topology, so mesh vertex/face counts differ from the
  pre-remesh pipeline by design.
- **TRELLIS.2 approximations:** narrow-band DC remesh vs cumesh; meshopt
  simplification/cleanup vs cumesh; xatlas vs cumesh unwrap; TELEA-approximation inpaint
  vs `cv2.inpaint`; synthesized occupancy guides vs upstream sparse caches.
- **TRELLIS.2 backend throughput:** one sampler step on q8\_0 (512, geometry only) takes
  22.0 s on CUDA, 55.9 s on Vulkan and 97.6 s on WGPU. Vulkan's residual gap is WDDM
  submission/fence latency: the shape VAE and DiTs issue tens of thousands of small
  dispatches and the backend spends ~32 s in blocking drains while GPU kernels sum to a
  few seconds. Candle now counts only reclaimable retained buffers per submission,
  retires the oldest submission incrementally when the allocator is under pressure, and
  keeps the measured-optimal 512 MiB in-flight budget; larger budgets, larger dispatch
  batches, a buffer pool and disabled drains all regressed.
  WGPU no longer runs the sparse VAE through the naive one-thread-per-output GEMM:
  F16 matmuls with ragged M/N/K use the register-tiled kernel, whose accumulators (and
  the F32 warptile accumulators) are now named scalars instead of dynamically indexed
  arrays that Naga placed in local memory. Together with the 256 MiB large-allocation
  flush threshold and deduplicated retained-byte accounting this cut the 1-step run from
  348 s to 98 s. Remaining WGPU cost is split between the WDDM per-pass latency (~7 ms
  per compute pass) and the exact-F32 DiT path: CUDA runs q8 GEMMs on int8 kernels while
  WGPU dequantizes packed weights once at load and multiplies dense F32 (stages are
  loaded/dropped sequentially; `LUX3D_WGPU_DEQUANT_WEIGHTS=0` restores packed weights,
  including the K-quant Q8K dispatch limit). Reduced-precision coop-matrix is not used
  for TRELLIS.2 because its O-Voxel occupancy is discrete and f16/TF32 drift changes the
  exported topology.
- **VGGT tracking:** explicit query points only. Automatic query selection (ALIKED/SP
  keypoints, DINO query-frame ranking), multi-query-frame reordering and non-visible track
  completion are not implemented.
- **VGGT contract drift:** the static contract has no `track_head` entry (the
  `NeuralSpec::track_head` field deserializes to `None`) and still lists the
  `vggt_track_head` exclusion while `runtime/vggt_track.rs` implements the head;
  `tests/contracts.rs`/`tests/model_spec.rs` assert the old metadata and need
  regeneration/update.
- **VGGT COLMAP export:** 3D points come from the point-head `world_points` (upstream
  `demo_colmap.py` unprojects depth), and point limiting is a deterministic evenly-spaced
  subsample instead of upstream's seeded random choice.
- **VGGT dtypes:** F32 is the parity mode. F16/BF16 are wired but the automated numeric
  suite is F32/CUDA-only; BF16 is unverified.
- **TripoSG scribble:** not implemented (weights unavailable).
- **TripoSG `--faces`:** meshopt quadric simplification approximates pymeshlab quadric
  edge collapse.
- **Server surfaces:** the HTTP mesh API keeps one sample, default postprocess/texture
  export, and has no `--texture-mesh`/per-stage/`--return-latent` knobs; TRELLIS.2 jobs
  write the vertex-color mesh + PBR sidecars, not an embedded-texture GLB.
- **Quantization:** TRELLIS.2 DiT-only GGUF policy is unchanged; the texturing-only
  `shape_enc` checkpoint is outside the canonical package and is never quantized.

---

## Summary Statistics

| Category | Functions | Test Coverage |
|----------|-----------|---------------|
| CLI Dispatch | 14 | Partial |
| Pi3 Pipeline | 28 | Minimal |
| Pi3X Pipeline | 24 | Minimal |
| TripoSR Pipeline | 22 | Minimal |
| VGGT Pipeline | 18 | CUDA parity |
| TripoSG Pipeline | 23 | CUDA golden + statistical |
| Attention Math | 10 | None |
| Resampling | 4 | None |
| Camera Ops | 4 | None |
| NN Blocks | 6 | None |
| Export | 4 | Via CLI |
| Geometry | 7 | Partial |
| Neural Stages | 6 | None |
| Weights | 8 | None |
| Contracts | 20 | Yes |
| Preprocess | 10 | Minimal |
| Path Utils | 3 | Yes |
| Error Types | 2 | N/A |
| Vision Preproc | 1 | None |
| **Total** | **~206** | **~10%** |

*This table predates sections 15-20. The working-tree additions and their suites are
inventoried in section 19; the older counts above are not recomputed.*

