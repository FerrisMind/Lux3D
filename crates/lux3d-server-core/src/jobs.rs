//! Async generation job store and inference execution.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, ensure, Context, Result};
use lux3d_core::export::{load_mesh, MeshFormat, PointCloudFormat};
use lux3d_core::runtime::{
    Pi3xInjectConditions, Trellis2RunOptions, Trellis2SamplerOverrides, Trellis2TexturingOptions,
    TripoSgRunOptions,
};
use lux3d_core::{export::pbrvox, Trellis2PipelineType};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    api_types::{
        GenerationError, GenerationObject, GenerationStatus, MeshGenerationOptions,
        MeshTexturingOptions, PointCloudGenerationOptions,
    },
    metrics::{record_generation_finished, record_generation_started},
    types::Lux3dState,
    upload::{normalize_source_for_frames, normalize_source_for_image},
    webhooks::deliver_generation_webhook,
};

#[derive(Debug, Clone)]
pub struct JobRecord {
    pub object: GenerationObject,
    pub output_path: Option<PathBuf>,
    pub content_type: String,
    pub download_filename: String,
    pub workspace: PathBuf,
    pub webhook_url: Option<String>,
    cancel_token: CancellationToken,
}

#[derive(Debug, Clone)]
pub struct JobStore {
    inner: Arc<RwLock<HashMap<String, JobRecord>>>,
}

impl Default for JobStore {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl JobStore {
    pub fn insert(&self, record: JobRecord) {
        if let Ok(mut jobs) = self.inner.write() {
            jobs.insert(record.object.id.clone(), record);
        }
    }

    pub fn get(&self, id: &str) -> Option<JobRecord> {
        self.inner
            .read()
            .ok()
            .and_then(|jobs| jobs.get(id).cloned())
    }

    pub fn list(&self, limit: usize) -> Vec<GenerationObject> {
        let Ok(jobs) = self.inner.read() else {
            return Vec::new();
        };
        let mut entries: Vec<_> = jobs.values().map(|record| record.object.clone()).collect();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.created_at));
        entries.truncate(limit);
        entries
    }

    pub fn active_count(&self) -> usize {
        self.inner
            .read()
            .ok()
            .map(|jobs| {
                jobs.values()
                    .filter(|record| {
                        matches!(record.object.status.as_str(), "queued" | "in_progress")
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    fn update<F>(&self, id: &str, update: F) -> Result<()>
    where
        F: FnOnce(&mut JobRecord),
    {
        let mut jobs = self
            .inner
            .write()
            .map_err(|_| anyhow::anyhow!("job store lock poisoned"))?;
        let record = jobs
            .get_mut(id)
            .with_context(|| format!("generation `{id}` not found"))?;
        update(record);
        Ok(())
    }

    pub fn mark_in_progress(&self, id: &str) -> Result<()> {
        self.update(id, |record| {
            record.object.status = GenerationStatus::InProgress.as_str().to_string();
            record.object.progress = 10;
        })
    }

    pub fn mark_completed(&self, id: &str, output_path: PathBuf) -> Result<()> {
        self.update(id, |record| {
            record.object.status = GenerationStatus::Completed.as_str().to_string();
            record.object.progress = 100;
            record.object.completed_at = Some(now_unix());
            record.object.error = None;
            record.output_path = Some(output_path);
        })
    }

    pub fn mark_failed(&self, id: &str, message: String) -> Result<()> {
        self.update(id, |record| {
            record.object.status = GenerationStatus::Failed.as_str().to_string();
            record.object.completed_at = Some(now_unix());
            record.object.error = Some(GenerationError {
                code: "generation_failed".to_string(),
                message,
            });
        })
    }

    pub fn mark_cancelled(&self, id: &str) -> Result<()> {
        self.update(id, |record| {
            record.object.status = GenerationStatus::Cancelled.as_str().to_string();
            record.object.completed_at = Some(now_unix());
            record.object.error = Some(GenerationError {
                code: "cancelled".to_string(),
                message: "generation was cancelled".to_string(),
            });
        })
    }

    pub fn request_cancel(&self, id: &str) -> Result<bool> {
        let jobs = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("job store lock poisoned"))?;
        let Some(record) = jobs.get(id) else {
            return Ok(false);
        };
        if matches!(
            record.object.status.as_str(),
            "completed" | "failed" | "cancelled"
        ) {
            return Ok(false);
        }
        record.cancel_token.cancel();
        Ok(true)
    }

    pub fn is_cancelled(&self, id: &str) -> bool {
        self.inner
            .read()
            .ok()
            .and_then(|jobs| {
                jobs.get(id)
                    .map(|record| record.cancel_token.is_cancelled())
            })
            .unwrap_or(false)
    }

    pub fn remove(&self, id: &str) -> Option<JobRecord> {
        let mut jobs = self.inner.write().ok()?;
        jobs.remove(id)
    }

    pub fn cleanup_expired(&self, now: u64) -> usize {
        let expired: Vec<String> = self
            .inner
            .read()
            .ok()
            .map(|jobs| {
                jobs.iter()
                    .filter_map(|(id, record)| {
                        record
                            .object
                            .expires_at
                            .filter(|expires_at| *expires_at <= now)
                            .map(|_| id.clone())
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut removed = 0;
        for id in expired {
            if let Some(record) = self.remove(&id) {
                let _ = fs::remove_dir_all(&record.workspace);
                removed += 1;
            }
        }
        removed
    }
}

pub fn new_job_id() -> String {
    format!("gen_{}", Uuid::new_v4().simple())
}

pub fn job_workspace(upload_dir: &Path, job_id: &str) -> PathBuf {
    upload_dir.join("jobs").join(job_id)
}

pub struct NewJobRecord {
    pub job_id: String,
    pub model: String,
    pub output_format: String,
    pub workspace: PathBuf,
    pub content_type: String,
    pub download_filename: String,
    pub ttl_secs: u64,
    pub webhook_url: Option<String>,
}

pub fn create_job_record(params: NewJobRecord) -> JobRecord {
    let created_at = now_unix();
    JobRecord {
        object: GenerationObject {
            id: params.job_id,
            object: "generation",
            status: GenerationStatus::Queued.as_str().to_string(),
            model: params.model,
            created_at,
            completed_at: None,
            expires_at: Some(created_at.saturating_add(params.ttl_secs)),
            progress: 0,
            output_format: params.output_format,
            error: None,
        },
        output_path: None,
        content_type: params.content_type,
        download_filename: params.download_filename,
        workspace: params.workspace,
        webhook_url: params.webhook_url,
        cancel_token: CancellationToken::new(),
    }
}

pub async fn spawn_point_cloud_generation(
    state: Arc<Lux3dState>,
    record: JobRecord,
    source: PathBuf,
    conditions: Option<PathBuf>,
    options: PointCloudGenerationOptions,
) -> Result<GenerationObject> {
    let model = record.object.model.clone();
    let format = PointCloudFormat::from_extension(&record.object.output_format)
        .map_err(anyhow::Error::from)?;
    ensure!(
        matches!(model.as_str(), "pi3" | "pi3x" | "vggt"),
        "unsupported point-cloud model `{model}`; expected \"pi3\", \"pi3x\" or \"vggt\""
    );
    if model == "vggt" {
        ensure!(
            conditions.is_none(),
            "`conditions` is not supported for `vggt`"
        );
        ensure!(
            !options.vo.unwrap_or(false),
            "`vo` is not supported for `vggt`"
        );
        ensure!(
            options.chunk_size.is_none(),
            "`chunk_size` is not supported for `vggt`"
        );
        ensure!(
            options.overlap.is_none(),
            "`overlap` is not supported for `vggt`"
        );
        ensure!(
            options.inject_condition.is_empty(),
            "`inject_condition` is not supported for `vggt`"
        );
    }

    let job_id = record.object.id.clone();
    state.jobs.insert(record);
    let response = state
        .jobs
        .get(&job_id)
        .expect("job just inserted")
        .object
        .clone();
    record_generation_started(&model);

    let state_for_task = state.clone();
    tokio::spawn(async move {
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", 0.0);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        if let Err(error) = state_for_task.jobs.mark_in_progress(&job_id) {
            tracing::error!("failed to mark job in progress: {error}");
            return;
        }

        let started = Instant::now();
        let cache_model = if model == "pi3" {
            "pi3"
        } else if model == "vggt" {
            "vggt"
        } else if options.vo.unwrap_or(false) {
            "pi3x_vo"
        } else {
            "pi3x"
        };
        let job_id_for_blocking = job_id.clone();

        let result = state_for_task
            .run_blocking({
                let workspace = state_for_task
                    .jobs
                    .get(&job_id)
                    .map(|record| record.workspace.clone());
                let source = source.clone();
                let options = options.clone();
                let model = model.clone();
                move |runtime| {
                    if runtime.jobs.is_cancelled(&job_id_for_blocking) {
                        bail!("generation was cancelled");
                    }
                    runtime
                        .models
                        .ensure_loaded(runtime, cache_model)
                        .with_context(|| format!("failed to load `{cache_model}` pipeline"))?;
                    let workspace =
                        workspace.ok_or_else(|| anyhow::anyhow!("job workspace missing"))?;
                    let source = normalize_source_for_frames(&source, &workspace)?;
                    let device = runtime.device()?;
                    let output_path = workspace.join(format!("{model}.{}", format.extension()));
                    match model.as_str() {
                        "pi3" => runtime.models.with_pi3(|pipeline| {
                            let inference = pipeline
                                .infer_from_path_with_interval(&source, options.interval, &device)
                                .map_err(anyhow::Error::from)?;
                            pipeline
                                .export_point_cloud(&inference, &output_path)
                                .map_err(anyhow::Error::from)?;
                            Ok(())
                        })?,
                        "pi3x" if options.vo.unwrap_or(false) => {
                            let inject_conditions =
                                parse_inject_conditions(&options.inject_condition);
                            runtime.models.with_pi3x_vo(|pipeline| {
                                let inference = pipeline
                                    .infer_from_path(
                                        &source,
                                        options.interval,
                                        options.chunk_size,
                                        options.overlap,
                                        options.conf_threshold,
                                        inject_conditions,
                                        &device,
                                    )
                                    .map_err(anyhow::Error::from)?;
                                pipeline
                                    .export_point_cloud(&inference, &output_path)
                                    .map_err(anyhow::Error::from)?;
                                Ok(())
                            })?
                        }
                        "pi3x" => runtime.models.with_pi3x(|pipeline| {
                            let inference = pipeline
                                .infer_from_path(
                                    &source,
                                    conditions.as_deref(),
                                    options.interval,
                                    &device,
                                )
                                .map_err(anyhow::Error::from)?;
                            pipeline
                                .export_point_cloud(&inference, &output_path)
                                .map_err(anyhow::Error::from)?;
                            Ok(())
                        })?,
                        "vggt" => runtime.models.with_vggt(|pipeline| {
                            let inference = pipeline
                                .infer_from_path(&source, options.interval, &device)
                                .map_err(anyhow::Error::from)?;
                            pipeline
                                .export_point_cloud_with_threshold(
                                    &inference,
                                    &output_path,
                                    options.conf_threshold.unwrap_or(1.0),
                                )
                                .map_err(anyhow::Error::from)?;
                            Ok(())
                        })?,
                        _ => bail!("unsupported point-cloud model `{model}`"),
                    }
                    Ok(output_path)
                }
            })
            .await;

        let elapsed = started.elapsed().as_secs_f64();
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", elapsed);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        match result {
            Ok(output_path) => {
                if let Err(error) = state_for_task.jobs.mark_completed(&job_id, output_path) {
                    tracing::error!("failed to mark job completed: {error}");
                } else {
                    record_generation_finished(&model, "completed", elapsed);
                    maybe_deliver_webhook(&state_for_task, &job_id, "generation.completed").await;
                }
            }
            Err(error) => {
                let status = if error.to_string().contains("cancelled") {
                    let _ = state_for_task.jobs.mark_cancelled(&job_id);
                    "cancelled"
                } else if let Err(update_error) =
                    state_for_task.jobs.mark_failed(&job_id, error.to_string())
                {
                    tracing::error!("failed to mark job failed: {update_error}");
                    "failed"
                } else {
                    "failed"
                };
                record_generation_finished(&model, status, elapsed);
                maybe_deliver_webhook(&state_for_task, &job_id, "generation.failed").await;
            }
        }
    });

    Ok(response)
}

/// Build the TripoSG run options from the reference defaults plus explicit
/// mesh-request overrides (`query_chunk_size` defaults to 16384). Pure helper
/// so the mapping and validation can be unit-tested without a server state.
fn triposg_options_from_mesh_options(options: &MeshGenerationOptions) -> Result<TripoSgRunOptions> {
    let defaults = TripoSgRunOptions::default();
    let run_options = TripoSgRunOptions {
        steps: options.steps.unwrap_or(defaults.steps),
        guidance_scale: options.guidance_scale.unwrap_or(defaults.guidance_scale),
        seed: options.seed.unwrap_or(defaults.seed),
        num_tokens: options.num_tokens.unwrap_or(defaults.num_tokens),
        dense_octree_depth: options
            .dense_octree_depth
            .unwrap_or(defaults.dense_octree_depth),
        hierarchical_octree_depth: options
            .hierarchical_octree_depth
            .unwrap_or(defaults.hierarchical_octree_depth),
        use_flash_decoder: options
            .use_flash_decoder
            .unwrap_or(defaults.use_flash_decoder),
        flash_octree_depth: options
            .flash_octree_depth
            .unwrap_or(defaults.flash_octree_depth),
        query_chunk_size: options
            .query_chunk_size
            .unwrap_or(defaults.query_chunk_size),
        timesteps: options.timesteps.clone(),
        bounds: options.bounds,
        faces: options.faces,
    };
    run_options.validate().map_err(anyhow::Error::from)?;
    Ok(run_options)
}

/// Per-stage sampler overrides from the mesh-request fields. All-`None`
/// yields `None` so the pipeline-config defaults stay in charge; the legacy
/// `steps` field is handled by the caller through the `*_steps` base counts.
fn sampler_overrides(
    steps: Option<usize>,
    guidance_strength: Option<f32>,
    guidance_rescale: Option<f32>,
    guidance_interval: Option<[f32; 2]>,
    rescale_t: Option<f32>,
) -> Option<Trellis2SamplerOverrides> {
    if steps.is_none()
        && guidance_strength.is_none()
        && guidance_rescale.is_none()
        && guidance_interval.is_none()
        && rescale_t.is_none()
    {
        return None;
    }
    Some(Trellis2SamplerOverrides {
        steps,
        guidance_strength,
        guidance_rescale,
        guidance_interval: guidance_interval.map(|[low, high]| (low, high)),
        rescale_t,
    })
}

/// Build the TRELLIS.2 run options from the reference defaults plus explicit
/// mesh-request overrides. Pure helper so the mapping and validation can be
/// unit-tested without a server state.
fn trellis2_options_from_mesh_options(
    options: &MeshGenerationOptions,
) -> Result<Trellis2RunOptions> {
    let defaults = Trellis2RunOptions::default();
    let pipeline_type = match options.pipeline_type.as_deref() {
        None => defaults.pipeline_type,
        Some("512") => Trellis2PipelineType::R512,
        Some("1024") => Trellis2PipelineType::R1024,
        Some("1024_cascade") => Trellis2PipelineType::Cascade1024,
        Some("1536_cascade") => Trellis2PipelineType::Cascade1536,
        Some(other) => bail!(
            "unknown trellis2 `pipeline_type` `{other}`; expected `512`, `1024`, \
             `1024_cascade` or `1536_cascade`"
        ),
    };
    let steps = options.steps.unwrap_or(defaults.ss_steps);

    let mut postprocess = defaults.postprocess.clone();
    if let Some(enabled) = options.postprocess {
        postprocess.enabled = enabled;
    }
    if let Some(value) = options.decimation_target {
        postprocess.decimation_target = value;
    }
    if let Some(value) = options.fill_hole_perimeter {
        postprocess.fill_hole_perimeter = value;
    }
    if let Some(value) = options.min_component_area {
        postprocess.min_component_area = value;
    }
    if let Some(remesh) = options.remesh {
        postprocess.remesh = remesh;
    }
    if let Some(value) = options.remesh_band {
        postprocess.remesh_band = value;
    }
    if let Some(value) = options.remesh_project {
        postprocess.remesh_project = value;
    }
    postprocess.validate().map_err(anyhow::Error::from)?;

    let mut texture_export = defaults.texture_export.clone();
    if let Some(value) = options.texture_size {
        texture_export.texture_size = value;
    }
    texture_export.validate().map_err(anyhow::Error::from)?;

    let num_samples = options.num_samples.unwrap_or(defaults.num_samples);
    ensure!(
        num_samples >= 1,
        "Trellis2 `num_samples` must be at least 1"
    );

    Ok(Trellis2RunOptions {
        seed: options.seed.unwrap_or(defaults.seed),
        pipeline_type,
        ss_steps: steps,
        shape_steps: options.steps.unwrap_or(defaults.shape_steps),
        tex_steps: options.steps.unwrap_or(defaults.tex_steps),
        ss_sampler: sampler_overrides(
            options.ss_steps,
            options.ss_guidance_strength,
            options.ss_guidance_rescale,
            options.ss_guidance_interval,
            options.ss_rescale_t,
        ),
        shape_sampler: sampler_overrides(
            options.shape_steps,
            options.shape_guidance_strength,
            options.shape_guidance_rescale,
            options.shape_guidance_interval,
            options.shape_rescale_t,
        ),
        tex_sampler: sampler_overrides(
            options.tex_steps,
            options.tex_guidance_strength,
            options.tex_guidance_rescale,
            options.tex_guidance_interval,
            options.tex_rescale_t,
        ),
        with_texture: options.texture.unwrap_or(defaults.with_texture),
        // CLI-only switch (TR6): the server keeps the reference default.
        low_vram: defaults.low_vram,
        max_num_tokens: options.max_num_tokens.unwrap_or(defaults.max_num_tokens),
        preprocess: defaults.preprocess,
        num_samples,
        postprocess,
        texture_export,
    })
}

/// Pure mesh-request guard shared by [`spawn_mesh_generation`]: the model
/// allow-list plus per-model option validation, unit-tested without a server
/// state.
fn validate_mesh_generation_request(model: &str, options: &MeshGenerationOptions) -> Result<()> {
    ensure!(
        matches!(model, "triposr" | "triposg" | "trellis2"),
        "unsupported mesh model `{model}`; expected `triposr`, `triposg` or `trellis2`"
    );
    if model == "triposr" {
        ensure!(
            options.mc_resolution >= 2,
            "mc_resolution must be at least 2"
        );
        for (name, present) in [
            ("steps", options.steps.is_some()),
            ("guidance_scale", options.guidance_scale.is_some()),
            ("seed", options.seed.is_some()),
            ("num_tokens", options.num_tokens.is_some()),
            ("dense_octree_depth", options.dense_octree_depth.is_some()),
            (
                "hierarchical_octree_depth",
                options.hierarchical_octree_depth.is_some(),
            ),
            ("use_flash_decoder", options.use_flash_decoder.is_some()),
            ("flash_octree_depth", options.flash_octree_depth.is_some()),
            ("query_chunk_size", options.query_chunk_size.is_some()),
            ("timesteps", options.timesteps.is_some()),
            ("bounds", options.bounds.is_some()),
            ("faces", options.faces.is_some()),
            ("pipeline_type", options.pipeline_type.is_some()),
            ("max_num_tokens", options.max_num_tokens.is_some()),
            ("texture", options.texture.is_some()),
            ("ss_steps", options.ss_steps.is_some()),
            (
                "ss_guidance_strength",
                options.ss_guidance_strength.is_some(),
            ),
            ("ss_guidance_rescale", options.ss_guidance_rescale.is_some()),
            (
                "ss_guidance_interval",
                options.ss_guidance_interval.is_some(),
            ),
            ("ss_rescale_t", options.ss_rescale_t.is_some()),
            ("shape_steps", options.shape_steps.is_some()),
            (
                "shape_guidance_strength",
                options.shape_guidance_strength.is_some(),
            ),
            (
                "shape_guidance_rescale",
                options.shape_guidance_rescale.is_some(),
            ),
            (
                "shape_guidance_interval",
                options.shape_guidance_interval.is_some(),
            ),
            ("shape_rescale_t", options.shape_rescale_t.is_some()),
            ("tex_steps", options.tex_steps.is_some()),
            (
                "tex_guidance_strength",
                options.tex_guidance_strength.is_some(),
            ),
            (
                "tex_guidance_rescale",
                options.tex_guidance_rescale.is_some(),
            ),
            (
                "tex_guidance_interval",
                options.tex_guidance_interval.is_some(),
            ),
            ("tex_rescale_t", options.tex_rescale_t.is_some()),
            ("num_samples", options.num_samples.is_some()),
            ("texture_size", options.texture_size.is_some()),
            ("postprocess", options.postprocess.is_some()),
            ("decimation_target", options.decimation_target.is_some()),
            ("fill_hole_perimeter", options.fill_hole_perimeter.is_some()),
            ("min_component_area", options.min_component_area.is_some()),
        ] {
            ensure!(!present, "`{name}` is not supported for `triposr`");
        }
        return Ok(());
    }
    if model == "trellis2" {
        // TRELLIS.2 owns the voxel grid (512 or 1024_cascade pipeline types),
        // so the TripoSR marching-cubes knobs and the TripoSG octree/flash
        // knobs must stay at their defaults here.
        ensure!(
            options.mc_resolution == 256,
            "`mc_resolution` is not supported for `trellis2`; only the default 256 is supported"
        );
        ensure!(
            options.mc_threshold == 25.0,
            "`mc_threshold` is not supported for `trellis2`; only the default 25.0 is supported"
        );
        for (name, present) in [
            ("guidance_scale", options.guidance_scale.is_some()),
            ("num_tokens", options.num_tokens.is_some()),
            ("dense_octree_depth", options.dense_octree_depth.is_some()),
            (
                "hierarchical_octree_depth",
                options.hierarchical_octree_depth.is_some(),
            ),
            ("use_flash_decoder", options.use_flash_decoder.is_some()),
            ("flash_octree_depth", options.flash_octree_depth.is_some()),
            ("query_chunk_size", options.query_chunk_size.is_some()),
            ("timesteps", options.timesteps.is_some()),
            ("bounds", options.bounds.is_some()),
            ("faces", options.faces.is_some()),
        ] {
            ensure!(!present, "`{name}` is not supported for `trellis2`");
        }
        return trellis2_options_from_mesh_options(options).map(|_| ());
    }
    // TripoSG owns the marching-cubes grid through its own octree-depth
    // options, so the TripoSR knobs must stay at their defaults here.
    ensure!(
        options.mc_resolution == 256,
        "`mc_resolution` is not supported for `triposg`; only the default 256 is supported"
    );
    ensure!(
        options.mc_threshold == 25.0,
        "`mc_threshold` is not supported for `triposg`; only the default 25.0 is supported"
    );
    for (name, present) in [
        ("pipeline_type", options.pipeline_type.is_some()),
        ("max_num_tokens", options.max_num_tokens.is_some()),
        ("texture", options.texture.is_some()),
        ("ss_steps", options.ss_steps.is_some()),
        (
            "ss_guidance_strength",
            options.ss_guidance_strength.is_some(),
        ),
        ("ss_guidance_rescale", options.ss_guidance_rescale.is_some()),
        (
            "ss_guidance_interval",
            options.ss_guidance_interval.is_some(),
        ),
        ("ss_rescale_t", options.ss_rescale_t.is_some()),
        ("shape_steps", options.shape_steps.is_some()),
        (
            "shape_guidance_strength",
            options.shape_guidance_strength.is_some(),
        ),
        (
            "shape_guidance_rescale",
            options.shape_guidance_rescale.is_some(),
        ),
        (
            "shape_guidance_interval",
            options.shape_guidance_interval.is_some(),
        ),
        ("shape_rescale_t", options.shape_rescale_t.is_some()),
        ("tex_steps", options.tex_steps.is_some()),
        (
            "tex_guidance_strength",
            options.tex_guidance_strength.is_some(),
        ),
        (
            "tex_guidance_rescale",
            options.tex_guidance_rescale.is_some(),
        ),
        (
            "tex_guidance_interval",
            options.tex_guidance_interval.is_some(),
        ),
        ("tex_rescale_t", options.tex_rescale_t.is_some()),
        ("num_samples", options.num_samples.is_some()),
        ("texture_size", options.texture_size.is_some()),
        ("postprocess", options.postprocess.is_some()),
        ("decimation_target", options.decimation_target.is_some()),
        ("fill_hole_perimeter", options.fill_hole_perimeter.is_some()),
        ("min_component_area", options.min_component_area.is_some()),
        ("remesh", options.remesh.is_some()),
        ("remesh_band", options.remesh_band.is_some()),
        ("remesh_project", options.remesh_project.is_some()),
    ] {
        ensure!(!present, "`{name}` is not supported for `triposg`");
    }
    triposg_options_from_mesh_options(options).map(|_| ())
}

pub async fn spawn_mesh_generation(
    state: Arc<Lux3dState>,
    record: JobRecord,
    source: PathBuf,
    options: MeshGenerationOptions,
) -> Result<GenerationObject> {
    let model = record.object.model.clone();
    validate_mesh_generation_request(&model, &options)?;
    let output_format = match options.output_format.as_deref() {
        Some(format) => MeshFormat::from_extension(format).map_err(anyhow::Error::from)?,
        None => MeshFormat::Glb,
    };
    let triposg_run_options = if model == "triposg" {
        Some(triposg_options_from_mesh_options(&options)?)
    } else {
        None
    };
    let trellis2_run_options = if model == "trellis2" {
        Some(trellis2_options_from_mesh_options(&options)?)
    } else {
        None
    };

    let job_id = record.object.id.clone();
    state.jobs.insert(record);
    let response = state
        .jobs
        .get(&job_id)
        .expect("job just inserted")
        .object
        .clone();
    record_generation_started(&model);

    let state_for_task = state.clone();
    tokio::spawn(async move {
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", 0.0);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        if let Err(error) = state_for_task.jobs.mark_in_progress(&job_id) {
            tracing::error!("failed to mark job in progress: {error}");
            return;
        }

        let started = Instant::now();
        let job_id_for_blocking = job_id.clone();
        let result = state_for_task
            .run_blocking({
                let workspace = state_for_task
                    .jobs
                    .get(&job_id)
                    .map(|record| record.workspace.clone());
                let source = source.clone();
                let model = model.clone();
                let mc_resolution = options.mc_resolution;
                let mc_threshold = options.mc_threshold;
                move |runtime| {
                    if runtime.jobs.is_cancelled(&job_id_for_blocking) {
                        bail!("generation was cancelled");
                    }
                    runtime
                        .models
                        .ensure_loaded(runtime, &model)
                        .with_context(|| format!("failed to load `{model}` pipeline"))?;
                    let workspace =
                        workspace.ok_or_else(|| anyhow::anyhow!("job workspace missing"))?;
                    fs::create_dir_all(&workspace).with_context(|| {
                        format!(
                            "failed to create generation workspace at `{}`",
                            workspace.display()
                        )
                    })?;
                    let source = normalize_source_for_image(&source)?;
                    let device = runtime.device()?;
                    let output_path =
                        workspace.join(format!("{model}.{}", output_format.extension()));
                    let mut served_output = output_path.clone();
                    match model.as_str() {
                        "triposr" => runtime.models.with_triposr(|pipeline| {
                            let inference = pipeline
                                .infer_from_path(&source, &device)
                                .map_err(anyhow::Error::from)?;
                            let mesh = pipeline
                                .extract_mesh(
                                    &inference.scene_codes,
                                    mc_resolution,
                                    mc_threshold,
                                    8192,
                                )
                                .map_err(anyhow::Error::from)?;
                            pipeline
                                .export_mesh(&mesh, &output_path)
                                .map_err(anyhow::Error::from)?;
                            Ok(())
                        })?,
                        "triposg" => {
                            let run_options = triposg_run_options
                                .expect("triposg options validated before the job was queued");
                            runtime.models.with_triposg(|pipeline| {
                                let inference = pipeline
                                    .infer_from_path(&source, &device, &run_options)
                                    .map_err(anyhow::Error::from)?;
                                let mesh = pipeline
                                    .extract_mesh(&inference.latents, &run_options)
                                    .map_err(anyhow::Error::from)?;
                                pipeline
                                    .export_mesh(&mesh, &output_path)
                                    .map_err(anyhow::Error::from)?;
                                Ok(())
                            })?
                        }
                        "trellis2" => {
                            let run_options = trellis2_run_options
                                .expect("trellis2 options validated before the job was queued");
                            served_output = runtime.models.with_trellis2(|pipeline| {
                                // `num_samples = 1` runs the single-sample
                                // surface (bit-identical to the pre-
                                // num_samples behavior); above 1 runs the
                                // sequential seed-offset emulation.
                                let stages_list = if run_options.num_samples == 1 {
                                    vec![pipeline.run_from_path_with_stages(
                                        &source,
                                        &device,
                                        &run_options,
                                    )?]
                                } else {
                                    pipeline.run_many_from_path(&source, &device, &run_options)?
                                };
                                ensure!(
                                    !stages_list.is_empty(),
                                    "trellis2 run produced no samples"
                                );
                                let sample_count = stages_list.len();
                                for (index, stages) in stages_list.iter().enumerate() {
                                    let sample_output =
                                        sample_output_path(&output_path, index, sample_count);
                                    let mesh = stages
                                        .mesh
                                        .as_ref()
                                        .expect("trellis2 run always produces a mesh");
                                    let textured = stages
                                        .textured
                                        .as_ref()
                                        .filter(|_| matches!(output_format, MeshFormat::Glb));
                                    if let Some(textured) = textured {
                                        // Upstream `to_glb` textured export:
                                        // embedded PBR textures + UVs.
                                        // `doubleSided = false` on the
                                        // remeshed branch (`postprocess.py:303`).
                                        let double_sided = !(run_options.postprocess.enabled
                                            && run_options.postprocess.remesh);
                                        pipeline
                                            .export
                                            .write_textured_glb(
                                                textured,
                                                &sample_output,
                                                double_sided,
                                            )
                                            .map_err(anyhow::Error::from)?;
                                        if run_options.with_texture
                                            && mesh.voxel_attr_channels == pbrvox::CHANNELS
                                        {
                                            pipeline
                                                .export
                                                .write_pbr_vox(
                                                    mesh,
                                                    &pbr_sidecar_stem(&sample_output),
                                                )
                                                .map_err(anyhow::Error::from)?;
                                        }
                                    } else if run_options.with_texture
                                        && mesh.voxel_attr_channels == pbrvox::CHANNELS
                                    {
                                        // The mesh plus the <stem>.pbrvox.bin /
                                        // <stem>.pbr.json sidecar pair; the
                                        // sidecars stay in the job workspace
                                        // (only the mesh is served through
                                        // /content).
                                        pipeline
                                            .export_mesh_with_pbr(mesh, &sample_output)
                                            .map_err(anyhow::Error::from)?;
                                    } else {
                                        pipeline
                                            .export_mesh(mesh, &sample_output)
                                            .map_err(anyhow::Error::from)?;
                                    }
                                }
                                // The job serves a single artifact; with
                                // num_samples > 1 that is sample 0, while the
                                // remaining samples stay in the workspace.
                                Ok(sample_output_path(&output_path, 0, sample_count))
                            })?;
                        }
                        other => bail!("unsupported mesh model `{other}`"),
                    }
                    Ok(served_output)
                }
            })
            .await;

        let elapsed = started.elapsed().as_secs_f64();
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", elapsed);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        match result {
            Ok(output_path) => {
                if let Err(error) = state_for_task.jobs.mark_completed(&job_id, output_path) {
                    tracing::error!("failed to mark job completed: {error}");
                } else {
                    record_generation_finished(&model, "completed", elapsed);
                    maybe_deliver_webhook(&state_for_task, &job_id, "generation.completed").await;
                }
            }
            Err(error) => {
                let status = if error.to_string().contains("cancelled") {
                    let _ = state_for_task.jobs.mark_cancelled(&job_id);
                    "cancelled"
                } else if let Err(update_error) =
                    state_for_task.jobs.mark_failed(&job_id, error.to_string())
                {
                    tracing::error!("failed to mark job failed: {update_error}");
                    "failed"
                } else {
                    "failed"
                };
                record_generation_finished(&model, status, elapsed);
                maybe_deliver_webhook(&state_for_task, &job_id, "generation.failed").await;
            }
        }
    });

    Ok(response)
}

/// Per-sample artifact path: a single sample keeps the exact `output` path;
/// multiple samples suffix the stem with `_<index>` (CLI parity).
fn sample_output_path(output: &Path, index: usize, count: usize) -> PathBuf {
    if count <= 1 {
        return output.to_path_buf();
    }
    let file_name = output
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("sample_{index}"));
    let suffixed = match file_name.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() && !extension.is_empty() => {
            format!("{stem}_{index}.{extension}")
        }
        _ => format!("{file_name}_{index}"),
    };
    output.with_file_name(suffixed)
}

/// The `<parent>/<stem>` prefix the PBR sidecar writer extends into
/// `<stem>.pbrvox.bin` / `<stem>.pbr.json`.
fn pbr_sidecar_stem(path: &Path) -> PathBuf {
    path.parent().unwrap_or_else(|| Path::new(".")).join(
        path.file_stem()
            .unwrap_or_else(|| std::ffi::OsStr::new("mesh")),
    )
}

/// Build the TRELLIS.2 texturing-pipeline options from the texturing-request
/// fields. Pure helper so the mapping and validation are unit-testable.
fn trellis2_texturing_options_from_request(
    options: &MeshTexturingOptions,
) -> Result<Trellis2TexturingOptions> {
    let defaults = Trellis2TexturingOptions::default();
    let texturing = Trellis2TexturingOptions {
        seed: options.seed.unwrap_or(defaults.seed),
        resolution: options.texture_resolution.unwrap_or(defaults.resolution),
        tex_sampler: Trellis2SamplerOverrides {
            steps: options.tex_steps,
            guidance_strength: options.tex_guidance_strength,
            guidance_rescale: options.tex_guidance_rescale,
            guidance_interval: options.tex_guidance_interval.map(|[low, high]| (low, high)),
            rescale_t: options.tex_rescale_t,
        },
        texture_size: options.texture_size.unwrap_or(defaults.texture_size),
        preprocess_image: options
            .preprocess_image
            .unwrap_or(defaults.preprocess_image),
        preprocess_mesh: options.preprocess_mesh.unwrap_or(defaults.preprocess_mesh),
    };
    texturing.validate().map_err(anyhow::Error::from)?;
    Ok(texturing)
}

/// Pure texturing-request guard: the family allow-list plus the GLB-only
/// output contract (the baked PBR atlas is embedded in the GLB).
fn validate_mesh_texturing_request(model: &str, options: &MeshTexturingOptions) -> Result<()> {
    ensure!(
        model == "trellis2",
        "unsupported texturing model `{model}`; expected `trellis2`"
    );
    if let Some(format) = options.output_format.as_deref() {
        ensure!(
            matches!(
                MeshFormat::from_extension(format).map_err(anyhow::Error::from)?,
                MeshFormat::Glb
            ),
            "`output_format` must be `glb` for mesh texturing; the baked PBR atlas is \
             embedded in the GLB"
        );
    }
    trellis2_texturing_options_from_request(options).map(|_| ())
}

pub async fn spawn_mesh_texturing(
    state: Arc<Lux3dState>,
    record: JobRecord,
    mesh: PathBuf,
    image: PathBuf,
    options: MeshTexturingOptions,
) -> Result<GenerationObject> {
    let model = record.object.model.clone();
    validate_mesh_texturing_request(&model, &options)?;
    let texturing_options = trellis2_texturing_options_from_request(&options)?;

    let job_id = record.object.id.clone();
    state.jobs.insert(record);
    let response = state
        .jobs
        .get(&job_id)
        .expect("job just inserted")
        .object
        .clone();
    record_generation_started(&model);

    let state_for_task = state.clone();
    tokio::spawn(async move {
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", 0.0);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        if let Err(error) = state_for_task.jobs.mark_in_progress(&job_id) {
            tracing::error!("failed to mark job in progress: {error}");
            return;
        }

        let started = Instant::now();
        let job_id_for_blocking = job_id.clone();
        let result = state_for_task
            .run_blocking({
                let workspace = state_for_task
                    .jobs
                    .get(&job_id)
                    .map(|record| record.workspace.clone());
                let mesh = mesh.clone();
                let image = image.clone();
                let texturing_options = texturing_options.clone();
                move |runtime| {
                    if runtime.jobs.is_cancelled(&job_id_for_blocking) {
                        bail!("generation was cancelled");
                    }
                    runtime
                        .models
                        .ensure_loaded(runtime, "trellis2")
                        .context("failed to load `trellis2` pipeline")?;
                    let workspace =
                        workspace.ok_or_else(|| anyhow::anyhow!("job workspace missing"))?;
                    fs::create_dir_all(&workspace).with_context(|| {
                        format!(
                            "failed to create generation workspace at `{}`",
                            workspace.display()
                        )
                    })?;
                    let device = runtime.device()?;
                    let output_path = workspace
                        .join(format!("trellis2-textured.{}", MeshFormat::Glb.extension()));
                    runtime.models.with_trellis2(|pipeline| {
                        let imported = load_mesh(&mesh).map_err(anyhow::Error::from)?;
                        ensure!(
                            !imported.faces.is_empty(),
                            "texturing mesh `{}` contains no triangles",
                            mesh.display()
                        );
                        let textured = pipeline
                            .texture_mesh_from_path(
                                &imported.vertices,
                                &imported.faces,
                                &image,
                                &texturing_options,
                                &device,
                            )
                            .map_err(anyhow::Error::from)?;
                        pipeline
                            .export
                            .write_textured_glb(&textured.mesh, &output_path, true)
                            .map_err(anyhow::Error::from)?;
                        Ok(())
                    })?;
                    Ok(output_path)
                }
            })
            .await;

        let elapsed = started.elapsed().as_secs_f64();
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", elapsed);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        match result {
            Ok(output_path) => {
                if let Err(error) = state_for_task.jobs.mark_completed(&job_id, output_path) {
                    tracing::error!("failed to mark job completed: {error}");
                } else {
                    record_generation_finished(&model, "completed", elapsed);
                    maybe_deliver_webhook(&state_for_task, &job_id, "generation.completed").await;
                }
            }
            Err(error) => {
                let status = if error.to_string().contains("cancelled") {
                    let _ = state_for_task.jobs.mark_cancelled(&job_id);
                    "cancelled"
                } else if let Err(update_error) =
                    state_for_task.jobs.mark_failed(&job_id, error.to_string())
                {
                    tracing::error!("failed to mark job failed: {update_error}");
                    "failed"
                } else {
                    "failed"
                };
                record_generation_finished(&model, status, elapsed);
                maybe_deliver_webhook(&state_for_task, &job_id, "generation.failed").await;
            }
        }
    });

    Ok(response)
}

async fn maybe_deliver_webhook(state: &Lux3dState, job_id: &str, event: &'static str) {
    let Some(record) = state.jobs.get(job_id) else {
        return;
    };
    let Some(webhook_url) = record.webhook_url.clone() else {
        return;
    };
    deliver_generation_webhook(&state.http_client, &webhook_url, event, &record.object).await;
}

fn parse_inject_conditions(values: &[String]) -> Pi3xInjectConditions {
    Pi3xInjectConditions {
        pose: values
            .iter()
            .any(|value| value.eq_ignore_ascii_case("pose")),
        depth: values
            .iter()
            .any(|value| value.eq_ignore_ascii_case("depth")),
        ray: values.iter().any(|value| {
            value.eq_ignore_ascii_case("ray") || value.eq_ignore_ascii_case("intrinsic")
        }),
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{
        sample_output_path, trellis2_options_from_mesh_options,
        trellis2_texturing_options_from_request, triposg_options_from_mesh_options,
        validate_mesh_generation_request, validate_mesh_texturing_request, MeshGenerationOptions,
        MeshTexturingOptions, Trellis2PipelineType, Trellis2RunOptions, Trellis2TexturingOptions,
        TripoSgRunOptions,
    };

    fn default_mesh_options() -> MeshGenerationOptions {
        MeshGenerationOptions::default()
    }

    #[test]
    fn triposg_options_default_to_reference_values() {
        let run = triposg_options_from_mesh_options(&default_mesh_options())
            .expect("default triposg options");
        assert_eq!(TripoSgRunOptions::default(), run);
        assert_eq!(
            (50usize, 7.0f32, 42u64, 2048usize, 8u32, 9u32, true, 9u32, 16384usize),
            (
                run.steps,
                run.guidance_scale,
                run.seed,
                run.num_tokens,
                run.dense_octree_depth,
                run.hierarchical_octree_depth,
                run.use_flash_decoder,
                run.flash_octree_depth,
                run.query_chunk_size
            )
        );
    }

    #[test]
    fn triposg_options_map_every_override() {
        let options = MeshGenerationOptions {
            steps: Some(12),
            guidance_scale: Some(3.5),
            seed: Some(7),
            num_tokens: Some(1024),
            dense_octree_depth: Some(5),
            hierarchical_octree_depth: Some(6),
            use_flash_decoder: Some(false),
            flash_octree_depth: Some(7),
            query_chunk_size: Some(4096),
            ..default_mesh_options()
        };
        let run = triposg_options_from_mesh_options(&options).expect("overridden triposg options");
        assert_eq!(12, run.steps);
        assert_eq!(3.5, run.guidance_scale);
        assert_eq!(7, run.seed);
        assert_eq!(1024, run.num_tokens);
        assert_eq!(5, run.dense_octree_depth);
        assert_eq!(6, run.hierarchical_octree_depth);
        assert!(!run.use_flash_decoder);
        assert_eq!(7, run.flash_octree_depth);
        assert_eq!(4096, run.query_chunk_size);
    }

    #[test]
    fn triposg_accepts_default_mesh_request() {
        validate_mesh_generation_request("triposg", &default_mesh_options())
            .expect("default triposg request");
    }

    #[test]
    fn triposg_rejects_non_default_marching_cubes_resolution() {
        let options = MeshGenerationOptions {
            mc_resolution: 128,
            ..default_mesh_options()
        };
        let error = validate_mesh_generation_request("triposg", &options)
            .expect_err("mc_resolution=128 must be rejected for triposg");
        let message = error.to_string();
        assert!(
            message.contains("`mc_resolution`"),
            "unexpected message: `{message}`"
        );
        assert!(
            message.contains("is not supported for `triposg`"),
            "unexpected message: `{message}`"
        );
    }

    #[test]
    fn triposg_rejects_non_default_marching_cubes_threshold() {
        let options = MeshGenerationOptions {
            mc_threshold: 10.0,
            ..default_mesh_options()
        };
        let error = validate_mesh_generation_request("triposg", &options)
            .expect_err("mc_threshold=10.0 must be rejected for triposg");
        let message = error.to_string();
        assert!(
            message.contains("`mc_threshold`"),
            "unexpected message: `{message}`"
        );
        assert!(
            message.contains("is not supported for `triposg`"),
            "unexpected message: `{message}`"
        );
    }

    #[test]
    fn triposg_rejects_trellis2_remesh_options() {
        for options in [
            MeshGenerationOptions {
                remesh: Some(false),
                ..default_mesh_options()
            },
            MeshGenerationOptions {
                remesh_band: Some(2.0),
                ..default_mesh_options()
            },
            MeshGenerationOptions {
                remesh_project: Some(0.5),
                ..default_mesh_options()
            },
        ] {
            let error = validate_mesh_generation_request("triposg", &options)
                .expect_err("trellis2 remesh knobs must be rejected for triposg");
            let message = error.to_string();
            assert!(
                message.contains("remesh") && message.contains("is not supported for `triposg`"),
                "unexpected message: `{message}`"
            );
        }
    }

    #[test]
    fn triposr_accepts_default_mesh_request() {
        validate_mesh_generation_request("triposr", &default_mesh_options())
            .expect("default triposr request");
    }

    #[test]
    fn triposr_rejects_triposg_only_steps() {
        let options = MeshGenerationOptions {
            steps: Some(2),
            ..default_mesh_options()
        };
        let error = validate_mesh_generation_request("triposr", &options)
            .expect_err("steps=Some(2) must be rejected for triposr");
        assert!(
            error
                .to_string()
                .contains("`steps` is not supported for `triposr`"),
            "unexpected message: `{}`",
            error
        );
    }

    #[test]
    fn unsupported_mesh_models_are_rejected() {
        for model in ["pi3", "pi3x", "pi3x_vo", "vggt", "", "triposrg"] {
            let error = validate_mesh_generation_request(model, &default_mesh_options())
                .expect_err("unsupported mesh model must be rejected");
            let expected = format!(
                "unsupported mesh model `{model}`; expected `triposr`, `triposg` or `trellis2`"
            );
            assert!(
                error.to_string().contains(&expected),
                "unexpected message for `{model}`: `{}`",
                error
            );
        }
    }

    #[test]
    fn trellis2_accepts_default_mesh_request() {
        validate_mesh_generation_request("trellis2", &default_mesh_options())
            .expect("default trellis2 request");
    }

    #[test]
    fn trellis2_options_default_to_reference_values() {
        let run = trellis2_options_from_mesh_options(&default_mesh_options())
            .expect("default trellis2 options");
        assert_eq!(Trellis2RunOptions::default(), run);
        assert_eq!(
            (42u64, "1024_cascade", 12usize, true, 49152usize),
            (
                run.seed,
                run.pipeline_type.as_str(),
                run.ss_steps,
                run.with_texture,
                run.max_num_tokens
            )
        );
    }

    #[test]
    fn trellis2_options_map_every_override() {
        let options = MeshGenerationOptions {
            steps: Some(4),
            seed: Some(7),
            pipeline_type: Some("512".to_string()),
            max_num_tokens: Some(8192),
            texture: Some(false),
            ..default_mesh_options()
        };
        let run =
            trellis2_options_from_mesh_options(&options).expect("overridden trellis2 options");
        assert_eq!(7, run.seed);
        assert_eq!(Trellis2PipelineType::R512, run.pipeline_type);
        assert_eq!(4, run.ss_steps);
        assert_eq!(4, run.shape_steps);
        assert_eq!(4, run.tex_steps);
        assert!(!run.with_texture);
        assert_eq!(8192, run.max_num_tokens);

        let cascade = trellis2_options_from_mesh_options(&MeshGenerationOptions {
            pipeline_type: Some("1024_cascade".to_string()),
            ..default_mesh_options()
        })
        .expect("cascade options");
        assert_eq!(Trellis2PipelineType::Cascade1024, cascade.pipeline_type);

        let single_1024 = trellis2_options_from_mesh_options(&MeshGenerationOptions {
            pipeline_type: Some("1024".to_string()),
            ..default_mesh_options()
        })
        .expect("single-resolution options parse (runtime rejects it at run time)");
        assert_eq!(Trellis2PipelineType::R1024, single_1024.pipeline_type);
    }

    #[test]
    fn trellis2_rejects_unknown_pipeline_type() {
        // `1536_cascade` is a valid variant now; use a genuinely unknown name.
        let options = MeshGenerationOptions {
            pipeline_type: Some("768_cascade".to_string()),
            ..default_mesh_options()
        };
        let error = trellis2_options_from_mesh_options(&options)
            .expect_err("unknown pipeline_type must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("unknown trellis2 `pipeline_type` `768_cascade`"),
            "unexpected message: `{message}`"
        );
        // All four reference variants (including `1536_cascade`) map cleanly.
        for name in ["512", "1024", "1024_cascade", "1536_cascade"] {
            let options = MeshGenerationOptions {
                pipeline_type: Some(name.to_string()),
                ..default_mesh_options()
            };
            let run = trellis2_options_from_mesh_options(&options)
                .unwrap_or_else(|error| panic!("`{name}` must map: {error}"));
            assert_eq!(1, run.num_samples, "`{name}` single-sample default");
            assert_eq!(None, run.ss_sampler);
            assert_eq!(None, run.shape_sampler);
            assert_eq!(None, run.tex_sampler);
        }
    }

    #[test]
    fn trellis2_rejects_foreign_mesh_options() {
        let cases: [(&str, MeshGenerationOptions); 7] = [
            (
                "mc_resolution",
                MeshGenerationOptions {
                    mc_resolution: 128,
                    ..default_mesh_options()
                },
            ),
            (
                "mc_threshold",
                MeshGenerationOptions {
                    mc_threshold: 10.0,
                    ..default_mesh_options()
                },
            ),
            (
                "guidance_scale",
                MeshGenerationOptions {
                    guidance_scale: Some(3.5),
                    ..default_mesh_options()
                },
            ),
            (
                "num_tokens",
                MeshGenerationOptions {
                    num_tokens: Some(1024),
                    ..default_mesh_options()
                },
            ),
            (
                "use_flash_decoder",
                MeshGenerationOptions {
                    use_flash_decoder: Some(true),
                    ..default_mesh_options()
                },
            ),
            (
                "flash_octree_depth",
                MeshGenerationOptions {
                    flash_octree_depth: Some(9),
                    ..default_mesh_options()
                },
            ),
            (
                "query_chunk_size",
                MeshGenerationOptions {
                    query_chunk_size: Some(4096),
                    ..default_mesh_options()
                },
            ),
        ];
        for (name, options) in cases {
            let error = validate_mesh_generation_request("trellis2", &options)
                .expect_err("foreign option must be rejected for trellis2");
            let message = error.to_string();
            assert!(
                message.contains(&format!("`{name}` is not supported for `trellis2`")),
                "unexpected message for {name}: `{message}`"
            );
        }
    }

    #[test]
    fn triposg_and_triposr_reject_trellis2_only_options() {
        for model in ["triposg", "triposr"] {
            for (name, options) in [
                (
                    "pipeline_type",
                    MeshGenerationOptions {
                        pipeline_type: Some("512".to_string()),
                        ..default_mesh_options()
                    },
                ),
                (
                    "max_num_tokens",
                    MeshGenerationOptions {
                        max_num_tokens: Some(8192),
                        ..default_mesh_options()
                    },
                ),
                (
                    "texture",
                    MeshGenerationOptions {
                        texture: Some(false),
                        ..default_mesh_options()
                    },
                ),
            ] {
                let error = validate_mesh_generation_request(model, &options)
                    .expect_err("trellis2-only option must be rejected");
                let message = error.to_string();
                assert!(
                    message.contains(&format!("`{name}` is not supported for `{model}`")),
                    "unexpected message for {model}/{name}: `{message}`"
                );
            }
        }
    }

    #[test]
    fn triposg_options_map_schedule_bounds_and_faces() {
        let options = MeshGenerationOptions {
            timesteps: Some(vec![900.0, 500.0, 100.0]),
            bounds: Some([-1.0, -0.5, -1.0, 1.0, 0.5, 1.0]),
            faces: Some(5_000),
            ..default_mesh_options()
        };
        let run = triposg_options_from_mesh_options(&options).expect("mapped triposg options");
        assert_eq!(Some(vec![900.0, 500.0, 100.0]), run.timesteps);
        assert_eq!(Some([-1.0, -0.5, -1.0, 1.0, 0.5, 1.0]), run.bounds);
        assert_eq!(Some(5_000), run.faces);
    }

    #[test]
    fn triposg_schedule_fields_are_rejected_by_other_families() {
        for model in ["triposr", "trellis2"] {
            for (name, options) in [
                (
                    "timesteps",
                    MeshGenerationOptions {
                        timesteps: Some(vec![900.0, 100.0]),
                        ..default_mesh_options()
                    },
                ),
                (
                    "bounds",
                    MeshGenerationOptions {
                        bounds: Some([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
                        ..default_mesh_options()
                    },
                ),
                (
                    "faces",
                    MeshGenerationOptions {
                        faces: Some(1_000),
                        ..default_mesh_options()
                    },
                ),
            ] {
                let error = validate_mesh_generation_request(model, &options)
                    .expect_err("triposg-only field must be rejected");
                let message = error.to_string();
                assert!(
                    message.contains(&format!("`{name}` is not supported for `{model}`")),
                    "unexpected message for {model}/{name}: `{message}`"
                );
            }
        }
    }

    #[test]
    fn trellis2_options_map_per_stage_sampler_overrides() {
        let options = MeshGenerationOptions {
            ss_steps: Some(6),
            ss_guidance_strength: Some(3.0),
            ss_guidance_rescale: Some(0.2),
            ss_guidance_interval: Some([0.1, 0.9]),
            ss_rescale_t: Some(5.0),
            shape_steps: Some(7),
            shape_guidance_strength: Some(4.0),
            shape_guidance_rescale: Some(0.3),
            shape_guidance_interval: Some([0.2, 0.8]),
            shape_rescale_t: Some(6.0),
            tex_steps: Some(8),
            tex_guidance_strength: Some(5.0),
            tex_guidance_rescale: Some(0.4),
            tex_guidance_interval: Some([0.3, 0.7]),
            tex_rescale_t: Some(7.0),
            ..default_mesh_options()
        };
        let run = trellis2_options_from_mesh_options(&options).expect("per-stage options");

        let ss = run.ss_sampler.expect("ss sampler overrides");
        assert_eq!(Some(6), ss.steps);
        assert_eq!(Some(3.0), ss.guidance_strength);
        assert_eq!(Some(0.2), ss.guidance_rescale);
        assert_eq!(Some((0.1, 0.9)), ss.guidance_interval);
        assert_eq!(Some(5.0), ss.rescale_t);

        let shape = run.shape_sampler.expect("shape sampler overrides");
        assert_eq!(Some(7), shape.steps);
        assert_eq!(Some(4.0), shape.guidance_strength);
        assert_eq!(Some(0.3), shape.guidance_rescale);
        assert_eq!(Some((0.2, 0.8)), shape.guidance_interval);
        assert_eq!(Some(6.0), shape.rescale_t);

        let tex = run.tex_sampler.expect("tex sampler overrides");
        assert_eq!(Some(8), tex.steps);
        assert_eq!(Some(5.0), tex.guidance_strength);
        assert_eq!(Some(0.4), tex.guidance_rescale);
        assert_eq!(Some((0.3, 0.7)), tex.guidance_interval);
        assert_eq!(Some(7.0), tex.rescale_t);
    }

    #[test]
    fn trellis2_options_map_num_samples_and_postprocess() {
        let options = MeshGenerationOptions {
            num_samples: Some(3),
            texture_size: Some(4096),
            postprocess: Some(false),
            decimation_target: Some(50_000),
            fill_hole_perimeter: Some(0.05),
            min_component_area: Some(1e-4),
            remesh: Some(false),
            remesh_band: Some(2.0),
            remesh_project: Some(0.5),
            ..default_mesh_options()
        };
        let run = trellis2_options_from_mesh_options(&options).expect("mapped trellis2 options");
        assert_eq!(3, run.num_samples);
        assert_eq!(4096, run.texture_export.texture_size);
        assert!(!run.postprocess.enabled);
        assert_eq!(50_000, run.postprocess.decimation_target);
        assert_eq!(0.05, run.postprocess.fill_hole_perimeter);
        assert_eq!(1e-4, run.postprocess.min_component_area);
        assert!(!run.postprocess.remesh);
        assert_eq!(2.0, run.postprocess.remesh_band);
        assert_eq!(0.5, run.postprocess.remesh_project);

        // The shipped defaults (remesh on, band 1, project 0) apply when the
        // request omits the fields.
        let defaults = trellis2_options_from_mesh_options(&default_mesh_options())
            .expect("default trellis2 options")
            .postprocess;
        assert!(defaults.remesh);
        assert_eq!(1.0, defaults.remesh_band);
        assert_eq!(0.0, defaults.remesh_project);
    }

    #[test]
    fn trellis2_rejects_invalid_remesh_thresholds() {
        for (band, project) in [
            (Some(-1.0), None),
            (Some(f32::NAN), None),
            (None, Some(f32::NAN)),
        ] {
            let options = MeshGenerationOptions {
                remesh_band: band,
                remesh_project: project,
                ..default_mesh_options()
            };
            let error = trellis2_options_from_mesh_options(&options)
                .expect_err("invalid remesh knob must be rejected");
            assert!(
                error.to_string().contains("remesh_"),
                "unexpected message: `{error}`"
            );
        }
    }

    #[test]
    fn trellis2_rejects_zero_num_samples_and_invalid_texture_size() {
        let zero = MeshGenerationOptions {
            num_samples: Some(0),
            ..default_mesh_options()
        };
        let error = trellis2_options_from_mesh_options(&zero)
            .expect_err("num_samples = 0 must be rejected");
        assert!(
            error
                .to_string()
                .contains("`num_samples` must be at least 1"),
            "unexpected message: `{error}`"
        );

        let bad_size = MeshGenerationOptions {
            texture_size: Some(1001),
            ..default_mesh_options()
        };
        let error = trellis2_options_from_mesh_options(&bad_size)
            .expect_err("non-multiple-of-8 texture_size must be rejected");
        assert!(
            error.to_string().contains("multiple of 8"),
            "unexpected message: `{error}`"
        );
    }

    #[test]
    fn texturing_options_default_to_reference_values() {
        let texturing = trellis2_texturing_options_from_request(&MeshTexturingOptions::default())
            .expect("default texturing options");
        assert_eq!(Trellis2TexturingOptions::default(), texturing);
        assert_eq!(
            (42u64, 1024u32, 2048usize, true, true),
            (
                texturing.seed,
                texturing.resolution,
                texturing.texture_size,
                texturing.preprocess_image,
                texturing.preprocess_mesh,
            )
        );
    }

    #[test]
    fn texturing_options_map_every_override() {
        let options = MeshTexturingOptions {
            seed: Some(7),
            texture_resolution: Some(1536),
            texture_size: Some(1024),
            tex_steps: Some(4),
            tex_guidance_strength: Some(2.5),
            tex_guidance_rescale: Some(0.1),
            tex_guidance_interval: Some([0.2, 0.6]),
            tex_rescale_t: Some(3.0),
            preprocess_image: Some(false),
            preprocess_mesh: Some(false),
            ..MeshTexturingOptions::default()
        };
        let texturing =
            trellis2_texturing_options_from_request(&options).expect("overridden texturing");
        assert_eq!(7, texturing.seed);
        assert_eq!(1536, texturing.resolution);
        assert_eq!(1024, texturing.texture_size);
        assert_eq!(Some(4), texturing.tex_sampler.steps);
        assert_eq!(Some(2.5), texturing.tex_sampler.guidance_strength);
        assert_eq!(Some(0.1), texturing.tex_sampler.guidance_rescale);
        assert_eq!(Some((0.2, 0.6)), texturing.tex_sampler.guidance_interval);
        assert_eq!(Some(3.0), texturing.tex_sampler.rescale_t);
        assert!(!texturing.preprocess_image);
        assert!(!texturing.preprocess_mesh);
    }

    #[test]
    fn texturing_request_validation_rejects_bad_models_and_formats() {
        let default = MeshTexturingOptions::default();
        validate_mesh_texturing_request("trellis2", &default).expect("default texturing request");

        let error = validate_mesh_texturing_request("triposg", &default)
            .expect_err("only trellis2 textures meshes");
        assert!(
            error
                .to_string()
                .contains("unsupported texturing model `triposg`"),
            "unexpected message: `{error}`"
        );

        let obj = MeshTexturingOptions {
            output_format: Some("obj".to_string()),
            ..MeshTexturingOptions::default()
        };
        let error = validate_mesh_texturing_request("trellis2", &obj)
            .expect_err("non-GLB texturing output must be rejected");
        assert!(
            error.to_string().contains("must be `glb`"),
            "unexpected message: `{error}`"
        );

        let unknown = MeshTexturingOptions {
            output_format: Some("fbx".to_string()),
            ..MeshTexturingOptions::default()
        };
        assert!(validate_mesh_texturing_request("trellis2", &unknown).is_err());
    }

    #[test]
    fn texturing_request_rejects_invalid_resolution_and_texture_size() {
        for (name, options) in [
            (
                "resolution",
                MeshTexturingOptions {
                    texture_resolution: Some(768),
                    ..MeshTexturingOptions::default()
                },
            ),
            (
                "texture_size",
                MeshTexturingOptions {
                    texture_size: Some(1001),
                    ..MeshTexturingOptions::default()
                },
            ),
        ] {
            let error = validate_mesh_texturing_request("trellis2", &options)
                .expect_err("invalid texturing option must be rejected");
            let message = error.to_string();
            assert!(
                message.contains(name) || message.contains("resolution") || message.contains("8"),
                "unexpected message for {name}: `{message}`"
            );
        }
    }

    #[test]
    fn sample_output_path_suffixes_multi_sample_runs() {
        let output = std::path::Path::new("trellis2.glb");
        assert_eq!(
            std::path::PathBuf::from("trellis2.glb"),
            sample_output_path(output, 0, 1)
        );
        assert_eq!(
            std::path::PathBuf::from("trellis2_0.glb"),
            sample_output_path(output, 0, 3)
        );
        assert_eq!(
            std::path::PathBuf::from("trellis2_2.glb"),
            sample_output_path(output, 2, 3)
        );
    }
}
