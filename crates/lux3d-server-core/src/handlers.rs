//! General Lux3D server route handlers.

use std::path::Path;

use anyhow::Context;
use axum::{
    body::Body,
    extract::{FromRequest, Path as AxumPath, Query, Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;

use lux3d_core::export::{MeshFormat, PointCloudFormat};

use crate::{
    api_types::{
        ErrorBody, ErrorResponse, GenerationListResponse, GenerationObject, GenerationStatus,
        MeshGenerationOptions, MeshGenerationRequest, MeshTexturingOptions, MeshTexturingRequest,
        ModelListResponse, ModelObject, PointCloudGenerationOptions, PointCloudGenerationRequest,
        RouteSummary, ServerInfoResponse,
    },
    errors::{conflict, internal_error, invalid_request, not_found},
    jobs::{
        create_job_record, job_workspace, new_job_id, spawn_mesh_generation, spawn_mesh_texturing,
        spawn_point_cloud_generation, NewJobRecord,
    },
    media_source::{
        resolve_conditions_path, resolve_mesh_path, resolve_source_path, MediaSourceInput,
        MeshSourceInput,
    },
    models::{ModelOperationRequest, ModelStatusResponse},
    route_registry::LUX3D_API_ROUTES,
    types::ExtractedLux3dState,
    upload::{parse_generation_upload, parse_texturing_upload},
};

#[derive(Debug, Deserialize)]
pub struct GenerationListQuery {
    #[serde(default = "default_generation_list_limit")]
    pub limit: usize,
}

fn default_generation_list_limit() -> usize {
    20
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/",
    responses((status = 200, description = "Server metadata", body = ServerInfoResponse))
)]
pub async fn root() -> Json<ServerInfoResponse> {
    Json(ServerInfoResponse {
        object: "lux3d.server",
        name: "Lux3D Server".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        routes: LUX3D_API_ROUTES
            .iter()
            .map(|route| RouteSummary {
                path: route.path.to_string(),
                methods: route.methods.to_string(),
            })
            .collect(),
    })
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/health",
    responses((status = 200, description = "Health check"))
)]
pub async fn health() -> &'static str {
    "OK"
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/v1/models",
    responses((status = 200, description = "Supported generation models", body = ModelListResponse))
)]
pub async fn models(State(state): ExtractedLux3dState) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list",
        data: vec![
            model_entry(&state, "pi3", "point_cloud", "ply"),
            model_entry(&state, "pi3x", "point_cloud", "ply"),
            model_entry(&state, "vggt", "point_cloud", "ply"),
            model_entry(&state, "triposr", "mesh", "glb"),
            model_entry(&state, "triposg", "mesh", "glb"),
            model_entry(&state, "trellis2", "mesh", "glb"),
        ],
    })
}

fn model_entry(
    state: &crate::types::Lux3dState,
    id: &str,
    kind: &str,
    output_format: &str,
) -> ModelObject {
    ModelObject {
        id: id.to_string(),
        object: "model",
        created: state.creation_time,
        owned_by: "local",
        kind: kind.to_string(),
        output_format: output_format.to_string(),
    }
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/v1/generations",
    responses((status = 200, description = "Recent generation jobs", body = GenerationListResponse))
)]
pub async fn list_generations(
    State(state): ExtractedLux3dState,
    Query(query): Query<GenerationListQuery>,
) -> Json<GenerationListResponse> {
    Json(GenerationListResponse {
        object: "list",
        data: state.jobs.list(query.limit),
    })
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/v1/generations/{id}",
    responses(
        (status = 200, description = "Generation job status", body = GenerationObject),
        (status = 404, description = "Job not found")
    )
)]
pub async fn get_generation(
    State(state): ExtractedLux3dState,
    AxumPath(id): AxumPath<String>,
) -> Response {
    match state.jobs.get(&id) {
        Some(record) => (StatusCode::OK, Json(record.object)).into_response(),
        None => not_found(format!("generation `{id}` not found")),
    }
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/v1/generations/{id}/content",
    responses(
        (status = 200, description = "Generated asset"),
        (status = 404, description = "Job not found"),
        (status = 409, description = "Job not completed")
    )
)]
pub async fn get_generation_content(
    State(state): ExtractedLux3dState,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let Some(record) = state.jobs.get(&id) else {
        return not_found(format!("generation `{id}` not found"));
    };

    if record.object.status != GenerationStatus::Completed.as_str() {
        return conflict(format!(
            "generation `{id}` is not completed yet (status={})",
            record.object.status
        ));
    }

    let Some(output_path) = record.output_path.as_deref() else {
        return conflict(format!("generation `{id}` has no output artifact"));
    };

    file_response(output_path, &record.content_type, &record.download_filename)
}

#[utoipa::path(
    post,
    tag = "Lux3D",
    path = "/v1/generations/{id}/cancel",
    responses(
        (status = 200, description = "Cancelled generation job", body = GenerationObject),
        (status = 404, description = "Job not found"),
        (status = 409, description = "Job cannot be cancelled")
    )
)]
pub async fn cancel_generation(
    State(state): ExtractedLux3dState,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let Some(record) = state.jobs.get(&id) else {
        return not_found(format!("generation `{id}` not found"));
    };

    if matches!(
        record.object.status.as_str(),
        "completed" | "failed" | "cancelled"
    ) {
        return conflict(format!(
            "generation `{id}` cannot be cancelled (status={})",
            record.object.status
        ));
    }

    if !state.jobs.request_cancel(&id).unwrap_or(false) {
        return conflict(format!("generation `{id}` cannot be cancelled"));
    }

    if let Err(error) = state.jobs.mark_cancelled(&id) {
        return internal_error(error.to_string());
    }

    match state.jobs.get(&id) {
        Some(record) => (StatusCode::OK, Json(record.object)).into_response(),
        None => not_found(format!("generation `{id}` not found")),
    }
}

#[utoipa::path(
    delete,
    tag = "Lux3D",
    path = "/v1/generations/{id}",
    responses(
        (status = 200, description = "Deleted generation job", body = GenerationObject),
        (status = 404, description = "Job not found")
    )
)]
pub async fn delete_generation(
    State(state): ExtractedLux3dState,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let Some(record) = state.jobs.remove(&id) else {
        return not_found(format!("generation `{id}` not found"));
    };
    let _ = std::fs::remove_dir_all(&record.workspace);
    (StatusCode::OK, Json(record.object)).into_response()
}

#[utoipa::path(
    post,
    tag = "Lux3D",
    path = "/v1/models/reload",
    request_body = ModelOperationRequest,
    responses((status = 200, description = "Model reload status", body = ModelStatusResponse))
)]
pub async fn reload_model(
    State(state): ExtractedLux3dState,
    Json(request): Json<ModelOperationRequest>,
) -> Json<ModelStatusResponse> {
    match state.models.reload(&state, &request.model_id) {
        Ok(status) => Json(ModelStatusResponse {
            model_id: request.model_id,
            status,
            error: None,
        }),
        Err(error) => Json(ModelStatusResponse {
            model_id: request.model_id,
            status: crate::models::ModelLoadStatus::InternalError,
            error: Some(error.to_string()),
        }),
    }
}

#[utoipa::path(
    post,
    tag = "Lux3D",
    path = "/v1/models/unload",
    request_body = ModelOperationRequest,
    responses((status = 200, description = "Model unload status", body = ModelStatusResponse))
)]
pub async fn unload_model(
    State(state): ExtractedLux3dState,
    Json(request): Json<ModelOperationRequest>,
) -> Json<ModelStatusResponse> {
    match state.models.unload(&request.model_id) {
        Ok(status) => Json(ModelStatusResponse {
            model_id: request.model_id,
            status,
            error: None,
        }),
        Err(error) => Json(ModelStatusResponse {
            model_id: request.model_id,
            status: crate::models::ModelLoadStatus::InternalError,
            error: Some(error.to_string()),
        }),
    }
}

#[utoipa::path(
    post,
    tag = "Lux3D",
    path = "/v1/models/status",
    request_body = ModelOperationRequest,
    responses((status = 200, description = "Model cache status", body = ModelStatusResponse))
)]
pub async fn model_status(
    State(state): ExtractedLux3dState,
    Json(request): Json<ModelOperationRequest>,
) -> Json<ModelStatusResponse> {
    Json(ModelStatusResponse {
        model_id: request.model_id.clone(),
        status: state.models.status(&request.model_id),
        error: None,
    })
}

#[utoipa::path(
    post,
    tag = "Lux3D",
    path = "/v1/point-clouds/generations",
    request_body(content = String, description = "multipart/form-data or JSON generation request"),
    responses(
        (status = 202, description = "Accepted generation job", body = GenerationObject),
        (status = 400, description = "Invalid request")
    )
)]
pub async fn create_point_cloud_generation(
    State(state): ExtractedLux3dState,
    request: Request,
) -> Response {
    if is_json_request(&request) {
        match axum::Json::<PointCloudGenerationRequest>::from_request(request, &state).await {
            Ok(Json(body)) => match create_point_cloud_generation_json(state, body).await {
                Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
                Err(error) => invalid_request(error.to_string()),
            },
            Err(rejection) => invalid_request(rejection.to_string()),
        }
    } else {
        match axum::extract::Multipart::from_request(request, &state).await {
            Ok(multipart) => {
                match create_point_cloud_generation_multipart(state, multipart).await {
                    Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
                    Err(error) => invalid_request(error.to_string()),
                }
            }
            Err(rejection) => invalid_request(rejection.to_string()),
        }
    }
}

#[utoipa::path(
    post,
    tag = "Lux3D",
    path = "/v1/meshes/generations",
    request_body(content = String, description = "multipart/form-data or JSON generation request"),
    responses(
        (status = 202, description = "Accepted generation job", body = GenerationObject),
        (status = 400, description = "Invalid request")
    )
)]
pub async fn create_mesh_generation(
    State(state): ExtractedLux3dState,
    request: Request,
) -> Response {
    if is_json_request(&request) {
        match axum::Json::<MeshGenerationRequest>::from_request(request, &state).await {
            Ok(Json(body)) => match create_mesh_generation_json(state, body).await {
                Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
                Err(error) => invalid_request(error.to_string()),
            },
            Err(rejection) => invalid_request(rejection.to_string()),
        }
    } else {
        match axum::extract::Multipart::from_request(request, &state).await {
            Ok(multipart) => match create_mesh_generation_multipart(state, multipart).await {
                Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
                Err(error) => invalid_request(error.to_string()),
            },
            Err(rejection) => invalid_request(rejection.to_string()),
        }
    }
}

#[utoipa::path(
    post,
    tag = "Lux3D",
    path = "/v1/meshes/texturings",
    request_body(content = String, description = "multipart/form-data or JSON texturing request"),
    responses(
        (status = 202, description = "Accepted texturing job", body = GenerationObject),
        (status = 400, description = "Invalid request")
    )
)]
pub async fn create_mesh_texturing(
    State(state): ExtractedLux3dState,
    request: Request,
) -> Response {
    if is_json_request(&request) {
        match axum::Json::<MeshTexturingRequest>::from_request(request, &state).await {
            Ok(Json(body)) => match create_mesh_texturing_json(state, body).await {
                Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
                Err(error) => invalid_request(error.to_string()),
            },
            Err(rejection) => invalid_request(rejection.to_string()),
        }
    } else {
        match axum::extract::Multipart::from_request(request, &state).await {
            Ok(multipart) => match create_mesh_texturing_multipart(state, multipart).await {
                Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
                Err(error) => invalid_request(error.to_string()),
            },
            Err(rejection) => invalid_request(rejection.to_string()),
        }
    }
}

async fn create_point_cloud_generation_multipart(
    state: std::sync::Arc<crate::types::Lux3dState>,
    multipart: axum::extract::Multipart,
) -> anyhow::Result<GenerationObject> {
    let job_id = new_job_id();
    let workspace = job_workspace(&state.upload_dir, &job_id);
    let (upload, options): (_, PointCloudGenerationOptions) =
        parse_generation_upload(multipart, &workspace).await?;
    let model = upload
        .model
        .context("missing required multipart field `model` (`pi3`, `pi3x` or `vggt`)")?;
    let format = point_cloud_output_format(&options)?;

    let record = create_job_record(NewJobRecord {
        job_id,
        model,
        output_format: format.extension().to_string(),
        workspace,
        content_type: format.content_type().to_string(),
        download_filename: format!("point-cloud.{}", format.extension()),
        ttl_secs: state.generation_defaults.job_ttl_secs,
        webhook_url: None,
    });

    spawn_point_cloud_generation(state, record, upload.source, upload.conditions, options).await
}

async fn create_point_cloud_generation_json(
    state: std::sync::Arc<crate::types::Lux3dState>,
    request: PointCloudGenerationRequest,
) -> anyhow::Result<GenerationObject> {
    let job_id = new_job_id();
    let workspace = job_workspace(&state.upload_dir, &job_id);
    let source = resolve_source_path(
        &state,
        &workspace,
        &MediaSourceInput {
            source_url: request.source_url,
            source_file_id: request.source_file_id,
            source_base64: request.source_base64,
            source_filename: request.source_filename,
            conditions_file_id: request.conditions_file_id.clone(),
        },
    )
    .await?;
    let conditions = resolve_conditions_path(&state, request.conditions_file_id.as_deref())?;
    let format = point_cloud_output_format(&request.options)?;

    let record = create_job_record(NewJobRecord {
        job_id,
        model: request.model,
        output_format: format.extension().to_string(),
        workspace,
        content_type: format.content_type().to_string(),
        download_filename: format!("point-cloud.{}", format.extension()),
        ttl_secs: state.generation_defaults.job_ttl_secs,
        webhook_url: request.webhook_url,
    });

    spawn_point_cloud_generation(state, record, source, conditions, request.options).await
}

/// Download name for mesh jobs: TripoSG/TRELLIS.2 export `<family>.<ext>`,
/// while the historical triposr default keeps its `mesh.<ext>` stem for old
/// clients.
fn mesh_download_filename(model: &str, format: MeshFormat) -> String {
    let stem = match model {
        "triposg" | "trellis2" => model,
        _ => "mesh",
    };
    format!("{stem}.{}", format.extension())
}

fn point_cloud_output_format(
    options: &PointCloudGenerationOptions,
) -> anyhow::Result<PointCloudFormat> {
    match options.output_format.as_deref() {
        Some(format) => PointCloudFormat::from_extension(format).map_err(anyhow::Error::from),
        None => Ok(PointCloudFormat::Ply),
    }
}

fn mesh_output_format(options: &MeshGenerationOptions) -> anyhow::Result<MeshFormat> {
    match options.output_format.as_deref() {
        Some(format) => MeshFormat::from_extension(format).map_err(anyhow::Error::from),
        None => Ok(MeshFormat::Glb),
    }
}

async fn create_mesh_generation_multipart(
    state: std::sync::Arc<crate::types::Lux3dState>,
    multipart: axum::extract::Multipart,
) -> anyhow::Result<GenerationObject> {
    let job_id = new_job_id();
    let workspace = job_workspace(&state.upload_dir, &job_id);
    let (upload, options): (_, MeshGenerationOptions) =
        parse_generation_upload(multipart, &workspace).await?;
    // Missing `model` keeps the documented `triposr` default; the download
    // name follows the requested model.
    let model = upload.model.unwrap_or_else(|| "triposr".to_string());
    let format = mesh_output_format(&options)?;
    let download_filename = mesh_download_filename(&model, format);

    let record = create_job_record(NewJobRecord {
        job_id,
        model,
        output_format: format.extension().to_string(),
        workspace,
        content_type: format.content_type().to_string(),
        download_filename,
        ttl_secs: state.generation_defaults.job_ttl_secs,
        webhook_url: None,
    });

    spawn_mesh_generation(state, record, upload.source, options).await
}

async fn create_mesh_generation_json(
    state: std::sync::Arc<crate::types::Lux3dState>,
    request: MeshGenerationRequest,
) -> anyhow::Result<GenerationObject> {
    let job_id = new_job_id();
    let workspace = job_workspace(&state.upload_dir, &job_id);
    let source = resolve_source_path(
        &state,
        &workspace,
        &MediaSourceInput {
            source_url: request.source_url,
            source_file_id: request.source_file_id,
            source_base64: request.source_base64,
            source_filename: request.source_filename,
            conditions_file_id: None,
        },
    )
    .await?;

    let format = mesh_output_format(&request.options)?;
    let download_filename = mesh_download_filename(&request.model, format);

    let record = create_job_record(NewJobRecord {
        job_id,
        model: request.model,
        output_format: format.extension().to_string(),
        workspace,
        content_type: format.content_type().to_string(),
        download_filename,
        ttl_secs: state.generation_defaults.job_ttl_secs,
        webhook_url: request.webhook_url,
    });

    spawn_mesh_generation(state, record, source, request.options).await
}

/// Texturing output format: the baked PBR atlas is embedded in the GLB, so
/// only `glb` (the default) is accepted.
fn texturing_output_format(options: &MeshTexturingOptions) -> anyhow::Result<MeshFormat> {
    match options.output_format.as_deref() {
        Some(format) => match MeshFormat::from_extension(format).map_err(anyhow::Error::from)? {
            MeshFormat::Glb => Ok(MeshFormat::Glb),
            other => anyhow::bail!(
                "`output_format` must be `glb` for mesh texturing (got `{}`); the baked \
                 PBR atlas is embedded in the GLB",
                other.extension()
            ),
        },
        None => Ok(MeshFormat::Glb),
    }
}

fn texturing_download_filename(format: MeshFormat) -> String {
    format!("trellis2-textured.{}", format.extension())
}

async fn create_mesh_texturing_multipart(
    state: std::sync::Arc<crate::types::Lux3dState>,
    multipart: axum::extract::Multipart,
) -> anyhow::Result<GenerationObject> {
    let job_id = new_job_id();
    let workspace = job_workspace(&state.upload_dir, &job_id);
    let (upload, options): (_, MeshTexturingOptions) =
        parse_texturing_upload(multipart, &workspace).await?;
    let model = upload.model.unwrap_or_else(|| "trellis2".to_string());
    let format = texturing_output_format(&options)?;

    let record = create_job_record(NewJobRecord {
        job_id,
        model,
        output_format: format.extension().to_string(),
        workspace,
        content_type: format.content_type().to_string(),
        download_filename: texturing_download_filename(format),
        ttl_secs: state.generation_defaults.job_ttl_secs,
        webhook_url: None,
    });

    spawn_mesh_texturing(state, record, upload.mesh, upload.image, options).await
}

async fn create_mesh_texturing_json(
    state: std::sync::Arc<crate::types::Lux3dState>,
    request: MeshTexturingRequest,
) -> anyhow::Result<GenerationObject> {
    let job_id = new_job_id();
    let workspace = job_workspace(&state.upload_dir, &job_id);
    let image = resolve_source_path(
        &state,
        &workspace,
        &MediaSourceInput {
            source_url: request.source_url,
            source_file_id: request.source_file_id,
            source_base64: request.source_base64,
            source_filename: request.source_filename,
            conditions_file_id: None,
        },
    )
    .await?;
    let mesh = resolve_mesh_path(
        &state,
        &workspace,
        &MeshSourceInput {
            mesh_url: request.mesh_url,
            mesh_file_id: request.mesh_file_id,
            mesh_base64: request.mesh_base64,
            mesh_filename: request.mesh_filename,
        },
    )
    .await?;
    let format = texturing_output_format(&request.options)?;

    let record = create_job_record(NewJobRecord {
        job_id,
        model: request.model,
        output_format: format.extension().to_string(),
        workspace,
        content_type: format.content_type().to_string(),
        download_filename: texturing_download_filename(format),
        ttl_secs: state.generation_defaults.job_ttl_secs,
        webhook_url: request.webhook_url,
    });

    spawn_mesh_texturing(state, record, mesh, image, request.options).await
}

fn is_json_request(request: &Request<Body>) -> bool {
    request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/json"))
}

fn file_response(path: &Path, content_type: &str, filename: &str) -> Response {
    match std::fs::read(path) {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, content_type),
                (
                    header::CONTENT_DISPOSITION,
                    &format!("attachment; filename=\"{filename}\""),
                ),
            ],
            bytes,
        )
            .into_response(),
        Err(error) => internal_error(format!(
            "failed to read generated asset at `{}`: {error}",
            path.display()
        )),
    }
}

pub fn error_response(status: StatusCode, error: anyhow::Error) -> Response {
    let body = Json(ErrorResponse {
        error: ErrorBody {
            message: error.to_string(),
        },
    });
    (status, body).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_jobs_default_to_glb_and_accept_explicit_formats() {
        let default = MeshGenerationOptions::default();
        assert_eq!(mesh_output_format(&default).unwrap(), MeshFormat::Glb);
        assert_eq!(
            mesh_download_filename("triposg", MeshFormat::Glb),
            "triposg.glb"
        );
        assert_eq!(
            mesh_download_filename("trellis2", MeshFormat::Glb),
            "trellis2.glb"
        );
        assert_eq!(
            mesh_download_filename("triposr", MeshFormat::Glb),
            "mesh.glb"
        );

        let obj = MeshGenerationOptions {
            output_format: Some("obj".to_string()),
            ..MeshGenerationOptions::default()
        };
        assert_eq!(mesh_output_format(&obj).unwrap(), MeshFormat::Obj);
        assert_eq!(
            mesh_download_filename("triposr", MeshFormat::Obj),
            "mesh.obj"
        );

        let invalid = MeshGenerationOptions {
            output_format: Some("fbx".to_string()),
            ..MeshGenerationOptions::default()
        };
        assert!(mesh_output_format(&invalid).is_err());
    }

    #[test]
    fn point_cloud_jobs_default_to_ply_and_accept_xyz() {
        let default = PointCloudGenerationOptions::default();
        assert_eq!(
            point_cloud_output_format(&default).unwrap(),
            PointCloudFormat::Ply
        );
        assert_eq!(PointCloudFormat::Ply.content_type(), "model/ply");

        let xyz = PointCloudGenerationOptions {
            output_format: Some("XYZ".to_string()),
            ..PointCloudGenerationOptions::default()
        };
        assert_eq!(
            point_cloud_output_format(&xyz).unwrap(),
            PointCloudFormat::Xyz
        );
        assert_eq!(
            PointCloudFormat::Xyz.content_type(),
            "text/plain; charset=utf-8"
        );

        let invalid = PointCloudGenerationOptions {
            output_format: Some("pcd".to_string()),
            ..PointCloudGenerationOptions::default()
        };
        assert!(point_cloud_output_format(&invalid).is_err());
    }

    #[test]
    fn texturing_jobs_are_glb_only() {
        let default = MeshTexturingOptions::default();
        assert_eq!(texturing_output_format(&default).unwrap(), MeshFormat::Glb);
        assert_eq!(
            texturing_download_filename(MeshFormat::Glb),
            "trellis2-textured.glb"
        );

        let explicit = MeshTexturingOptions {
            output_format: Some("GLB".to_string()),
            ..MeshTexturingOptions::default()
        };
        assert_eq!(texturing_output_format(&explicit).unwrap(), MeshFormat::Glb);

        for format in ["obj", "ply", "stl"] {
            let invalid = MeshTexturingOptions {
                output_format: Some(format.to_string()),
                ..MeshTexturingOptions::default()
            };
            let error = texturing_output_format(&invalid)
                .expect_err("non-GLB texturing output must be rejected");
            assert!(
                error.to_string().contains("must be `glb`"),
                "unexpected message for {format}: `{error}`"
            );
        }
    }
}
