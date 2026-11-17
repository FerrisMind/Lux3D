//! OpenAPI document generation for the Lux3D server.

use utoipa::OpenApi;

use crate::{
    api_types::{
        ErrorBody, ErrorResponse, GenerationError, GenerationListResponse, GenerationObject,
        MeshGenerationOptions, MeshGenerationRequest, MeshTexturingOptions, MeshTexturingRequest,
        ModelListResponse, ModelObject, PointCloudGenerationOptions, PointCloudGenerationRequest,
        RouteSummary, ServerInfoResponse,
    },
    errors::{ApiErrorBody, ApiErrorResponse},
    files::{FileListResponse, FileObject},
    handlers,
    models::{ModelOperationRequest, ModelStatusResponse},
    system::{DoctorCheck, DoctorReport, SystemInfoResponse},
};

#[derive(OpenApi)]
#[openapi(
    paths(
        handlers::root,
        handlers::health,
        handlers::models,
        handlers::reload_model,
        handlers::unload_model,
        handlers::model_status,
        handlers::list_generations,
        handlers::get_generation,
        handlers::cancel_generation,
        handlers::delete_generation,
        handlers::get_generation_content,
        handlers::create_point_cloud_generation,
        handlers::create_mesh_generation,
        handlers::create_mesh_texturing,
        crate::files::upload_file,
        crate::files::list_files,
        crate::files::get_file,
        crate::files::get_file_content,
        crate::files::delete_file,
        crate::stream::stream_generation,
        crate::system::system_info,
        crate::system::system_doctor,
    ),
    components(schemas(
        ServerInfoResponse,
        RouteSummary,
        ModelListResponse,
        ModelObject,
        ModelOperationRequest,
        ModelStatusResponse,
        GenerationListResponse,
        GenerationObject,
        GenerationError,
        PointCloudGenerationOptions,
        PointCloudGenerationRequest,
        MeshGenerationOptions,
        MeshGenerationRequest,
        MeshTexturingOptions,
        MeshTexturingRequest,
        FileListResponse,
        FileObject,
        SystemInfoResponse,
        DoctorReport,
        DoctorCheck,
        ErrorResponse,
        ErrorBody,
        ApiErrorResponse,
        ApiErrorBody,
    )),
    tags(
        (name = "Lux3D", description = "OpenAI-style 3D generation server endpoints")
    ),
    info(
        title = "Lux3D Server API",
        version = "0.1.0",
        description = "Async HTTP API for Pi3/Pi3X/VGGT point clouds and TripoSR/TripoSG/TRELLIS.2 \
                       meshes, including the TRELLIS.2 mesh-texturing pipeline."
    )
)]
struct ApiDoc;

pub fn get_openapi_doc(base_path: Option<&str>) -> utoipa::openapi::OpenApi {
    let mut doc = ApiDoc::openapi();
    if let Some(prefix) = base_path.filter(|path| !path.is_empty()) {
        doc.servers = Some(vec![utoipa::openapi::Server::new(prefix.to_string())]);
    }
    doc
}

#[cfg(test)]
mod tests {
    use super::get_openapi_doc;

    #[test]
    fn description_names_every_mesh_family() {
        let doc = get_openapi_doc(None);
        let description = doc
            .info
            .description
            .as_deref()
            .expect("openapi info description");
        assert!(
            description.contains("TripoSR"),
            "description: {description}"
        );
        assert!(
            description.contains("TripoSG"),
            "description: {description}"
        );
        assert!(
            description.contains("TRELLIS.2"),
            "description: {description}"
        );
    }

    #[test]
    fn texturing_path_and_schemas_are_registered() {
        let doc = get_openapi_doc(None);
        assert!(
            doc.paths.paths.contains_key("/v1/meshes/texturings"),
            "texturing path missing from the OpenAPI document"
        );
        let components = doc.components.expect("openapi components");
        for schema in ["MeshTexturingRequest", "MeshTexturingOptions"] {
            assert!(
                components.schemas.contains_key(schema),
                "schema `{schema}` missing from the OpenAPI document"
            );
        }
    }

    #[test]
    fn mesh_option_schemas_document_request_caps() {
        let json = serde_json::to_string(&get_openapi_doc(None)).expect("serialize openapi");
        for cap in ["At most 64", "At most 1000"] {
            assert!(
                json.contains(cap),
                "request cap description `{cap}` missing from the OpenAPI document"
            );
        }
    }

    #[test]
    fn mesh_option_schemas_document_the_new_fields() {
        let json = serde_json::to_string(&get_openapi_doc(None)).expect("serialize openapi");
        for field in [
            "timesteps",
            "bounds",
            "faces",
            "ss_steps",
            "ss_guidance_strength",
            "ss_guidance_interval",
            "shape_steps",
            "shape_rescale_t",
            "tex_steps",
            "tex_guidance_rescale",
            "num_samples",
            "texture_size",
            "postprocess",
            "decimation_target",
            "fill_hole_perimeter",
            "min_component_area",
            "remesh",
            "remesh_band",
            "remesh_project",
            "texture_resolution",
            "tex_rescale_t",
        ] {
            assert!(
                json.contains(&format!("\"{field}\"")),
                "field `{field}` missing from the OpenAPI document"
            );
        }
    }
}
