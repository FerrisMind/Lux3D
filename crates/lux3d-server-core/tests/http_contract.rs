//! Router-level HTTP contract tests for the generation and texturing surfaces.
//!
//! Each test starts a real Axum server on an ephemeral loopback port against a
//! CPU state whose model directory is deliberately missing. Requests that are
//! accepted create jobs that fail asynchronously; these tests assert only the
//! synchronous HTTP contract (status codes and validation messages).

use std::path::PathBuf;

use lux3d_server_core::{
    DeviceBackend, ObservabilityConfig, lux3d_for_server_builder::Lux3dForServerBuilder,
    lux3d_server_router_builder::Lux3dServerRouterBuilder,
};
use reqwest::{Response, header::CONTENT_TYPE};
use serde_json::{Value, json};

const BOUNDARY: &str = "lux3d-test-boundary";

struct TestServer {
    base_url: String,
    root_dir: PathBuf,
    jobs_dir: PathBuf,
    handle: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn start() -> Self {
        let root_dir = std::env::temp_dir().join(format!(
            "lux3d-server-http-contract-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let upload_dir = root_dir.join("uploads");
        std::fs::create_dir_all(&upload_dir).expect("create upload directory");

        let observability = ObservabilityConfig {
            access_log: false,
            access_log_health: false,
            access_log_format: Default::default(),
            request_id_header: false,
            metrics: false,
        };
        let state = Lux3dForServerBuilder::new()
            .with_upload_dir(upload_dir.clone())
            .with_model_path(root_dir.join("missing-model"))
            .with_device_backend(DeviceBackend::Cpu)
            .with_observability(observability)
            .build()
            .expect("build lux3d state");
        let router = Lux3dServerRouterBuilder::new()
            .with_lux3d(state)
            .with_spawn_background_tasks(false)
            .build()
            .await
            .expect("build lux3d router");

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let address = listener.local_addr().expect("local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        Self {
            base_url: format!("http://{address}"),
            jobs_dir: upload_dir.join("jobs"),
            root_dir,
            handle,
        }
    }

    async fn post_json(&self, path: &str, body: &Value) -> Response {
        reqwest::Client::new()
            .post(format!("{}{path}", self.base_url))
            .json(body)
            .send()
            .await
            .expect("send JSON request")
    }

    async fn post_multipart(&self, path: &str, body: Vec<u8>) -> Response {
        reqwest::Client::new()
            .post(format!("{}{path}", self.base_url))
            .header(
                CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(body)
            .send()
            .await
            .expect("send multipart request")
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
        let _ = std::fs::remove_dir_all(&self.root_dir);
    }
}

async fn error_message(response: Response) -> String {
    let body: Value = response.json().await.expect("error response is JSON");
    body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

fn multipart_texturing_body(image: &[u8], mesh_filename: &str, mesh: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, filename, bytes) in [
        ("source", "source.png", image),
        ("mesh", mesh_filename, mesh),
    ] {
        body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n\
                 Content-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\ntrellis2\r\n");
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    body
}

fn texturing_json(mesh_filename: Option<&str>) -> Value {
    let mut body = json!({
        "model": "trellis2",
        "source_base64": "aW1hZ2U=",
        "mesh_base64": "bWVzaA==",
    });
    if let Some(filename) = mesh_filename {
        body["mesh_filename"] = json!(filename);
    }
    body
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn texturing_route_accepts_json_and_multipart_requests() {
    let server = TestServer::start().await;

    let response = server
        .post_json("/v1/meshes/texturings", &texturing_json(Some("mesh.glb")))
        .await;
    assert_eq!(
        202,
        response.status().as_u16(),
        "JSON texturing request must create a job"
    );
    let body: Value = response.json().await.expect("job JSON");
    assert_eq!("generation", body["object"]);
    assert!(
        body["id"].as_str().is_some_and(|id| id.starts_with("gen_")),
        "unexpected job body: {body}"
    );

    let response = server
        .post_multipart(
            "/v1/meshes/texturings",
            multipart_texturing_body(b"image", "mesh.glb", b"mesh"),
        )
        .await;
    assert_eq!(
        202,
        response.status().as_u16(),
        "multipart texturing request must create a job"
    );
    let body: Value = response.json().await.expect("job JSON");
    assert_eq!("generation", body["object"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn texturing_route_rejects_invalid_model_and_format_before_media_io() {
    let server = TestServer::start().await;

    let invalid_model = json!({
        "model": "triposr",
        "source_base64": "aW1hZ2U=",
        "mesh_base64": "bWVzaA==",
        "mesh_filename": "mesh.glb",
    });
    let response = server
        .post_json("/v1/meshes/texturings", &invalid_model)
        .await;
    assert_eq!(400, response.status().as_u16());
    let message = error_message(response).await;
    assert!(
        message.contains("unsupported texturing model `triposr`"),
        "unexpected message: `{message}`"
    );

    let invalid_format = json!({
        "model": "trellis2",
        "source_base64": "aW1hZ2U=",
        "mesh_base64": "bWVzaA==",
        "mesh_filename": "mesh.glb",
        "options": {"output_format": "obj"},
    });
    let response = server
        .post_json("/v1/meshes/texturings", &invalid_format)
        .await;
    assert_eq!(400, response.status().as_u16());
    let message = error_message(response).await;
    assert!(
        message.contains("must be `glb`"),
        "unexpected message: `{message}`"
    );

    assert!(
        !server.jobs_dir.exists(),
        "invalid texturing requests must not create a job workspace"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn texturing_route_rejects_unreadable_mesh_inputs_with_400() {
    let server = TestServer::start().await;

    // `mesh_base64` without a filename cannot be dispatched to a reader.
    let response = server
        .post_json("/v1/meshes/texturings", &texturing_json(None))
        .await;
    assert_eq!(400, response.status().as_u16());
    let message = error_message(response).await;
    assert!(
        message.contains("requires `mesh_filename`"),
        "unexpected message: `{message}`"
    );

    // An extensionless URL must be rejected before any outbound fetch (the
    // port is a discard port; a fetch attempt would surface a transport
    // error instead of the extension message).
    let extensionless_url = json!({
        "model": "trellis2",
        "source_base64": "aW1hZ2U=",
        "mesh_url": "http://127.0.0.1:9/mesh",
    });
    let response = server
        .post_json("/v1/meshes/texturings", &extensionless_url)
        .await;
    assert_eq!(400, response.status().as_u16());
    let message = error_message(response).await;
    assert!(
        message.contains("`mesh_url` path"),
        "unexpected message: `{message}`"
    );

    // A known field name with an unsupported extension is also rejected.
    let response = server
        .post_json("/v1/meshes/texturings", &texturing_json(Some("mesh.bin")))
        .await;
    assert_eq!(400, response.status().as_u16());
    let message = error_message(response).await;
    assert!(
        message.contains("`mesh_filename` `mesh.bin`"),
        "unexpected message: `{message}`"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mesh_generation_route_enforces_request_caps() {
    let server = TestServer::start().await;

    // Over-limit requests fail with 400 before media resolution.
    let over_limit_cases = [
        (
            "trellis2 num_samples",
            json!({"model": "trellis2", "options": {"num_samples": 65}}),
            "`num_samples` must be at most 64",
        ),
        (
            "trellis2 ss_steps",
            json!({"model": "trellis2", "options": {"ss_steps": 1001}}),
            "`ss_steps` must be at most 1000",
        ),
        (
            "trellis2 shape_steps",
            json!({"model": "trellis2", "options": {"shape_steps": 1001}}),
            "`shape_steps` must be at most 1000",
        ),
        (
            "trellis2 tex_steps",
            json!({"model": "trellis2", "options": {"tex_steps": 1001}}),
            "`tex_steps` must be at most 1000",
        ),
        (
            "triposg steps",
            json!({"model": "triposg", "options": {"steps": 1001}}),
            "`steps` must be at most 1000",
        ),
        (
            "triposg timesteps",
            json!({
                "model": "triposg",
                "options": {"timesteps": vec![900.0_f32; 1001]},
            }),
            "`timesteps` must contain at most 1000 entries",
        ),
    ];
    for (name, body, expected) in over_limit_cases {
        let response = server.post_json("/v1/meshes/generations", &body).await;
        assert_eq!(400, response.status().as_u16(), "{name} must be rejected");
        let message = error_message(response).await;
        assert!(
            message.contains(expected),
            "unexpected message for {name}: `{message}`"
        );
    }

    // Boundary values are accepted (202) and create jobs.
    let boundary_cases = [
        (
            "trellis2 num_samples",
            json!({"model": "trellis2", "options": {"num_samples": 64}}),
        ),
        (
            "trellis2 steps",
            json!({"model": "trellis2", "options": {"steps": 1000}}),
        ),
        (
            "trellis2 ss_steps",
            json!({"model": "trellis2", "options": {"ss_steps": 1000}}),
        ),
        (
            "triposg steps",
            json!({"model": "triposg", "options": {"steps": 1000}}),
        ),
        (
            "triposg timesteps",
            json!({
                "model": "triposg",
                "options": {"timesteps": vec![900.0_f32; 1000]},
            }),
        ),
    ];
    for (name, mut body) in boundary_cases {
        body["source_base64"] = json!("aW1hZ2U=");
        let response = server.post_json("/v1/meshes/generations", &body).await;
        assert_eq!(
            202,
            response.status().as_u16(),
            "{name} at the cap must be accepted"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn texturing_route_enforces_tex_steps_cap() {
    let server = TestServer::start().await;

    let over_limit = json!({
        "model": "trellis2",
        "source_base64": "aW1hZ2U=",
        "mesh_base64": "bWVzaA==",
        "mesh_filename": "mesh.glb",
        "options": {"tex_steps": 1001},
    });
    let response = server.post_json("/v1/meshes/texturings", &over_limit).await;
    assert_eq!(400, response.status().as_u16());
    let message = error_message(response).await;
    assert!(
        message.contains("`tex_steps` must be at most 1000"),
        "unexpected message: `{message}`"
    );

    let at_limit = json!({
        "model": "trellis2",
        "source_base64": "aW1hZ2U=",
        "mesh_base64": "bWVzaA==",
        "mesh_filename": "mesh.glb",
        "options": {"tex_steps": 1000},
    });
    let response = server.post_json("/v1/meshes/texturings", &at_limit).await;
    assert_eq!(202, response.status().as_u16());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn triposr_route_rejects_remesh_options() {
    let server = TestServer::start().await;

    for options in [
        json!({"remesh": false}),
        json!({"remesh_band": 2.0}),
        json!({"remesh_project": 0.5}),
    ] {
        let body = json!({
            "model": "triposr",
            "source_base64": "aW1hZ2U=",
            "options": options,
        });
        let response = server.post_json("/v1/meshes/generations", &body).await;
        assert_eq!(400, response.status().as_u16());
        let message = error_message(response).await;
        assert!(
            message.contains("is not supported for `triposr`"),
            "unexpected message: `{message}`"
        );
    }
}
