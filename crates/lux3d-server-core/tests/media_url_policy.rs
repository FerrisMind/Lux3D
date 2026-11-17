//! Outbound media URL policy contract tests.
//!
//! These tests exercise the real reqwest client configured by
//! [`Lux3dForServerBuilder`] against a local Axum server: the literal-IP
//! filter, the redirect hop limit and the streaming byte cap. They are
//! hermetic (loopback only) and explicitly opt into private URLs except for
//! the test that asserts the production default.

use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use lux3d_server_core::{
    DeviceBackend, MediaFetchConfig, ObservabilityConfig,
    lux3d_for_server_builder::Lux3dForServerBuilder,
    media_source::{MediaSourceInput, resolve_source_path},
};

struct MediaServer {
    base_url: String,
    root_dir: PathBuf,
    loop_hits: Arc<AtomicUsize>,
    handle: tokio::task::JoinHandle<()>,
}

impl MediaServer {
    async fn start() -> Self {
        let root_dir = std::env::temp_dir().join(format!(
            "lux3d-media-policy-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let loop_hits = Arc::new(AtomicUsize::new(0));

        let app = Router::new()
            .route("/ok", get(|| async { "ok" }))
            .route("/loop", get(redirect_loop))
            .route("/big", get(big_stream))
            .route("/small", get(small_stream))
            .with_state(loop_hits.clone());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let address = listener.local_addr().expect("local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        Self {
            base_url: format!("http://{address}"),
            root_dir,
            loop_hits,
            handle,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }
}

impl Drop for MediaServer {
    fn drop(&mut self) {
        self.handle.abort();
        let _ = std::fs::remove_dir_all(&self.root_dir);
    }
}

async fn redirect_loop(State(hits): State<Arc<AtomicUsize>>) -> Response {
    hits.fetch_add(1, Ordering::SeqCst);
    (StatusCode::FOUND, [(header::LOCATION, "/loop")]).into_response()
}

/// Chunked responses with no `Content-Length`, so the cap must be enforced
/// while reading the stream.
async fn big_stream() -> Response {
    stream_response(4, 512)
}

async fn small_stream() -> Response {
    stream_response(2, 128)
}

fn stream_response(chunks: usize, chunk_len: usize) -> Response {
    let body = (0..chunks)
        .map(|_| Ok::<Bytes, std::io::Error>(Bytes::from(vec![7_u8; chunk_len])))
        .collect::<Vec<_>>();
    Response::builder()
        .status(StatusCode::OK)
        .body(Body::from_stream(futures::stream::iter(body)))
        .expect("stream body response")
}

fn state(root: &Path, media_fetch: MediaFetchConfig) -> lux3d_server_core::SharedLux3dState {
    let observability = ObservabilityConfig {
        access_log: false,
        access_log_health: false,
        access_log_format: Default::default(),
        request_id_header: false,
        metrics: false,
    };
    Lux3dForServerBuilder::new()
        .with_upload_dir(root.join("uploads"))
        .with_model_path(root.join("missing-model"))
        .with_device_backend(DeviceBackend::Cpu)
        .with_observability(observability)
        .with_media_fetch_config(media_fetch)
        .build()
        .expect("build lux3d state")
}

fn source_input(url: String) -> MediaSourceInput {
    MediaSourceInput {
        source_url: Some(url),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn private_literal_url_is_rejected_by_default() {
    let server = MediaServer::start().await;
    let state = state(&server.root_dir, MediaFetchConfig::default());

    let error = resolve_source_path(
        &state,
        &server.root_dir.join("workspace"),
        &source_input(server.url("/ok")),
    )
    .await
    .expect_err("loopback literal must be rejected without the opt-in");
    let message = error.to_string();
    assert!(
        message.contains("not a public address"),
        "unexpected message: `{message}`"
    );
    assert!(
        message.contains("--allow-private-urls"),
        "the error must point at the opt-in: `{message}`"
    );
    assert_eq!(
        0,
        server.loop_hits.load(Ordering::SeqCst),
        "no request may reach the server"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn private_literal_url_is_allowed_with_opt_in() {
    let server = MediaServer::start().await;
    let mut media_fetch = MediaFetchConfig::default();
    media_fetch.allow_private_urls = true;
    let state = state(&server.root_dir, media_fetch);

    let workspace = server.root_dir.join("workspace-ok");
    let path = resolve_source_path(&state, &workspace, &source_input(server.url("/ok")))
        .await
        .expect("the opt-in allows loopback media URLs");
    assert_eq!(
        "ok",
        std::fs::read_to_string(&path).expect("downloaded body")
    );
    assert_eq!(path, workspace.join("ok"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn redirect_chain_stops_after_the_hop_limit() {
    let server = MediaServer::start().await;
    let mut media_fetch = MediaFetchConfig::default();
    media_fetch.allow_private_urls = true;
    let state = state(&server.root_dir, media_fetch);

    let error = resolve_source_path(
        &state,
        &server.root_dir.join("workspace-loop"),
        &source_input(server.url("/loop")),
    )
    .await
    .expect_err("an endless redirect chain must fail");
    // The redirect-policy error is the source of the request context.
    let message = format!("{error:#}");
    assert!(
        message.contains("too many redirects"),
        "unexpected message: `{message}`"
    );
    assert_eq!(
        6,
        server.loop_hits.load(Ordering::SeqCst),
        "one initial request plus five followed redirects must reach the server"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_body_cap_rejects_oversized_responses() {
    let server = MediaServer::start().await;
    let mut media_fetch = MediaFetchConfig::default();
    media_fetch.allow_private_urls = true;
    media_fetch.max_download_bytes = 1024;
    let state = state(&server.root_dir, media_fetch);

    let workspace = server.root_dir.join("workspace-big");
    let error = resolve_source_path(&state, &workspace, &source_input(server.url("/big")))
        .await
        .expect_err("a body over the cap must fail even without Content-Length");
    let message = error.to_string();
    assert!(
        message.contains("download cap"),
        "unexpected message: `{message}`"
    );
    assert!(
        !workspace.join("big").exists(),
        "a rejected download must not leave a partial file"
    );

    // The same server route below the cap succeeds, so the limit is not
    // rejecting chunked bodies unconditionally.
    let workspace = server.root_dir.join("workspace-small");
    let path = resolve_source_path(&state, &workspace, &source_input(server.url("/small")))
        .await
        .expect("a body below the cap must be accepted");
    assert_eq!(
        256,
        std::fs::metadata(&path).expect("downloaded file").len()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn declared_content_length_over_the_cap_is_rejected_early() {
    let server = MediaServer::start().await;
    let mut media_fetch = MediaFetchConfig::default();
    media_fetch.allow_private_urls = true;
    media_fetch.max_download_bytes = 1024;
    let state = state(&server.root_dir, media_fetch);

    // `/big-length` returns a single 2048-byte body with Content-Length.
    let app = Router::new().route("/big-length", get(|| async { vec![0_u8; 2048] }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind content-length port");
    let address: SocketAddr = listener.local_addr().expect("local address");
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let error = resolve_source_path(
        &state,
        &server.root_dir.join("workspace-length"),
        &source_input(format!("http://{address}/big-length")),
    )
    .await
    .expect_err("declared oversized body must fail");
    assert!(
        error.to_string().contains("declares 2048 bytes"),
        "unexpected message: `{error}`"
    );
    handle.abort();
}
