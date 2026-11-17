use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;
use lux3d_server_core::{
    config::{AccessLogFormat, ObservabilityConfig},
    lux3d_for_server_builder::Lux3dForServerBuilder,
    lux3d_server_router_builder::Lux3dServerRouterBuilder,
};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "lux3d-server",
    about = "HTTP server for Lux3D Pi3, Pi3X, TripoSR, VGGT, and TripoSG reconstruction."
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    #[arg(long, env = "LUX3D_MODEL_PATH")]
    model_path: Option<PathBuf>,

    #[arg(long, env = "LUX3D_CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    #[arg(long, env = "LUX3D_RMBG_MODEL_PATH")]
    rmbg_model_path: Option<PathBuf>,

    #[arg(long, env = "LUX3D_TRELLIS2_RMBG_MODEL_PATH")]
    trellis2_rmbg_model_path: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    cuda_device: usize,

    #[arg(long)]
    upload_dir: Option<PathBuf>,

    /// Allow `source_url` / `mesh_url` fetches to non-public addresses
    /// (loopback, RFC1918, link-local, ULA, ...) for LAN deployments. Only the
    /// address filter is disabled: http/https, credential, redirect and
    /// timeout/size rules still apply. Equivalent to
    /// `LUX3D_ALLOW_PRIVATE_URLS=1`.
    #[arg(
        long,
        env = "LUX3D_ALLOW_PRIVATE_URLS",
        action = clap::ArgAction::Set,
        value_parser = clap::builder::BoolishValueParser::new(),
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = false
    )]
    allow_private_urls: bool,

    #[arg(long)]
    cors_origin: Vec<String>,

    #[arg(long, default_value_t = 86_400)]
    job_ttl_secs: u64,

    #[arg(long, default_value_t = 300)]
    cleanup_interval_secs: u64,

    #[arg(long, default_value_t = true)]
    access_log: bool,

    #[arg(long, default_value_t = false)]
    access_log_health: bool,

    #[arg(long, value_enum, default_value_t = CliAccessLogFormat::Text)]
    access_log_format: CliAccessLogFormat,

    #[arg(long, default_value_t = true)]
    request_id_header: bool,

    #[arg(long, default_value_t = true)]
    metrics: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CliAccessLogFormat {
    Text,
    Json,
}

impl From<CliAccessLogFormat> for AccessLogFormat {
    fn from(value: CliAccessLogFormat) -> Self {
        match value {
            CliAccessLogFormat::Text => Self::Text,
            CliAccessLogFormat::Json => Self::Json,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // One server process serves every family, but candle's GEMM switches are
    // process-wide (vulkan reads them once at the first matmul, wgpu at device
    // creation). Reduced precision cannot be enabled per request without
    // changing TRELLIS.2's output, so the server keeps the exact-fp32 default;
    // operators who want the validated pi3/pi3x fast paths set
    // CANDLE_VULKAN_F32_UNALIGNED_COOPMAT=1 and/or CANDLE_WGPU_COOP_MATMUL=1
    // explicitly for a dedicated deployment.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("lux3d_server=info".parse()?))
        .init();

    let args = Args::parse();
    let addr = SocketAddr::new(
        args.host
            .parse()
            .map_err(|error| anyhow::anyhow!("invalid host `{host}`: {error}", host = args.host))?,
        args.port,
    );

    let observability = ObservabilityConfig {
        access_log: args.access_log,
        access_log_health: args.access_log_health,
        access_log_format: args.access_log_format.into(),
        request_id_header: args.request_id_header,
        metrics: args.metrics,
    };

    let lux3d = Lux3dForServerBuilder::new()
        .with_model_path_optional(args.model_path)
        .with_cache_dir_optional(args.cache_dir)
        .with_rmbg_model_path_optional(args.rmbg_model_path)
        .with_trellis2_rmbg_model_path_optional(args.trellis2_rmbg_model_path)
        .with_cuda_device(args.cuda_device)
        .with_upload_dir_optional(args.upload_dir)
        .with_allow_private_urls(args.allow_private_urls)
        .with_job_ttl_secs(args.job_ttl_secs)
        .with_cleanup_interval_secs(args.cleanup_interval_secs)
        .with_observability(observability)
        .build()?;

    let mut router_builder = Lux3dServerRouterBuilder::new().with_lux3d(lux3d);
    if !args.cors_origin.is_empty() {
        router_builder = router_builder.with_allowed_origins(args.cors_origin);
    }
    let router = router_builder.build().await?;

    let reduced_precision = [
        "CANDLE_VULKAN_F32_UNALIGNED_COOPMAT",
        "CANDLE_WGPU_COOP_MATMUL",
    ]
    .iter()
    .any(|key| std::env::var_os(key).is_some());
    tracing::info!(
        "GEMM precision policy: {}",
        if reduced_precision {
            "reduced-precision opt-in enabled by environment (pi3/pi3x fast paths; TRELLIS.2 output may change)"
        } else {
            "exact fp32 (server default; enable CANDLE_VULKAN_F32_UNALIGNED_COOPMAT \
             or CANDLE_WGPU_COOP_MATMUL for the validated pi3/pi3x fast paths)"
        }
    );
    tracing::info!("Lux3D server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}
