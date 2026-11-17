//! Server-level configuration and defaults.

use std::time::Duration;

/// Default TTL for generation jobs and uploaded files (24 hours).
pub const DEFAULT_JOB_TTL_SECS: u64 = 86_400;

/// Default interval for background cleanup tasks.
pub const DEFAULT_CLEANUP_INTERVAL_SECS: u64 = 300;

/// Default maximum concurrent GPU inference jobs.
pub const DEFAULT_MAX_CONCURRENT_JOBS: usize = 1;

/// Default maximum entries returned by list endpoints.
pub const DEFAULT_LIST_LIMIT: usize = 20;

/// Default maximum upload size for the Files API (64 MiB).
pub const DEFAULT_MAX_FILE_BYTES: usize = 64 * 1024 * 1024;

/// Default connect timeout for outbound media URL fetches (10 s).
pub const DEFAULT_MEDIA_FETCH_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Default total timeout for an outbound media URL fetch (30 s), including
/// reading the response body.
pub const DEFAULT_MEDIA_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Default streaming byte cap for an outbound media URL fetch (64 MiB).
pub const DEFAULT_MEDIA_FETCH_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of redirects followed for an outbound media URL fetch.
pub const MEDIA_FETCH_MAX_REDIRECTS: usize = 5;

/// Limits and opt-ins for outbound `source_url` / `mesh_url` fetches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaFetchConfig {
    /// Disable only the non-public address filter (server
    /// `--allow-private-urls` / `LUX3D_ALLOW_PRIVATE_URLS=1`) so LAN hosts can
    /// be fetched. Scheme, credential, redirect, timeout and size rules keep
    /// applying.
    pub allow_private_urls: bool,
    /// Timeout for establishing the TCP/TLS connection.
    pub connect_timeout: Duration,
    /// Total request timeout, including body reads.
    pub request_timeout: Duration,
    /// Hard cap enforced while streaming the response body. `Content-Length`
    /// is only used as an early check; the cap is always enforced on bytes
    /// actually read.
    pub max_download_bytes: usize,
}

impl Default for MediaFetchConfig {
    fn default() -> Self {
        Self {
            allow_private_urls: false,
            connect_timeout: DEFAULT_MEDIA_FETCH_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_MEDIA_FETCH_TIMEOUT,
            max_download_bytes: DEFAULT_MEDIA_FETCH_MAX_BYTES,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationDefaults {
    pub job_ttl_secs: u64,
    pub max_concurrent_jobs: usize,
    pub cleanup_interval: Duration,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            job_ttl_secs: DEFAULT_JOB_TTL_SECS,
            max_concurrent_jobs: DEFAULT_MAX_CONCURRENT_JOBS,
            cleanup_interval: Duration::from_secs(DEFAULT_CLEANUP_INTERVAL_SECS),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AccessLogFormat {
    #[default]
    Text,
    Json,
}

#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub access_log: bool,
    pub access_log_health: bool,
    pub access_log_format: AccessLogFormat,
    pub request_id_header: bool,
    pub metrics: bool,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            access_log: true,
            access_log_health: false,
            access_log_format: AccessLogFormat::Text,
            request_id_header: true,
            metrics: true,
        }
    }
}
