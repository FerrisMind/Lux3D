//! Resolve generation sources from URLs, file IDs, base64 payloads, or local paths.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use url::Url;

use crate::{files::resolve_file_path, types::Lux3dState, url_policy::validate_media_url};

#[derive(Debug, Clone, Default)]
pub struct MediaSourceInput {
    pub source_url: Option<String>,
    pub source_file_id: Option<String>,
    pub source_base64: Option<String>,
    pub source_filename: Option<String>,
    pub conditions_file_id: Option<String>,
}

/// Mesh input for the texturing endpoint: same resolution rules as
/// [`MediaSourceInput`] but with `mesh_*` field names.
#[derive(Debug, Clone, Default)]
pub struct MeshSourceInput {
    pub mesh_url: Option<String>,
    pub mesh_file_id: Option<String>,
    pub mesh_base64: Option<String>,
    pub mesh_filename: Option<String>,
}

pub async fn resolve_source_path(
    state: &Lux3dState,
    workspace: &Path,
    input: &MediaSourceInput,
) -> Result<PathBuf> {
    resolve_media_path(
        state,
        workspace,
        input.source_url.as_deref(),
        input.source_file_id.as_deref(),
        input.source_base64.as_deref(),
        input.source_filename.as_deref(),
        "one of `source`, `source_url`, `source_file_id`, or `source_base64` is required",
    )
    .await
}

pub async fn resolve_mesh_path(
    state: &Lux3dState,
    workspace: &Path,
    input: &MeshSourceInput,
) -> Result<PathBuf> {
    validate_mesh_source_input(input)?;
    let path = resolve_media_path(
        state,
        workspace,
        input.mesh_url.as_deref(),
        input.mesh_file_id.as_deref(),
        input.mesh_base64.as_deref(),
        input.mesh_filename.as_deref(),
        "one of `mesh`, `mesh_url`, `mesh_file_id`, or `mesh_base64` is required",
    )
    .await?;
    ensure_supported_mesh_path(&path)?;
    Ok(path)
}

/// Mesh readers dispatch on the file extension (`load_mesh`), so reject an
/// input that cannot be read before any job is created. Accepts the readers
/// currently implemented by `lux3d-core`: `.glb`, `.obj`, `.ply` and `.stl`.
pub(crate) fn ensure_supported_mesh_path(path: &Path) -> Result<()> {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    require_supported_mesh_filename(filename, "mesh input filename")
}

/// Validate the effective mesh filename before downloading or writing
/// anything. Mirrors the `resolve_media_path` precedence (`url`, then
/// `file_id`, then `base64`); a resolved file-id path is checked by
/// [`ensure_supported_mesh_path`] after the lookup.
fn validate_mesh_source_input(input: &MeshSourceInput) -> Result<()> {
    if let Some(url) = input.mesh_url.as_deref() {
        match input.mesh_filename.as_deref() {
            Some(filename) => {
                require_supported_mesh_filename(filename, "`mesh_filename`")?;
            }
            None => {
                let filename = infer_filename_from_url(url).ok_or_else(|| {
                    anyhow::anyhow!(
                        "`mesh_url` `{url}` has no filename; pass `mesh_filename` with a \
                         supported mesh extension (.glb, .obj, .ply or .stl)"
                    )
                })?;
                require_supported_mesh_filename(&filename, "`mesh_url` path")?;
            }
        }
    } else if input.mesh_file_id.is_none() {
        if input.mesh_base64.is_some() {
            let filename = input.mesh_filename.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "`mesh_base64` requires `mesh_filename` with a supported mesh extension \
                     (.glb, .obj, .ply or .stl)"
                )
            })?;
            require_supported_mesh_filename(filename, "`mesh_filename`")?;
        }
    }
    Ok(())
}

fn require_supported_mesh_filename(filename: &str, context: &str) -> Result<()> {
    ensure!(
        is_supported_mesh_filename(filename),
        "{context} `{filename}` has no supported mesh extension; use .glb, .obj, .ply or .stl"
    );
    Ok(())
}

fn is_supported_mesh_filename(filename: &str) -> bool {
    let extension = Path::new(filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.trim_start_matches('.').to_ascii_lowercase());
    matches!(extension.as_deref(), Some("glb" | "obj" | "ply" | "stl"))
}

#[allow(clippy::too_many_arguments)]
async fn resolve_media_path(
    state: &Lux3dState,
    workspace: &Path,
    url: Option<&str>,
    file_id: Option<&str>,
    base64_payload: Option<&str>,
    filename: Option<&str>,
    missing_message: &str,
) -> Result<PathBuf> {
    if let Some(url) = url {
        return fetch_url_to_workspace(state, workspace, url, filename).await;
    }
    if let Some(file_id) = file_id {
        let source_path = resolve_file_path(state, file_id)?;
        return copy_file_to_workspace(workspace, &source_path);
    }
    if let Some(encoded) = base64_payload {
        return write_base64_to_workspace(workspace, encoded, filename);
    }
    bail!("{missing_message}")
}

pub fn resolve_conditions_path(
    state: &Lux3dState,
    file_id: Option<&str>,
) -> Result<Option<PathBuf>> {
    match file_id {
        Some(id) => resolve_file_path(state, id).map(Some),
        None => Ok(None),
    }
}

async fn fetch_url_to_workspace(
    state: &Lux3dState,
    workspace: &Path,
    url: &str,
    filename: Option<&str>,
) -> Result<PathBuf> {
    let parsed_url = validate_media_url(url, state.media_fetch.allow_private_urls)?;
    let max_bytes = state.media_fetch.max_download_bytes;

    let mut response = state
        .media_http_client
        .get(parsed_url.clone())
        .send()
        .await
        .with_context(|| format!("failed to fetch source URL `{url}`"))?
        .error_for_status()
        .with_context(|| format!("source URL `{url}` returned an error status"))?;
    if response.status().is_redirection() {
        bail!(
            "source URL `{url}` returned {} without a usable redirect location",
            response.status()
        );
    }

    // `Content-Length` is only an early rejection; the cap is enforced on the
    // bytes actually streamed below, so chunked or lying responses cannot
    // exceed it.
    if let Some(content_length) = response.content_length() {
        if content_length > max_bytes as u64 {
            bail!(
                "source URL `{url}` declares {content_length} bytes, above the {max_bytes} byte download cap"
            );
        }
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .with_context(|| format!("failed to read response body from `{url}`"))?
    {
        if bytes.len().saturating_add(chunk.len()) > max_bytes {
            bail!("source URL `{url}` exceeds the {max_bytes} byte download cap");
        }
        bytes.extend_from_slice(&chunk);
    }

    fs::create_dir_all(workspace).with_context(|| {
        format!(
            "failed to create generation workspace at `{}`",
            workspace.display()
        )
    })?;

    let filename = filename
        .map(ToString::to_string)
        .or_else(|| infer_filename_from_url(parsed_url.as_str()))
        .unwrap_or_else(|| "source.bin".to_string());
    let path = workspace.join(sanitize_filename(&filename));
    fs::write(&path, &bytes)
        .with_context(|| format!("failed to write downloaded source to `{}`", path.display()))?;
    Ok(path)
}

fn copy_file_to_workspace(workspace: &Path, source: &Path) -> Result<PathBuf> {
    fs::create_dir_all(workspace).with_context(|| {
        format!(
            "failed to create generation workspace at `{}`",
            workspace.display()
        )
    })?;
    let filename = source
        .file_name()
        .and_then(|name| name.to_str())
        .map(sanitize_filename)
        .unwrap_or_else(|| "source.bin".to_string());
    let path = workspace.join(filename);
    fs::copy(source, &path).with_context(|| {
        format!(
            "failed to copy source `{}` into workspace at `{}`",
            source.display(),
            path.display()
        )
    })?;
    Ok(path)
}

fn write_base64_to_workspace(
    workspace: &Path,
    encoded: &str,
    filename: Option<&str>,
) -> Result<PathBuf> {
    let bytes = STANDARD
        .decode(encoded.trim())
        .context("failed to decode source_base64 payload")?;
    fs::create_dir_all(workspace).with_context(|| {
        format!(
            "failed to create generation workspace at `{}`",
            workspace.display()
        )
    })?;
    let filename = filename.unwrap_or("source.bin");
    let path = workspace.join(sanitize_filename(filename));
    fs::write(&path, bytes)
        .with_context(|| format!("failed to write base64 source to `{}`", path.display()))?;
    Ok(path)
}

fn infer_filename_from_url(url: &str) -> Option<String> {
    Url::parse(url)
        .ok()
        .and_then(|parsed| {
            parsed
                .path_segments()
                .and_then(|mut segments| segments.next_back().map(str::to_string))
        })
        .filter(|name| !name.is_empty())
}

fn sanitize_filename(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "source.bin".to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MeshSourceInput, ensure_supported_mesh_path, is_supported_mesh_filename,
        validate_mesh_source_input,
    };
    use std::path::Path;

    fn mesh_input(
        url: Option<&str>,
        file_id: Option<&str>,
        base64: Option<&str>,
        filename: Option<&str>,
    ) -> MeshSourceInput {
        MeshSourceInput {
            mesh_url: url.map(str::to_string),
            mesh_file_id: file_id.map(str::to_string),
            mesh_base64: base64.map(str::to_string),
            mesh_filename: filename.map(str::to_string),
        }
    }

    #[test]
    fn supported_mesh_extensions_are_case_insensitive() {
        for filename in ["mesh.glb", "mesh.OBJ", "mesh.Ply", "mesh.stl"] {
            assert!(
                is_supported_mesh_filename(filename),
                "`{filename}` must be accepted"
            );
        }
        for filename in ["mesh", "mesh.bin", "mesh.gltf", "mesh.png"] {
            assert!(
                !is_supported_mesh_filename(filename),
                "`{filename}` must be rejected"
            );
        }
    }

    #[test]
    fn mesh_url_requires_a_supported_path_extension() {
        validate_mesh_source_input(&mesh_input(
            Some("https://example.test/mesh.glb?token=abc"),
            None,
            None,
            None,
        ))
        .expect("URL with a supported mesh path extension");

        let error = validate_mesh_source_input(&mesh_input(
            Some("https://example.test/mesh"),
            None,
            None,
            None,
        ))
        .expect_err("extensionless mesh URL must be rejected");
        assert!(
            error.to_string().contains("`mesh_url` path"),
            "unexpected message: `{error}`"
        );

        validate_mesh_source_input(&mesh_input(
            Some("https://example.test/download"),
            None,
            None,
            Some("mesh.stl"),
        ))
        .expect("mesh_filename can supply the reader extension");
    }

    #[test]
    fn mesh_base64_requires_a_supported_mesh_filename() {
        let error = validate_mesh_source_input(&mesh_input(None, None, Some("aGVsbG8="), None))
            .expect_err("base64 mesh without a filename must be rejected");
        assert!(
            error.to_string().contains("requires `mesh_filename`"),
            "unexpected message: `{error}`"
        );

        let error =
            validate_mesh_source_input(&mesh_input(None, None, Some("aGVsbG8="), Some("mesh.bin")))
                .expect_err("base64 mesh with an unknown extension must be rejected");
        assert!(
            error.to_string().contains("`mesh_filename` `mesh.bin`"),
            "unexpected message: `{error}`"
        );

        validate_mesh_source_input(&mesh_input(None, None, Some("aGVsbG8="), Some("mesh.glb")))
            .expect("base64 mesh with a supported filename");
    }

    #[test]
    fn file_id_mesh_inputs_are_checked_after_resolution() {
        // The stored upload filename is only known after the file lookup, so
        // the pre-download validation deliberately skips `mesh_file_id`.
        validate_mesh_source_input(&mesh_input(None, Some("file_123"), None, None))
            .expect("file-id input is validated from the resolved path");

        ensure_supported_mesh_path(Path::new("mesh.glb")).expect("supported resolved path");
        let error = ensure_supported_mesh_path(Path::new("upload-123.bin"))
            .expect_err("resolved path without a supported extension must be rejected");
        assert!(
            error.to_string().contains("mesh input filename"),
            "unexpected message: `{error}`"
        );
    }
}
