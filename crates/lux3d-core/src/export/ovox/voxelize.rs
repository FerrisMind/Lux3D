//! Textured-mesh → sparse volumetric PBR attribute voxelization.
//!
//! Faithful CPU port of the official o_voxel surface:
//!
//! * wrapper semantics (settings, defaults, attribute quantization):
//!   `o-voxel/o_voxel/convert/volumetic_attr.py:42-302`
//!   (`textured_mesh_to_volumetric_attr`),
//! * kernel: `o-voxel/src/convert/volumetic_attr.cpp:291-871`
//!   (`voxelize_trimesh_pbr_impl`) — per-triangle 3-axis scan-line
//!   voxelization with 4-connected 2x2 stamps, barycentric UV/weight
//!   interpolation at each voxel center, mipmapped linear texture sampling,
//!   and weighted accumulation followed by division by the accumulated
//!   weight.
//!
//! The geometry (occupancy / dual vertices) side reuses
//! [`crate::geometry::mesh_to_flexible_dual_grid`], whose scan-line stamping
//! is the same algorithm (both come from the same upstream kernel family), so
//! the material voxel set matches the geometry voxel set coordinate for
//! coordinate.
//!
//! # Documented deviations
//!
//! * Non-square / non-power-of-two textures sample the base level instead of
//!   calling `build_mipmaps`, which in the official CPU kernel throws
//!   (`volumetic_attr.cpp:174-176`). POT-square textures build the identical
//!   box-filtered mip chain.
//! * A triangle whose UV parameterization is degenerate (`|det| < 1e-6`) uses
//!   the upstream default tangent/bitangent (`volumetic_attr.cpp:92-99`) but
//!   guards the zero-length normalization that would otherwise produce `NaN`
//!   tangents; `mip_length` stays the upstream `1e6` fallback either way.
//! * The per-triangle voxel set is accumulated in sorted coordinate order
//!   instead of the official `std::unordered_set` iteration order. Each
//!   voxel is visited at most once per triangle, so the cross-triangle
//!   accumulation order (the only thing that affects floating-point sums) is
//!   unchanged; the output is deterministic.
//! * `emissive` / `normal` attributes are produced (the official converter
//!   returns them) but are not part of the 6-channel PBR layout consumed by
//!   `to_glb`.

use std::collections::{BTreeSet, HashMap};

use crate::error::{Lux3dError, Result};

/// glTF/o_voxel alpha mode (`ALPHA_MODE_ENUM`, `volumetic_attr.py:18-22`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OvoxAlphaMode {
    /// `OPAQUE` (0): alpha stays 1, the alpha texture is ignored.
    Opaque,
    /// `MASK` (1): alpha is thresholded by the cutoff once the texture /
    /// factor are applied.
    Mask,
    /// `BLEND` (2): alpha is sampled without thresholding.
    Blend,
}

impl OvoxAlphaMode {
    /// Numeric code used by the official kernel (`0|1|2`).
    pub const fn code(self) -> u8 {
        match self {
            Self::Opaque => 0,
            Self::Mask => 1,
            Self::Blend => 2,
        }
    }
}

/// Decoded RGBA texture plane.
///
/// Row 0 is the TOP of the image (PIL / NumPy layout), exactly what the
/// official `np.array(PIL image)` produces; the sampler converts a
/// lower-left UV `v` into this layout with `y = (1 - v) * height`
/// (`volumetic_attr.cpp:217-218`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OvoxTexture {
    /// Image width in texels.
    pub width: usize,
    /// Image height in texels.
    pub height: usize,
    /// Row-major RGBA bytes, `width * height * 4`.
    pub rgba: Vec<u8>,
}

impl OvoxTexture {
    /// Build a decoded RGBA texture, validating the buffer length.
    pub fn new(width: usize, height: usize, rgba: Vec<u8>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: "o_voxel textures need a positive width and height".to_string(),
            });
        }
        let expected =
            width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(4))
                .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                    message: "o_voxel texture dimensions overflow".to_string(),
                })?;
        if rgba.len() != expected {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "o_voxel texture holds {} RGBA bytes; expected {expected} for {width}x{height}",
                    rgba.len()
                ),
            });
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }
}

/// PBR material surface consumed by the volumetric voxelizer: the fields the
/// official kernel reads from a trimesh `PBRMaterial`
/// (`volumetic_attr.py:205-229`).
///
/// Channel usage follows the official kernel: base color RGB, metallic from
/// the metallic-roughness texture's BLUE channel, roughness from GREEN, alpha
/// from the base-color texture's ALPHA channel, emissive RGB, normal RGB.
#[derive(Debug, Clone, PartialEq)]
pub struct OvoxMaterial {
    /// `baseColorFactor` RGBA in `[0, 1]` (RGB multiplies the base texture;
    /// A is the alpha factor).
    pub base_color_factor: [f32; 4],
    /// `baseColorTexture` (RGB sampled; alpha used when `alpha_mode` is not
    /// `Opaque`).
    pub base_color_texture: Option<OvoxTexture>,
    /// `metallicFactor` (upstream default 1.0).
    pub metallic_factor: f32,
    /// `metallicRoughnessTexture` (BLUE channel = metallic).
    pub metallic_texture: Option<OvoxTexture>,
    /// `roughnessFactor` (upstream default 1.0).
    pub roughness_factor: f32,
    /// `metallicRoughnessTexture` (GREEN channel = roughness).
    pub roughness_texture: Option<OvoxTexture>,
    /// `emissiveFactor` (upstream default `[0, 0, 0]`).
    pub emissive_factor: [f32; 3],
    /// `emissiveTexture` (RGB).
    pub emissive_texture: Option<OvoxTexture>,
    /// glTF `alphaMode` (upstream `OPAQUE` when absent).
    pub alpha_mode: OvoxAlphaMode,
    /// glTF `alphaCutoff` (upstream default 0.5).
    pub alpha_cutoff: f32,
    /// Alpha plane sampled when `alpha_mode != Opaque` (the base-color
    /// texture's alpha channel).
    pub alpha_texture: Option<OvoxTexture>,
    /// `normalTexture` (RGB, tangent space).
    pub normal_texture: Option<OvoxTexture>,
}

impl Default for OvoxMaterial {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: None,
            metallic_factor: 1.0,
            metallic_texture: None,
            roughness_factor: 1.0,
            roughness_texture: None,
            emissive_factor: [0.0, 0.0, 0.0],
            emissive_texture: None,
            alpha_mode: OvoxAlphaMode::Opaque,
            alpha_cutoff: 0.5,
            alpha_texture: None,
            normal_texture: None,
        }
    }
}

impl OvoxMaterial {
    /// Validate the factors and every attached texture.
    pub fn validate(&self) -> Result<()> {
        for value in self
            .base_color_factor
            .iter()
            .chain([self.metallic_factor, self.roughness_factor].iter())
            .chain(self.emissive_factor.iter())
            .chain([self.alpha_cutoff].iter())
        {
            if !value.is_finite() {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: "o_voxel material factors must be finite".to_string(),
                });
            }
        }
        for texture in [
            &self.base_color_texture,
            &self.metallic_texture,
            &self.roughness_texture,
            &self.emissive_texture,
            &self.alpha_texture,
            &self.normal_texture,
        ]
        .into_iter()
        .flatten()
        {
            let expected = texture.width * texture.height * 4;
            if texture.rgba.len() != expected {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: "o_voxel material texture buffer length mismatch".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Resolved voxel-grid placement (the upstream wrapper's `voxel_size`,
/// `grid_size` and `aabb` trio after auto-padding and cross-filling,
/// `volumetic_attr.py:87-145`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelizationSettings {
    /// Cell size per axis (the kernel's `delta_p`).
    pub voxel_size: [f32; 3],
    /// Grid range per axis (`[0, grid_size)`).
    pub grid_size: [u32; 3],
    /// Axis-aligned box the mesh was shifted into.
    pub aabb: [[f32; 3]; 2],
}

/// Resolve the upstream wrapper's settings from the mesh bounds.
///
/// * One of `voxel_size` / `grid_size` must be given.
/// * When `aabb` is `None`, the mesh bounds are expanded by half a cell:
///   `voxel_size` mode uses `padding = ceil(extent / voxel_size) *
///   voxel_size - extent`, `grid_size` mode uses
///   `padding = extent / (grid_size - 1)` (`volumetic_attr.py:122-137`).
/// * A missing `voxel_size` is `extent / grid_size`; a missing `grid_size`
///   is `round(extent / voxel_size)` (torch `.round().int()` semantics, i.e.
///   ties to even).
pub fn resolve_voxelization_settings(
    bounds: [[f32; 3]; 2],
    voxel_size: Option<[f32; 3]>,
    grid_size: Option<[u32; 3]>,
    aabb: Option<[[f32; 3]; 2]>,
) -> Result<VoxelizationSettings> {
    if voxel_size.is_none() && grid_size.is_none() {
        return Err(Lux3dError::InvalidInput(
            "o_voxel voxelization needs either a voxel_size or a grid_size",
        ));
    }
    if let Some(voxel_size) = &voxel_size {
        for size in voxel_size {
            if !size.is_finite() || *size <= 0.0 {
                return Err(Lux3dError::InvalidInput(
                    "o_voxel voxel_size must be finite and positive on every axis",
                ));
            }
        }
    }
    if let Some(grid_size) = &grid_size {
        for size in grid_size {
            if *size == 0 {
                return Err(Lux3dError::InvalidInput(
                    "o_voxel grid_size must be positive on every axis",
                ));
            }
        }
    }
    for corner in &bounds {
        if corner.iter().any(|value| !value.is_finite()) {
            return Err(Lux3dError::InvalidInput(
                "o_voxel voxelization bounds must be finite",
            ));
        }
    }

    let resolved_aabb = match aabb {
        Some(aabb) => {
            for axis in 0..3 {
                if !aabb[0][axis].is_finite()
                    || !aabb[1][axis].is_finite()
                    || !(aabb[0][axis] < aabb[1][axis])
                {
                    return Err(Lux3dError::InvalidInput(
                        "o_voxel aabb must be finite with min < max on every axis",
                    ));
                }
            }
            aabb
        }
        None => {
            let mut min = bounds[0];
            let mut max = bounds[1];
            if let Some(voxel_size) = voxel_size {
                for axis in 0..3 {
                    let extent = max[axis] - min[axis];
                    let padding = (extent / voxel_size[axis]).ceil() * voxel_size[axis] - extent;
                    min[axis] -= padding * 0.5;
                    max[axis] += padding * 0.5;
                }
            } else if let Some(grid_size) = grid_size {
                for axis in 0..3 {
                    if grid_size[axis] < 2 {
                        return Err(Lux3dError::InvalidInput(
                            "o_voxel auto aabb needs grid_size >= 2 per axis",
                        ));
                    }
                    let padding = (max[axis] - min[axis]) / (grid_size[axis] - 1) as f32;
                    min[axis] -= padding * 0.5;
                    max[axis] += padding * 0.5;
                }
            }
            for axis in 0..3 {
                if !min[axis].is_finite()
                    || !max[axis].is_finite()
                    || !(min[axis] < max[axis])
                {
                    return Err(Lux3dError::InvalidInput(
                        "o_voxel auto aabb is degenerate along an axis; pass an explicit aabb",
                    ));
                }
            }
            [min, max]
        }
    };

    let resolved_voxel_size = match voxel_size {
        Some(voxel_size) => voxel_size,
        None => {
            let grid_size = grid_size.expect("one of voxel_size/grid_size is set");
            [
                (resolved_aabb[1][0] - resolved_aabb[0][0]) / grid_size[0] as f32,
                (resolved_aabb[1][1] - resolved_aabb[0][1]) / grid_size[1] as f32,
                (resolved_aabb[1][2] - resolved_aabb[0][2]) / grid_size[2] as f32,
            ]
        }
    };
    if resolved_voxel_size
        .iter()
        .any(|size| !size.is_finite() || *size <= 0.0)
    {
        return Err(Lux3dError::InvalidInput(
            "o_voxel voxelization produced a degenerate voxel size",
        ));
    }

    let resolved_grid_size = match grid_size {
        Some(grid_size) => grid_size,
        None => {
            let mut grid = [0u32; 3];
            for axis in 0..3 {
                let size =
                    (resolved_aabb[1][axis] - resolved_aabb[0][axis]) / resolved_voxel_size[axis];
                grid[axis] = size.round_ties_even().max(1.0) as u32;
            }
            grid
        }
    };

    Ok(VoxelizationSettings {
        voxel_size: resolved_voxel_size,
        grid_size: resolved_grid_size,
        aabb: resolved_aabb,
    })
}

/// Sparse volumetric PBR attributes in the official kernel's float space
/// (pre-quantization). Rows align with [`MaterialVoxelization::coords`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MaterialVoxelization {
    /// Occupied voxel coordinates, `0 <= coord < grid_size`, in the
    /// kernel's hash-table insertion order.
    pub coords: Vec<[i32; 3]>,
    /// `[N, 3]` base color RGB in `[0, 1]`.
    pub base_color: Vec<[f32; 3]>,
    /// `[N]` metallic in `[0, 1]`.
    pub metallic: Vec<f32>,
    /// `[N]` roughness in `[0, 1]`.
    pub roughness: Vec<f32>,
    /// `[N, 3]` emissive RGB.
    pub emissive: Vec<[f32; 3]>,
    /// `[N]` alpha in `[0, 1]` (1 for `Opaque` materials).
    pub alpha: Vec<f32>,
    /// `[N, 3]` normal (interpolated vertex normal, or the normal-mapped
    /// normal when a normal texture is present).
    pub normal: Vec<[f32; 3]>,
}

impl MaterialVoxelization {
    /// Number of occupied voxels.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Whether no voxel was produced.
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

/// A prepared texture: the base level plus the box-filtered mip chain
/// (`mipmaps[0]` is half resolution, exactly the official layout).
struct PreparedTexture<'a> {
    base: &'a OvoxTexture,
    mips: Vec<(usize, usize, Vec<u8>)>,
}

impl<'a> PreparedTexture<'a> {
    fn new(texture: &'a OvoxTexture) -> Self {
        let mut mips: Vec<(usize, usize, Vec<u8>)> = Vec::new();
        // The official `build_mipmaps` requires square power-of-two textures;
        // anything else falls back to base-level sampling (documented).
        if texture.width == texture.height && texture.width.is_power_of_two() {
            let mut cur_w = texture.width;
            let mut cur_h = texture.height;
            loop {
                let next_w = cur_w >> 1;
                let next_h = cur_h >> 1;
                if next_w == 0 || next_h == 0 {
                    break;
                }
                let current: &[u8] = match mips.last() {
                    Some((_, _, data)) => data,
                    None => texture.rgba.as_slice(),
                };
                let mut next = vec![0_u8; next_w * next_h * 4];
                for y in 0..next_h {
                    for x in 0..next_w {
                        for channel in 0..4usize {
                            let xx = x << 1;
                            let yy = y << 1;
                            let mut sum = 0usize;
                            for dy in 0..2usize {
                                for dx in 0..2usize {
                                    let offset = ((yy + dy) * cur_w + (xx + dx)) * 4 + channel;
                                    sum += current[offset] as usize;
                                }
                            }
                            next[(y * next_w + x) * 4 + channel] = (sum / 4) as u8;
                        }
                    }
                }
                mips.push((next_w, next_h, next));
                cur_w = next_w;
                cur_h = next_h;
            }
        }
        Self { base: texture, mips }
    }
}

struct PreparedMaterial<'a> {
    material: &'a OvoxMaterial,
    base_color: Option<PreparedTexture<'a>>,
    metallic: Option<PreparedTexture<'a>>,
    roughness: Option<PreparedTexture<'a>>,
    emissive: Option<PreparedTexture<'a>>,
    alpha: Option<PreparedTexture<'a>>,
    normal: Option<PreparedTexture<'a>>,
}

impl<'a> PreparedMaterial<'a> {
    fn new(material: &'a OvoxMaterial) -> Self {
        Self {
            material,
            base_color: material.base_color_texture.as_ref().map(PreparedTexture::new),
            metallic: material.metallic_texture.as_ref().map(PreparedTexture::new),
            roughness: material.roughness_texture.as_ref().map(PreparedTexture::new),
            emissive: material.emissive_texture.as_ref().map(PreparedTexture::new),
            alpha: if material.alpha_mode == OvoxAlphaMode::Opaque {
                None
            } else {
                material.alpha_texture.as_ref().map(PreparedTexture::new)
            },
            normal: material.normal_texture.as_ref().map(PreparedTexture::new),
        }
    }
}

/// Wrap a texel index into `[0, size)` with the official modes
/// (`volumetic_attr.cpp:154-167`); the trimesh wrapper only ever uses
/// REPEAT (0).
#[inline]
fn wrap_texcoord(value: i64, size: i64) -> i64 {
    (value % size + size) % size
}

/// Bilinear texture sample at `(u, v)` in the PIL/NumPy row layout (row 0 is
/// the image top), matching `sample_texture` (`volumetic_attr.cpp:210-247`).
fn sample_level(
    width: usize,
    height: usize,
    data: &[u8],
    u: f32,
    v: f32,
    out: &mut [f32; 4],
) {
    let width_i = width as i64;
    let height_i = height as i64;
    let x = u * width as f32;
    let y = (1.0 - v) * height as f32;
    let x_low_raw = (x - 0.5).floor() as i64;
    let y_low_raw = (y - 0.5).floor() as i64;
    let x_low = wrap_texcoord(x_low_raw, width_i);
    let x_high = wrap_texcoord(x_low_raw + 1, width_i);
    let y_low = wrap_texcoord(y_low_raw, height_i);
    let y_high = wrap_texcoord(y_low_raw + 1, height_i);
    let w_x = x - x_low_raw as f32 - 0.5;
    let w_y = y - y_low_raw as f32 - 0.5;
    let index = |x: i64, y: i64| ((y as usize * width + x as usize) * 4) as usize;
    let texel = |x: i64, y: i64, channel: usize| data[index(x, y) + channel] as f32;
    for (channel, value) in out.iter_mut().enumerate() {
        let sample = (1.0 - w_x) * (1.0 - w_y) * texel(x_low, y_low, channel)
            + w_x * (1.0 - w_y) * texel(x_high, y_low, channel)
            + (1.0 - w_x) * w_y * texel(x_low, y_high, channel)
            + w_x * w_y * texel(x_high, y_high, channel);
        *value = sample / 255.0;
    }
}

/// Mipmapped linear sample: `mip_level = log2(mip_length * height) +
/// mip_offset`, clamped to the chain (or the base level for non-finite /
/// non-positive levels), with linear blending between adjacent levels — the
/// exact `sample_texture_mipmap` (`volumetic_attr.cpp:250-288`).
fn sample_mipmap(
    texture: &PreparedTexture<'_>,
    u: f32,
    v: f32,
    mip_length: f32,
    mip_offset: f32,
) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    let height = texture.base.height as f32;
    let mip_level = (mip_length * height).log2() + mip_offset;
    if !mip_level.is_finite() || mip_level <= 0.0 || texture.mips.is_empty() {
        sample_level(
            texture.base.width,
            texture.base.height,
            &texture.base.rgba,
            u,
            v,
            &mut out,
        );
        return out;
    }
    if mip_level >= texture.mips.len() as f32 {
        let (width, height, data) = texture.mips.last().expect("non-empty chain");
        sample_level(*width, *height, data, u, v, &mut out);
        return out;
    }
    let lower = mip_level.floor() as usize;
    let upper = lower + 1;
    let fraction = mip_level - lower as f32;
    let lower_sample = if lower == 0 {
        let mut sample = [0.0f32; 4];
        sample_level(
            texture.base.width,
            texture.base.height,
            &texture.base.rgba,
            u,
            v,
            &mut sample,
        );
        sample
    } else {
        let (width, height, data) = &texture.mips[lower - 1];
        let mut sample = [0.0f32; 4];
        sample_level(*width, *height, data, u, v, &mut sample);
        sample
    };
    let (upper_width, upper_height, upper_data) = &texture.mips[upper - 1];
    let mut upper_sample = [0.0f32; 4];
    sample_level(*upper_width, *upper_height, upper_data, u, v, &mut upper_sample);
    for channel in 0..4 {
        out[channel] = (1.0 - fraction) * lower_sample[channel] + fraction * upper_sample[channel];
    }
    out
}

/// Per-voxel weighted accumulators (insertion ordered).
struct Accumulator {
    weight: f32,
    base_color: [f32; 3],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    alpha: f32,
    normal: [f32; 3],
}

/// The official C-style `int(double)` truncation followed by a clamp.
#[inline]
fn clamp_i(value: f64, low: i32, high: i32) -> i32 {
    (value as i32).clamp(low, high)
}

#[inline]
fn lerp_scalar(a: f64, b: f64, t: f64, value_a: f64, value_b: f64) -> f64 {
    if a == b {
        return value_a;
    }
    let alpha = (t - a) / (b - a);
    (1.0 - alpha) * value_a + alpha * value_b
}

#[inline]
fn lerp_pair(a: f64, b: f64, t: f64, value_a: [f64; 2], value_b: [f64; 2]) -> [f64; 2] {
    if a == b {
        return value_a;
    }
    let alpha = (t - a) / (b - a);
    [
        (1.0 - alpha) * value_a[0] + alpha * value_b[0],
        (1.0 - alpha) * value_a[1] + alpha * value_b[1],
    ]
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(a: [f32; 3]) -> f32 {
    dot3(a, a).sqrt()
}

#[inline]
fn normalize3(a: [f32; 3]) -> [f32; 3] {
    let length = norm3(a);
    [a[0] / length, a[1] / length, a[2] / length]
}

/// TBN + texture-space mip length of one triangle
/// (`compute_TBN`, `volumetic_attr.cpp:78-111`).
fn compute_tbn(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    uv0: [f32; 2],
    uv1: [f32; 2],
    uv2: [f32; 2],
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 2]) {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
    let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];
    let n = normalize3(cross3(e1, e2));
    let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
    if !det.is_finite() || det.abs() < 1e-6 {
        return (
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            n,
            [1e6, 1e6],
        );
    }
    let inv_det = 1.0 / det;
    let mut t = [
        duv2[1] * e1[0] - duv1[1] * e2[0],
        duv2[1] * e1[1] - duv1[1] * e2[1],
        duv2[1] * e1[2] - duv1[1] * e2[2],
    ];
    let mut b = [
        duv1[0] * e2[0] - duv2[0] * e1[0],
        duv1[0] * e2[1] - duv2[0] * e1[1],
        duv1[0] * e2[2] - duv2[0] * e1[2],
    ];
    let t_norm = norm3(t);
    let b_norm = norm3(b);
    if t_norm > 0.0 && t_norm.is_finite() {
        t = [t[0] / t_norm, t[1] / t_norm, t[2] / t_norm];
    } else {
        t = [1.0, 0.0, 0.0];
    }
    if b_norm > 0.0 && b_norm.is_finite() {
        b = [b[0] / b_norm, b[1] / b_norm, b[2] / b_norm];
    } else {
        b = [0.0, 1.0, 0.0];
    }
    let mip_length = if t_norm > 0.0 && b_norm > 0.0 {
        [inv_det * t_norm, inv_det * b_norm]
    } else {
        [1e6, 1e6]
    };
    (t, b, n, mip_length)
}

/// Project a point onto the triangle plane and return barycentric
/// coordinates plus the signed plane distance (`project_onto_triangle`,
/// `volumetic_attr.cpp:125-151`).
fn project_onto_triangle(
    point: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    normal: [f32; 3],
) -> [f32; 4] {
    let distance = dot3(sub3(point, a), normal);
    let projected = [
        point[0] - distance * normal[0],
        point[1] - distance * normal[1],
        point[2] - distance * normal[2],
    ];
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(projected, a);
    let d00 = dot3(ab, ab);
    let d01 = dot3(ab, ac);
    let d11 = dot3(ac, ac);
    let d20 = dot3(ap, ab);
    let d21 = dot3(ap, ac);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    [u, v, w, distance]
}

/// Collect the voxels the scan-line stamps of one triangle touch
/// (`scan_line_fill`, `volumetic_attr.cpp:431-482`). The `(dx, dy) = (0, 0)`
/// stamp is the `intersected` corner, matching the geometry voxelizer.
fn scan_line_fill_triangle(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    voxel_size: [f32; 3],
    grid_size: [u32; 3],
    voxels: &mut BTreeSet<[i32; 3]>,
) {
    let grid_max = [
        grid_size[0] as i32,
        grid_size[1] as i32,
        grid_size[2] as i32,
    ];
    for ax2 in 0..3usize {
        let ax0 = (ax2 + 1) % 3;
        let ax1 = (ax2 + 2) % 3;
        let mut t = [
            [v0[ax0] as f64, v0[ax1] as f64, v0[ax2] as f64],
            [v1[ax0] as f64, v1[ax1] as f64, v1[ax2] as f64],
            [v2[ax0] as f64, v2[ax1] as f64, v2[ax2] as f64],
        ];
        t.sort_by(|a, b| a[1].total_cmp(&b[1]));

        let vs1 = voxel_size[ax1] as f64;
        let start = clamp_i(t[0][1] / vs1, 0, grid_max[ax1] - 1);
        let mid = clamp_i(t[1][1] / vs1, 0, grid_max[ax1] - 1);
        let end = clamp_i(t[2][1] / vs1, 0, grid_max[ax1] - 1);

        for (row_start, row_end, t0, t1, t2) in [
            (start, mid, t[0], t[1], t[2]),
            (mid, end, t[2], t[1], t[0]),
        ] {
            let vs0 = voxel_size[ax0] as f64;
            let vs2 = voxel_size[ax2] as f64;
            for y_idx in row_start..row_end {
                let y = ((y_idx + 1) as f32 * voxel_size[ax1]) as f64;
                let mut t3 = lerp_pair(t0[1], t1[1], y, [t0[0], t0[2]], [t1[0], t1[2]]);
                let mut t4 = lerp_pair(t0[1], t2[1], y, [t0[0], t0[2]], [t2[0], t2[2]]);
                if t3[0] > t4[0] {
                    std::mem::swap(&mut t3, &mut t4);
                }
                let line_start = clamp_i(t3[0] / vs0, 0, grid_max[ax0] - 1);
                let line_end = clamp_i(t4[0] / vs0, 0, grid_max[ax0] - 1);
                for x_idx in line_start..line_end {
                    let x = ((x_idx + 1) as f32 * voxel_size[ax0]) as f64;
                    let z = lerp_scalar(t3[0], t4[0], x, t3[1], t4[1]);
                    let z_idx = (z / vs2) as i32;
                    if z_idx < 0 || z_idx >= grid_max[ax2] {
                        continue;
                    }
                    for dx in 0..2i32 {
                        for dy in 0..2i32 {
                            let mut coord = [0i32; 3];
                            coord[ax0] = x_idx + dx;
                            coord[ax1] = y_idx + dy;
                            coord[ax2] = z_idx;
                            voxels.insert(coord);
                        }
                    }
                }
            }
        }
    }
}

/// Voxelize a textured triangle mesh into sparse volumetric PBR attributes —
/// the official `textured_mesh_to_volumetric_attr` kernel behaviour
/// (`volumetic_attr.py:42-302`, `volumetic_attr.cpp:291-871`).
///
/// * `vertices`: flat `[V, 3]` world-space positions.
/// * `faces`: flat zero-based triangle indices (`3 * F`).
/// * `corner_uvs`: `3 * F` per-corner UVs in the trimesh convention (origin
///   lower-left, `v` up), aligned with `faces`.
///
///   The official wrapper fills zero UVs when a geometry has none
///   (`volumetic_attr.py:207-208`); callers should do the same for
///   untextured inputs.
/// * `material_ids`: `F` material indices; `materials` holds at least one
///   entry (an unused default material is harmless).
///
/// Coordinates are shifted by `settings.aabb[0]` internally, exactly like
/// the wrapper's `triangles = ... - aabb[0]`.
#[allow(clippy::too_many_arguments)]
pub fn textured_mesh_to_volumetric_attr(
    vertices: &[[f32; 3]],
    faces: &[u32],
    corner_uvs: &[[f32; 2]],
    corner_normals: &[[f32; 3]],
    material_ids: &[usize],
    materials: &[OvoxMaterial],
    settings: &VoxelizationSettings,
    mip_level_offset: f32,
) -> Result<MaterialVoxelization> {
    if faces.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "o_voxel material voxelization face buffer length must be a multiple of 3",
        ));
    }
    if corner_uvs.len() != faces.len() || corner_normals.len() != faces.len() {
        return Err(Lux3dError::InvalidInput(
            "o_voxel material voxelization needs one UV and one normal per face corner",
        ));
    }
    let face_count = faces.len() / 3;
    if material_ids.len() != face_count {
        return Err(Lux3dError::InvalidInput(
            "o_voxel material voxelization needs one material id per face",
        ));
    }
    if materials.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "o_voxel material voxelization needs at least one material",
        ));
    }
    if vertices.is_empty() || faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "o_voxel material voxelization requires a non-empty mesh",
        ));
    }
    if !mip_level_offset.is_finite() {
        return Err(Lux3dError::InvalidInput(
            "o_voxel mip_level_offset must be finite",
        ));
    }
    let vertex_count = vertices.len();
    for &index in faces {
        if index as usize >= vertex_count {
            return Err(Lux3dError::InvalidInput(
                "o_voxel material voxelization face index outside the vertex buffer",
            ));
        }
    }
    if vertices.iter().any(|vertex| vertex.iter().any(|v| !v.is_finite())) {
        return Err(Lux3dError::InvalidInput(
            "o_voxel material voxelization requires finite vertex positions",
        ));
    }
    for material in materials {
        material.validate()?;
    }
    for &material_id in material_ids {
        if material_id >= materials.len() {
            return Err(Lux3dError::InvalidInput(
                "o_voxel material voxelization material id outside the material list",
            ));
        }
    }

    // Shift into the voxel frame (`triangles - aabb[0]`).
    let shifted: Vec<[f32; 3]> = vertices
        .iter()
        .map(|v| sub3(*v, settings.aabb[0]))
        .collect();
    let grid_max = [
        settings.grid_size[0] as i32,
        settings.grid_size[1] as i32,
        settings.grid_size[2] as i32,
    ];
    let prepared: Vec<PreparedMaterial<'_>> =
        materials.iter().map(PreparedMaterial::new).collect();

    let mut map: HashMap<[i32; 3], usize> = HashMap::new();
    let mut coords: Vec<[i32; 3]> = Vec::new();
    let mut rows: Vec<Accumulator> = Vec::new();

    for (face_index, triangle) in faces.chunks_exact(3).enumerate() {
        let v0 = shifted[triangle[0] as usize];
        let v1 = shifted[triangle[1] as usize];
        let v2 = shifted[triangle[2] as usize];
        let n0 = corner_normals[face_index * 3];
        let n1 = corner_normals[face_index * 3 + 1];
        let n2 = corner_normals[face_index * 3 + 2];
        let uv0 = corner_uvs[face_index * 3];
        let uv1 = corner_uvs[face_index * 3 + 1];
        let uv2 = corner_uvs[face_index * 3 + 2];
        let (t, b, face_normal, mip_lengths) = compute_tbn(v0, v1, v2, uv0, uv1, uv2);
        let mip_length = settings
            .voxel_size
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            / (mip_lengths[0] * mip_lengths[1]).sqrt();
        let material_index = material_ids[face_index];
        let prepared_material = &prepared[material_index];
        let material = prepared_material.material;

        let mut voxels = BTreeSet::new();
        scan_line_fill_triangle(
            v0,
            v1,
            v2,
            settings.voxel_size,
            settings.grid_size,
            &mut voxels,
        );

        for coord in voxels {
            let center = [
                (coord[0] as f32 + 0.5) * settings.voxel_size[0],
                (coord[1] as f32 + 0.5) * settings.voxel_size[1],
                (coord[2] as f32 + 0.5) * settings.voxel_size[2],
            ];
            let barycentric = project_onto_triangle(center, v0, v1, v2, face_normal);
            let uv = [
                barycentric[0] * uv0[0] + barycentric[1] * uv1[0] + barycentric[2] * uv2[0],
                barycentric[0] * uv0[1] + barycentric[1] * uv1[1] + barycentric[2] * uv2[1],
            ];
            let interpolated_normal = [
                barycentric[0] * n0[0] + barycentric[1] * n1[0] + barycentric[2] * n2[0],
                barycentric[0] * n0[1] + barycentric[1] * n1[1] + barycentric[2] * n2[1],
                barycentric[0] * n0[2] + barycentric[1] * n1[2] + barycentric[2] * n2[2],
            ];
            let weight = 1.0 - barycentric[3];

            let mut base_color = [1.0f32; 3];
            if let Some(texture) = &prepared_material.base_color {
                let sample = sample_mipmap(texture, uv[0], uv[1], mip_length, mip_level_offset);
                base_color = [sample[0], sample[1], sample[2]];
            }
            for channel in 0..3 {
                base_color[channel] *= material.base_color_factor[channel];
            }

            let mut metallic = 1.0f32;
            if let Some(texture) = &prepared_material.metallic {
                let sample = sample_mipmap(texture, uv[0], uv[1], mip_length, mip_level_offset);
                metallic = sample[2];
            }
            metallic *= material.metallic_factor;

            let mut roughness = 1.0f32;
            if let Some(texture) = &prepared_material.roughness {
                let sample = sample_mipmap(texture, uv[0], uv[1], mip_length, mip_level_offset);
                roughness = sample[1];
            }
            roughness *= material.roughness_factor;

            let mut emissive = [1.0f32; 3];
            if let Some(texture) = &prepared_material.emissive {
                let sample = sample_mipmap(texture, uv[0], uv[1], mip_length, mip_level_offset);
                emissive = [sample[0], sample[1], sample[2]];
            }
            for channel in 0..3 {
                emissive[channel] *= material.emissive_factor[channel];
            }

            let mut alpha = 1.0f32;
            if material.alpha_mode != OvoxAlphaMode::Opaque {
                if let Some(texture) = &prepared_material.alpha {
                    let sample =
                        sample_mipmap(texture, uv[0], uv[1], mip_length, mip_level_offset);
                    alpha = sample[3];
                }
                alpha *= material.base_color_factor[3];
                if material.alpha_mode == OvoxAlphaMode::Mask {
                    alpha = if alpha < material.alpha_cutoff {
                        0.0
                    } else {
                        1.0
                    };
                }
            }

            let mut normal = interpolated_normal;
            if let Some(texture) = &prepared_material.normal {
                let sample = sample_mipmap(texture, uv[0], uv[1], mip_length, mip_level_offset);
                let decoded = [
                    sample[0] * 2.0 - 1.0,
                    sample[1] * 2.0 - 1.0,
                    sample[2] * 2.0 - 1.0,
                ];
                let mapped = [
                    decoded[0] * t[0] + decoded[1] * b[0] + decoded[2] * interpolated_normal[0],
                    decoded[0] * t[1] + decoded[1] * b[1] + decoded[2] * interpolated_normal[1],
                    decoded[0] * t[2] + decoded[1] * b[2] + decoded[2] * interpolated_normal[2],
                ];
                let length = norm3(mapped);
                normal = if length > 0.0 && length.is_finite() {
                    [mapped[0] / length, mapped[1] / length, mapped[2] / length]
                } else {
                    interpolated_normal
                };
            }

            let row = match map.get(&coord) {
                Some(&row) => row,
                None => {
                    let row = rows.len();
                    map.insert(coord, row);
                    coords.push(coord);
                    rows.push(Accumulator {
                        weight: 0.0,
                        base_color: [0.0; 3],
                        metallic: 0.0,
                        roughness: 0.0,
                        emissive: [0.0; 3],
                        alpha: 0.0,
                        normal: [0.0; 3],
                    });
                    row
                }
            };
            let accumulator = &mut rows[row];
            accumulator.weight += weight;
            for channel in 0..3 {
                accumulator.base_color[channel] += base_color[channel] * weight;
                accumulator.emissive[channel] += emissive[channel] * weight;
                accumulator.normal[channel] += normal[channel] * weight;
            }
            accumulator.metallic += metallic * weight;
            accumulator.roughness += roughness * weight;
            accumulator.alpha += alpha * weight;
        }
    }

    let mut out = MaterialVoxelization {
        coords: Vec::with_capacity(coords.len()),
        base_color: Vec::with_capacity(coords.len()),
        metallic: Vec::with_capacity(coords.len()),
        roughness: Vec::with_capacity(coords.len()),
        emissive: Vec::with_capacity(coords.len()),
        alpha: Vec::with_capacity(coords.len()),
        normal: Vec::with_capacity(coords.len()),
    };
    for (coord, accumulator) in coords.into_iter().zip(rows.iter()) {
        if coord.iter().any(|&c| c < 0) || (0..3).any(|axis| coord[axis] >= grid_max[axis]) {
            return Err(Lux3dError::InvalidInput(
                "o_voxel material voxelization produced a coordinate outside the grid",
            ));
        }
        let divisor = accumulator.weight;
        let normalize = |value: f32| {
            let normalized = value / divisor;
            if normalized.is_finite() {
                normalized
            } else {
                0.0
            }
        };
        out.coords.push(coord);
        out.base_color.push([
            normalize(accumulator.base_color[0]),
            normalize(accumulator.base_color[1]),
            normalize(accumulator.base_color[2]),
        ]);
        out.metallic.push(normalize(accumulator.metallic));
        out.roughness.push(normalize(accumulator.roughness));
        out.emissive.push([
            normalize(accumulator.emissive[0]),
            normalize(accumulator.emissive[1]),
            normalize(accumulator.emissive[2]),
        ]);
        out.alpha.push(normalize(accumulator.alpha));
        out.normal.push([
            normalize(accumulator.normal[0]),
            normalize(accumulator.normal[1]),
            normalize(accumulator.normal[2]),
        ]);
    }
    Ok(out)
}

/// Quantize a float attribute to the official `clamp(value * 255, 0, 255)`
/// byte (`volumetic_attr.py:293-300`). `NaN` maps to 0, matching NumPy's
/// invalid float→uint8 cast.
#[inline]
pub fn quantize_attr(value: f32) -> u8 {
    if value.is_nan() {
        return 0;
    }
    (value * 255.0).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_material() -> OvoxMaterial {
        OvoxMaterial {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: Some(OvoxTexture::new(2, 2, {
                let mut rgba = vec![0u8; 2 * 2 * 4];
                for (index, texel) in rgba.chunks_exact_mut(4).enumerate() {
                    texel[0] = (index as u8) * 40;
                    texel[1] = 10;
                    texel[2] = 20;
                    texel[3] = 255;
                }
                rgba
            })
            .unwrap()),
            metallic_factor: 0.25,
            roughness_factor: 0.75,
            emissive_factor: [0.1, 0.2, 0.3],
            ..OvoxMaterial::default()
        }
    }

    /// One quad in the z = 0.5 plane with full [0,1]^2 UVs.
    fn quad_fixture() -> (Vec<[f32; 3]>, Vec<u32>, Vec<[f32; 2]>, Vec<[f32; 3]>) {
        let vertices = vec![
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
        ];
        let faces = vec![0u32, 1, 2, 0, 2, 3];
        let uvs = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];
        let normals = vec![[0.0, 0.0, 1.0]; 6];
        (vertices, faces, uvs, normals)
    }

    fn unit_settings() -> VoxelizationSettings {
        VoxelizationSettings {
            voxel_size: [0.125; 3],
            grid_size: [8, 8, 8],
            aabb: [[0.0; 3], [1.0; 3]],
        }
    }

    #[test]
    fn quad_voxelization_matches_material_factors_and_layout() {
        let (vertices, faces, uvs, normals) = quad_fixture();
        let materials = vec![solid_material()];
        let result = textured_mesh_to_volumetric_attr(
            &vertices,
            &faces,
            &uvs,
            &normals,
            &[0, 0],
            &materials,
            &unit_settings(),
            0.0,
        )
        .expect("voxelization");

        assert!(!result.is_empty());
        // The quad lies at z = 0.5 -> voxel row 4 on the z axis.
        for coord in &result.coords {
            assert_eq!(coord[2], 4, "voxel {coord:?} off the surface slab");
        }
        for index in 0..result.len() {
            let base = result.base_color[index];
            assert!(base.iter().all(|v| v.is_finite()));
            assert!((result.metallic[index] - 0.25).abs() < 1e-5);
            assert!((result.roughness[index] - 0.75).abs() < 1e-5);
            assert!((result.alpha[index] - 1.0).abs() < 1e-6);
            assert!((result.emissive[index][0] - 0.1).abs() < 1e-5);
            assert!((result.emissive[index][1] - 0.2).abs() < 1e-5);
            assert!((result.emissive[index][2] - 0.3).abs() < 1e-5);
            // Base color comes from the 2x2 texture (red channel 0/40/80/120
            // depending on which texel the UV lands in).
            assert!(base[1] > 0.0 && base[1] < 0.05, "green channel {base:?}");
        }
        // Default normal is the interpolated vertex normal.
        for normal in &result.normal {
            assert!((normal[2] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn alpha_mask_thresholds() {
        let (vertices, faces, uvs, normals) = quad_fixture();
        let material = OvoxMaterial {
            alpha_mode: OvoxAlphaMode::Mask,
            alpha_texture: Some(
                OvoxTexture::new(1, 1, vec![0, 0, 0, 0]).expect("transparent 1x1"),
            ),
            alpha_cutoff: 0.5,
            ..OvoxMaterial::default()
        };
        let result = textured_mesh_to_volumetric_attr(
            &vertices,
            &faces,
            &uvs,
            &normals,
            &[0, 0],
            &[material],
            &unit_settings(),
            0.0,
        )
        .expect("voxelization");
        assert!(result.alpha.iter().all(|alpha| *alpha == 0.0));
    }

    #[test]
    fn settings_resolve_grid_and_voxel_modes() {
        let bounds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let grid = resolve_voxelization_settings(bounds, None, Some([8, 8, 8]), None).unwrap();
        // grid-mode auto padding: 1 / (8 - 1) per axis, split in half.
        let pad = 1.0 / 7.0 / 2.0;
        assert!((grid.aabb[0][0] + pad).abs() < 1e-6);
        assert!((grid.aabb[1][0] - (1.0 + pad)).abs() < 1e-6);
        assert!((grid.voxel_size[0] - (1.0 + 2.0 * pad) / 8.0).abs() < 1e-6);

        let explicit = resolve_voxelization_settings(
            bounds,
            None,
            Some([8, 8, 8]),
            Some([[0.0; 3], [1.0; 3]]),
        )
        .unwrap();
        assert_eq!(explicit.voxel_size, [0.125; 3]);

        let voxel = resolve_voxelization_settings(
            bounds,
            Some([0.125; 3]),
            None,
            Some([[0.0; 3], [1.0; 3]]),
        )
        .unwrap();
        assert_eq!(voxel.grid_size, [8, 8, 8]);

        assert!(resolve_voxelization_settings(bounds, None, None, None).is_err());
        assert!(
            resolve_voxelization_settings(bounds, Some([0.0; 3]), None, Some([[0.0; 3], [1.0; 3]]))
                .is_err()
        );
    }

    #[test]
    fn non_power_of_two_textures_sample_base_level() {
        // 3x3 texture: the official kernel would throw building mipmaps; the
        // documented fallback samples the base level.
        let texture = OvoxTexture::new(3, 3, vec![255u8; 3 * 3 * 4]).unwrap();
        let prepared = PreparedTexture::new(&texture);
        assert!(prepared.mips.is_empty());
        let sample = sample_mipmap(&prepared, 0.5, 0.5, 0.001, 0.0);
        assert!((sample[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn power_of_two_textures_build_official_mip_chain() {
        let texture = OvoxTexture::new(4, 4, vec![128u8; 4 * 4 * 4]).unwrap();
        let prepared = PreparedTexture::new(&texture);
        assert_eq!(prepared.mips.len(), 2);
        assert_eq!((prepared.mips[0].0, prepared.mips[0].1), (2, 2));
        assert_eq!((prepared.mips[1].0, prepared.mips[1].1), (1, 1));
        // Uniform texture: every mip level keeps the same value.
        let sample = sample_mipmap(&prepared, 0.25, 0.75, 1.0, 0.0);
        assert!((sample[0] - 128.0 / 255.0).abs() < 1e-6);
    }
}
