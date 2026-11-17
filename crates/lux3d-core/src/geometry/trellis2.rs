//! TRELLIS.2 geometry stage: portable dual-grid mesh buffers.
//!
//! The runtime pipeline decodes the shape SLAT and extracts the O-Voxel
//! dual-grid mesh (`runtime::trellis2_mesh`); this module holds the plain
//! CPU buffers plus voxel payload that the export stage and the
//! `Trellis2Mesh` contract consume. The reference coordinate frame is kept
//! unchanged in the buffers: vertices live in the reference `[-0.5, 0.5]^3`
//! box with no axis remap (the GLB writer applies the upstream glTF axis
//! conversion at export time only).
//!
//! When the run options enable mesh post-processing
//! (`runtime::Trellis2MeshPostprocessOptions`), the pipeline rewrites the
//! `vertices`/`faces` of the freshly assembled buffers through
//! [`Trellis2GeometryStage::postprocess_decoded_mesh`] BEFORE the texture
//! stage attaches per-vertex colors; the voxel payload and `resolution` are
//! never touched by that pass.
//!
//! The post-process has two mutually exclusive branches, mirroring upstream
//! `to_glb` (`postprocess.py:108-186`): the shared pre-branch
//! `fill_holes(fill_hole_perimeter)`, then either the narrow-band
//! dual-contouring remesh + single simplify (default; shipped entry-point
//! values `remesh=True, band=1, project=0`) or the cleanup/simplify chain.
//! Remesh input is always the canonical `[-0.5, 0.5]^3` frame the dual-grid
//! extraction produces, with `resolution = mesh.resolution` — exactly the
//! `aabb`/`grid_size` the official call sites pass (`example.py:33-48`,
//! `app.py:493-512`).

use crate::geometry::{
    compact_mesh, fill_holes_with_centroids, postprocess_mesh, remesh_narrow_band_dc,
    simplify_mesh, MeshBvh, MeshEditResult, RemeshDcOptions,
};
use crate::runtime::Trellis2MeshPostprocessOptions;
use crate::{contracts::Trellis2Mesh, error::Lux3dError, ModelFamily, Result};

/// Canonical AABB of the decoded dual-grid mesh (upstream `to_glb` call
/// sites pass `[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]`, `example.py:42`,
/// `app.py:502`). The remesh grid center is its midpoint and its scale the
/// largest extent (`postprocess.py:166-168`).
const CANONICAL_AABB: [[f32; 3]; 2] = [[-0.5; 3], [0.5; 3]];

/// CPU mesh + voxel payload produced by the TRELLIS.2 shape decode.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Trellis2MeshBuffers {
    /// Voxel grid resolution the mesh was extracted at (512 for the 512
    /// pipeline).
    pub resolution: u32,
    /// Flat `[x, y, z, ...]` f32 vertex positions, one triple per active
    /// voxel (dual-grid vertices; may include unused voxels).
    pub vertices: Vec<f32>,
    /// Flat zero-based triangle indices.
    pub faces: Vec<i64>,
    /// Flat `[x, y, z, ...]` i32 active-voxel coordinates (row order matches
    /// `voxel_attrs` rows).
    pub voxel_coords: Vec<i32>,
    /// Flat `[N, attrs]` decoded per-voxel attributes (7 channels for the
    /// shape decode; 6 PBR channels when the texture stage ran).
    pub voxel_attrs: Vec<f32>,
    /// Attribute channel count (`voxel_attrs.len() / voxel_count`).
    pub voxel_attr_channels: usize,
    /// Optional per-vertex base colors sampled from the decoded PBR
    /// attributes (flat `[V, 3]` RGB rows in [0, 1]; `None` for
    /// geometry-only runs).
    pub vertex_colors: Option<Vec<f32>>,
}

impl Trellis2MeshBuffers {
    /// Number of dual-grid vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / 3
    }

    /// Number of triangles.
    pub fn face_count(&self) -> usize {
        self.faces.len() / 3
    }

    /// Number of active voxels carrying attributes.
    pub fn voxel_count(&self) -> usize {
        if self.voxel_attr_channels == 0 {
            0
        } else {
            self.voxel_attrs.len() / self.voxel_attr_channels
        }
    }

    /// Axis-aligned vertex bounds `(min, max)`, or `None` for an empty mesh.
    pub fn bbox(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let mut any = false;
        for vertex in self.vertices.chunks_exact(3) {
            any = true;
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        if any {
            Some((min, max))
        } else {
            None
        }
    }

    /// True when any vertex is non-finite.
    pub fn has_non_finite_vertex(&self) -> bool {
        self.vertices.iter().any(|v| !v.is_finite())
    }

    /// Structural validation: buffer length divisibility, finite vertices,
    /// face indices in range, voxel payload consistency.
    pub fn validate(&self) -> Result<()> {
        if self.vertices.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 mesh vertex buffer length must be a multiple of 3",
            ));
        }
        if self.faces.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 mesh face buffer length must be a multiple of 3",
            ));
        }
        if self.has_non_finite_vertex() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 mesh contains a non-finite vertex",
            ));
        }
        let vertex_count = self.vertex_count();
        for &index in &self.faces {
            if index < 0 || (vertex_count > 0 && index as usize >= vertex_count) {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 mesh face index outside the vertex buffer",
                ));
            }
        }
        if self.voxel_coords.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 voxel coordinate buffer length must be a multiple of 3",
            ));
        }
        if self.voxel_attr_channels > 0 && self.voxel_attrs.len() % self.voxel_attr_channels != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 voxel attribute buffer length does not match the channel count",
            ));
        }
        if let Some(colors) = &self.vertex_colors {
            if colors.len() != self.vertex_count() * 3 {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 vertex color buffer must hold three channels per vertex",
                ));
            }
        }
        Ok(())
    }

    /// Convert into the serialized `Trellis2Mesh` contract struct.
    pub fn to_contract(&self) -> Result<Trellis2Mesh> {
        self.validate()?;
        Ok(Trellis2Mesh {
            family: ModelFamily::Trellis2,
            vertices: self.vertices.clone(),
            faces: self.faces.clone(),
            voxel_coords: self.voxel_coords.clone(),
            voxel_attrs: self.voxel_attrs.clone(),
            resolution: self.resolution,
        })
    }
}

/// Static geometry-stage descriptor for the TRELLIS.2 family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Trellis2GeometryStage;

impl Trellis2GeometryStage {
    /// Assemble mesh buffers from the decode products. Faces and vertices
    /// come straight from the dual-grid extraction; the voxel payload keeps
    /// the active-voxel coordinates and decoded attributes (the texture
    /// stage swaps them for the 6 PBR channels and fills `vertex_colors`).
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        &self,
        resolution: u32,
        vertices: Vec<f32>,
        faces: Vec<i64>,
        voxel_coords: Vec<i32>,
        voxel_attrs: Vec<f32>,
        voxel_attr_channels: usize,
    ) -> Result<Trellis2MeshBuffers> {
        let buffers = Trellis2MeshBuffers {
            resolution,
            vertices,
            faces,
            voxel_coords,
            voxel_attrs,
            voxel_attr_channels,
            vertex_colors: None,
        };
        buffers.validate()?;
        Ok(buffers)
    }

    /// Upstream-style geometry post-process of an assembled TRELLIS.2 mesh:
    /// either the narrow-band dual-contouring remesh + single simplify pass
    /// (`options.remesh`, the default) or `mesh_ops::postprocess_mesh` (the
    /// reference `to_glb` cleanup order, both simplify passes included) over
    /// the decoded dual-grid buffers. Both branches start from the upstream
    /// shared `fill_holes(fill_hole_perimeter)` pre-pass
    /// (`postprocess.py:108-113`).
    ///
    /// * `enabled = false` returns the input struct untouched (byte-identical
    ///   to the pre-postprocess pipeline).
    /// * `faces` are converted `i64 -> u32` with a bounds check (a negative
    ///   or out-of-range index fails the call instead of being silently
    ///   dropped).
    /// * The remesh branch rebuilds the topology from scratch: vertices are
    ///   QEF-placed in the grid cells, so pre-attached vertex colors cannot
    ///   be transferred (the call fails descriptively; attach colors after
    ///   post-processing, as the production path does).
    /// * `voxel_coords` / `voxel_attrs` / `voxel_attr_channels` /
    ///   `resolution` pass through untouched; the post-process only rewrites
    ///   `vertices` / `faces` (and may change their counts — filled holes add
    ///   centroid vertices, decimation and cleanup remove faces and
    ///   vertices, remesh replaces the whole topology).
    /// * `vertex_colors` is `None` on the production path (colors attach
    ///   AFTER this pass, so the final vertices are sampled directly). A
    ///   pre-attached color buffer is transferred through
    ///   `MeshEditResult::vertex_remap` in the cleanup branch; that remap
    ///   indexes the post-process intermediate buffer (input vertices plus
    ///   hole-fill centroids), so a remap entry pointing past the input
    ///   vertex count (a fill centroid) has no source color and the call
    ///   fails rather than guessing a color. Attach colors after
    ///   post-processing instead.
    pub fn postprocess_decoded_mesh(
        &self,
        mesh: Trellis2MeshBuffers,
        options: &Trellis2MeshPostprocessOptions,
    ) -> Result<Trellis2MeshBuffers> {
        options.validate()?;
        if !options.enabled {
            return Ok(mesh);
        }
        let vertex_count = mesh.vertex_count();
        if vertex_count > u32::MAX as usize {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 mesh post-process supports at most u32::MAX vertices",
            ));
        }
        if options.remesh && mesh.vertex_colors.is_some() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 remesh rebuilds the topology, so pre-attached vertex colors cannot \
                 be transferred; attach colors after post-processing",
            ));
        }
        let mut faces_u32 = Vec::with_capacity(mesh.faces.len());
        for &index in &mesh.faces {
            let index = usize::try_from(index).map_err(|_| {
                Lux3dError::InvalidInput(
                    "Trellis2 mesh post-process rejected a negative face index",
                )
            })?;
            if index >= vertex_count {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 mesh post-process rejected a face index outside the vertex buffer",
                ));
            }
            faces_u32.push(index as u32);
        }
        let edited = if options.remesh {
            postprocess_mesh_remesh(&mesh.vertices, &faces_u32, options, mesh.resolution)?
        } else {
            postprocess_mesh(
                &mesh.vertices,
                &faces_u32,
                options.fill_hole_perimeter,
                options.decimation_target,
                options.min_component_area,
            )
        };

        // Defensive transfer of a pre-attached color buffer. `vertex_remap`
        // indexes the post-process intermediate buffer; entries past the
        // input vertex count are hole-fill centroids with no source color.
        let vertex_colors = match &mesh.vertex_colors {
            Some(colors) => {
                if colors.len() != vertex_count * 3 {
                    return Err(Lux3dError::InvalidInput(
                        "Trellis2 mesh post-process needs three color channels per input vertex",
                    ));
                }
                let mut remapped = Vec::with_capacity(edited.vertex_remap.len() * 3);
                for &old in &edited.vertex_remap {
                    let old = old as usize;
                    if old >= vertex_count {
                        return Err(Lux3dError::InvalidInput(
                            "Trellis2 mesh post-process filled a hole with a new vertex, so the \
                             pre-attached vertex colors cannot be remapped; attach colors after \
                             post-processing",
                        ));
                    }
                    remapped.extend_from_slice(&colors[old * 3..old * 3 + 3]);
                }
                Some(remapped)
            }
            None => None,
        };

        let out = Trellis2MeshBuffers {
            resolution: mesh.resolution,
            vertices: edited.vertices,
            faces: edited.faces.iter().map(|&index| index as i64).collect(),
            voxel_coords: mesh.voxel_coords,
            voxel_attrs: mesh.voxel_attrs,
            voxel_attr_channels: mesh.voxel_attr_channels,
            vertex_colors,
        };
        out.validate()?;
        Ok(out)
    }
}

/// Upstream `to_glb` remesh branch (`postprocess.py:171-186`): shared
/// `fill_holes` pre-pass, triangle BVH build over the cleaned mesh, the
/// narrow-band dual-contouring remesh on the canonical `[-0.5, 0.5]^3`
/// frame, then a single `simplify(decimation_target)` pass.
///
/// Grid arguments exactly as the official call site: `resolution =
/// grid_size.max()` (the mesh extraction resolution),
/// `center = aabb.mean(dim=0)`, `scale = (resolution + 3 * band) / resolution
/// * (aabb[1] - aabb[0]).max()` — [`RemeshDcOptions::from_aabb`] applies that
/// arithmetic and [`crate::geometry::narrow_band_grid_scale`] owns the
/// expansion formula.
///
/// The upstream remesh branch skips the duplicate-face / non-manifold /
/// small-component / orientation-unification chain, so this helper does not
/// run it. The remesher emits quads; they are split along the 0-2 diagonal
/// (the same diagonal convention as the O-Voxel quad splitter) because the
/// meshopt simplifier and the export writers are triangle-only.
fn postprocess_mesh_remesh(
    vertices: &[f32],
    faces: &[u32],
    options: &Trellis2MeshPostprocessOptions,
    resolution: u32,
) -> Result<MeshEditResult> {
    let (vertices, faces) = fill_holes_with_centroids(vertices, faces, options.fill_hole_perimeter);
    let bvh = MeshBvh::build(&vertices, &faces)?;
    let remesh_options = RemeshDcOptions::from_aabb(
        CANONICAL_AABB,
        resolution,
        options.remesh_band,
        options.remesh_project,
    );
    let remeshed = remesh_narrow_band_dc(&vertices, &faces, &bvh, &remesh_options)?;
    if remeshed.faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 remesh produced no faces (empty or fully degenerate input surface)",
        ));
    }
    debug_assert_eq!(remeshed.faces.len() % 4, 0, "the remesher emits quads");
    let mut triangles = Vec::with_capacity(remeshed.faces.len() / 2);
    for quad in remeshed.faces.chunks_exact(4) {
        triangles.extend_from_slice(&[quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]);
    }
    Ok(if options.decimation_target > 0 {
        simplify_mesh(&remeshed.vertices, &triangles, options.decimation_target)
    } else {
        compact_mesh(&remeshed.vertices, &triangles)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube: 8 vertices, 12 triangles, outward winding (same layout as
    /// the `mesh_ops` tests).
    fn cube_vertices() -> Vec<f32> {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .iter()
        .flatten()
        .copied()
        .collect()
    }

    fn cube_faces() -> Vec<i64> {
        [
            [0, 2, 1],
            [0, 3, 2], // bottom (-z)
            [4, 5, 6],
            [4, 6, 7], // top (+z)
            [0, 1, 5],
            [0, 5, 4], // front (-y)
            [2, 3, 7],
            [2, 7, 6], // back (+y)
            [0, 4, 7],
            [0, 7, 3], // left (-x)
            [1, 2, 6],
            [1, 6, 5], // right (+x)
        ]
        .iter()
        .flatten()
        .map(|&i| i as i64)
        .collect()
    }

    /// Assembled-buffer fixture: the cube (top face = 2 triangles removed,
    /// square hole with perimeter 4.0) plus a tiny disconnected 0.05 cube
    /// far away (fresh vertex slots 8..16, total area 0.015).
    fn holey_cube_with_tiny_component() -> Trellis2MeshBuffers {
        let faces = cube_faces();
        // Drop the whole top quad (flat indices 6..12).
        let mut faces: Vec<i64> = faces[..6]
            .iter()
            .chain(faces[12..].iter())
            .copied()
            .collect();
        let mut vertices = cube_vertices();
        // Tiny cube at (10, 10, 10), side 0.05, vertices 8..16.
        for v in 0..8 {
            let base = [10.0f32, 10.0, 10.0];
            for axis in 0..3 {
                vertices.push(base[axis] + vertices[v * 3 + axis] * 0.05);
            }
        }
        for face in cube_faces() {
            faces.push(face + 8);
        }
        Trellis2MeshBuffers {
            resolution: 512,
            vertices,
            faces,
            voxel_coords: vec![0, 0, 0, 1, 0, 0],
            voxel_attrs: vec![0.5; 14],
            voxel_attr_channels: 7,
            vertex_colors: None,
        }
    }

    #[test]
    fn trellis2_postprocess_options_defaults_match_shipped_entry_points() {
        let options = Trellis2MeshPostprocessOptions::default();
        assert!(options.enabled);
        assert_eq!(3e-2, options.fill_hole_perimeter);
        assert_eq!(1_000_000, options.decimation_target);
        assert_eq!(1e-5, options.min_component_area);
        // Shipped entry points: example.py:43 / app.py:503 pass
        // remesh=True, remesh_band=1, remesh_project=0.
        assert!(options.remesh);
        assert_eq!(1.0, options.remesh_band);
        assert_eq!(0.0, options.remesh_project);
        options.validate().expect("defaults valid");
    }

    #[test]
    fn trellis2_postprocess_options_validate_thresholds() {
        // Negative / non-finite thresholds are rejected while enabled.
        for mut options in [
            Trellis2MeshPostprocessOptions {
                fill_hole_perimeter: -1.0,
                ..Trellis2MeshPostprocessOptions::default()
            },
            Trellis2MeshPostprocessOptions {
                fill_hole_perimeter: f32::NAN,
                ..Trellis2MeshPostprocessOptions::default()
            },
            Trellis2MeshPostprocessOptions {
                fill_hole_perimeter: f32::INFINITY,
                ..Trellis2MeshPostprocessOptions::default()
            },
            Trellis2MeshPostprocessOptions {
                min_component_area: -1e-6,
                ..Trellis2MeshPostprocessOptions::default()
            },
            Trellis2MeshPostprocessOptions {
                min_component_area: f32::NEG_INFINITY,
                ..Trellis2MeshPostprocessOptions::default()
            },
            Trellis2MeshPostprocessOptions {
                remesh_band: -0.5,
                ..Trellis2MeshPostprocessOptions::default()
            },
            Trellis2MeshPostprocessOptions {
                remesh_band: f32::NAN,
                ..Trellis2MeshPostprocessOptions::default()
            },
            Trellis2MeshPostprocessOptions {
                remesh_project: f32::NAN,
                ..Trellis2MeshPostprocessOptions::default()
            },
            Trellis2MeshPostprocessOptions {
                remesh_project: f32::INFINITY,
                ..Trellis2MeshPostprocessOptions::default()
            },
        ] {
            options.validate().expect_err("invalid threshold rejected");
            // The same garbage values are allowed while disabled (unused).
            options.enabled = false;
            options.validate().expect("disabled skips checks");
        }
        // Remesh parameters are unused (and unchecked) when the cleanup
        // branch is selected.
        Trellis2MeshPostprocessOptions {
            remesh: false,
            remesh_band: f32::NAN,
            remesh_project: f32::NAN,
            ..Trellis2MeshPostprocessOptions::default()
        }
        .validate()
        .expect("remesh knobs unused in the cleanup branch");
        // Zero thresholds are the documented "off" values; decimation
        // target 0 is the documented "skip simplification" value; a zero
        // remesh band is a valid (sub-cell) band.
        Trellis2MeshPostprocessOptions {
            fill_hole_perimeter: 0.0,
            min_component_area: 0.0,
            decimation_target: 0,
            remesh_band: 0.0,
            ..Trellis2MeshPostprocessOptions::default()
        }
        .validate()
        .expect("zero thresholds valid");
    }

    #[test]
    fn trellis2_postprocess_disabled_keeps_mesh_byte_identical() {
        let stage = Trellis2GeometryStage;
        let input = holey_cube_with_tiny_component();
        let output = stage
            .postprocess_decoded_mesh(
                input.clone(),
                &Trellis2MeshPostprocessOptions {
                    enabled: false,
                    ..Trellis2MeshPostprocessOptions::default()
                },
            )
            .expect("disabled pass-through");
        assert_eq!(input, output);
    }

    #[test]
    fn trellis2_postprocess_fills_hole_and_drops_tiny_component() {
        let stage = Trellis2GeometryStage;
        let input = holey_cube_with_tiny_component();
        let input_faces = input.face_count();
        let output = stage
            .postprocess_decoded_mesh(
                input.clone(),
                &Trellis2MeshPostprocessOptions {
                    fill_hole_perimeter: 4.001,
                    // Above the tiny cube's 0.015 total area: it is removed.
                    min_component_area: 0.02,
                    // Far above any face count: no decimation.
                    decimation_target: 1_000_000,
                    remesh: false,
                    ..Trellis2MeshPostprocessOptions::default()
                },
            )
            .expect("postprocess");

        // The square hole (perimeter 4.0 < threshold) is closed with a
        // centroid fan; the tiny component is gone.
        assert_eq!(output.face_count(), 14, "10 kept + 4 fan triangles");
        assert_eq!(output.vertex_count(), 9, "8 cube vertices + 1 centroid");

        // Every face index is in range and the mesh validates.
        let vertex_count = output.vertex_count();
        assert!(output
            .faces
            .iter()
            .all(|&i| i >= 0 && (i as usize) < vertex_count));
        output.validate().expect("buffers valid");

        // The voxel payload, resolution and channel count pass through
        // untouched; colors stay unattached before the texture stage.
        assert_eq!(output.resolution, input.resolution);
        assert_eq!(output.voxel_coords, input.voxel_coords);
        assert_eq!(output.voxel_attrs, input.voxel_attrs);
        assert_eq!(output.voxel_attr_channels, input.voxel_attr_channels);
        assert_eq!(None, output.vertex_colors);
        assert_eq!(input_faces, 22);
    }

    #[test]
    fn trellis2_postprocess_respects_decimation_target_and_index_range() {
        let stage = Trellis2GeometryStage;
        let input = holey_cube_with_tiny_component();
        let output = stage
            .postprocess_decoded_mesh(
                input,
                &Trellis2MeshPostprocessOptions {
                    fill_hole_perimeter: 4.001,
                    min_component_area: 0.02,
                    decimation_target: 8,
                    remesh: false,
                    ..Trellis2MeshPostprocessOptions::default()
                },
            )
            .expect("postprocess with decimation");
        let face_count = output.face_count();
        assert!(
            face_count > 0 && face_count <= 8,
            "decimation target respected, got {face_count}"
        );
        let vertex_count = output.vertex_count();
        assert!(output
            .faces
            .iter()
            .all(|&i| i >= 0 && (i as usize) < vertex_count));
        assert_eq!(None, output.vertex_colors);
    }

    /// Decimation target 0 = cleanup-only: hole filling and component
    /// removal still run, no simplify pass touches the mesh.
    #[test]
    fn trellis2_postprocess_zero_target_skips_simplification() {
        let stage = Trellis2GeometryStage;
        let input = holey_cube_with_tiny_component();
        let output = stage
            .postprocess_decoded_mesh(
                input,
                &Trellis2MeshPostprocessOptions {
                    fill_hole_perimeter: 4.001,
                    min_component_area: 0.02,
                    decimation_target: 0,
                    remesh: false,
                    ..Trellis2MeshPostprocessOptions::default()
                },
            )
            .expect("cleanup-only postprocess");
        assert_eq!(output.face_count(), 14);
        assert_eq!(output.vertex_count(), 9);
    }

    #[test]
    fn trellis2_postprocess_rejects_out_of_range_face_indices() {
        let stage = Trellis2GeometryStage;
        let mut input = holey_cube_with_tiny_component();
        input.faces.push(0);
        input.faces.push(1);
        input.faces.push(999);
        let error = stage
            .postprocess_decoded_mesh(input, &Trellis2MeshPostprocessOptions::default())
            .expect_err("out-of-range face must fail");
        assert!(error
            .to_string()
            .contains("face index outside the vertex buffer"));
    }

    #[test]
    fn trellis2_postprocess_transfers_pre_attached_colors_when_possible() {
        let stage = Trellis2GeometryStage;
        // Closed cube (no hole, no decimation): the remap stays inside the
        // input vertex buffer, so colors transfer.
        let input = Trellis2MeshBuffers {
            resolution: 8,
            vertices: cube_vertices(),
            faces: cube_faces(),
            voxel_coords: vec![0, 0, 0],
            voxel_attrs: vec![0.5; 7],
            voxel_attr_channels: 7,
            vertex_colors: Some((0..8).flat_map(|i| [i as f32; 3]).collect()),
        };
        let output = stage
            .postprocess_decoded_mesh(
                input.clone(),
                &Trellis2MeshPostprocessOptions {
                    decimation_target: 1_000_000,
                    remesh: false,
                    ..Trellis2MeshPostprocessOptions::default()
                },
            )
            .expect("closed cube keeps colors");
        let colors = output.vertex_colors.clone().expect("colors remapped");
        assert_eq!(colors.len(), output.vertex_count() * 3);
        // Every output vertex color equals the color of its remap source
        // (first-use order on a closed undecimated cube is identity here).
        for (index, row) in colors.chunks_exact(3).enumerate() {
            assert_eq!(row, &[index as f32; 3]);
        }

        // A hole that closes with a NEW centroid vertex has no source color:
        // the call fails instead of guessing.
        let mut holey = input;
        holey.faces.truncate(30); // drop the top quad
        let error = stage
            .postprocess_decoded_mesh(
                holey,
                &Trellis2MeshPostprocessOptions {
                    fill_hole_perimeter: 4.001,
                    decimation_target: 1_000_000,
                    remesh: false,
                    ..Trellis2MeshPostprocessOptions::default()
                },
            )
            .expect_err("centroid fill has no source color");
        assert!(error.to_string().contains("vertex colors"));
    }

    /// Closed cube in the canonical `[-0.5, 0.5]^3` frame the dual-grid
    /// extraction produces (the remesh grid is centered on that AABB).
    fn canonical_cube_input() -> Trellis2MeshBuffers {
        let vertices: Vec<f32> = cube_vertices()
            .chunks_exact(3)
            .flat_map(|v| [v[0] - 0.5, v[1] - 0.5, v[2] - 0.5])
            .collect();
        Trellis2MeshBuffers {
            resolution: 16,
            vertices,
            faces: cube_faces(),
            voxel_coords: vec![0, 0, 0],
            voxel_attrs: vec![0.5; 7],
            voxel_attr_channels: 7,
            vertex_colors: None,
        }
    }

    /// Every undirected triangle edge must be used exactly twice (closed,
    /// manifold surface).
    fn assert_closed_triangle_mesh(mesh: &Trellis2MeshBuffers) {
        use std::collections::HashMap;
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        for face in mesh.faces.chunks_exact(3) {
            let (a, b, c) = (face[0] as u32, face[1] as u32, face[2] as u32);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = (u.min(v), u.max(v));
                *edges.entry(key).or_insert(0) += 1;
            }
        }
        assert!(!edges.is_empty(), "mesh has no edges");
        for (edge, count) in edges {
            assert_eq!(2, count, "edge {edge:?} used {count} times");
        }
    }

    /// Default branch (remesh) rebuilds the topology: the output vertex /
    /// face counts differ from the input, the surface stays closed and the
    /// voxel payload passes through untouched.
    #[test]
    fn trellis2_remesh_branch_rebuilds_topology_and_stays_closed() {
        let stage = Trellis2GeometryStage;
        let input = canonical_cube_input();
        let options = Trellis2MeshPostprocessOptions {
            // Keep every remeshed triangle (no simplification) so the
            // watertight quad-derived surface is observable.
            decimation_target: 0,
            ..Trellis2MeshPostprocessOptions::default()
        };
        assert!(options.remesh, "shipped default is remesh on");
        let output = stage
            .postprocess_decoded_mesh(input.clone(), &options)
            .expect("remesh postprocess");

        assert_ne!(input.face_count(), output.face_count());
        assert_ne!(input.vertex_count(), output.vertex_count());
        assert!(output.face_count() > 0);
        assert_eq!(0, output.faces.len() % 3);
        assert!(output
            .faces
            .iter()
            .all(|&i| i >= 0 && (i as usize) < output.vertex_count()));
        assert_closed_triangle_mesh(&output);
        output.validate().expect("buffers valid");

        // Voxel payload, resolution and channel count are untouched.
        assert_eq!(input.resolution, output.resolution);
        assert_eq!(input.voxel_coords, output.voxel_coords);
        assert_eq!(input.voxel_attrs, output.voxel_attrs);
        assert_eq!(input.voxel_attr_channels, output.voxel_attr_channels);
        assert_eq!(None, output.vertex_colors);
    }

    /// `remesh = false` is the pre-remesh-port pipeline, byte-for-byte:
    /// the same `mesh_ops::postprocess_mesh` call the old code made.
    #[test]
    fn trellis2_no_remesh_branch_is_byte_identical_to_cleanup_pipeline() {
        let stage = Trellis2GeometryStage;
        let input = holey_cube_with_tiny_component();
        let options = Trellis2MeshPostprocessOptions {
            fill_hole_perimeter: 4.001,
            min_component_area: 0.02,
            decimation_target: 32,
            remesh: false,
            ..Trellis2MeshPostprocessOptions::default()
        };
        let output = stage
            .postprocess_decoded_mesh(input.clone(), &options)
            .expect("cleanup postprocess");

        let faces_u32: Vec<u32> = input.faces.iter().map(|&i| i as u32).collect();
        let expected = postprocess_mesh(
            &input.vertices,
            &faces_u32,
            options.fill_hole_perimeter,
            options.decimation_target,
            options.min_component_area,
        );
        let expected_faces: Vec<i64> = expected.faces.iter().map(|&i| i as i64).collect();
        assert_eq!(expected.vertices, output.vertices);
        assert_eq!(expected_faces, output.faces);
    }

    /// The remesh branch feeds the remeshed surface through a single
    /// simplify pass toward `decimation_target`.
    #[test]
    fn trellis2_remesh_branch_respects_decimation_target() {
        let stage = Trellis2GeometryStage;
        let output = stage
            .postprocess_decoded_mesh(
                canonical_cube_input(),
                &Trellis2MeshPostprocessOptions {
                    decimation_target: 8,
                    ..Trellis2MeshPostprocessOptions::default()
                },
            )
            .expect("remesh with decimation");
        let face_count = output.face_count();
        assert!(
            face_count > 0 && face_count <= 8,
            "decimation target respected, got {face_count}"
        );
        assert_eq!(None, output.vertex_colors);
    }

    /// Remesh replaces every vertex, so pre-attached colors have no remap
    /// source: the call fails descriptively instead of guessing.
    #[test]
    fn trellis2_remesh_branch_rejects_pre_attached_colors() {
        let stage = Trellis2GeometryStage;
        let mut input = canonical_cube_input();
        input.vertex_colors = Some(vec![0.5; input.vertex_count() * 3]);
        let error = stage
            .postprocess_decoded_mesh(input, &Trellis2MeshPostprocessOptions::default())
            .expect_err("remesh cannot remap vertex colors");
        assert!(error.to_string().contains("remesh rebuilds the topology"));
    }
}
