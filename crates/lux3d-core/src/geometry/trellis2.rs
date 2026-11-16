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
//! [`Trellis2GeometryStage::postprocess_decoded_mesh`] (hole filling,
//! decimation, cleanup, orientation unification) BEFORE the texture stage
//! attaches per-vertex colors; the voxel payload and `resolution` are never
//! touched by that pass.

use crate::geometry::postprocess_mesh;
use crate::runtime::Trellis2MeshPostprocessOptions;
use crate::{ModelFamily, Result, contracts::Trellis2Mesh, error::Lux3dError};

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
        if any { Some((min, max)) } else { None }
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
    /// `mesh_ops::postprocess_mesh` (the reference `to_glb` cumesh order,
    /// both simplify passes included) over the decoded dual-grid buffers.
    ///
    /// * `enabled = false` returns the input struct untouched (byte-identical
    ///   to the pre-postprocess pipeline).
    /// * `faces` are converted `i64 -> u32` with a bounds check (a negative
    ///   or out-of-range index fails the call instead of being silently
    ///   dropped).
    /// * `voxel_coords` / `voxel_attrs` / `voxel_attr_channels` /
    ///   `resolution` pass through untouched; the post-process only rewrites
    ///   `vertices` / `faces` (and may change their counts — filled holes add
    ///   centroid vertices, decimation and cleanup remove faces and
    ///   vertices).
    /// * `vertex_colors` is `None` on the production path (colors attach
    ///   AFTER this pass, so the final vertices are sampled directly). A
    ///   pre-attached color buffer is transferred through
    ///   `MeshEditResult::vertex_remap`; that remap indexes the post-process
    ///   intermediate buffer (input vertices plus hole-fill centroids), so a
    ///   remap entry pointing past the input vertex count (a fill centroid)
    ///   has no source color and the call fails rather than guessing a
    ///   color. Attach colors after post-processing instead.
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
        let edited = postprocess_mesh(
            &mesh.vertices,
            &faces_u32,
            options.fill_hole_perimeter,
            options.decimation_target,
            options.min_component_area,
        );

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
    fn trellis2_postprocess_options_defaults_match_upstream_to_glb() {
        let options = Trellis2MeshPostprocessOptions::default();
        assert!(options.enabled);
        assert_eq!(3e-2, options.fill_hole_perimeter);
        assert_eq!(1_000_000, options.decimation_target);
        assert_eq!(1e-5, options.min_component_area);
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
        ] {
            options.validate().expect_err("invalid threshold rejected");
            // The same garbage values are allowed while disabled (unused).
            options.enabled = false;
            options.validate().expect("disabled skips checks");
        }
        // Zero thresholds are the documented "off" values; decimation
        // target 0 is the documented "skip simplification" value.
        Trellis2MeshPostprocessOptions {
            fill_hole_perimeter: 0.0,
            min_component_area: 0.0,
            decimation_target: 0,
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
        assert!(
            output
                .faces
                .iter()
                .all(|&i| i >= 0 && (i as usize) < vertex_count)
        );
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
        assert!(
            output
                .faces
                .iter()
                .all(|&i| i >= 0 && (i as usize) < vertex_count)
        );
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
        assert!(
            error
                .to_string()
                .contains("face index outside the vertex buffer")
        );
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
                    ..Trellis2MeshPostprocessOptions::default()
                },
            )
            .expect_err("centroid fill has no source color");
        assert!(error.to_string().contains("vertex colors"));
    }
}
