mod dual_grid;
mod mesh_bvh;
mod mesh_ops;
mod pbr_bake;
mod pi3;
mod remesh_dc;
mod trellis2;
mod triposg;
mod triposr;
mod uv_atlas;
mod vggt;

pub use dual_grid::{
    mesh_to_flexible_dual_grid, DualGridVoxelization, MeshToDualGridOptions, MAX_GRID_SIZE_PER_AXIS,
};
pub use mesh_bvh::{ClosestPoint, MeshBvh};
pub use mesh_ops::{
    compact_mesh, fill_holes, fill_holes_with_centroids, merge_close_vertices, postprocess_mesh,
    remove_duplicate_faces, remove_small_connected_components, repair_non_manifold_edges,
    simplify_mesh, unify_face_orientations, MeshEditResult,
};
pub use pbr_bake::{
    bake_pbr_texture, bake_pbr_texture_in_grid, bake_pbr_texture_with_snap,
    bake_pbr_texture_with_snap_in_grid, BakeStats, BakedPbrTexture, PbrSampleGrid,
    COVERAGE_PAD_TEXELS,
};
pub use pi3::{Pi3GeometryStage, Pi3PointCloudCpu};
pub use remesh_dc::{narrow_band_grid_scale, remesh_narrow_band_dc, RemeshDcMesh, RemeshDcOptions};
pub use trellis2::{Trellis2GeometryStage, Trellis2MeshBuffers};
pub use triposg::{TripoSgGeometryStage, TripoSgMeshCpu};
pub use triposr::{TripoGeometryStage, TripoMeshCpu};
pub use uv_atlas::{
    area_weighted_vertex_normals, uv_unwrap, welded_vertex_normals, UvAtlas,
};
pub use vggt::{
    rescale_intrinsics_to_original, shift_point2d_to_original, unproject_depth_to_world,
    VggtGeometryStage, VggtPointCloudCpu,
};
