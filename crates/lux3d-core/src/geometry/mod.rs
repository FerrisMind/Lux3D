mod dual_grid;
mod mesh_bvh;
mod mesh_ops;
mod pbr_bake;
mod pi3;
mod trellis2;
mod triposg;
mod triposr;
mod uv_atlas;
mod vggt;

pub use dual_grid::{
    DualGridVoxelization, MAX_GRID_SIZE_PER_AXIS, MeshToDualGridOptions, mesh_to_flexible_dual_grid,
};
pub use mesh_bvh::{ClosestPoint, MeshBvh};
pub use mesh_ops::{
    MeshEditResult, compact_mesh, fill_holes, fill_holes_with_centroids, merge_close_vertices,
    postprocess_mesh, remove_duplicate_faces, remove_small_connected_components,
    repair_non_manifold_edges, simplify_mesh, unify_face_orientations,
};
pub use pbr_bake::{BakeStats, BakedPbrTexture, COVERAGE_PAD_TEXELS, bake_pbr_texture};
pub use pi3::{Pi3GeometryStage, Pi3PointCloudCpu};
pub use trellis2::{Trellis2GeometryStage, Trellis2MeshBuffers};
pub use triposg::{TripoSgGeometryStage, TripoSgMeshCpu};
pub use triposr::{TripoGeometryStage, TripoMeshCpu};
pub use uv_atlas::{UvAtlas, area_weighted_vertex_normals, uv_unwrap};
pub use vggt::{
    VggtGeometryStage, VggtPointCloudCpu, rescale_intrinsics_to_original,
    shift_point2d_to_original, unproject_depth_to_world,
};
