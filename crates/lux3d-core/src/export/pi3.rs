use std::path::Path;

use candle_core::Tensor;

use crate::{
    contracts::{ExportPlan, ExportStage, Pi3PointCloud},
    geometry::Pi3PointCloudCpu,
    Result,
};

use super::point_cloud::{write_ply, write_point_cloud, PointCloudFormat, PointCloudView};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pi3ExportStage;

impl ExportStage<Pi3PointCloud> for Pi3ExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &Pi3PointCloud) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "ply".to_string(),
            alternate_extensions: vec!["xyz".to_string()],
            utility_notes: vec![
                "sigmoid(confidence_logits) > confidence_threshold (default 0.1)".to_string(),
                "depth_edge(local_points[..., 2])".to_string(),
                "write_ply(points, rgb); .xyz is a plain-text alternative".to_string(),
            ],
        })
    }
}

impl Pi3ExportStage {
    /// Compute the export mask with an explicit sigmoid-confidence gate
    /// (`sigmoid(confidence_logits) > confidence_threshold`) AND the local-depth
    /// edge mask. `0.1` preserves the reference behavior
    /// (`sigmoid(confidence_logits) > 0.1`).
    pub fn export_mask_with_threshold(
        &self,
        local_points: &Tensor,
        confidence_logits: &Tensor,
        confidence_threshold: f32,
    ) -> Result<Tensor> {
        crate::runtime::point_camera_ops::export_mask_from_local_points_and_confidence_with_threshold(
            local_points,
            confidence_logits,
            confidence_threshold,
        )
        .map_err(|source| crate::error::Lux3dError::CanonicalWeightsValidation {
            message: format!("Pi3 export mask computation failed: {source}"),
        })
    }

    /// Write in the format selected by `path`'s extension (`.ply` or `.xyz`).
    pub fn write_point_cloud(
        &self,
        point_cloud: &Pi3PointCloudCpu,
        path: &Path,
    ) -> Result<PointCloudFormat> {
        let view = PointCloudView {
            points: &point_cloud.points,
            colors: &point_cloud.colors,
        };
        write_point_cloud(&view, path)
    }

    /// Write an ASCII PLY point cloud, preserving the historical output.
    pub fn write_ply(&self, point_cloud: &Pi3PointCloudCpu, path: &Path) -> Result<()> {
        let view = PointCloudView {
            points: &point_cloud.points,
            colors: &point_cloud.colors,
        };
        write_ply(&view, path)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::Pi3ExportStage;

    fn kept_points(mask: &Tensor) -> usize {
        mask.to_dtype(DType::U8)
            .expect("mask u8")
            .flatten_all()
            .expect("mask flatten")
            .to_vec1::<u8>()
            .expect("mask vec")
            .into_iter()
            .filter(|value| *value != 0)
            .count()
    }

    #[test]
    fn export_mask_threshold_filters_confidence() {
        let device = Device::Cpu;
        // One 1x4 frame grid of non-edge points (constant depth 1).
        let local_points = Tensor::ones((1, 1, 1, 4, 3), DType::F32, &device).expect("points");
        // sigmoid(-1) ~= 0.269, sigmoid(0) = 0.5, sigmoid(0.5) ~= 0.622,
        // sigmoid(1) ~= 0.731.
        let confidence_logits =
            Tensor::from_vec(vec![-1.0f32, 0.0, 0.5, 1.0], (1, 1, 1, 4, 1), &device)
                .expect("logits");
        let stage = Pi3ExportStage;

        let default_mask = stage
            .export_mask_with_threshold(&local_points, &confidence_logits, 0.1)
            .expect("default mask");
        assert_eq!(4, kept_points(&default_mask));

        let strict_mask = stage
            .export_mask_with_threshold(&local_points, &confidence_logits, 0.5)
            .expect("strict mask");
        assert_eq!(3, kept_points(&strict_mask));

        let no_mask = stage
            .export_mask_with_threshold(&local_points, &confidence_logits, 0.99)
            .expect("empty mask");
        assert_eq!(0, kept_points(&no_mask));
    }
}
