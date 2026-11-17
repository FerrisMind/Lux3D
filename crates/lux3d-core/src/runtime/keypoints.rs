//! Shared keypoint-detector types for VGGT tracking stage A.
//!
//! The official tracker selects query points with ALIKED + SuperPoint
//! (`vggt/dependency/track_predict.py:67-69,172`). Both extractors return the
//! same payload: pixel-center keypoints in the input image, their detection
//! scores, and L2-normalized descriptors. This module keeps that payload
//! backend-neutral so [`super::aliked`] and [`super::superpoint`] can share
//! it and the VGGT integration can concatenate both detector outputs exactly
//! like `extract_keypoints` (`vggt/dependency/vggsfm_utils.py:227-255`).

/// Keypoints, scores and descriptors produced by one detector pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeypointSet {
    /// Pixel-center coordinates in the detector input image, `[x, y]`.
    pub keypoints: Vec<[f32; 2]>,
    /// Detection score per keypoint, in the detector's own scale.
    pub scores: Vec<f32>,
    /// Row-major descriptor matrix, `keypoints.len() * descriptor_dim` values.
    pub descriptors: Vec<f32>,
    /// Descriptor width per keypoint (0 when descriptors were not requested).
    pub descriptor_dim: usize,
}

impl KeypointSet {
    /// Appends `other` after `self`, matching `torch.cat` along the keypoint
    /// axis in `extract_keypoints`.
    pub fn append(&mut self, other: &KeypointSet) {
        let previous_len = self.keypoints.len();
        let previous_dim = self.descriptor_dim;
        let previous_valid = self.descriptors.len() == previous_len * previous_dim;
        self.keypoints.extend_from_slice(&other.keypoints);
        self.scores.extend_from_slice(&other.scores);

        let other_valid = other.descriptors.len() == other.keypoints.len() * other.descriptor_dim;
        if previous_len == 0 && previous_dim == 0 {
            // No rows to preserve: adopt the other descriptor matrix.
            if other_valid {
                self.descriptors = other.descriptors.clone();
                self.descriptor_dim = other.descriptor_dim;
            }
            return;
        }
        if !previous_valid || !other_valid || other.descriptor_dim == 0 {
            self.descriptors.clear();
            self.descriptor_dim = 0;
            return;
        }
        if previous_dim == other.descriptor_dim {
            self.descriptors.extend_from_slice(&other.descriptors);
            return;
        }
        // Mixed descriptor widths cannot share one matrix. Preserve the
        // keypoints/scores and drop descriptors instead of writing a
        // malformed matrix; callers that need descriptors use one detector.
        self.descriptors.clear();
        self.descriptor_dim = 0;
    }

    /// Number of keypoints.
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
}

/// Detector call options shared by ALIKED and SuperPoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeypointOptions {
    /// `max_num_keypoints` (official demo passes `--max_query_pts`, 4096).
    pub max_keypoints: usize,
    /// `detection_threshold` (official tracker uses `det_thres=0.005`,
    /// `vggsfm_utils.py:189`).
    pub detection_threshold: f32,
    /// Long-side resize before the detector (`Extractor.preprocess_conf`
    /// uses 1024). `None` keeps the input resolution (tests).
    pub resize_long_side: Option<usize>,
}

impl Default for KeypointOptions {
    fn default() -> Self {
        Self {
            max_keypoints: 4096,
            detection_threshold: 0.005,
            resize_long_side: Some(1024),
        }
    }
}
