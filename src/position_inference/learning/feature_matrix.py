from typing import Dict, List, Optional, Tuple
import numpy as np

from position_inference.data.schemas import TrackSummary
from position_inference.geometry.spatial_features import compute_spatial_features


FEATURE_NAMES = [
    "x_norm",
    "y_norm",
    "depth_los",
    "lateral_offset",
    "dist_center",
    "dist_qb",
    "min_nn_dist",
    "x_rank",
    "y_rank",
    "bbox_height_norm",
    "bbox_width_norm",
    "bbox_aspect_ratio",
    "presnap_motion",
    "action_has_c_anchor",
    "action_has_qb_anchor",
    "action_has_motion",
]


def extract_track_features(
    track_summary: TrackSummary,
    spatial_features: Dict[str, float],
    action_scores: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Extracts a 1D feature vector for a single track.
    """
    if action_scores is None:
        action_scores = {}

    feat_vals = [
        spatial_features.get("x_norm", 0.0),
        spatial_features.get("y_norm", 0.0),
        spatial_features.get("depth_los", 0.0),
        spatial_features.get("lateral_offset", 0.0),
        spatial_features.get("dist_center", 0.0),
        spatial_features.get("dist_qb", 0.0),
        spatial_features.get("min_nn_dist", 0.0),
        spatial_features.get("x_rank", 0.0),
        spatial_features.get("y_rank", 0.0),
        spatial_features.get("bbox_height_norm", 1.0),
        spatial_features.get("bbox_width_norm", 0.5),
        spatial_features.get("bbox_aspect_ratio", 0.5),
        spatial_features.get("presnap_motion", 0.0),
        1.0 if action_scores.get("C", 0.0) >= 0.9 else 0.0,
        1.0 if action_scores.get("QB", 0.0) >= 0.9 else 0.0,
        1.0 if action_scores.get("WR", 0.0) >= 0.8 else 0.0,
    ]

    return np.array(feat_vals, dtype=np.float32)
