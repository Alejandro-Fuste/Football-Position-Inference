from typing import Dict, List, Optional, Tuple
import numpy as np

from position_inference.data.schemas import TrackSummary


def compute_spatial_features(
    track_summaries: Dict[int, TrackSummary],
    center_track_id: Optional[int] = None,
    qb_track_id: Optional[int] = None,
    direction: str = "left",
) -> Dict[int, Dict[str, float]]:
    """
    Computes Center-relative normalized coordinates, lateral offset, depth from LOS,
    and spatial ranks for candidate player tracks.
    """
    features: Dict[int, Dict[str, float]] = {}

    player_summaries = {tid: t for tid, t in track_summaries.items() if t.label == "player" and t.presnap_median_footpoint}
    if not player_summaries:
        return features

    heights = [t.median_bbox_height for t in player_summaries.values()]
    scale = float(np.median(heights)) if heights else 100.0
    if scale <= 0:
        scale = 100.0

    if center_track_id and center_track_id in player_summaries:
        center_fp = player_summaries[center_track_id].presnap_median_footpoint
    else:
        fps = [t.presnap_median_footpoint for t in player_summaries.values()]
        center_fp = (float(np.median([fp[0] for fp in fps])), float(np.median([fp[1] for fp in fps])))

    cx, cy = center_fp
    qb_fp = player_summaries[qb_track_id].presnap_median_footpoint if qb_track_id and qb_track_id in player_summaries else None

    all_tids = list(player_summaries.keys())
    x_ranks = {tid: rank for rank, tid in enumerate(sorted(all_tids, key=lambda t: player_summaries[t].presnap_median_footpoint[0]))}
    y_ranks = {tid: rank for rank, tid in enumerate(sorted(all_tids, key=lambda t: player_summaries[t].presnap_median_footpoint[1]))}

    for tid, summary in player_summaries.items():
        px, py = summary.presnap_median_footpoint

        dx = (px - cx) / scale
        dy = (py - cy) / scale

        # Transform to depth_los (positive = defense ahead of LOS, negative = offense backfield)
        # and lateral_offset (negative = right side of offense, positive = left side of offense)
        if direction == "left":
            depth_los = -dx
            lateral_offset = dy
        elif direction == "right":
            depth_los = dx
            lateral_offset = -dy
        elif direction == "up":
            depth_los = -dy
            lateral_offset = dx
        elif direction == "down":
            depth_los = dy
            lateral_offset = -dx
        else:
            depth_los = -dx
            lateral_offset = dy

        dist_center = float(np.sqrt(dx ** 2 + dy ** 2))

        dist_qb = 0.0
        if qb_fp:
            qbx, qby = qb_fp
            dist_qb = float(np.sqrt(((px - qbx) / scale) ** 2 + ((py - qby) / scale) ** 2))

        other_fps = [player_summaries[o].presnap_median_footpoint for o in all_tids if o != tid]
        if other_fps:
            nn_dists = [np.sqrt(((px - ox) / scale) ** 2 + ((py - oy) / scale) ** 2) for ox, oy in other_fps]
            min_nn_dist = float(min(nn_dists))
        else:
            min_nn_dist = 0.0

        feat_dict = {
            "x_norm": float(dx),
            "y_norm": float(dy),
            "depth_los": float(depth_los),
            "lateral_offset": float(lateral_offset),
            "dist_center": dist_center,
            "dist_qb": dist_qb,
            "min_nn_dist": min_nn_dist,
            "x_rank": float(x_ranks[tid]),
            "y_rank": float(y_ranks[tid]),
            "bbox_height_norm": float(summary.median_bbox_height / scale),
            "bbox_width_norm": float(summary.median_bbox_width / scale),
            "bbox_aspect_ratio": float(summary.median_bbox_width / max(summary.median_bbox_height, 1.0)),
            "presnap_motion": float(summary.presnap_motion or 0.0),
        }

        features[tid] = feat_dict

    return features
