from typing import Dict, List, Optional, Tuple
import numpy as np

from position_inference.data.schemas import TrackSummary


def _position_footpoint(summary: TrackSummary):
    return (
        summary.formation_anchor_footpoint
        or summary.presnap_median_footpoint
        or summary.median_footpoint
    )


def compute_spatial_features(
    track_summaries: Dict[int, TrackSummary],
    center_track_id: Optional[int] = None,
    qb_track_id: Optional[int] = None,
    direction: str = "left",
) -> Dict[int, Dict[str, float]]:
    """Compute normalized formation geometry from immutable position anchors.

    Position inference is anchored to the earliest reliable formation location. Later
    pre-snap statistics remain available for motion/quality diagnostics, but they do not
    move an already-established track's position geometry.
    """
    features: Dict[int, Dict[str, float]] = {}

    player_summaries = {
        tid: t
        for tid, t in track_summaries.items()
        if t.label == "player" and _position_footpoint(t)
    }
    if not player_summaries:
        return features

    heights = [t.median_bbox_height for t in player_summaries.values()]
    scale = float(np.median(heights)) if heights else 100.0
    if scale <= 0:
        scale = 100.0

    if center_track_id and center_track_id in player_summaries:
        center_fp = _position_footpoint(player_summaries[center_track_id])
    else:
        fps = [_position_footpoint(t) for t in player_summaries.values()]
        center_fp = (
            float(np.median([fp[0] for fp in fps])),
            float(np.median([fp[1] for fp in fps])),
        )

    cx, cy = center_fp
    c_arr = np.array([cx, cy], dtype=np.float64)

    qb_fp = (
        _position_footpoint(player_summaries[qb_track_id])
        if qb_track_id and qb_track_id in player_summaries
        else None
    )

    if qb_fp and (qb_fp[0] != cx or qb_fp[1] != cy):
        qb_arr = np.array(qb_fp, dtype=np.float64)
        u_vec = c_arr - qb_arr
        norm_u = np.linalg.norm(u_vec)
        u_hat = u_vec / norm_u if norm_u > 0 else np.array([-1.0, 0.0])
    else:
        if direction == "left":
            u_hat = np.array([-1.0, 0.0])
        elif direction == "right":
            u_hat = np.array([1.0, 0.0])
        elif direction == "up":
            u_hat = np.array([0.0, -1.0])
        elif direction == "down":
            u_hat = np.array([0.0, 1.0])
        else:
            u_hat = np.array([-1.0, 0.0])

    v_hat = np.array([u_hat[1], -u_hat[0]], dtype=np.float64)

    all_tids = list(player_summaries.keys())
    x_ranks = {
        tid: rank
        for rank, tid in enumerate(
            sorted(all_tids, key=lambda t: _position_footpoint(player_summaries[t])[0])
        )
    }
    y_ranks = {
        tid: rank
        for rank, tid in enumerate(
            sorted(all_tids, key=lambda t: _position_footpoint(player_summaries[t])[1])
        )
    }

    for tid, summary in player_summaries.items():
        fp = _position_footpoint(summary)
        px, py = fp
        p_arr = np.array([px, py], dtype=np.float64)
        diff = p_arr - c_arr

        lat_proj = float(np.dot(diff, v_hat) / scale)
        depth_backfield_proj = float(np.dot(diff, -u_hat) / scale)

        depth_los = -depth_backfield_proj
        depth_offense = depth_backfield_proj
        lateral_offense = lat_proj
        dist_center = float(np.linalg.norm(diff) / scale)

        dist_qb = 0.0
        if qb_fp:
            dist_qb = float(np.linalg.norm(p_arr - np.array(qb_fp)) / scale)

        other_fps = [
            _position_footpoint(player_summaries[o])
            for o in all_tids
            if o != tid
        ]
        if other_fps:
            nn_dists = [np.linalg.norm(p_arr - np.array(o_fp)) / scale for o_fp in other_fps]
            min_nn_dist = float(min(nn_dists))
        else:
            min_nn_dist = 0.0

        dx = (px - cx) / scale
        dy = (py - cy) / scale

        features[tid] = {
            "x_norm": float(dx),
            "y_norm": float(dy),
            "depth_los": float(depth_los),
            "depth_offense": float(depth_offense),
            "lateral_offset": float(lateral_offense),
            "lateral_offense": float(lateral_offense),
            "dist_center": dist_center,
            "dist_qb": dist_qb,
            "min_nn_dist": min_nn_dist,
            "x_rank": float(x_ranks[tid]),
            "y_rank": float(y_ranks[tid]),
            "bbox_height_norm": float(summary.median_bbox_height / scale),
            "bbox_width_norm": float(summary.median_bbox_width / scale),
            "bbox_aspect_ratio": float(summary.median_bbox_width / max(summary.median_bbox_height, 1.0)),
            "presnap_motion": float(summary.presnap_motion or 0.0),
            "formation_anchor_frame": float(summary.formation_anchor_frame if summary.formation_anchor_frame is not None else -1),
        }

    return features
