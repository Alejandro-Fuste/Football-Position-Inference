from typing import Dict, Optional
import numpy as np

from position_inference.config import get_scoring_weights
from position_inference.data.schemas import TrackSummary
from position_inference.learning.feature_matrix import extract_track_features
from position_inference.learning.role_model import ViewSpecificRoleModel


def _sideline_geometry_scores(depth_los: float, depth_off: float, lat_off: float, dist_c: float) -> Dict[str, float]:
    abs_lat = abs(lat_off)
    scores: Dict[str, float] = {}
    scores["C"] = 1.0 if dist_c < 0.25 and abs(depth_off) < 0.3 else max(0.0, 1.0 - 2.5 * dist_c)
    scores["QB"] = 1.0 if 1.2 <= depth_off <= 2.6 and abs_lat <= 0.8 else (0.5 if depth_off >= 1.0 and abs_lat <= 1.0 else 0.1)
    scores["RB"] = 1.0 if (depth_off >= 2.2 and abs_lat <= 1.2) else (0.5 if (depth_off >= 1.5 and abs_lat <= 1.2) else 0.1)
    scores["FB"] = 0.8 if (1.0 <= depth_off <= 2.2 and abs_lat <= 1.0) else 0.1

    is_on_ol_band = abs_lat <= 1.2 and -1.0 <= depth_off <= 1.5 and dist_c > 0.2
    scores["LT"] = 1.0 if (is_on_ol_band and 0.4 <= lat_off <= 1.2) else (0.4 if lat_off > 0 and is_on_ol_band else 0.1)
    scores["LG"] = 1.0 if (is_on_ol_band and 0.1 <= lat_off <= 0.6) else (0.4 if lat_off > 0 and is_on_ol_band else 0.1)
    scores["RG"] = 1.0 if (is_on_ol_band and -0.7 <= lat_off <= -0.1) else (0.4 if lat_off < 0 and is_on_ol_band else 0.1)
    scores["RT"] = 1.0 if (is_on_ol_band and -1.2 <= lat_off <= -0.4) else (0.4 if lat_off < 0 and is_on_ol_band else 0.1)

    scores["TE"] = 0.95 if (0.8 <= abs_lat <= 1.6 and -1.2 <= depth_off <= 1.0) else 0.15
    scores["WR"] = 1.0 if abs_lat >= 1.5 else max(0.1, (abs_lat - 0.4) / 1.5)

    scores["DE"] = 1.0 if (-0.4 <= depth_los <= 1.8 and 0.5 <= abs_lat <= 1.3) else 0.15
    scores["DT"] = 1.0 if (-0.2 <= depth_los <= 1.2 and abs_lat <= 0.5) else 0.15
    is_second_level = 1.9 <= depth_los <= 3.8 and abs_lat <= 1.6
    is_outside_lb = 0.4 <= depth_los <= 2.5 and 1.4 <= abs_lat <= 2.4
    scores["LB"] = 1.0 if (is_second_level or is_outside_lb) else 0.15
    scores["CB"] = 1.0 if (-0.3 <= depth_los <= 3.8 and abs_lat >= 2.0) else (0.4 if abs_lat >= 2.0 else 0.15)

    if depth_los >= 2.8:
        if lat_off > 0.5:
            scores["FS"], scores["SS"] = 1.0, 0.3
        else:
            scores["SS"], scores["FS"] = 1.0, 0.3
        scores["SAF"] = 1.0
    else:
        scores["FS"] = scores["SS"] = scores["SAF"] = 0.1
    return scores


def _endzone_geometry_scores(depth_los: float, depth_off: float, lat_off: float, dist_c: float) -> Dict[str, float]:
    """Endzone-specific geometry priors.

    ``depth_off`` is positive on the offense/backfield side of the Center and negative
    across the LOS on the defensive side. ``depth_los`` is the exact inverse. The OL and
    defensive-front scoring bands therefore must not overlap broadly; otherwise a DE just
    across the line can receive the same tackle score as a true offensive lineman.
    """
    abs_lat = abs(lat_off)
    scores: Dict[str, float] = {}

    scores["C"] = 1.0 if dist_c < 0.30 and abs(depth_off) < 0.45 else max(0.0, 1.0 - 2.2 * dist_c)
    scores["QB"] = 1.0 if 1.0 <= depth_off <= 3.2 and abs_lat <= 1.0 else (0.45 if depth_off >= 0.8 and abs_lat <= 1.2 else 0.08)
    scores["RB"] = 1.0 if depth_off >= 2.0 and abs_lat <= 1.35 else (0.55 if depth_off >= 1.3 and abs_lat <= 1.5 else 0.08)
    scores["FB"] = 0.75 if 1.0 <= depth_off <= 2.4 and abs_lat <= 1.2 else 0.08

    # Offensive linemen should be on (or slightly behind) the Center's LOS row.
    # Allow a small negative tolerance for detector/perspective noise, but do not let
    # clearly defensive-side tracks compete strongly for OL slots.
    on_ol_band = -0.20 <= depth_off <= 0.90 and dist_c > 0.18 and abs_lat <= 1.75
    wrong_side_for_ol = depth_off < -0.20

    if wrong_side_for_ol:
        scores["LT"] = scores["LG"] = scores["RG"] = scores["RT"] = 0.02
    else:
        scores["LT"] = 0.98 if on_ol_band and lat_off > 0.45 else (0.68 if on_ol_band and lat_off > 0 else 0.06)
        scores["LG"] = 0.98 if on_ol_band and 0.05 < lat_off <= 0.75 else (0.64 if on_ol_band and lat_off > 0 else 0.06)
        scores["RG"] = 0.98 if on_ol_band and -0.80 <= lat_off < -0.05 else (0.64 if on_ol_band and lat_off < 0 else 0.06)
        scores["RT"] = 0.98 if on_ol_band and lat_off < -0.45 else (0.68 if on_ol_band and lat_off < 0 else 0.06)

    # TE is attached/near-attached to the offensive surface and likewise should not be
    # scored strongly when the track is clearly across the LOS.
    scores["TE"] = 0.92 if -0.20 <= depth_off <= 1.0 and 0.9 <= abs_lat <= 2.0 else 0.08
    scores["WR"] = 0.65 if abs_lat >= 2.2 else (0.30 if abs_lat >= 1.6 else 0.08)

    # Defensive front should be on the defensive side of the Center. A small tolerance
    # around zero handles neutral-zone/perspective noise without making OL and DL symmetric.
    on_def_front = -0.10 <= depth_los <= 1.75
    scores["DT"] = 0.98 if on_def_front and abs_lat <= 0.70 else 0.06
    scores["DE"] = 0.98 if on_def_front and 0.50 <= abs_lat <= 1.90 else 0.06

    scores["LB"] = 0.96 if 1.30 <= depth_los <= 4.2 and abs_lat <= 2.3 else (0.35 if 0.75 <= depth_los <= 3.0 else 0.08)
    scores["CB"] = 0.60 if abs_lat >= 2.2 and depth_los >= 0.0 else (0.22 if abs_lat >= 1.8 else 0.06)

    if depth_los >= 3.0:
        scores["SAF"] = 0.75
        if lat_off > 0.4:
            scores["FS"], scores["SS"] = 0.70, 0.40
        else:
            scores["SS"], scores["FS"] = 0.70, 0.40
    else:
        scores["FS"] = scores["SS"] = scores["SAF"] = 0.06
    return scores


def compute_candidate_role_scores(
    track_summaries: Dict[int, TrackSummary],
    spatial_features: Dict[int, Dict[str, float]],
    action_role_scores: Dict[int, Dict[str, float]],
    view: str = "sideline",
    learned_model: Optional[ViewSpecificRoleModel] = None,
) -> Dict[int, Dict[str, float]]:
    """Integrate action semantics, view-specific geometry, and optional learned probabilities."""
    weights = get_scoring_weights().get(f"{view}_weights", get_scoring_weights().get("weights", {}))
    w_action = weights.get("action_semantics", 0.45)
    w_geom = weights.get("geometry", 0.35)
    w_model = weights.get("learned_model", 0.20)

    candidate_scores: Dict[int, Dict[str, float]] = {}
    learned_probs: Dict[int, Dict[str, float]] = {}

    if learned_model and learned_model.is_fitted:
        tids = list(track_summaries.keys())
        X_feats = np.array([
            extract_track_features(track_summaries[t], spatial_features.get(t, {}), action_role_scores.get(t, {}))
            for t in tids
        ])
        for tid, prob_map in zip(tids, learned_model.predict_proba(X_feats)):
            learned_probs[tid] = prob_map

    for tid, summary in track_summaries.items():
        if summary.label != "player" or getattr(summary, "validity_score", 1.0) < 0.30:
            continue

        s_feat = spatial_features.get(tid, {})
        a_score = action_role_scores.get(tid, {})
        l_prob = learned_probs.get(tid, {})
        depth_los = s_feat.get("depth_los", 0.0)
        depth_off = s_feat.get("depth_offense", -depth_los)
        lat_off = s_feat.get("lateral_offense", s_feat.get("lateral_offset", 0.0))
        dist_c = s_feat.get("dist_center", 0.0)

        if view == "endzone":
            geom_scores = _endzone_geometry_scores(depth_los, depth_off, lat_off, dist_c)
        else:
            geom_scores = _sideline_geometry_scores(depth_los, depth_off, lat_off, dist_c)

        combined_scores: Dict[str, float] = {}
        all_roles = set(geom_scores) | set(a_score) | set(l_prob)
        active_w_action = w_action if any(sc > 0.0 for sc in a_score.values()) else 0.0
        active_w_model = w_model if l_prob and any(sc > 0.0 for sc in l_prob.values()) else 0.0
        active_w_geom = w_geom
        total_w = active_w_action + active_w_geom + active_w_model or 1.0

        for role in all_roles:
            sc_a = a_score.get(role, 0.0)
            sc_g = geom_scores.get(role, 0.0)
            sc_m = l_prob.get(role, 0.0)
            if sc_a >= 0.99:
                combined = 1.0
            else:
                combined = (active_w_action * sc_a + active_w_geom * sc_g + active_w_model * sc_m) / total_w
            combined_scores[role] = float(min(1.0, max(0.0, combined)))

        candidate_scores[tid] = combined_scores

    return candidate_scores
