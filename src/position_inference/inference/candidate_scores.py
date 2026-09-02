from typing import Dict, List, Optional
import numpy as np

from position_inference.config import get_scoring_weights
from position_inference.data.schemas import TrackSummary
from position_inference.learning.feature_matrix import extract_track_features
from position_inference.learning.role_model import ViewSpecificRoleModel


def compute_candidate_role_scores(
    track_summaries: Dict[int, TrackSummary],
    spatial_features: Dict[int, Dict[str, float]],
    action_role_scores: Dict[int, Dict[str, float]],
    view: str = "sideline",
    learned_model: Optional[ViewSpecificRoleModel] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Integrates evidence from action anchors, spatial geometry, and learned models into candidate role scores.
    """
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
        probs_list = learned_model.predict_proba(X_feats)
        for tid, prob_map in zip(tids, probs_list):
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
        abs_lat = abs(lat_off)
        dist_c = s_feat.get("dist_center", 0.0)

        geom_scores: Dict[str, float] = {}

        # Offense heuristics
        geom_scores["C"] = 1.0 if dist_c < 0.25 and abs(depth_off) < 0.3 else max(0.0, 1.0 - 2.5 * dist_c)
        geom_scores["QB"] = 1.0 if 1.2 <= depth_off <= 2.6 and abs_lat <= 0.8 else (0.5 if depth_off >= 1.0 and abs_lat <= 1.0 else 0.1)

        # RB: backfield runner inside the tackle box
        geom_scores["RB"] = 1.0 if (depth_off >= 2.2 and abs_lat <= 1.2) else (0.5 if (depth_off >= 1.5 and abs_lat <= 1.2) else 0.1)
        geom_scores["FB"] = 0.8 if (1.0 <= depth_off <= 2.2 and abs_lat <= 1.0) else 0.1

        is_on_ol_band = (abs_lat <= 1.2 and -1.0 <= depth_off <= 1.5 and dist_c > 0.2)
        geom_scores["LT"] = 1.0 if (is_on_ol_band and 0.4 <= lat_off <= 1.2) else (0.4 if lat_off > 0 and is_on_ol_band else 0.1)
        geom_scores["LG"] = 1.0 if (is_on_ol_band and 0.1 <= lat_off <= 0.6) else (0.4 if lat_off > 0 and is_on_ol_band else 0.1)
        geom_scores["RG"] = 1.0 if (is_on_ol_band and -0.7 <= lat_off <= -0.1) else (0.4 if lat_off < 0 and is_on_ol_band else 0.1)
        geom_scores["RT"] = 1.0 if (is_on_ol_band and -1.2 <= lat_off <= -0.4) else (0.4 if lat_off < 0 and is_on_ol_band else 0.1)

        geom_scores["TE"] = 0.95 if (0.8 <= abs_lat <= 1.6 and -1.2 <= depth_off <= 1.0) else 0.15
        # WR: wide receivers on perimeter (or in motion)
        geom_scores["WR"] = 1.0 if abs_lat >= 1.5 else max(0.1, (abs_lat - 0.4) / 1.5)

        # Defense heuristics
        geom_scores["DE"] = 1.0 if (-0.4 <= depth_los <= 1.8 and 0.5 <= abs_lat <= 1.3) else 0.15
        geom_scores["DT"] = 1.0 if (-0.2 <= depth_los <= 1.2 and abs_lat <= 0.5) else 0.15

        # LB: second level interior or walked-up outside linebacker
        is_second_level = (1.5 <= depth_los <= 3.8 and abs_lat <= 1.6)
        is_outside_lb = (0.4 <= depth_los <= 2.5 and 1.4 <= abs_lat <= 2.4)
        geom_scores["LB"] = 1.0 if (is_second_level or is_outside_lb) else 0.15

        # CB: wide cornerbacks
        geom_scores["CB"] = 1.0 if (depth_los >= -0.3 and abs_lat >= 2.2) else 0.15

        # FS vs SS:
        if depth_los >= 2.8:
            if lat_off > 0.5:
                geom_scores["FS"] = 1.0
                geom_scores["SS"] = 0.3
            else:
                geom_scores["SS"] = 1.0
                geom_scores["FS"] = 0.3
            geom_scores["SAF"] = 1.0
        else:
            geom_scores["FS"] = 0.1
            geom_scores["SS"] = 0.1
            geom_scores["SAF"] = 0.1

        combined_scores: Dict[str, float] = {}

        all_roles = set(geom_scores.keys()) | set(a_score.keys()) | set(l_prob.keys())
        for r in all_roles:
            sc_a = a_score.get(r, 0.0)
            sc_g = geom_scores.get(r, 0.0)
            sc_m = l_prob.get(r, 0.0)

            if sc_a >= 0.99:
                comb = 1.0
            else:
                comb = w_action * sc_a + w_geom * sc_g + w_model * sc_m

            combined_scores[r] = float(comb)

        candidate_scores[tid] = combined_scores

    return candidate_scores
