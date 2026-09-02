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
    w_action = weights.get("action_semantics", 0.35)
    w_geom = weights.get("geometry", 0.25)
    w_model = weights.get("learned_model", 0.20)

    candidate_scores: Dict[int, Dict[str, float]] = {}

    # Extract learned model probabilities if model is provided
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
        if summary.label != "player" or summary.validity_score < 0.30:
            continue

        s_feat = spatial_features.get(tid, {})
        a_score = action_role_scores.get(tid, {})
        l_prob = learned_probs.get(tid, {})

        depth = s_feat.get("depth_los", 0.0)
        lat = abs(s_feat.get("lateral_offset", 0.0))
        dist_c = s_feat.get("dist_center", 0.0)

        # Baseline geometry heuristics
        geom_scores: Dict[str, float] = {}

        # Offense heuristics
        geom_scores["C"] = 1.0 if dist_c < 0.3 and abs(depth) < 0.2 else max(0.0, 1.0 - 2.0 * dist_c)
        geom_scores["QB"] = 1.0 if depth < -0.3 and dist_c < 0.8 else max(0.0, 1.0 - dist_c)
        geom_scores["LT"] = 1.0 if abs(depth) < 0.3 and 0.4 <= lat <= 1.5 and s_feat.get("lateral_offset", 0.0) < 0 else 0.2
        geom_scores["LG"] = 1.0 if abs(depth) < 0.3 and 0.1 <= lat <= 0.8 and s_feat.get("lateral_offset", 0.0) < 0 else 0.2
        geom_scores["RG"] = 1.0 if abs(depth) < 0.3 and 0.1 <= lat <= 0.8 and s_feat.get("lateral_offset", 0.0) > 0 else 0.2
        geom_scores["RT"] = 1.0 if abs(depth) < 0.3 and 0.4 <= lat <= 1.5 and s_feat.get("lateral_offset", 0.0) > 0 else 0.2
        geom_scores["WR"] = 1.0 if lat >= 1.5 else max(0.0, (lat - 0.5) / 1.5)
        geom_scores["TE"] = 0.8 if 0.8 <= lat <= 2.0 and abs(depth) < 0.4 else 0.2
        geom_scores["RB"] = 0.8 if depth < -0.5 else 0.1

        # Defense heuristics
        geom_scores["DE"] = 1.0 if 0.2 <= depth <= 0.8 and 0.8 <= lat <= 2.2 else 0.2
        geom_scores["DT"] = 1.0 if 0.1 <= depth <= 0.6 and lat <= 0.8 else 0.2
        geom_scores["LB"] = 1.0 if 0.6 <= depth <= 2.0 else 0.2
        geom_scores["CB"] = 1.0 if depth >= 0.5 and lat >= 1.8 else 0.2
        geom_scores["FS"] = 1.0 if depth >= 2.5 and lat <= 1.5 else 0.1
        geom_scores["SS"] = 1.0 if 1.5 <= depth <= 3.0 else 0.1

        combined_scores: Dict[str, float] = {}

        all_roles = set(geom_scores.keys()) | set(a_score.keys()) | set(l_prob.keys())
        for r in all_roles:
            sc_a = a_score.get(r, 0.0)
            sc_g = geom_scores.get(r, 0.0)
            sc_m = l_prob.get(r, 0.0)

            # Hard anchor override
            if sc_a >= 0.99:
                comb = 1.0
            else:
                comb = w_action * sc_a + w_geom * sc_g + w_model * sc_m

            combined_scores[r] = float(comb)

        candidate_scores[tid] = combined_scores

    return candidate_scores
