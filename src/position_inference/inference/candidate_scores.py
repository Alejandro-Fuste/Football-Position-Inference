from typing import Dict, Optional, Set
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
    """Endzone-specific geometry priors with explicit OL/TE/DL separation."""
    abs_lat = abs(lat_off)
    scores: Dict[str, float] = {}

    scores["C"] = 1.0 if dist_c < 0.30 and abs(depth_off) < 0.45 else max(0.0, 1.0 - 2.2 * dist_c)
    scores["QB"] = 1.0 if 1.0 <= depth_off <= 3.2 and abs_lat <= 1.0 else (0.45 if depth_off >= 0.8 and abs_lat <= 1.2 else 0.08)
    scores["RB"] = 1.0 if depth_off >= 2.0 and abs_lat <= 1.35 else (0.55 if depth_off >= 1.3 and abs_lat <= 1.5 else 0.08)
    scores["FB"] = 0.75 if 1.0 <= depth_off <= 2.4 and abs_lat <= 1.2 else 0.08

    on_ol_band = -0.65 <= depth_off <= 1.00 and dist_c > 0.18 and abs_lat <= 1.55
    scores["LT"] = 0.92 if on_ol_band and lat_off > 0.35 else (0.45 if on_ol_band and lat_off > 0 else 0.04)
    scores["LG"] = 0.92 if on_ol_band and 0.05 < lat_off <= 0.80 else (0.45 if on_ol_band and lat_off > 0 else 0.04)
    scores["RG"] = 0.92 if on_ol_band and -0.85 <= lat_off < -0.05 else (0.45 if on_ol_band and lat_off < 0 else 0.04)
    scores["RT"] = 0.92 if on_ol_band and lat_off < -0.35 else (0.45 if on_ol_band and lat_off < 0 else 0.04)

    scores["TE"] = 0.94 if -0.40 <= depth_off <= 1.10 and 1.05 <= abs_lat <= 2.20 else (0.25 if -0.40 <= depth_off <= 1.10 and 0.85 <= abs_lat < 1.05 else 0.07)
    scores["WR"] = 0.65 if abs_lat >= 2.2 else (0.30 if abs_lat >= 1.6 else 0.08)

    on_def_front = -0.10 <= depth_los <= 1.90
    scores["DT"] = 0.98 if on_def_front and abs_lat <= 0.75 else 0.06
    scores["DE"] = 0.98 if on_def_front and 0.50 <= abs_lat <= 2.00 else 0.06
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


def _infer_structural_endzone_ol_candidates(
    spatial_features: Dict[int, Dict[str, float]],
    action_role_scores: Dict[int, Dict[str, float]],
) -> Set[int]:
    """Infer the four non-center offensive linemen as a structural row around Center.

    The endzone projection can place a true OL slightly to either side of the Center's
    zero-depth line, so a fixed depth sign is not reliable. Instead, select two candidates
    on each lateral side that are closest to the Center row, while excluding tracks with
    strong semantic evidence for QB/WR/RB/FB/TE. CP-SAT still decides the exact LT/LG/RG/RT
    assignment using lateral ordering; this helper only supplies the OL family candidates.
    """
    excluded_roles = ("QB", "WR", "RB", "FB", "TE")
    candidates_by_side = {"left": [], "right": []}

    for tid, feat in spatial_features.items():
        dist_c = feat.get("dist_center", 0.0)
        lat = feat.get("lateral_offense", 0.0)
        depth = feat.get("depth_offense", 0.0)
        abs_lat = abs(lat)

        # Center itself and very wide/deep players are not non-center OL candidates.
        if dist_c < 0.15 or abs_lat > 1.85 or abs(depth) > 1.35:
            continue

        a_scores = action_role_scores.get(tid, {})
        if max((a_scores.get(role, 0.0) for role in excluded_roles), default=0.0) >= 0.40:
            continue

        # Prefer tracks aligned with the Center row. Slightly favor reasonable OL width
        # so interior defenders stacked almost directly over Center do not dominate.
        row_cost = abs(depth) + 0.08 * abs(abs_lat - 0.75)
        side = "left" if lat > 0 else "right"
        candidates_by_side[side].append((row_cost, tid))

    selected: Set[int] = set()
    for side in ("left", "right"):
        ranked = sorted(candidates_by_side[side], key=lambda item: (item[0], item[1]))
        selected.update(tid for _, tid in ranked[:2])
    return selected


def _apply_endzone_role_family_semantics(
    geom_scores: Dict[str, float], action_scores: Dict[str, float]
) -> Dict[str, float]:
    adjusted = dict(geom_scores)
    te_evidence = action_scores.get("TE", 0.0)
    ol_family_evidence = action_scores.get("OL", 0.0)

    if te_evidence >= 0.35 and te_evidence >= ol_family_evidence + 0.15:
        for role in ("LT", "LG", "RG", "RT"):
            adjusted[role] *= 0.20
        adjusted["TE"] = max(adjusted.get("TE", 0.0), 0.70)

    if ol_family_evidence >= 0.50 and ol_family_evidence >= te_evidence + 0.15:
        for role in ("LT", "LG", "RG", "RT"):
            adjusted[role] = max(adjusted.get(role, 0.0), 0.55)
        adjusted["TE"] *= 0.60

    return adjusted


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
    structural_ol_tids = (
        _infer_structural_endzone_ol_candidates(spatial_features, action_role_scores)
        if view == "endzone"
        else set()
    )

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
            geom_scores = _apply_endzone_role_family_semantics(
                _endzone_geometry_scores(depth_los, depth_off, lat_off, dist_c),
                a_score,
            )

            if dist_c >= 0.15:
                if tid in structural_ol_tids:
                    # Structural row membership is stronger than the noisy signed depth
                    # threshold. Preserve side-specific exact-position differences but
                    # make the OL family clearly preferable to DE/DT for these tracks.
                    for role in ("LT", "LG", "RG", "RT"):
                        geom_scores[role] = max(geom_scores.get(role, 0.0), 0.72)
                    geom_scores["DE"] *= 0.20
                    geom_scores["DT"] *= 0.20
                    geom_scores["LB"] *= 0.35
                else:
                    # Non-row tracks should not fill fixed OL slots merely because the
                    # solver needs five active linemen; missingness is preferable to a
                    # clearly non-OL player stealing a tackle/guard slot.
                    for role in ("LT", "LG", "RG", "RT"):
                        geom_scores[role] *= 0.18
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
