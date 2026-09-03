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
    abs_lat = abs(lat_off)
    scores: Dict[str, float] = {}

    scores["C"] = 1.0 if dist_c < 0.30 and abs(depth_off) < 0.45 else max(0.0, 1.0 - 2.2 * dist_c)
    scores["QB"] = 1.0 if 1.0 <= depth_off <= 3.2 and abs_lat <= 1.0 else (0.45 if depth_off >= 0.8 and abs_lat <= 1.2 else 0.08)
    scores["RB"] = 1.0 if depth_off >= 2.0 and abs_lat <= 1.35 else (0.55 if depth_off >= 1.3 and abs_lat <= 1.5 else 0.08)
    scores["FB"] = 0.75 if 1.0 <= depth_off <= 2.4 and abs_lat <= 1.2 else 0.08

    near_center_row = abs(depth_off) <= 1.60 and dist_c > 0.18
    base_ol = 0.35 if near_center_row else 0.04
    scores["LT"] = scores["LG"] = scores["RG"] = scores["RT"] = base_ol

    scores["TE"] = 0.94 if -0.50 <= depth_off <= 1.20 and 1.05 <= abs_lat <= 2.40 else (0.25 if -0.50 <= depth_off <= 1.20 and 0.85 <= abs_lat < 1.05 else 0.07)
    scores["WR"] = 0.65 if abs_lat >= 2.4 else (0.30 if abs_lat >= 1.7 else 0.08)

    on_def_front = -0.10 <= depth_los <= 1.90
    scores["DT"] = 0.98 if on_def_front and abs_lat <= 0.80 else 0.06
    scores["DE"] = 0.98 if on_def_front and 0.50 <= abs_lat <= 2.20 else 0.06
    scores["LB"] = 0.96 if 1.30 <= depth_los <= 4.2 and abs_lat <= 2.5 else (0.35 if 0.75 <= depth_los <= 3.0 else 0.08)
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


def _is_presnap_solver_eligible(summary: Optional[TrackSummary], snap_frame: Optional[int]) -> bool:
    if summary is None:
        return False
    if summary.label != "player" or getattr(summary, "validity_score", 1.0) < 0.30:
        return False
    if not (summary.presnap_median_footpoint or summary.median_footpoint):
        return False
    if summary.presnap_median_footpoint is None and snap_frame is not None and summary.first_frame > snap_frame:
        return False
    return True


def _presnap_stability_penalty(summary: TrackSummary, snap_frame: Optional[int]) -> float:
    """Penalize tracks that only become observable immediately before the snap.

    Structural OL roles should favor players that are established throughout the pre-snap
    formation. A defender emerging from occlusion just before the snap remains eligible for
    defensive assignment, but should not outrank a stable guard simply because the two are
    geometrically stacked in the endzone projection.
    """
    if snap_frame is None or snap_frame <= 0:
        return 0.0

    presnap_frames = [f for f in summary.frames_present if f <= snap_frame]
    if not presnap_frames:
        return 2.0

    first = min(presnap_frames)
    observed_span = max(1, snap_frame - first + 1)
    full_span = max(1, snap_frame + 1)
    span_fraction = min(1.0, observed_span / full_span)

    # No penalty for a track visible through at least half of the pre-snap window.
    # Scale smoothly up to a strong penalty for tracks appearing only at the end.
    if span_fraction >= 0.50:
        return 0.0
    return 2.0 * (0.50 - span_fraction) / 0.50


def _infer_structural_endzone_ol_roles(
    spatial_features: Dict[int, Dict[str, float]],
    action_role_scores: Dict[int, Dict[str, float]],
    track_summaries: Dict[int, TrackSummary],
    snap_frame: Optional[int] = None,
) -> Dict[int, str]:
    """Infer LT/LG/RG/RT relationally around the anchored Center."""
    excluded_roles = ("QB", "WR", "RB", "FB", "TE")
    by_side = {"left": [], "right": []}

    for tid, feat in spatial_features.items():
        summary = track_summaries.get(tid)
        if not _is_presnap_solver_eligible(summary, snap_frame):
            continue

        dist_c = feat.get("dist_center", 0.0)
        lat = feat.get("lateral_offense", 0.0)
        depth = feat.get("depth_offense", 0.0)

        if dist_c < 0.15 or abs(depth) > 1.80 or abs(lat) < 0.05:
            continue

        a_scores = action_role_scores.get(tid, {})
        if max((a_scores.get(role, 0.0) for role in excluded_roles), default=0.0) >= 0.40:
            continue

        defensive_side_penalty = 1.50 * max(0.0, -depth)
        offensive_side_bonus = 0.10 * max(0.0, min(depth, 0.80))
        stability_penalty = _presnap_stability_penalty(summary, snap_frame)
        row_cost = abs(depth) + defensive_side_penalty - offensive_side_bonus + stability_penalty
        side = "left" if lat > 0 else "right"
        by_side[side].append((row_cost, abs(lat), tid))

    roles: Dict[int, str] = {}
    for side in ("left", "right"):
        selected = sorted(by_side[side], key=lambda item: (item[0], item[1], item[2]))[:2]
        if len(selected) < 2:
            continue
        selected_by_width = sorted(selected, key=lambda item: item[1])
        inner_tid = selected_by_width[0][2]
        outer_tid = selected_by_width[1][2]
        if side == "left":
            roles[inner_tid] = "LG"
            roles[outer_tid] = "LT"
        else:
            roles[inner_tid] = "RG"
            roles[outer_tid] = "RT"
    return roles


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
    snap_frame: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    weights = get_scoring_weights().get(f"{view}_weights", get_scoring_weights().get("weights", {}))
    w_action = weights.get("action_semantics", 0.45)
    w_geom = weights.get("geometry", 0.35)
    w_model = weights.get("learned_model", 0.20)

    candidate_scores: Dict[int, Dict[str, float]] = {}
    learned_probs: Dict[int, Dict[str, float]] = {}
    structural_ol_roles = (
        _infer_structural_endzone_ol_roles(
            spatial_features,
            action_role_scores,
            track_summaries,
            snap_frame=snap_frame,
        )
        if view == "endzone"
        else {}
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

            expected_ol_role = structural_ol_roles.get(tid)
            if expected_ol_role:
                for role in ("LT", "LG", "RG", "RT"):
                    geom_scores[role] = max(geom_scores.get(role, 0.0), 0.20)
                geom_scores[expected_ol_role] = max(geom_scores.get(expected_ol_role, 0.0), 0.98)
                geom_scores["DE"] *= 0.18
                geom_scores["DT"] *= 0.18
                geom_scores["LB"] *= 0.30
            elif dist_c >= 0.15:
                for role in ("LT", "LG", "RG", "RT"):
                    geom_scores[role] *= 0.12
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
