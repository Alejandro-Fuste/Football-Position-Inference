from typing import Dict, List, Optional, Tuple

from position_inference.data.schemas import ActionAnnotation
from position_inference.semantics.action_rules import match_action_rule


def extract_semantic_anchors(
    action_annotations: List[ActionAnnotation],
    play_type: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Extracts Center anchor, QB anchor, per-track action role scores, and per-track side evidence.
    Returns (center_track_id, qb_track_id, track_action_role_scores, track_side_scores).
    """
    center_track_id: Optional[int] = None
    qb_track_id: Optional[int] = None

    track_action_scores: Dict[int, Dict[str, float]] = {}
    track_side_scores: Dict[int, Dict[str, float]] = {}

    for act in action_annotations:
        if act.actor_track_id is None:
            continue

        tid = act.actor_track_id
        rule = match_action_rule(act.action, play_type)

        if not rule:
            continue

        mode = rule.get("mode", "soft")
        roles = rule.get("roles", {})
        side = rule.get("side", {})

        # Side evidence contributions
        tid_side = track_side_scores.setdefault(tid, {})
        for s_name, s_wt in side.items():
            current_s = tid_side.get(s_name, 0.0)
            tid_side[s_name] = max(current_s, s_wt)

        # Hard anchors
        if mode == "hard_anchor":
            if "C" in roles and roles["C"] >= 0.99:
                center_track_id = tid
                tid_side["offense"] = 1.0
            if "QB" in roles and roles["QB"] >= 0.99:
                qb_track_id = tid
                tid_side["offense"] = 1.0

        # Score contributions
        tid_scores = track_action_scores.setdefault(tid, {})
        for role, weight in roles.items():
            current = tid_scores.get(role, 0.0)
            tid_scores[role] = max(current, weight)

    return center_track_id, qb_track_id, track_action_scores, track_side_scores
