from typing import Dict, List, Optional, Tuple

from position_inference.data.schemas import ActionAnnotation
from position_inference.semantics.action_rules import match_action_rule


def extract_semantic_anchors(
    action_annotations: List[ActionAnnotation],
    play_type: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int], Dict[int, Dict[str, float]]]:
    """
    Extracts Center anchor, QB anchor, and per-track action role scores.
    Returns (center_track_id, qb_track_id, track_action_role_scores).
    """
    center_track_id: Optional[int] = None
    qb_track_id: Optional[int] = None

    track_action_scores: Dict[int, Dict[str, float]] = {}

    for act in action_annotations:
        if act.actor_track_id is None:
            continue

        tid = act.actor_track_id
        rule = match_action_rule(act.action, play_type)

        if not rule:
            continue

        mode = rule.get("mode", "soft")
        roles = rule.get("roles", {})

        # Hard anchors
        if mode == "hard_anchor":
            if "C" in roles and roles["C"] >= 0.99:
                center_track_id = tid
            if "QB" in roles and roles["QB"] >= 0.99:
                qb_track_id = tid

        # Score contributions
        tid_scores = track_action_scores.setdefault(tid, {})
        for role, weight in roles.items():
            current = tid_scores.get(role, 0.0)
            tid_scores[role] = max(current, weight)

    return center_track_id, qb_track_id, track_action_scores
