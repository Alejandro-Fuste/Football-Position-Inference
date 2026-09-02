from typing import Dict, List, Optional, Set, Tuple

from position_inference.data.schemas import ActionAnnotation, TrackSummary
from position_inference.geometry.spatial_features import compute_spatial_features


def partition_teams(
    track_summaries: Dict[int, TrackSummary],
    action_annotations: List[ActionAnnotation] = None,
    track_side_scores: Optional[Dict[int, Dict[str, float]]] = None,
    center_track_id: Optional[int] = None,
    qb_track_id: Optional[int] = None,
    direction: str = "left",
) -> Tuple[List[int], List[int]]:
    """Build offense/defense *eligibility pools* for the joint optimizer.

    Strong semantic anchors are side-locked. Geometrically clear backfield/second-level
    tracks receive a preferred side, while ambiguous trench, wing, and unresolved tracks
    remain eligible for BOTH sides so CP-SAT can make the final side/role decision.

    This function deliberately does not force every unresolved player onto defense.
    """
    if action_annotations is None:
        action_annotations = []
    if track_side_scores is None:
        track_side_scores = {}

    player_tids = [
        tid
        for tid, t in track_summaries.items()
        if t.label == "player" and getattr(t, "validity_score", 1.0) >= 0.30
    ]

    spatial_feats = compute_spatial_features(
        track_summaries,
        center_track_id=center_track_id,
        qb_track_id=qb_track_id,
        direction=direction,
    )

    offense_seeds: Set[int] = set()
    defense_seeds: Set[int] = set()

    if center_track_id and center_track_id in player_tids:
        offense_seeds.add(center_track_id)
    if qb_track_id and qb_track_id in player_tids:
        offense_seeds.add(qb_track_id)

    # Strong action-side evidence locks a track to one side. Lower-confidence action
    # evidence is intentionally left ambiguous for the global solver.
    for act in action_annotations:
        tid = act.actor_track_id
        if not tid or tid not in player_tids:
            continue
        s_scores = track_side_scores.get(tid, {})
        off_score = s_scores.get("offense", 0.0)
        def_score = s_scores.get("defense", 0.0)
        if off_score >= 0.85 and off_score >= def_score + 0.15:
            offense_seeds.add(tid)
        elif def_score >= 0.85 and def_score >= off_score + 0.15:
            defense_seeds.add(tid)

    offense_tids: Set[int] = set(offense_seeds)
    defense_tids: Set[int] = set(defense_seeds)

    for tid in player_tids:
        if tid in offense_seeds or tid in defense_seeds:
            continue

        feat = spatial_feats.get(tid, {})
        depth_off = feat.get("depth_offense", 0.0)
        abs_lat = abs(feat.get("lateral_offense", 0.0))

        # Clearly offensive backfield tracks may be preferred to offense, but remain
        # defense-eligible only when they are close enough to the neutral zone to be
        # geometrically ambiguous.
        if depth_off >= 1.25:
            offense_tids.add(tid)
            if depth_off <= 1.75:
                defense_tids.add(tid)
            continue

        # Clearly defensive second/deep level players can be preferred to defense.
        if depth_off <= -1.75:
            defense_tids.add(tid)
            if depth_off >= -2.25 and abs_lat <= 2.5:
                offense_tids.add(tid)
            continue

        # Neutral-zone/trench players are deliberately eligible for BOTH sides. This
        # is especially important in endzone views where OL and DL overlap heavily.
        if abs_lat <= 1.8 and -1.75 < depth_off < 1.25:
            offense_tids.add(tid)
            defense_tids.add(tid)
            continue

        # Wide/wing tracks can be WR/TE or CB/S depending on depth and camera view.
        # Keep them on both candidate sides unless semantics side-lock them.
        offense_tids.add(tid)
        defense_tids.add(tid)

    # Safety net: any valid player-like track not classified above remains eligible
    # for BOTH sides rather than being silently converted into a defender.
    for tid in player_tids:
        if tid not in offense_tids and tid not in defense_tids:
            offense_tids.add(tid)
            defense_tids.add(tid)

    # Strong seeds stay side-locked.
    off_final = sorted(offense_tids - defense_seeds)
    def_final = sorted(defense_tids - offense_seeds)

    return off_final, def_final
