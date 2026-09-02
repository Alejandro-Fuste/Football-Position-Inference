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
    view: str = "sideline",
) -> Tuple[List[int], List[int]]:
    """Build offense/defense eligibility pools for the joint optimizer.

    Sideline geometry is sufficiently discriminative to retain the prior preferred-side
    partitioning that already produces the JetSweep_1 golden result. Endzone geometry is
    much more compressed in the trench, so ambiguous/unresolved endzone tracks remain
    eligible for BOTH sides unless strong semantic evidence side-locks them.
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

    # Strong action-side evidence locks a track to one side.
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

    # Preserve the previously successful sideline partition behavior. This avoids
    # broadening sideline eligibility enough for defenders to steal WR/TE slots.
    if view != "endzone":
        offense_tids: Set[int] = set(offense_seeds)
        defense_tids: Set[int] = set(defense_seeds)

        for tid in player_tids:
            if tid in offense_seeds or tid in defense_seeds:
                continue
            feat = spatial_feats.get(tid, {})
            depth_off = feat.get("depth_offense", 0.0)
            abs_lat = abs(feat.get("lateral_offense", 0.0))

            if depth_off >= 0.5:
                offense_tids.add(tid)
            elif depth_off <= -1.2 and abs_lat <= 2.0:
                defense_tids.add(tid)

        trench_tids = [
            t for t in player_tids
            if t not in offense_seeds and t not in defense_seeds
            and abs(spatial_feats.get(t, {}).get("lateral_offense", 0.0)) <= 1.4
        ]
        for tid in trench_tids:
            depth_off = spatial_feats.get(tid, {}).get("depth_offense", 0.0)
            if depth_off >= -0.65:
                offense_tids.add(tid)
            if depth_off <= 0.3:
                defense_tids.add(tid)

        left_wing = [
            t for t in player_tids
            if spatial_feats.get(t, {}).get("lateral_offense", 0.0) > 1.4
        ]
        right_wing = [
            t for t in player_tids
            if spatial_feats.get(t, {}).get("lateral_offense", 0.0) < -1.4
        ]

        for wing in (left_wing, right_wing):
            wing_sorted = sorted(
                wing,
                key=lambda t: spatial_feats.get(t, {}).get("depth_offense", -99.0),
                reverse=True,
            )
            for rank, tid in enumerate(wing_sorted):
                if tid in offense_seeds or tid in defense_seeds:
                    continue
                if rank == 0 or (rank < len(wing_sorted) / 2 and len(offense_tids) < 11):
                    offense_tids.add(tid)
                else:
                    defense_tids.add(tid)

        # Preserve legacy fallback for sideline only because it is part of the
        # already-validated JetSweep_1 behavior.
        for tid in player_tids:
            if tid not in offense_tids and tid not in defense_tids:
                defense_tids.add(tid)

        return (
            sorted(offense_tids - defense_seeds),
            sorted(defense_tids - offense_seeds),
        )

    # Endzone: retain ambiguous tracks on both sides. This lets CP-SAT resolve OL/DL
    # overlap instead of making an irreversible side decision before optimization.
    offense_tids = set(offense_seeds)
    defense_tids = set(defense_seeds)

    for tid in player_tids:
        if tid in offense_seeds or tid in defense_seeds:
            continue

        feat = spatial_feats.get(tid, {})
        depth_off = feat.get("depth_offense", 0.0)
        abs_lat = abs(feat.get("lateral_offense", 0.0))

        if depth_off >= 1.8:
            offense_tids.add(tid)
            continue
        if depth_off <= -2.4 and abs_lat <= 2.5:
            defense_tids.add(tid)
            continue

        offense_tids.add(tid)
        defense_tids.add(tid)

    # Endzone unresolved tracks remain dual-eligible rather than default-defense.
    for tid in player_tids:
        if tid not in offense_tids and tid not in defense_tids:
            offense_tids.add(tid)
            defense_tids.add(tid)

    return (
        sorted(offense_tids - defense_seeds),
        sorted(defense_tids - offense_seeds),
    )
