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
    """
    Partitions valid player tracks into Offense and Defense candidate pools.
    Borderline trench tracks without hard offense seeds are made available to defense
    so the joint optimizer can decide their role.
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

    # 1. Action side seeds
    for act in action_annotations:
        if act.actor_track_id and act.actor_track_id in player_tids:
            s_scores = track_side_scores.get(act.actor_track_id, {})
            if s_scores.get("offense", 0.0) >= 0.70:
                offense_seeds.add(act.actor_track_id)
            elif s_scores.get("defense", 0.0) >= 0.70:
                defense_seeds.add(act.actor_track_id)

    offense_tids: Set[int] = set(offense_seeds)
    defense_tids: Set[int] = set(defense_seeds)

    # 2. Backfield players: clearly on offense side (depth_offense >= 0.5)
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

    # 3. Trench players (abs_lat <= 1.4, -0.7 <= depth_off < 0.5)
    trench_tids = [
        t for t in player_tids
        if t not in offense_seeds and t not in defense_seeds
        and abs(spatial_feats.get(t, {}).get("lateral_offense", 0.0)) <= 1.4
    ]
    for tid in trench_tids:
        feat = spatial_feats.get(tid, {})
        depth_off = feat.get("depth_offense", 0.0)
        if depth_off >= -0.65:
            offense_tids.add(tid)
        # Trench players on LOS without hard offense seeds are also eligible for defense
        if depth_off <= 0.3:
            defense_tids.add(tid)

    # 4. Wing matchups: for wide players (abs_lat > 1.4)
    left_wing = [
        t for t in player_tids
        if spatial_feats.get(t, {}).get("lateral_offense", 0.0) > 1.4
    ]
    right_wing = [
        t for t in player_tids
        if spatial_feats.get(t, {}).get("lateral_offense", 0.0) < -1.4
    ]

    for wing in [left_wing, right_wing]:
        if not wing:
            continue
        wing_sorted = sorted(wing, key=lambda t: spatial_feats.get(t, {}).get("depth_offense", -99.0), reverse=True)
        for rank, tid in enumerate(wing_sorted):
            if tid in offense_seeds or tid in defense_seeds:
                continue
            if rank == 0 or (rank < len(wing_sorted) / 2 and len(offense_tids) < 11):
                offense_tids.add(tid)
            else:
                defense_tids.add(tid)

    # 5. Any remaining unassigned players go to defense
    for tid in player_tids:
        if tid not in offense_tids and tid not in defense_tids:
            defense_tids.add(tid)

    off_final = sorted(list(offense_tids - defense_seeds))
    def_final = sorted(list(defense_tids - offense_seeds))

    return off_final, def_final
