from typing import Dict, List, Optional, Set, Tuple

from position_inference.data.schemas import ActionAnnotation, TrackSummary
from position_inference.geometry.spatial_features import compute_spatial_features


def partition_teams(
    track_summaries: Dict[int, TrackSummary],
    action_annotations: List[ActionAnnotation] = None,
    center_track_id: Optional[int] = None,
    qb_track_id: Optional[int] = None,
    direction: str = "left",
) -> Tuple[List[int], List[int]]:
    """
    Partitions valid player tracks into Offense and Defense candidate pools.
    Uses seed action anchors and spatial formation clustering around Center/LOS.
    """
    if action_annotations is None:
        action_annotations = []

    player_tids = [tid for tid, t in track_summaries.items() if t.label == "player" and t.validity_score >= 0.30]

    offense_seeds: Set[int] = set()
    defense_seeds: Set[int] = set()

    if center_track_id and center_track_id in player_tids:
        offense_seeds.add(center_track_id)
    if qb_track_id and qb_track_id in player_tids:
        offense_seeds.add(qb_track_id)

    # Add all actor_track_ids referenced in Key Actions to offense seeds
    for act in action_annotations:
        if act.actor_track_id in player_tids:
            offense_seeds.add(act.actor_track_id)

    spatial_feats = compute_spatial_features(
        track_summaries,
        center_track_id=center_track_id,
        qb_track_id=qb_track_id,
        direction=direction,
    )

    # Calculate distance to Center for all tracks
    center_dists = {tid: spatial_feats.get(tid, {}).get("dist_center", 99.0) for tid in player_tids}

    # Offense includes:
    # 1. All action seed tracks
    # 2. Tracks behind LOS (depth_los <= -0.5)
    # 3. Tracks on LOS within tight lateral cluster around Center (abs(depth_los) <= 1.0 and abs(lateral_offset) <= 3.5)
    offense_tids: List[int] = list(offense_seeds)
    defense_tids: List[int] = []

    for tid in player_tids:
        if tid in offense_seeds:
            continue

        feat = spatial_feats.get(tid, {})
        depth = feat.get("depth_los", 0.0)
        lat = abs(feat.get("lateral_offset", 0.0))

        # Offense criteria:
        if depth <= -0.5:
            offense_tids.append(tid)
        elif abs(depth) <= 1.0 and lat <= 3.5:
            offense_tids.append(tid)
        else:
            defense_tids.append(tid)

    # Cap offense at 11 players
    if len(offense_tids) > 11:
        # Keep seeds first, then sort remaining by distance to Center/QB
        non_seed_off = [t for t in offense_tids if t not in offense_seeds]
        non_seed_off.sort(key=lambda t: center_dists.get(t, 99.0))

        keep_count = max(0, 11 - len(offense_seeds))
        kept_non_seeds = non_seed_off[:keep_count]
        excess_non_seeds = non_seed_off[keep_count:]

        offense_tids = list(offense_seeds) + kept_non_seeds
        defense_tids.extend(excess_non_seeds)

    return sorted(list(set(offense_tids))), sorted(list(set(defense_tids)))
