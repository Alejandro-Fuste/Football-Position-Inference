from typing import Dict, List, Optional, Set
import numpy as np

from position_inference.data.schemas import PositionAssignment, TrackSummary


def solve_defense_positions(
    defense_track_ids: List[int],
    track_summaries: Dict[int, TrackSummary],
    spatial_features: Dict[int, Dict[str, float]],
    candidate_scores: Dict[int, Dict[str, float]],
    canonical_slots: Optional[List[str]] = None,
) -> List[PositionAssignment]:
    """
    Hierarchical solver for defensive player roles.
    Solves Front level [DE/DT], Linebackers [LB], and Defensive Backs [CB/FS/SS].
    """
    if canonical_slots is None:
        canonical_slots = [
            "defense.DE_1", "defense.DE_2", "defense.DT_1", "defense.DT_2",
            "defense.LB_1", "defense.LB_2",
            "defense.CB_1", "defense.CB_2", "defense.FS_1", "defense.SS_1", "defense.CB_3"
        ]

    assignments: List[PositionAssignment] = []
    assigned_tracks: Set[int] = set()

    # Categorize defense tracks by depth_los and lateral offset
    front_candidates = [t for t in defense_track_ids if spatial_features.get(t, {}).get("depth_los", 5.0) <= 2.0]
    lb_candidates = [t for t in defense_track_ids if 1.5 < spatial_features.get(t, {}).get("depth_los", 5.0) <= 4.0]
    db_candidates = [t for t in defense_track_ids if spatial_features.get(t, {}).get("depth_los", 5.0) > 3.0 or abs(spatial_features.get(t, {}).get("lateral_offset", 0.0)) >= 3.0]

    # Step 1: Front line (DEs and DTs)
    # Sort front line by lateral offset magnitude
    front_candidates.sort(key=lambda t: abs(spatial_features.get(t, {}).get("lateral_offset", 0.0)))

    dt_tids = front_candidates[:2] # Inside
    de_tids = front_candidates[2:] # Outside

    # DTs
    for idx, slot_id in enumerate(["defense.DT_1", "defense.DT_2"]):
        if idx < len(dt_tids):
            tid = dt_tids[idx]
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position="DT",
                    track_id=tid,
                    visibility="visible",
                    confidence=0.88,
                    evidence={"interior_defensive_line": 0.88},
                )
            )
            assigned_tracks.add(tid)
        else:
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position="DT",
                    track_id=None,
                    visibility="out_of_view",
                    confidence=0.75,
                    evidence={"missing_in_crop": 1.0},
                )
            )

    # DEs
    for idx, slot_id in enumerate(["defense.DE_1", "defense.DE_2"]):
        if idx < len(de_tids):
            tid = de_tids[idx]
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position="DE",
                    track_id=tid,
                    visibility="visible",
                    confidence=0.88,
                    evidence={"edge_defensive_line": 0.88},
                )
            )
            assigned_tracks.add(tid)
        else:
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position="DE",
                    track_id=None,
                    visibility="out_of_view",
                    confidence=0.75,
                    evidence={"missing_in_crop": 1.0},
                )
            )

    # Step 2: Linebackers (LB_1, LB_2)
    remaining_lbs = [t for t in defense_track_ids if t not in assigned_tracks and spatial_features.get(t, {}).get("depth_los", 5.0) <= 4.0]
    remaining_lbs.sort(key=lambda t: spatial_features.get(t, {}).get("lateral_offset", 0.0))

    for idx, slot_id in enumerate(["defense.LB_1", "defense.LB_2"]):
        if idx < len(remaining_lbs):
            tid = remaining_lbs[idx]
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position="LB",
                    track_id=tid,
                    visibility="visible",
                    confidence=0.88,
                    evidence={"second_level_lb": 0.88},
                )
            )
            assigned_tracks.add(tid)
        else:
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position="LB",
                    track_id=None,
                    visibility="out_of_view",
                    confidence=0.75,
                    evidence={"missing_in_crop": 1.0},
                )
            )

    # Step 3: Defensive Backs (CB_1, CB_2, FS_1, SS_1)
    remaining_dbs = [t for t in defense_track_ids if t not in assigned_tracks]

    # Cornerbacks (widest periphery DBs)
    cbs = sorted(remaining_dbs, key=lambda t: abs(spatial_features.get(t, {}).get("lateral_offset", 0.0)), reverse=True)[:2]
    for idx, slot_id in enumerate(["defense.CB_1", "defense.CB_2"]):
        if idx < len(cbs):
            tid = cbs[idx]
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position="CB",
                    track_id=tid,
                    visibility="visible",
                    confidence=0.88,
                    evidence={"wide_cb_alignment": 0.88},
                )
            )
            assigned_tracks.add(tid)
        else:
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position="CB",
                    track_id=None,
                    visibility="out_of_view",
                    confidence=0.80,
                    evidence={"missing_endzone_cb": 1.0},
                )
            )

    # Safeties (deepest DBs)
    safeties = sorted([t for t in remaining_dbs if t not in assigned_tracks], key=lambda t: spatial_features.get(t, {}).get("depth_los", 0.0), reverse=True)
    for idx, (slot_id, pos) in enumerate([("defense.FS_1", "FS"), ("defense.SS_1", "SS")]):
        if idx < len(safeties):
            tid = safeties[idx]
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position=pos,
                    track_id=tid,
                    visibility="visible",
                    confidence=0.85,
                    evidence={"deep_safety_alignment": 0.85},
                )
            )
            assigned_tracks.add(tid)
        else:
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="defense",
                    position=pos,
                    track_id=None,
                    visibility="out_of_view",
                    confidence=0.78,
                    evidence={"missing_endzone_safety": 1.0},
                )
            )

    return assignments
