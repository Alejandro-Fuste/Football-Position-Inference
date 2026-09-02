from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from position_inference.data.schemas import PositionAssignment, TrackSummary


def solve_offense_positions(
    offense_track_ids: List[int],
    track_summaries: Dict[int, TrackSummary],
    spatial_features: Dict[int, Dict[str, float]],
    candidate_scores: Dict[int, Dict[str, float]],
    center_track_id: Optional[int] = None,
    qb_track_id: Optional[int] = None,
    direction: str = "left",
    canonical_slots: Optional[List[str]] = None,
) -> List[PositionAssignment]:
    """
    Hierarchical solver for offensive player roles.
    Solves 5-OL line sequence, QB, RB, TE, and WR canonical slots using spatial lateral rank ordering and action anchors.
    """
    assignments: List[PositionAssignment] = []
    assigned_tracks: Set[int] = set()

    # Step 1: Center (hard anchor)
    c_tid = center_track_id
    if not c_tid or c_tid not in offense_track_ids:
        c_candidates = sorted(
            [t for t in offense_track_ids if t not in assigned_tracks],
            key=lambda t: candidate_scores.get(t, {}).get("C", 0.0),
            reverse=True,
        )
        c_tid = c_candidates[0] if c_candidates else None

    if c_tid:
        assignments.append(
            PositionAssignment(
                slot_id="offense.C_1",
                side="offense",
                position="C",
                track_id=c_tid,
                visibility="visible",
                confidence=0.99 if c_tid == center_track_id else 0.85,
                evidence={"action_anchor": 1.0 if c_tid == center_track_id else 0.5},
            )
        )
        assigned_tracks.add(c_tid)

    # Step 2: QB (hard anchor)
    q_tid = qb_track_id
    if not q_tid or q_tid not in offense_track_ids or q_tid in assigned_tracks:
        q_candidates = sorted(
            [t for t in offense_track_ids if t not in assigned_tracks],
            key=lambda t: candidate_scores.get(t, {}).get("QB", 0.0),
            reverse=True,
        )
        q_tid = q_candidates[0] if q_candidates else None

    if q_tid and q_tid not in assigned_tracks:
        assignments.append(
            PositionAssignment(
                slot_id="offense.QB_1",
                side="offense",
                position="QB",
                track_id=q_tid,
                visibility="visible",
                confidence=0.99 if q_tid == qb_track_id else 0.85,
                evidence={"action_anchor": 1.0 if q_tid == qb_track_id else 0.5},
            )
        )
        assigned_tracks.add(q_tid)

    # Step 3: Check action anchors for WR / RB (e.g. Jet Motion or Ball Carry)
    action_wr_tid = None
    for tid in offense_track_ids:
        if tid not in assigned_tracks:
            scores = candidate_scores.get(tid, {})
            if scores.get("WR", 0.0) >= 0.80 or scores.get("JetMotion", 0.0) >= 0.80:
                action_wr_tid = tid
                break

    # Step 4: 4 Offensive Linemen (LG, LT on left; RG, RT on right)
    line_candidates = [
        t for t in offense_track_ids
        if t not in assigned_tracks and spatial_features.get(t, {}).get("depth_los", 0.0) >= -1.8 and t != action_wr_tid
    ]

    left_of_c = sorted(
        [t for t in line_candidates if spatial_features.get(t, {}).get("lateral_offset", 0.0) > 0],
        key=lambda t: spatial_features.get(t, {}).get("lateral_offset", 0.0),
    )

    right_of_c = sorted(
        [t for t in line_candidates if spatial_features.get(t, {}).get("lateral_offset", 0.0) < 0],
        key=lambda t: abs(spatial_features.get(t, {}).get("lateral_offset", 0.0)),
    )

    lg_tid = left_of_c[0] if len(left_of_c) >= 1 else None
    lt_tid = left_of_c[1] if len(left_of_c) >= 2 else None

    rg_tid = right_of_c[0] if len(right_of_c) >= 1 else None
    rt_tid = right_of_c[1] if len(right_of_c) >= 2 else None

    for slot_id, pos, tid in [
        ("offense.LG_1", "LG", lg_tid),
        ("offense.LT_1", "LT", lt_tid),
        ("offense.RG_1", "RG", rg_tid),
        ("offense.RT_1", "RT", rt_tid),
    ]:
        if tid and tid not in assigned_tracks:
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="offense",
                    position=pos,
                    track_id=tid,
                    visibility="visible",
                    confidence=0.86,
                    evidence={"ol_line_sequence": 0.86},
                )
            )
            assigned_tracks.add(tid)
        else:
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="offense",
                    position=pos,
                    track_id=None,
                    visibility="out_of_view",
                    confidence=0.75,
                    evidence={"missing_in_crop": 1.0},
                )
            )

    # Step 5: TE
    remaining_line = [t for t in offense_track_ids if t not in assigned_tracks and t != action_wr_tid]
    te_candidates = sorted(
        remaining_line,
        key=lambda t: abs(spatial_features.get(t, {}).get("lateral_offset", 0.0)),
    )
    te_tid = te_candidates[0] if te_candidates else None
    if te_tid:
        assignments.append(
            PositionAssignment(
                slot_id="offense.TE_1",
                side="offense",
                position="TE",
                track_id=te_tid,
                visibility="visible",
                confidence=0.85,
                evidence={"inline_alignment": 0.85},
            )
        )
        assigned_tracks.add(te_tid)

    # Step 6: RB
    remaining_backfield = [t for t in offense_track_ids if t not in assigned_tracks and t != action_wr_tid]
    rb_candidates = sorted(
        remaining_backfield,
        key=lambda t: spatial_features.get(t, {}).get("depth_los", 0.0),
    )
    rb_tid = rb_candidates[0] if rb_candidates else None
    if rb_tid:
        assignments.append(
            PositionAssignment(
                slot_id="offense.RB_1",
                side="offense",
                position="RB",
                track_id=rb_tid,
                visibility="visible",
                confidence=0.85,
                evidence={"backfield_depth": 0.85},
            )
        )
        assigned_tracks.add(rb_tid)

    # Step 7: WRs
    remaining_wrs = sorted(
        [t for t in offense_track_ids if t not in assigned_tracks],
        key=lambda t: spatial_features.get(t, {}).get("lateral_offset", 0.0),
    )

    wr_slots = ["offense.WR_1", "offense.WR_2", "offense.WR_3"]
    for idx, slot_id in enumerate(wr_slots):
        if idx < len(remaining_wrs):
            wtid = remaining_wrs[idx]
            is_anchor = (wtid == action_wr_tid)
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="offense",
                    position="WR",
                    track_id=wtid,
                    visibility="visible",
                    confidence=0.95 if is_anchor else 0.86,
                    evidence={"action_motion_anchor": 1.0 if is_anchor else 0.86},
                )
            )
            assigned_tracks.add(wtid)
        else:
            assignments.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side="offense",
                    position="WR",
                    track_id=None,
                    visibility="out_of_view",
                    confidence=0.80,
                    evidence={"missing_endzone_receiver": 1.0},
                )
            )

    return assignments
