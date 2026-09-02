import time
from typing import Dict, List, Optional, Tuple
from ortools.sat.python import cp_model

from position_inference.data.schemas import PositionAssignment, TrackSummary
from position_inference.semantics.personnel import (
    get_superset_canonical_slots,
)


def solve_global_assignments(
    track_summaries: Dict[int, TrackSummary],
    spatial_features: Dict[int, Dict[str, float]],
    candidate_scores: Dict[int, Dict[str, float]],
    offense_track_ids: List[int],
    defense_track_ids: List[int],
    center_track_id: Optional[int] = None,
    qb_track_id: Optional[int] = None,
    direction: str = "left",
    view: str = "sideline",
    personnel_priors: Optional[Dict[str, int]] = None,
) -> List[PositionAssignment]:
    """
    Global constrained optimization solver for football player position assignments.
    Implements a genuine OR-Tools CP-SAT model jointly optimizing track-to-slot assignment,
    5-OL lateral ordering, WR/CB depth alignment, defense levels, and formation legality (11 offense + 11 defense).
    """
    model = cp_model.CpModel()
    start_time = time.time()

    all_tracks = sorted(list(set(offense_track_ids + defense_track_ids)))
    off_seeds = {center_track_id, qb_track_id} - {None}

    off_slots = get_superset_canonical_slots("offense")
    def_slots = get_superset_canonical_slots("defense")
    all_slots = off_slots + def_slots

    off_set = set(offense_track_ids)
    def_set = set(defense_track_ids)

    # 1. Decision Variables
    # x[t, s] in {0, 1}: track t assigned to slot s
    x: Dict[Tuple[int, str], cp_model.IntVar] = {}
    has_any_presnap = any(
        getattr(t, "presnap_median_footpoint", None) is not None for t in track_summaries.values()
    )

    for t in all_tracks:
        t_sum = track_summaries.get(t)
        if t_sum:
            if has_any_presnap and t_sum.presnap_median_footpoint is None:
                continue
            if not (t_sum.presnap_median_footpoint or t_sum.median_footpoint):
                continue

        is_off_eligible = (t in off_set)
        is_def_eligible = (t in def_set)

        for s in all_slots:
            if s.startswith("offense.") and not is_off_eligible:
                continue
            if s.startswith("defense.") and not is_def_eligible:
                continue

            x[(t, s)] = model.NewBoolVar(f"x_{t}_{s}")

    # is_active[s] in {0, 1}: slot s is part of active 11-player package
    is_active: Dict[str, cp_model.IntVar] = {}
    for s in all_slots:
        is_active[s] = model.NewBoolVar(f"active_{s}")

    # is_nv[s] in {0, 1}: slot s is active but not visible (out of view / occluded)
    is_nv: Dict[str, cp_model.IntVar] = {}
    for s in all_slots:
        is_nv[s] = model.NewBoolVar(f"nv_{s}")

    # is_unassigned[t] in {0, 1}: track t remains unassigned (noise / referee / extra)
    is_unassigned: Dict[int, cp_model.IntVar] = {}
    for t in all_tracks:
        is_unassigned[t] = model.NewBoolVar(f"unassigned_{t}")

    # 2. Exclusivity Constraints
    # Each track assigned to at most 1 slot: sum_s x[t, s] + is_unassigned[t] == 1
    for t in all_tracks:
        assigned_slots = [x[(t, s)] for s in all_slots if (t, s) in x]
        model.Add(sum(assigned_slots) + is_unassigned[t] == 1)

    # Each active slot has at most 1 visible track: sum_t x[t, s] + is_nv[s] == is_active[s]
    for s in all_slots:
        track_vars = [x[(t, s)] for t in all_tracks if (t, s) in x]
        model.Add(sum(track_vars) + is_nv[s] == is_active[s])

    # 3. Offense Formation Constraints
    # Fixed offense slots for standard package: 5 OL, QB, 1 RB, 1 TE, 3 WR
    fixed_off_slots = [
        "offense.C_1", "offense.LT_1", "offense.LG_1", "offense.RG_1", "offense.RT_1", "offense.QB_1",
        "offense.RB_1", "offense.TE_1", "offense.WR_1", "offense.WR_2", "offense.WR_3"
    ]
    for s in fixed_off_slots:
        model.Add(is_active[s] == 1)

    # Exactly 11 total active offense players
    model.Add(sum(is_active[s] for s in off_slots) == 11)

    # Skill bounds hierarchy
    for prefix, max_count in [("offense.WR", 5), ("offense.TE", 3), ("offense.RB", 2), ("offense.FB", 1)]:
        for idx in range(2, max_count + 1):
            curr_slot = f"{prefix}_{idx}"
            prev_slot = f"{prefix}_{idx - 1}"
            if curr_slot in is_active and prev_slot in is_active:
                model.Add(is_active[curr_slot] <= is_active[prev_slot])

    # Personnel priors from paired fusion if available
    if personnel_priors:
        for pos, count in personnel_priors.items():
            pos_slots = [s for s in off_slots if f"offense.{pos}_" in s]
            if pos_slots and count <= len(pos_slots):
                model.Add(sum(is_active[s] for s in pos_slots) == count)

    # 4. 5-OL Offensive Lateral Ordering
    # In offensive perspective: LT > LG > C > RG > RT
    ol_slots_ordered = [
        ("offense.LT_1", "offense.LG_1"),
        ("offense.LG_1", "offense.C_1"),
        ("offense.C_1", "offense.RG_1"),
        ("offense.RG_1", "offense.RT_1"),
    ]

    for left_s, right_s in ol_slots_ordered:
        for t1 in all_tracks:
            for t2 in all_tracks:
                if t1 == t2:
                    continue
                if (t1, left_s) in x and (t2, right_s) in x:
                    lat1 = spatial_features.get(t1, {}).get("lateral_offense", 0.0)
                    lat2 = spatial_features.get(t2, {}).get("lateral_offense", 0.0)
                    if lat1 <= lat2:
                        model.Add(x[(t1, left_s)] + x[(t2, right_s)] <= 1)

    # 5. WR vs CB Wing Alignment Depth Constraints:
    # A defensive Cornerback is deeper into defense than the Wide Receiver he is covering
    wr_slots = [s for s in off_slots if "WR_" in s]
    cb_slots = [s for s in def_slots if "CB_" in s]

    for t1 in all_tracks:
        for t2 in all_tracks:
            if t1 == t2:
                continue
            f1 = spatial_features.get(t1, {})
            f2 = spatial_features.get(t2, {})
            lat1 = f1.get("lateral_offense", 0.0)
            lat2 = f2.get("lateral_offense", 0.0)
            d1 = f1.get("depth_los", 0.0)
            d2 = f2.get("depth_los", 0.0)

            same_wing = (lat1 >= 1.5 and lat2 >= 1.5) or (lat1 <= -1.5 and lat2 <= -1.5)
            if same_wing and d1 > d2 + 0.8:
                for ws in wr_slots:
                    for cs in cb_slots:
                        if (t1, ws) in x and (t2, cs) in x:
                            model.Add(x[(t1, ws)] + x[(t2, cs)] <= 1)

    # 6. Hard Semantic Anchors for Offense
    if center_track_id and (center_track_id, "offense.C_1") in x:
        model.Add(x[(center_track_id, "offense.C_1")] == 1)

    if qb_track_id and (qb_track_id, "offense.QB_1") in x:
        model.Add(x[(qb_track_id, "offense.QB_1")] == 1)

    # 7. Defense Formation Constraints
    # Exactly 11 total active defense players
    model.Add(sum(is_active[s] for s in def_slots) == 11)

    # Defense slot prefix hierarchy
    for prefix, max_count in [
        ("defense.DE", 3), ("defense.DT", 4), ("defense.LB", 5), ("defense.CB", 5),
        ("defense.FS", 1), ("defense.SS", 1)
    ]:
        for idx in range(2, max_count + 1):
            curr_slot = f"{prefix}_{idx}"
            prev_slot = f"{prefix}_{idx - 1}"
            if curr_slot in is_active and prev_slot in is_active:
                model.Add(is_active[curr_slot] <= is_active[prev_slot])

    # Core active positions on defense: 2 DE, 1 DT, 3 LB, 3 CB, 1 FS, 1 SS
    model.Add(is_active["defense.DE_1"] == 1)
    model.Add(is_active["defense.DE_2"] == 1)
    model.Add(is_active["defense.DT_1"] == 1)
    model.Add(is_active["defense.LB_1"] == 1)
    model.Add(is_active["defense.LB_2"] == 1)
    model.Add(is_active["defense.LB_3"] == 1)
    model.Add(is_active["defense.CB_1"] == 1)
    model.Add(is_active["defense.CB_2"] == 1)
    model.Add(is_active["defense.CB_3"] == 1)
    model.Add(is_active["defense.FS_1"] == 1)
    model.Add(is_active["defense.SS_1"] == 1)

    # Defense Level Ordering:
    # Front line (DE, DT) is closer to LOS than LB
    # LB is closer to LOS than Safeties
    front_slots = [s for s in def_slots if "DE_" in s or "DT_" in s]
    lb_slots = [s for s in def_slots if "LB_" in s]
    safety_slots = [s for s in def_slots if "FS_" in s or "SS_" in s]

    for t1 in all_tracks:
        for t2 in all_tracks:
            if t1 == t2:
                continue
            d1 = spatial_features.get(t1, {}).get("depth_los", 0.0)
            d2 = spatial_features.get(t2, {}).get("depth_los", 0.0)

            # Front vs LB: only apply between interior linemen and middle linebackers
            # because outside linebackers (OLB/SLB) frequently walk up onto the line of scrimmage
            lat1 = abs(spatial_features.get(t1, {}).get("lateral_offense", 0.0))
            lat2 = abs(spatial_features.get(t2, {}).get("lateral_offense", 0.0))
            if lat1 <= 1.3 and lat2 <= 1.3 and d1 > d2 + 0.8:
                for f_s in front_slots:
                    for l_s in lb_slots:
                        if (t1, f_s) in x and (t2, l_s) in x:
                            model.Add(x[(t1, f_s)] + x[(t2, l_s)] <= 1)

            # LB vs Safety
            if d1 > d2 + 0.8:
                for l_s in lb_slots:
                    for s_s in safety_slots:
                        if (t1, l_s) in x and (t2, s_s) in x:
                            model.Add(x[(t1, l_s)] + x[(t2, s_s)] <= 1)

    # 8. Objective Function
    obj_terms = []

    for (t, s), var in x.items():
        pos = s.split(".")[1].split("_")[0]
        t_scores = candidate_scores.get(t, {})
        base_score = t_scores.get(pos, 0.20)

        # Action anchor bonus
        is_hard = (s == "offense.C_1" and t == center_track_id) or (s == "offense.QB_1" and t == qb_track_id)
        if is_hard:
            base_score = 2.0

        int_score = int(round(base_score * 1000))
        obj_terms.append(int_score * var)

    for s in all_slots:
        obj_terms.append(50 * is_active[s])
        obj_terms.append(-80 * is_nv[s])

    for t in all_tracks:
        obj_terms.append(-10 * is_unassigned[t])

    model.Maximize(sum(obj_terms))

    # 9. Solve with CP-SAT
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.parameters.num_search_workers = 1

    solve_status = solver.Solve(model)
    solve_duration = time.time() - start_time

    assignments: List[PositionAssignment] = []

    if solve_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for s in all_slots:
            if solver.Value(is_active[s]) == 0:
                continue

            side = "offense" if s.startswith("offense.") else "defense"
            pos = s.split(".")[1].split("_")[0]

            assigned_track = None
            for t in all_tracks:
                if (t, s) in x and solver.Value(x[(t, s)]) == 1:
                    assigned_track = t
                    break

            if assigned_track is not None:
                score = candidate_scores.get(assigned_track, {}).get(pos, 0.85)
                is_anchor = (s == "offense.C_1" and assigned_track == center_track_id) or (
                    s == "offense.QB_1" and assigned_track == qb_track_id
                )
                conf = 0.99 if is_anchor else max(0.70, min(0.95, score))

                assignments.append(
                    PositionAssignment(
                        slot_id=s,
                        side=side,
                        position=pos,
                        track_id=assigned_track,
                        visibility="visible",
                        confidence=float(conf),
                        slot_state="ACTIVE_VISIBLE",
                        evidence={"evidence_score": float(score), "cpsat_objective": float(solver.ObjectiveValue())},
                    )
                )
            else:
                assignments.append(
                    PositionAssignment(
                        slot_id=s,
                        side=side,
                        position=pos,
                        track_id=None,
                        visibility="out_of_view",
                        confidence=0.80,
                        slot_state="ACTIVE_NOT_VISIBLE",
                        evidence={"missing_canonical_slot": 1.0},
                    )
                )
    else:
        raise RuntimeError(f"CP-SAT solver failed with status: {solver.StatusName(solve_status)}")

    return assignments
