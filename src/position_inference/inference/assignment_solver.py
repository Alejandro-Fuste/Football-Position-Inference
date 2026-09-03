import time
from typing import Dict, List, Optional, Tuple
from ortools.sat.python import cp_model

from position_inference.data.schemas import PositionAssignment, TrackSummary
from position_inference.semantics.personnel import (
    get_fixed_positions,
    get_personnel_bounds,
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
    solver_pass: int = 1,
    snap_frame: Optional[int] = None,
) -> List[PositionAssignment]:
    """Global constrained optimizer for football position assignments.

    Pass 1 infers personnel freely within configured bounds. When Pass 2 receives a paired
    personnel prior, those formation counts are enforced as constraints because the paired
    pipeline has already resolved the formation package (normally from the full-width sideline
    view). Track identities remain view-local and are still chosen independently by this solve.
    """
    model = cp_model.CpModel()
    start_time = time.time()

    all_tracks = sorted(set(offense_track_ids + defense_track_ids))
    off_slots = get_superset_canonical_slots("offense")
    def_slots = get_superset_canonical_slots("defense")
    all_slots = off_slots + def_slots
    off_set = set(offense_track_ids)
    def_set = set(defense_track_ids)

    x: Dict[Tuple[int, str], cp_model.IntVar] = {}

    for t in all_tracks:
        t_sum = track_summaries.get(t)
        if t_sum:
            if not (t_sum.presnap_median_footpoint or t_sum.median_footpoint):
                continue

            # Prefer actual pre-snap geometry. If that is unavailable, permit the median
            # fallback only for a track that already existed by the snap. This preserves
            # sparse-but-real pre-snap players while preventing late ID fragments / false
            # positives from competing for canonical pre-snap positions.
            if t_sum.presnap_median_footpoint is None and snap_frame is not None:
                if t_sum.first_frame > snap_frame:
                    continue

        is_off_eligible = t in off_set
        is_def_eligible = t in def_set
        for s in all_slots:
            if s.startswith("offense.") and not is_off_eligible:
                continue
            if s.startswith("defense.") and not is_def_eligible:
                continue
            x[(t, s)] = model.NewBoolVar(f"x_{t}_{s}")

    is_active = {s: model.NewBoolVar(f"active_{s}") for s in all_slots}
    is_nv = {s: model.NewBoolVar(f"nv_{s}") for s in all_slots}
    is_unassigned = {t: model.NewBoolVar(f"unassigned_{t}") for t in all_tracks}

    for t in all_tracks:
        assigned_slots = [x[(t, s)] for s in all_slots if (t, s) in x]
        model.Add(sum(assigned_slots) + is_unassigned[t] == 1)

    for s in all_slots:
        track_vars = [x[(t, s)] for t in all_tracks if (t, s) in x]
        model.Add(sum(track_vars) + is_nv[s] == is_active[s])

    fixed_off_dict = get_fixed_positions("offense")
    fixed_off_slots = [f"offense.{pos}_1" for pos in fixed_off_dict]
    for s in fixed_off_slots:
        if s in is_active:
            model.Add(is_active[s] == 1)

    model.Add(sum(is_active[s] for s in off_slots) == 11)
    skill_slots = [s for s in off_slots if s not in fixed_off_slots]
    model.Add(sum(is_active[s] for s in skill_slots) == 5)

    off_bounds = get_personnel_bounds("offense")
    for pos in ("RB", "FB", "TE", "WR"):
        pos_slots = [s for s in skill_slots if f"offense.{pos}_" in s]
        bounds = off_bounds.get(pos, {"min": 0, "max": len(pos_slots)})
        model.Add(sum(is_active[s] for s in pos_slots) >= bounds["min"])
        model.Add(sum(is_active[s] for s in pos_slots) <= bounds["max"])

    for pos in ("WR", "TE", "RB"):
        pos_slots = [s for s in skill_slots if f"offense.{pos}_" in s]
        for idx in range(1, len(pos_slots)):
            model.Add(is_active[pos_slots[idx]] <= is_active[pos_slots[idx - 1]])

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

    if center_track_id and (center_track_id, "offense.C_1") in x:
        model.Add(x[(center_track_id, "offense.C_1")] == 1)
    if qb_track_id and (qb_track_id, "offense.QB_1") in x:
        model.Add(x[(qb_track_id, "offense.QB_1")] == 1)

    model.Add(sum(is_active[s] for s in def_slots) == 11)
    def_bounds = get_personnel_bounds("defense")
    for pos in ("DE", "DT", "LB", "CB", "FS", "SS"):
        pos_slots = [s for s in def_slots if f"defense.{pos}_" in s]
        bounds = def_bounds.get(pos, {"min": 0, "max": len(pos_slots)})
        model.Add(sum(is_active[s] for s in pos_slots) >= bounds["min"])
        model.Add(sum(is_active[s] for s in pos_slots) <= bounds["max"])

    front_slots = [s for s in def_slots if "DE_" in s or "DT_" in s]
    lb_slots = [s for s in def_slots if "LB_" in s]
    db_slots = [s for s in def_slots if "CB_" in s or "FS_" in s or "SS_" in s]
    safety_slots = [s for s in def_slots if "FS_" in s or "SS_" in s]
    model.Add(sum(is_active[s] for s in front_slots) >= 3)
    model.Add(sum(is_active[s] for s in front_slots) <= 5)
    model.Add(sum(is_active[s] for s in lb_slots) >= 1)
    model.Add(sum(is_active[s] for s in lb_slots) <= 4)
    model.Add(sum(is_active[s] for s in db_slots) >= 4)
    model.Add(sum(is_active[s] for s in db_slots) <= 6)
    model.Add(sum(is_active[s] for s in safety_slots) >= 1)
    model.Add(sum(is_active[s] for s in safety_slots) <= 2)

    for pos in ("DE", "DT", "LB", "CB"):
        pos_slots = [s for s in def_slots if f"defense.{pos}_" in s]
        for idx in range(1, len(pos_slots)):
            model.Add(is_active[pos_slots[idx]] <= is_active[pos_slots[idx - 1]])

    if solver_pass >= 2 and personnel_priors:
        for pos, target_count in personnel_priors.items():
            pos_slots = [s for s in all_slots if f".{pos}_" in s]
            if pos_slots and 0 <= int(target_count) <= len(pos_slots):
                model.Add(sum(is_active[s] for s in pos_slots) == int(target_count))

    for t1 in all_tracks:
        for t2 in all_tracks:
            if t1 == t2:
                continue
            d1 = spatial_features.get(t1, {}).get("depth_los", 0.0)
            d2 = spatial_features.get(t2, {}).get("depth_los", 0.0)
            lat1 = abs(spatial_features.get(t1, {}).get("lateral_offense", 0.0))
            lat2 = abs(spatial_features.get(t2, {}).get("lateral_offense", 0.0))
            if lat1 <= 1.3 and lat2 <= 1.3 and d1 > d2 + 0.8:
                for front_s in front_slots:
                    for lb_s in lb_slots:
                        if (t1, front_s) in x and (t2, lb_s) in x:
                            model.Add(x[(t1, front_s)] + x[(t2, lb_s)] <= 1)
            if d1 > d2 + 0.8:
                for lb_s in lb_slots:
                    for safety_s in safety_slots:
                        if (t1, lb_s) in x and (t2, safety_s) in x:
                            model.Add(x[(t1, lb_s)] + x[(t2, safety_s)] <= 1)

    obj_terms = []
    for (t, s), var in x.items():
        pos = s.split(".")[1].split("_")[0]
        score = candidate_scores.get(t, {}).get(pos, 0.0)
        obj_terms.append(int(score * 1000) * var)

    for s in all_slots:
        pos = s.split(".")[1].split("_")[0]
        if view == "endzone":
            if pos in {"C", "LT", "LG", "RG", "RT", "QB", "TE", "DE", "DT", "LB"}:
                nv_penalty = 500
            elif pos in {"RB", "FB"}:
                nv_penalty = 350
            else:
                nv_penalty = 90
        elif s.startswith("offense.") and pos in {"C", "LT", "LG", "RG", "RT", "QB"}:
            nv_penalty = 350
        else:
            nv_penalty = 150
        obj_terms.append(-nv_penalty * is_nv[s])

    for var in is_unassigned.values():
        obj_terms.append(-50 * var)

    if personnel_priors and solver_pass < 2:
        for pos, target_count in personnel_priors.items():
            pos_slots = [s for s in all_slots if f".{pos}_" in s]
            for idx, slot in enumerate(pos_slots, start=1):
                obj_terms.append((120 if idx <= target_count else -120) * is_active[slot])

    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10.0
    solve_status = solver.Solve(model)

    assignments: List[PositionAssignment] = []
    if solve_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT solver failed with status: {solver.StatusName(solve_status)}")

    for s in all_slots:
        active_val = solver.Value(is_active[s])
        side = "offense" if s.startswith("offense.") else "defense"
        pos = s.split(".")[1].split("_")[0]

        if active_val == 0:
            assignments.append(
                PositionAssignment(
                    slot_id=s,
                    side=side,
                    position=pos,
                    track_id=None,
                    visibility="unknown",
                    confidence=0.0,
                    slot_state="INACTIVE_SLOT",
                )
            )
            continue

        assigned_track = next(
            (t for t in all_tracks if (t, s) in x and solver.Value(x[(t, s)]) == 1),
            None,
        )

        if assigned_track is not None:
            assigned_score = candidate_scores.get(assigned_track, {}).get(pos, 0.0)
            t_scores = candidate_scores.get(assigned_track, {})
            legal_positions = (
                {"C", "LT", "LG", "RG", "RT", "QB", "RB", "FB", "TE", "WR"}
                if side == "offense"
                else {"DE", "DT", "LB", "CB", "FS", "SS"}
            )
            alt_positions = sorted(
                [(p, sc) for p, sc in t_scores.items() if p != pos and p in legal_positions and sc > 0.0],
                key=lambda item: item[1],
                reverse=True,
            )
            alt_pos = alt_positions[0][0] if alt_positions else None
            alt_score = alt_positions[0][1] if alt_positions else 0.0
            team_tracks = off_set if side == "offense" else def_set
            competing_scores = [
                candidate_scores.get(other_t, {}).get(pos, 0.0)
                for other_t in team_tracks
                if other_t != assigned_track and (other_t, s) in x
            ]
            max_competing = max(competing_scores) if competing_scores else 0.0
            best_alternative_score = max(alt_score, max_competing)
            score_margin = max(0.0, assigned_score - best_alternative_score)

            is_anchor = (
                (s == "offense.C_1" and assigned_track == center_track_id)
                or (s == "offense.QB_1" and assigned_track == qb_track_id)
            )
            if is_anchor:
                conf = 0.99
            elif score_margin >= 0.35:
                conf = min(0.96, assigned_score + 0.08 * score_margin)
            elif score_margin < 0.10:
                conf = max(0.0, assigned_score - 0.25 * (0.10 - score_margin))
            else:
                conf = min(0.90, assigned_score)

            assignments.append(
                PositionAssignment(
                    slot_id=s,
                    side=side,
                    position=pos,
                    track_id=assigned_track,
                    visibility="visible",
                    confidence=float(conf),
                    slot_state="ACTIVE_VISIBLE",
                    assigned_score=float(assigned_score),
                    best_alternative_score=float(best_alternative_score),
                    score_margin=float(score_margin),
                    alternative_position=alt_pos,
                    evidence={
                        "evidence_score": float(assigned_score),
                        "score_margin": float(score_margin),
                        "cpsat_objective": float(solver.ObjectiveValue()),
                        "solver_pass": float(solver_pass),
                    },
                )
            )
            continue

        prior_supported = bool(personnel_priors and personnel_priors.get(pos, 0) > 0)
        if view == "endzone" and pos in {"WR", "CB", "FS", "SS"}:
            nv_conf = 0.60 + (0.15 if prior_supported else 0.0)
        elif prior_supported:
            nv_conf = 0.40
        else:
            nv_conf = 0.25

        assignments.append(
            PositionAssignment(
                slot_id=s,
                side=side,
                position=pos,
                track_id=None,
                visibility="out_of_view",
                confidence=float(min(0.80, nv_conf)),
                slot_state="ACTIVE_NOT_VISIBLE",
                evidence={
                    "missing_canonical_slot": 1.0,
                    "prior_supported": 1.0 if prior_supported else 0.0,
                    "solver_pass": float(solver_pass),
                },
            )
        )

    _ = time.time() - start_time
    return assignments
