from typing import Any, Dict, List, Optional, Set

from position_inference.config import get_personnel_constraints
from position_inference.data.schemas import PositionAssignment


def get_personnel_bounds(side: str) -> Dict[str, Dict[str, int]]:
    """
    Returns min/max count constraints for positions on a given side from configuration.
    """
    cfg = get_personnel_constraints()
    if side == "offense":
        off_cfg = cfg.get("offense", {})
        bounds = {}
        # Fixed positions have min=max=fixed count
        for pos, count in off_cfg.get("fixed_positions", {}).items():
            bounds[pos] = {"min": count, "max": count}
        for pos, b in off_cfg.get("skill_bounds", {}).items():
            bounds[pos] = {"min": b.get("min", 0), "max": b.get("max", 5)}
        return bounds
    else:
        def_cfg = cfg.get("defense", {})
        bounds = {}
        for pos, b in def_cfg.get("bounds", {}).items():
            bounds[pos] = {"min": b.get("min", 0), "max": b.get("max", 5)}
        return bounds


def get_fixed_positions(side: str) -> Dict[str, int]:
    """Returns mapping of guaranteed fixed positions (e.g. 5 OL + QB for offense)."""
    cfg = get_personnel_constraints()
    if side == "offense":
        return dict(cfg.get("offense", {}).get("fixed_positions", {}))
    return {}


def get_superset_canonical_slots(side: str) -> List[str]:
    """
    Returns all potential canonical slot IDs for a given side,
    dynamically derived from the maximum bounds in personnel_constraints.yaml.
    """
    cfg = get_personnel_constraints()
    slots: List[str] = []

    if side == "offense":
        off_cfg = cfg.get("offense", {})
        # Guaranteed fixed positions
        for pos in ("C", "LT", "LG", "RG", "RT", "QB"):
            slots.append(f"offense.{pos}_1")
        # Skill positions up to configured max
        skill_bounds = off_cfg.get("skill_bounds", {})
        for pos in ("RB", "FB", "TE", "WR"):
            b = skill_bounds.get(pos, {})
            max_c = b.get("max", 2 if pos == "RB" else 1 if pos == "FB" else 3 if pos == "TE" else 5)
            for idx in range(1, max_c + 1):
                slots.append(f"offense.{pos}_{idx}")
    else:
        def_cfg = cfg.get("defense", {})
        def_bounds = def_cfg.get("bounds", {})
        for pos in ("DE", "DT", "LB", "CB", "FS", "SS"):
            b = def_bounds.get(pos, {})
            max_c = b.get("max", 3 if pos == "DE" else 4 if pos == "DT" else 5 if pos in ("LB", "CB") else 1)
            for idx in range(1, max_c + 1):
                slots.append(f"defense.{pos}_{idx}")

    return slots


def get_canonical_slots(side: str, personnel_name: str = "11") -> List[str]:
    """
    Returns list of 11 canonical slot IDs for Offense or Defense given package name.
    """
    if side == "offense":
        if personnel_name == "12":  # 1 RB, 2 TE, 2 WR
            return [
                "offense.C_1", "offense.LT_1", "offense.LG_1", "offense.RG_1", "offense.RT_1",
                "offense.QB_1", "offense.RB_1", "offense.TE_1", "offense.TE_2",
                "offense.WR_1", "offense.WR_2"
            ]
        elif personnel_name == "21":  # 2 RB, 1 TE, 2 WR
            return [
                "offense.C_1", "offense.LT_1", "offense.LG_1", "offense.RG_1", "offense.RT_1",
                "offense.QB_1", "offense.RB_1", "offense.FB_1", "offense.TE_1",
                "offense.WR_1", "offense.WR_2"
            ]
        elif personnel_name == "10":  # 1 RB, 0 TE, 4 WR
            return [
                "offense.C_1", "offense.LT_1", "offense.LG_1", "offense.RG_1", "offense.RT_1",
                "offense.QB_1", "offense.RB_1",
                "offense.WR_1", "offense.WR_2", "offense.WR_3", "offense.WR_4"
            ]
        else:  # Default 11 personnel (1 RB, 1 TE, 3 WR)
            return [
                "offense.C_1", "offense.LT_1", "offense.LG_1", "offense.RG_1", "offense.RT_1",
                "offense.QB_1", "offense.RB_1", "offense.TE_1",
                "offense.WR_1", "offense.WR_2", "offense.WR_3"
            ]
    else:
        # Default defense (e.g. 2 DE, 1 DT, 3 LB, 3 CB, 1 FS, 1 SS = 11 players)
        return [
            "defense.DE_1", "defense.DE_2", "defense.DT_1",
            "defense.LB_1", "defense.LB_2", "defense.LB_3",
            "defense.CB_1", "defense.CB_2", "defense.CB_3",
            "defense.FS_1", "defense.SS_1",
        ]


def extract_personnel_hypothesis(assignments: List[PositionAssignment]) -> Dict[str, int]:
    """
    Counts active positions (both visible and not_visible) from assignments.
    Excludes INACTIVE_SLOT.
    """
    counts: Dict[str, int] = {}
    for a in assignments:
        if getattr(a, "slot_state", "ACTIVE_VISIBLE") != "INACTIVE_SLOT":
            counts[a.position] = counts.get(a.position, 0) + 1
    return counts
