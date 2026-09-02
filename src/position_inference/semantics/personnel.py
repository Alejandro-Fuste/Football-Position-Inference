from typing import Dict, List, Optional, Set

from position_inference.data.schemas import PositionAssignment

SUPERSET_OFFENSE_SLOTS = [
    "offense.C_1", "offense.LT_1", "offense.LG_1", "offense.RG_1", "offense.RT_1", "offense.QB_1",
    "offense.RB_1", "offense.RB_2",
    "offense.FB_1",
    "offense.TE_1", "offense.TE_2", "offense.TE_3",
    "offense.WR_1", "offense.WR_2", "offense.WR_3", "offense.WR_4", "offense.WR_5",
]

SUPERSET_DEFENSE_SLOTS = [
    "defense.DE_1", "defense.DE_2", "defense.DE_3",
    "defense.DT_1", "defense.DT_2", "defense.DT_3", "defense.DT_4",
    "defense.LB_1", "defense.LB_2", "defense.LB_3", "defense.LB_4", "defense.LB_5",
    "defense.CB_1", "defense.CB_2", "defense.CB_3", "defense.CB_4", "defense.CB_5",
    "defense.FS_1",
    "defense.SS_1",
]


def get_superset_canonical_slots(side: str) -> List[str]:
    """Returns all potential canonical slot IDs for a given side."""
    return list(SUPERSET_OFFENSE_SLOTS) if side == "offense" else list(SUPERSET_DEFENSE_SLOTS)


def get_canonical_slots(side: str, personnel_name: str = "11") -> List[str]:
    """
    Returns full list of 11 canonical slot IDs for Offense or Defense given package name.
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


def get_canonical_slots_from_counts(side: str, position_counts: Dict[str, int]) -> List[str]:
    """
    Builds exact list of active canonical slot IDs from position counts.
    E.g. {"C": 1, "LT": 1, "LG": 1, "RG": 1, "RT": 1, "QB": 1, "RB": 1, "TE": 1, "WR": 3}
    -> ['offense.C_1', ..., 'offense.WR_3']
    """
    slots = []
    prefix = f"{side}."
    for pos, count in sorted(position_counts.items()):
        for idx in range(1, count + 1):
            slots.append(f"{prefix}{pos}_{idx}")
    return slots


def extract_personnel_hypothesis(assignments: List[PositionAssignment]) -> Dict[str, int]:
    """
    Counts active positions (both visible and not_visible) from assignments.
    """
    counts: Dict[str, int] = {}
    for a in assignments:
        if getattr(a, "slot_state", "ACTIVE_VISIBLE") != "INACTIVE_SLOT":
            counts[a.position] = counts.get(a.position, 0) + 1
    return counts
