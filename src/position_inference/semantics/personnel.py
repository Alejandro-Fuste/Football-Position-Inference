from typing import Dict, List


def get_canonical_slots(
    side: str,
    personnel_name: str = "11",
) -> List[str]:
    """
    Returns full list of 11 canonical slot IDs for Offense or Defense.
    """
    if side == "offense":
        if personnel_name == "12": # 1 RB, 2 TE, 2 WR
            return [
                "offense.C_1", "offense.LT_1", "offense.LG_1", "offense.RG_1", "offense.RT_1",
                "offense.QB_1", "offense.RB_1", "offense.TE_1", "offense.TE_2",
                "offense.WR_1", "offense.WR_2"
            ]
        elif personnel_name == "21": # 2 RB (or 1 RB + 1 FB), 1 TE, 2 WR
            return [
                "offense.C_1", "offense.LT_1", "offense.LG_1", "offense.RG_1", "offense.RT_1",
                "offense.QB_1", "offense.RB_1", "offense.FB_1", "offense.TE_1",
                "offense.WR_1", "offense.WR_2"
            ]
        else: # Default 11 personnel (1 RB, 1 TE, 3 WR)
            return [
                "offense.C_1", "offense.LT_1", "offense.LG_1", "offense.RG_1", "offense.RT_1",
                "offense.QB_1", "offense.RB_1", "offense.TE_1",
                "offense.WR_1", "offense.WR_2", "offense.WR_3"
            ]
    else: # Defense (e.g. 4-2-5 or 4-3-4)
        return [
            "defense.DE_1", "defense.DE_2", "defense.DT_1", "defense.DT_2",
            "defense.LB_1", "defense.LB_2",
            "defense.CB_1", "defense.CB_2", "defense.FS_1", "defense.SS_1", "defense.CB_3"
        ]
