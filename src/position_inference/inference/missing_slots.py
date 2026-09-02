from typing import List

from position_inference.data.schemas import PositionAssignment


def complete_missing_slots(
    assignments: List[PositionAssignment],
    expected_slots: List[str],
    side: str = "offense",
) -> List[PositionAssignment]:
    """
    Ensures all expected canonical slots are represented in the assignment list,
    creating `track_id=None` (`not_visible`) assignments for unassigned slots.
    """
    assigned_slots = {a.slot_id for a in assignments}
    completed = list(assignments)

    for slot_id in expected_slots:
        if slot_id not in assigned_slots:
            pos = slot_id.split(".")[1].rsplit("_", 1)[0]
            completed.append(
                PositionAssignment(
                    slot_id=slot_id,
                    side=side,
                    position=pos,
                    track_id=None,
                    visibility="out_of_view",
                    confidence=0.80,
                    evidence={"missing_canonical_slot": 1.0},
                )
            )

    return completed
