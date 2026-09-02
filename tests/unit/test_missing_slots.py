from position_inference.data.schemas import PositionAssignment
from position_inference.inference import complete_missing_slots


def test_complete_missing_slots():
    existing = [
        PositionAssignment("offense.C_1", "offense", "C", 7, "visible", 0.99),
        PositionAssignment("offense.QB_1", "offense", "QB", 17, "visible", 0.99),
    ]

    expected = ["offense.C_1", "offense.QB_1", "offense.WR_1", "offense.WR_2"]
    completed = complete_missing_slots(existing, expected, "offense")

    assert len(completed) == 4
    wr1 = next(a for a in completed if a.slot_id == "offense.WR_1")
    assert wr1.track_id is None
    assert wr1.track_id_display == "not_visible"
    assert wr1.visibility == "out_of_view"
