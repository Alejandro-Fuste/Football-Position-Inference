from position_inference.data.schemas import PositionAssignment, ViewInferenceResult
from position_inference.inference import fuse_paired_views


def test_fuse_paired_views():
    s_assigns = [
        PositionAssignment("offense.C_1", "offense", "C", 7, "visible", 0.99),
        PositionAssignment("offense.QB_1", "offense", "QB", 17, "visible", 0.99),
    ]
    e_assigns = [
        PositionAssignment("offense.C_1", "offense", "C", 13, "visible", 0.99),
        PositionAssignment("offense.QB_1", "offense", "QB", 11, "visible", 0.99),
    ]

    s_res = ViewInferenceResult("JetSweep_1", "sideline", 0.98, "left", 0.90, s_assigns)
    e_res = ViewInferenceResult("JetSweep_2", "endzone", 0.95, "left", 0.90, e_assigns)

    s_fused, e_fused, warnings = fuse_paired_views(s_res, e_res)

    assert len(warnings) == 0
    assert s_fused.status == "AUTO_ACCEPTED"
    assert e_fused.status == "AUTO_ACCEPTED"
