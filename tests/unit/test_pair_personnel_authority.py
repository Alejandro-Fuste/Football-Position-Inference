from position_inference.data.schemas import ViewInferenceResult
from position_inference.inference.paired_fusion import _build_shared_personnel_prior


def _result(video_id, view, confidence):
    result = ViewInferenceResult(
        video_id=video_id,
        view=view,
        view_confidence=0.98,
        offense_direction="left" if view == "sideline" else "down",
        offense_direction_confidence=0.90,
        assignments=[],
    )
    result.confidence = confidence
    return result


def test_sideline_personnel_counts_win_even_when_endzone_scalar_confidence_is_slightly_higher():
    sideline = _result("JetSweep_1", "sideline", 0.24)
    endzone = _result("JetSweep_2", "endzone", 0.26)

    sideline_hyp = {
        "RB": 1,
        "TE": 1,
        "WR": 3,
        "DE": 2,
        "DT": 1,
        "LB": 3,
        "CB": 3,
        "FS": 1,
        "SS": 1,
    }
    endzone_hyp = {
        "WR": 5,
        "DE": 1,
        "DT": 3,
        "LB": 1,
        "CB": 5,
        "SS": 1,
    }

    shared, source = _build_shared_personnel_prior(
        sideline,
        endzone,
        sideline_hyp,
        endzone_hyp,
    )

    assert source == "sideline"
    assert shared == sideline_hyp
    assert shared["DE"] == 2
    assert shared["DT"] == 1
    assert shared["LB"] == 3
    assert shared["WR"] == 3
