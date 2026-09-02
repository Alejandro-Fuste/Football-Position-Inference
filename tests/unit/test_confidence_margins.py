import pytest
from position_inference.data.schemas import PositionAssignment, ViewInferenceResult
from position_inference.inference.confidence import evaluate_result_confidence


def test_confidence_uncalibrated_conservative():
    # When calibrated: false, status should be at most REVIEW_RECOMMENDED even if confidence is high
    assignments = [
        PositionAssignment(
            slot_id="offense.C_1", side="offense", position="C", track_id=1,
            visibility="visible", confidence=0.99, score_margin=0.5
        ),
        PositionAssignment(
            slot_id="offense.QB_1", side="offense", position="QB", track_id=2,
            visibility="visible", confidence=0.99, score_margin=0.5
        ),
    ]

    result = ViewInferenceResult(
        video_id="test_vid", view="sideline", view_confidence=0.98,
        offense_direction="left", offense_direction_confidence=0.95,
        assignments=assignments, status="AUTO_ACCEPTED"
    )

    evaluated = evaluate_result_confidence(result)
    assert not evaluated.confidence_calibrated
    assert evaluated.status == "REVIEW_RECOMMENDED"


def test_confidence_unknown_view_triggers_warning():
    assignments = [
        PositionAssignment(
            slot_id="offense.C_1", side="offense", position="C", track_id=1,
            visibility="visible", confidence=0.99, score_margin=0.5
        ),
        PositionAssignment(
            slot_id="offense.QB_1", side="offense", position="QB", track_id=2,
            visibility="visible", confidence=0.99, score_margin=0.5
        ),
    ]

    result = ViewInferenceResult(
        video_id="test_vid", view="unknown", view_confidence=0.50,
        offense_direction="left", offense_direction_confidence=0.95,
        assignments=assignments, status="AUTO_ACCEPTED"
    )

    evaluated = evaluate_result_confidence(result)
    assert "unresolved_camera_view" in evaluated.warnings
    assert evaluated.status == "HUMAN_REQUIRED"
