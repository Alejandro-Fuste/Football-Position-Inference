import pytest
from pathlib import Path

from position_inference.data.action_loader import (
    ActionVideoNotFoundError,
    filter_actions_for_video,
    load_action_annotations,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "jetsweep_pair_001_002"


def test_keyactions_matching_exact():
    actions = load_action_annotations(FIXTURES_DIR / "key_actions.csv")
    matched = filter_actions_for_video(actions, "JetSweep_1")
    assert len(matched) > 0
    assert all(a.video_id == "JetSweep_1" for a in matched)


def test_keyactions_matching_missing_error():
    actions = load_action_annotations(FIXTURES_DIR / "key_actions.csv")
    with pytest.raises(ActionVideoNotFoundError) as exc_info:
        filter_actions_for_video(actions, "NonExistentVideo_99")

    assert "NonExistentVideo_99" in str(exc_info.value)
    assert exc_info.value.video_id == "NonExistentVideo_99"


def test_keyactions_matching_allow_missing():
    actions = load_action_annotations(FIXTURES_DIR / "key_actions.csv")
    res = filter_actions_for_video(actions, "NonExistentVideo_99", allow_missing_actions=True)
    assert res == []
