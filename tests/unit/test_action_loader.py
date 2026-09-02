import os
from pathlib import Path
import pytest

from position_inference.data import load_action_annotations, ActionAnnotation

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "jetsweep_pair_001_002"
ENV_DATA = os.environ.get("POSITION_INFERENCE_TEST_DATA")
DATA_DIR = Path(ENV_DATA) if ENV_DATA else FIXTURES_DIR


def test_load_action_annotations():
    actions_csv = DATA_DIR / "key_actions.csv"
    if not actions_csv.exists():
        pytest.skip("KeyActions CSV missing")

    actions = load_action_annotations(actions_csv)
    assert len(actions) > 0
    assert isinstance(actions[0], ActionAnnotation)

    js1_actions = [a for a in actions if a.video_id == "JetSweep_1"]
    assert len(js1_actions) > 0
    snap_act = next((a for a in js1_actions if a.action == "Ball Snap"), None)
    assert snap_act is not None
    assert snap_act.actor_track_id == 7
