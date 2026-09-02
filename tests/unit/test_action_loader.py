from pathlib import Path
import pytest

from position_inference.data import load_action_annotations, ActionAnnotation

DATA_DIR = Path("/Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Combine_Tracks_and_Actions/data")


def test_load_action_annotations():
    actions_csv = DATA_DIR / "key_actions" / "JetSweep.csv"
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
