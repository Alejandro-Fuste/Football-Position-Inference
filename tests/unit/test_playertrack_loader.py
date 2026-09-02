import os
from pathlib import Path
import pytest

from position_inference.data import load_ground_truth_roles, GroundTruthRole

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "jetsweep_pair_001_002"
ENV_DATA = os.environ.get("POSITION_INFERENCE_TEST_DATA")
DATA_DIR = Path(ENV_DATA) if ENV_DATA else FIXTURES_DIR


def test_load_ground_truth_roles():
    gt_csv = DATA_DIR / "player_tracks.csv"
    if not gt_csv.exists():
        pytest.skip("PlayerTrack CSV missing")

    gt_roles = load_ground_truth_roles(gt_csv)
    assert len(gt_roles) > 0
    assert isinstance(gt_roles[0], GroundTruthRole)

    js1_roles = [g for g in gt_roles if g.video_id == "JetSweep_1"]
    assert len(js1_roles) == 22

    c_role = next(g for g in js1_roles if g.position == "C")
    assert c_role.track_id == 7
    assert c_role.track_state == "VISIBLE"

    qb_role = next(g for g in js1_roles if g.position == "QB")
    assert qb_role.track_id == 17
    assert qb_role.track_state == "VISIBLE"
