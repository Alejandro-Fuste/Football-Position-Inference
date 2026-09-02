from pathlib import Path
import pytest

from position_inference.data import load_ground_truth_roles, GroundTruthRole

DATA_DIR = Path("/Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Combine_Tracks_and_Actions/data")


def test_load_ground_truth_roles():
    gt_csv = DATA_DIR / "player_tracks" / "JetSweep.csv"
    if not gt_csv.exists():
        pytest.skip("PlayerTrack CSV missing")

    gt_roles = load_ground_truth_roles(gt_csv)
    assert len(gt_roles) > 0
    assert isinstance(gt_roles[0], GroundTruthRole)

    js1_roles = [g for g in gt_roles if g.video_id == "JetSweep_1"]
    assert len(js1_roles) == 22

    c_role = next(g for g in js1_roles if g.position == "C")
    assert c_role.track_id == 7

    qb_role = next(g for g in js1_roles if g.position == "QB")
    assert qb_role.track_id == 17
