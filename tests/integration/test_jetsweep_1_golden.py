from pathlib import Path
import pytest

from position_inference.data import (
    load_action_annotations,
    load_ground_truth_roles,
    load_mot_detections,
)
from position_inference.evaluation import evaluate_predictions
from position_inference.inference import fuse_paired_views
from position_inference.output import (
    write_inference_json,
    write_playertrack_csv,
    write_review_report_markdown,
)
from position_inference.pipeline import infer_video_positions

DATA_DIR = Path("/Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Combine_Tracks_and_Actions/data")


@pytest.mark.integration
def test_jetsweep_1_golden_inference(tmp_path):
    mot_path_s = DATA_DIR / "tracking" / "JetSweep" / "JetSweep_1_cvat_mot.zip"
    mot_path_e = DATA_DIR / "tracking" / "JetSweep" / "JetSweep_2_cvat_mot.zip"
    actions_path = DATA_DIR / "key_actions" / "JetSweep.csv"
    gt_path = DATA_DIR / "player_tracks" / "JetSweep.csv"

    assert mot_path_s.exists(), "Sideline MOT zip missing"
    assert mot_path_e.exists(), "Endzone MOT zip missing"
    assert actions_path.exists(), "KeyActions CSV missing"
    assert gt_path.exists(), "PlayerTrack CSV missing"

    # 1. Infer Sideline clip
    s_result = infer_video_positions(mot_path_s, actions_path, video_id="JetSweep_1")
    assert s_result.video_id == "JetSweep_1"
    assert s_result.view == "sideline"
    assert len(s_result.assignments) >= 22

    # 2. Infer Endzone clip
    e_result = infer_video_positions(mot_path_e, actions_path, video_id="JetSweep_2")
    assert e_result.video_id == "JetSweep_2"

    # 3. Fuse paired views
    s_fused, e_fused, warnings = fuse_paired_views(s_result, e_result)
    assert s_fused is not None
    assert e_fused is not None

    # 4. Compare with Ground Truth
    gt_all = load_ground_truth_roles(gt_path)
    gt_js1 = [g for g in gt_all if g.video_id == "JetSweep_1"]
    assert len(gt_js1) == 22, "Ground truth for JetSweep_1 should contain 22 roles"

    metrics = evaluate_predictions(gt_js1, s_fused.assignments)
    assert metrics["center_accuracy"] == 1.0, "Center should be correctly inferred"
    assert metrics["qb_accuracy"] == 1.0, "QB should be correctly inferred"
    assert metrics["high_confidence_precision"] >= 0.90, "High confidence assignments should have >=90% precision"

    # 5. Verify outputs written cleanly
    out_csv = tmp_path / "JetSweep_1_playertrack.csv"
    out_json = tmp_path / "JetSweep_1_inference.json"
    out_md = tmp_path / "JetSweep_1_review.md"

    write_playertrack_csv(s_fused, out_csv)
    write_inference_json(s_fused, out_json)
    write_review_report_markdown(s_fused, out_md)

    assert out_csv.exists()
    assert out_json.exists()
    assert out_md.exists()
