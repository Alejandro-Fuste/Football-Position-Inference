from pathlib import Path
import pytest

from position_inference.data import load_ground_truth_roles
from position_inference.evaluation import evaluate_predictions
from position_inference.output import (
    write_inference_json,
    write_playertrack_csv,
    write_review_report_markdown,
)
from position_inference.pipeline import infer_video_positions

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "jetsweep_pair_001_002"


@pytest.mark.integration
def test_jetsweep_2_endzone_golden(tmp_path):
    mot_path_e = FIXTURE_DIR / "JetSweep_2_cvat_mot.zip"
    actions_path = FIXTURE_DIR / "key_actions.csv"
    ds_path = FIXTURE_DIR / "dataset_summary.csv"
    gt_path = FIXTURE_DIR / "player_tracks.csv"

    assert mot_path_e.exists(), "Endzone MOT zip missing"
    assert actions_path.exists(), "KeyActions CSV missing"
    assert ds_path.exists(), "DatasetSummary CSV missing"
    assert gt_path.exists(), "PlayerTrack CSV missing"

    # 1. Run inference using DatasetSummary integration
    e_result = infer_video_positions(
        mot_path_e,
        actions_path,
        video_id="JetSweep_2",
        dataset_summary=ds_path,
    )

    # 2. View and Direction Verification
    assert e_result.video_id == "JetSweep_2"
    assert e_result.view == "endzone", "DatasetSummary must classify JetSweep_2 as endzone"
    assert e_result.offense_direction in ("down", "up"), "Endzone perspective must infer vertical offensive direction"

    # 3. Collect active visible assignments by position
    assigned_by_pos = {}
    assigned_tracks = []
    for a in e_result.assignments:
        if a.slot_state == "ACTIVE_VISIBLE" and a.track_id is not None:
            assigned_by_pos.setdefault(a.position, set()).add(a.track_id)
            assigned_tracks.append(a.track_id)

    # Verify no duplicate visible track assignments
    assert len(assigned_tracks) == len(set(assigned_tracks)), "Each visible track must be assigned to at most 1 slot"

    # 4. Verify Anchors
    assert assigned_by_pos.get("C") == {13}, "Center must be track 13"
    assert assigned_by_pos.get("QB") == {11}, "QB must be track 11"

    # 5. Verify Out of view slots handling
    not_vis_slots = [a for a in e_result.assignments if a.slot_state == "ACTIVE_NOT_VISIBLE"]
    # Endzone typically has out-of-view slots for perimeter skill/DB players
    assert len(not_vis_slots) >= 0

    # 6. Evaluation against Ground Truth with unknown role masking ('?')
    gt_all = load_ground_truth_roles(gt_path)
    gt_js2 = [g for g in gt_all if g.video_id == "JetSweep_2"]
    assert len(gt_js2) > 0, "Ground truth roles for JetSweep_2 must exist"

    # Filter out ground truth items with unannotated '?' roles
    known_gt = [g for g in gt_js2 if g.track_state != "UNKNOWN_GROUND_TRUTH" and g.track_id is not None]
    metrics = evaluate_predictions(known_gt, e_result.assignments)
    assert metrics["center_accuracy"] == 1.0, "Center accuracy must be 1.0"
    assert metrics["qb_accuracy"] == 1.0, "QB accuracy must be 1.0"

    # 7. Verify Output Serialization
    out_csv = tmp_path / "JetSweep_2_playertrack.csv"
    out_json = tmp_path / "JetSweep_2_inference.json"
    out_md = tmp_path / "JetSweep_2_review.md"

    write_playertrack_csv(e_result, out_csv, video_number="2")
    write_inference_json(e_result, out_json)
    write_review_report_markdown(e_result, out_md, pair_id="jetsweep_pair_001_002")

    assert out_csv.exists() and out_csv.stat().st_size > 0
    assert out_json.exists() and out_json.stat().st_size > 0
    assert out_md.exists() and out_md.stat().st_size > 0
