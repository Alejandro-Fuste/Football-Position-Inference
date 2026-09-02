import json
from pathlib import Path
import pytest

from position_inference.inference import fuse_paired_views_two_pass
from position_inference.output import (
    write_inference_json,
    write_playertrack_csv,
    write_review_report_markdown,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "jetsweep_pair_001_002"


@pytest.mark.integration
def test_jetsweep_pair_golden(tmp_path):
    mot_path_s = FIXTURE_DIR / "JetSweep_1_cvat_mot.zip"
    mot_path_e = FIXTURE_DIR / "JetSweep_2_cvat_mot.zip"
    actions_path = FIXTURE_DIR / "key_actions.csv"
    ds_path = FIXTURE_DIR / "dataset_summary.csv"

    assert mot_path_s.exists(), "Sideline MOT zip missing"
    assert mot_path_e.exists(), "Endzone MOT zip missing"
    assert actions_path.exists(), "KeyActions CSV missing"
    assert ds_path.exists(), "DatasetSummary CSV missing"

    # 1. Execute full Two-Pass Paired-View Inference
    s_fused, e_fused, pair_summary = fuse_paired_views_two_pass(
        sideline_mot=mot_path_s,
        endzone_mot=mot_path_e,
        action_source=actions_path,
        sideline_id="JetSweep_1",
        endzone_id="JetSweep_2",
        dataset_summary=ds_path,
        pair_id="jetsweep_pair_001_002",
    )

    # 2. Camera View Verification via DatasetSummary
    assert s_fused.view == "sideline", "JetSweep_1 must be sideline view"
    assert e_fused.view == "endzone", "JetSweep_2 must be endzone view"

    # 3. Two-Pass Execution & Provenance
    assert s_fused.solver_pass == 2, "Final sideline output must come from Pass 2"
    assert e_fused.solver_pass == 2, "Final endzone output must come from Pass 2"

    assert s_fused.preliminary_personnel_hypothesis is not None
    assert e_fused.preliminary_personnel_hypothesis is not None
    assert s_fused.paired_personnel_prior is not None
    assert e_fused.paired_personnel_prior is not None

    # 4. Independent Track IDs (No Cross-View ID Contamination)
    s_tracks = {a.track_id for a in s_fused.assignments if a.track_id is not None}
    e_tracks = {a.track_id for a in e_fused.assignments if a.track_id is not None}

    # Center tracks must match their respective independent video track IDs
    s_c = [a.track_id for a in s_fused.assignments if a.position == "C" and a.slot_state == "ACTIVE_VISIBLE"]
    e_c = [a.track_id for a in e_fused.assignments if a.position == "C" and a.slot_state == "ACTIVE_VISIBLE"]
    assert s_c == [7], "Sideline center must be track 7"
    assert e_c == [13], "Endzone center must be track 13 (independent tracking space)"

    # QB tracks must match their respective independent video track IDs
    s_qb = [a.track_id for a in s_fused.assignments if a.position == "QB" and a.slot_state == "ACTIVE_VISIBLE"]
    e_qb = [a.track_id for a in e_fused.assignments if a.position == "QB" and a.slot_state == "ACTIVE_VISIBLE"]
    assert s_qb == [17], "Sideline QB must be track 17"
    assert e_qb == [11], "Endzone QB must be track 11 (independent tracking space)"

    # 5. Full Pair Summary Verification
    expected_keys = [
        "pair_id",
        "sideline_video_id",
        "endzone_video_id",
        "sideline_view",
        "endzone_view",
        "preliminary_sideline_personnel",
        "preliminary_endzone_personnel",
        "shared_personnel_prior",
        "final_sideline_personnel",
        "final_endzone_personnel",
        "pair_resolution_margin",
        "pair_status",
        "pair_warnings",
        "confidence_calibrated",
    ]
    for k in expected_keys:
        assert k in pair_summary, f"pair_summary missing required key: {k}"

    assert pair_summary["sideline_view"] == "sideline"
    assert pair_summary["endzone_view"] == "endzone"
    assert not pair_summary["confidence_calibrated"]

    # 6. Verify Artifact Export
    write_playertrack_csv(s_fused, tmp_path / "JetSweep_1_playertrack.csv", video_number="JetSweep_1")
    write_inference_json(s_fused, tmp_path / "JetSweep_1_inference.json", pair_id="jetsweep_pair_001_002")
    write_review_report_markdown(s_fused, tmp_path / "JetSweep_1_review.md", pair_id="jetsweep_pair_001_002")

    write_playertrack_csv(e_fused, tmp_path / "JetSweep_2_playertrack.csv", video_number="JetSweep_2")
    write_inference_json(e_fused, tmp_path / "JetSweep_2_inference.json", pair_id="jetsweep_pair_001_002")
    write_review_report_markdown(e_fused, tmp_path / "JetSweep_2_review.md", pair_id="jetsweep_pair_001_002")

    with open(tmp_path / "pair_summary.json", "w", encoding="utf-8") as f:
        json.dump(pair_summary, f, indent=2)

    assert (tmp_path / "pair_summary.json").exists()
    assert (tmp_path / "JetSweep_1_playertrack.csv").exists()
    assert (tmp_path / "JetSweep_2_playertrack.csv").exists()
    assert (tmp_path / "JetSweep_1_inference.json").exists()
    assert (tmp_path / "JetSweep_2_inference.json").exists()
    assert (tmp_path / "JetSweep_1_review.md").exists()
    assert (tmp_path / "JetSweep_2_review.md").exists()
