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


def _visible_by_position(result):
    out = {}
    for a in result.assignments:
        if a.slot_state == "ACTIVE_VISIBLE" and a.track_id is not None:
            out.setdefault(a.position, set()).add(a.track_id)
    return out


@pytest.mark.integration
def test_jetsweep_pair_golden(tmp_path):
    mot_path_s = FIXTURE_DIR / "JetSweep_1_cvat_mot.zip"
    mot_path_e = FIXTURE_DIR / "JetSweep_2_cvat_mot.zip"
    actions_path = FIXTURE_DIR / "key_actions.csv"
    ds_path = FIXTURE_DIR / "dataset_summary.csv"

    for path in (mot_path_s, mot_path_e, actions_path, ds_path):
        assert path.exists(), f"Required paired fixture input missing: {path.name}"

    s_fused, e_fused, pair_summary = fuse_paired_views_two_pass(
        sideline_mot=mot_path_s,
        endzone_mot=mot_path_e,
        action_source=actions_path,
        sideline_id="JetSweep_1",
        endzone_id="JetSweep_2",
        dataset_summary=ds_path,
        pair_id="jetsweep_pair_001_002",
    )

    assert s_fused.view == "sideline"
    assert e_fused.view == "endzone"
    assert s_fused.solver_pass == 2
    assert e_fused.solver_pass == 2
    assert s_fused.preliminary_personnel_hypothesis is not None
    assert e_fused.preliminary_personnel_hypothesis is not None
    assert s_fused.paired_personnel_prior is not None
    assert e_fused.paired_personnel_prior is not None

    # The sideline view is the formation-count authority for this dataset pair.
    assert pair_summary["shared_personnel_source"] == "sideline"
    assert pair_summary["shared_personnel_prior"] == pair_summary["preliminary_sideline_personnel"]

    s_pos = _visible_by_position(s_fused)
    e_pos = _visible_by_position(e_fused)

    # Sideline strict golden mapping must remain correct after paired Pass 2.
    assert s_pos.get("C") == {7}
    assert s_pos.get("QB") == {17}
    assert s_pos.get("LT") == {5}
    assert s_pos.get("LG") == {3}
    assert s_pos.get("RG") == {9}
    assert s_pos.get("RT") == {13}
    assert s_pos.get("TE") == {12}
    assert s_pos.get("RB") == {20}
    assert s_pos.get("WR") == {1, 19, 21}
    assert s_pos.get("DE") == {6, 8}
    assert s_pos.get("DT") == {4}
    assert s_pos.get("LB") == {10, 15, 18}
    assert s_pos.get("CB") == {14, 16, 22}
    assert s_pos.get("FS") == {2}
    assert s_pos.get("SS") == {11}

    # Endzone final Pass 2 must match every known JetSweep_2 annotation.
    # RB, DT, and a third WR are unresolved in the source sheet and are not asserted.
    assert e_pos.get("C") == {13}
    assert e_pos.get("QB") == {11}
    assert e_pos.get("LT") == {12}
    assert e_pos.get("LG") == {21}
    assert e_pos.get("RG") == {9}
    assert e_pos.get("RT") == {14}
    assert e_pos.get("TE") == {15}
    assert {5, 6}.issubset(e_pos.get("WR", set()))
    assert {8, 16}.issubset(e_pos.get("DE", set()))
    assert {7, 10}.issubset(e_pos.get("LB", set()))
    assert {1, 3}.issubset(e_pos.get("CB", set()))
    assert e_pos.get("FS") == {2}
    assert e_pos.get("SS") == {4}

    # Known-visible endzone trench roles may not be converted to not_visible by pairing.
    e_nv = {a.position for a in e_fused.assignments if a.slot_state == "ACTIVE_NOT_VISIBLE"}
    for pos in ("C", "LT", "LG", "RG", "RT", "QB", "TE"):
        assert pos not in e_nv, f"Known-visible endzone role {pos} was incorrectly marked not_visible"

    expected_keys = [
        "pair_id",
        "sideline_video_id",
        "endzone_video_id",
        "sideline_view",
        "endzone_view",
        "preliminary_sideline_personnel",
        "preliminary_endzone_personnel",
        "shared_personnel_prior",
        "shared_personnel_source",
        "personnel_disagreements",
        "final_sideline_personnel",
        "final_endzone_personnel",
        "pair_resolution_margin",
        "pair_status",
        "pair_warnings",
        "confidence_calibrated",
    ]
    for key in expected_keys:
        assert key in pair_summary, f"pair_summary missing required key: {key}"

    assert pair_summary["sideline_view"] == "sideline"
    assert pair_summary["endzone_view"] == "endzone"
    assert not pair_summary["confidence_calibrated"]

    write_playertrack_csv(s_fused, tmp_path / "JetSweep_1_playertrack.csv", video_number="JetSweep_1")
    write_inference_json(s_fused, tmp_path / "JetSweep_1_inference.json", pair_id="jetsweep_pair_001_002")
    write_review_report_markdown(s_fused, tmp_path / "JetSweep_1_review.md", pair_id="jetsweep_pair_001_002")
    write_playertrack_csv(e_fused, tmp_path / "JetSweep_2_playertrack.csv", video_number="JetSweep_2")
    write_inference_json(e_fused, tmp_path / "JetSweep_2_inference.json", pair_id="jetsweep_pair_001_002")
    write_review_report_markdown(e_fused, tmp_path / "JetSweep_2_review.md", pair_id="jetsweep_pair_001_002")

    with open(tmp_path / "pair_summary.json", "w", encoding="utf-8") as f:
        json.dump(pair_summary, f, indent=2)

    for filename in (
        "pair_summary.json",
        "JetSweep_1_playertrack.csv",
        "JetSweep_2_playertrack.csv",
        "JetSweep_1_inference.json",
        "JetSweep_2_inference.json",
        "JetSweep_1_review.md",
        "JetSweep_2_review.md",
    ):
        assert (tmp_path / filename).exists()
