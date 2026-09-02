from pathlib import Path
import pytest

from position_inference.data.dataset_summary import load_dataset_summary, resolve_video_metadata
from position_inference.data.schemas import VideoMetadata

FIXTURE_DS = Path(__file__).resolve().parent.parent / "fixtures" / "jetsweep_pair_001_002" / "dataset_summary.csv"


def test_resolve_exact_metadata():
    meta = resolve_video_metadata(FIXTURE_DS, "JetSweep_1")
    assert meta is not None
    assert meta.video_id == "JetSweep_1"
    assert meta.view_raw == "sideline"


def test_resolve_mp4_extension():
    meta = resolve_video_metadata(FIXTURE_DS, "JetSweep_1.mp4")
    assert meta is not None
    assert meta.video_id == "JetSweep_1"
    assert meta.view_raw == "sideline"


def test_resolve_case_insensitive():
    meta = resolve_video_metadata(FIXTURE_DS, "jetsweep_2")
    assert meta is not None
    assert meta.video_id == "JetSweep_2"
    assert meta.view_raw == "endzone"


def test_resolve_numeric_clip_order():
    meta = resolve_video_metadata(FIXTURE_DS, "1")
    assert meta is not None
    assert meta.dataset_order == 1


def test_resolve_no_ambiguous_substring_match():
    # JetSweep_1 should NOT match JetSweep_10, 11, etc.
    meta_1 = resolve_video_metadata(FIXTURE_DS, "JetSweep_1")
    meta_10 = resolve_video_metadata(FIXTURE_DS, "JetSweep_10")
    assert meta_1 is not None
    assert meta_10 is not None
    assert meta_1.video_id != meta_10.video_id
    assert meta_1.dataset_order == 1
    assert meta_10.dataset_order == 10


def test_resolve_missing_returns_none():
    meta = resolve_video_metadata(FIXTURE_DS, "NonExistent_Clip_9999")
    assert meta is None
