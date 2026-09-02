import pytest
from pathlib import Path

from position_inference.data import load_mot_detections, MotDetection

DATA_DIR = Path("/Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Combine_Tracks_and_Actions/data")


def test_mot_loader_zip():
    mot_zip = DATA_DIR / "tracking" / "JetSweep" / "JetSweep_1_cvat_mot.zip"
    if not mot_zip.exists():
        pytest.skip("MOT zip file not available")

    dets = load_mot_detections(mot_zip)
    assert len(dets) > 0
    assert isinstance(dets[0], MotDetection)
    assert dets[0].frame >= 1
    assert len(dets[0].bbox_xywh) == 4
