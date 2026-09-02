import os
from pathlib import Path
import pytest

from position_inference.data import load_mot_detections, MotDetection

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "jetsweep_pair_001_002"
ENV_DATA = os.environ.get("POSITION_INFERENCE_TEST_DATA")
DATA_DIR = Path(ENV_DATA) if ENV_DATA else FIXTURES_DIR


def test_mot_loader_zip():
    mot_zip = DATA_DIR / "JetSweep_1_cvat_mot.zip"
    if not mot_zip.exists():
        pytest.skip("MOT zip file not available")

    dets = load_mot_detections(mot_zip)
    assert len(dets) > 0
    assert isinstance(dets[0], MotDetection)
    assert dets[0].frame >= 1
    assert len(dets[0].bbox_xywh) == 4
