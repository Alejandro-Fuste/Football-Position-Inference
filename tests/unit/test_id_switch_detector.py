from position_inference.data.schemas import MotDetection, TrackSummary
from position_inference.quality import detect_id_switches


def test_detect_id_switches():
    dets = [
        MotDetection(1, 5, "player", (100.0, 100.0, 50.0, 100.0)),
        MotDetection(2, 5, "player", (102.0, 101.0, 50.0, 100.0)),
        MotDetection(3, 5, "player", (500.0, 500.0, 50.0, 100.0)), # Sudden 400px jump
    ]

    summary = TrackSummary(
        track_id=5, label="player", frames_present=[1, 2, 3], detections=dets,
        first_frame=1, last_frame=3, num_boxes=3, coverage_ratio=1.0,
        median_bbox_height=100.0, median_bbox_width=50.0
    )

    switches = detect_id_switches({5: summary}, max_jump_pixels=100.0)
    assert len(switches) == 1
    assert switches[0]["track_id"] == 5
    assert switches[0]["frame"] == 3
