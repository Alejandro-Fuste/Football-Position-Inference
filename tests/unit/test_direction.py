from position_inference.data.schemas import TrackSummary
from position_inference.geometry import infer_offensive_direction


def test_infer_offensive_direction_cq():
    c_summary = TrackSummary(
        track_id=7, label="player", frames_present=[1], detections=[], first_frame=1, last_frame=1,
        num_boxes=1, coverage_ratio=1.0, median_bbox_height=100, median_bbox_width=50,
        presnap_median_footpoint=(1200.0, 500.0)
    )
    q_summary = TrackSummary(
        track_id=17, label="player", frames_present=[1], detections=[], first_frame=1, last_frame=1,
        num_boxes=1, coverage_ratio=1.0, median_bbox_height=100, median_bbox_width=50,
        presnap_median_footpoint=(1400.0, 500.0) # QB behind Center (x > cx)
    )

    summaries = {7: c_summary, 17: q_summary}
    pred = infer_offensive_direction(summaries, center_track_id=7, qb_track_id=17)

    assert pred.direction == "left"
    assert pred.confidence >= 0.90
