from typing import List, Dict
import numpy as np

from position_inference.data.schemas import MotDetection, TrackSummary
from position_inference.geometry.footpoints import compute_footpoint


def summarize_tracks(detections: List[MotDetection], total_frames: int = None) -> Dict[int, TrackSummary]:
    """
    Groups MOT detections by track_id and calculates statistical track summaries.
    """
    by_track: Dict[int, List[MotDetection]] = {}
    for det in detections:
        by_track.setdefault(det.track_id, []).append(det)

    if not total_frames:
        total_frames = max((d.frame for d in detections), default=1)

    summaries: Dict[int, TrackSummary] = {}

    for track_id, det_list in sorted(by_track.items()):
        det_list.sort(key=lambda d: d.frame)
        label = det_list[0].label
        frames_present = [d.frame for d in det_list]
        first_frame = frames_present[0]
        last_frame = frames_present[-1]
        num_boxes = len(det_list)

        coverage_ratio = num_boxes / max(1, (last_frame - first_frame + 1))

        heights = [d.bbox_xywh[3] for d in det_list]
        widths = [d.bbox_xywh[2] for d in det_list]
        footpoints = [compute_footpoint(d.bbox_xywh) for d in det_list]

        med_height = float(np.median(heights)) if heights else 0.0
        med_width = float(np.median(widths)) if widths else 0.0

        med_fp_x = float(np.median([fp[0] for fp in footpoints])) if footpoints else 0.0
        med_fp_y = float(np.median([fp[1] for fp in footpoints])) if footpoints else 0.0
        med_fp = (med_fp_x, med_fp_y)

        summaries[track_id] = TrackSummary(
            track_id=track_id,
            label=label,
            frames_present=frames_present,
            detections=det_list,
            first_frame=first_frame,
            last_frame=last_frame,
            num_boxes=num_boxes,
            coverage_ratio=coverage_ratio,
            median_bbox_height=med_height,
            median_bbox_width=med_width,
            median_footpoint=med_fp,
            validity_score=1.0,
            validity_flags=[],
        )

    return summaries
