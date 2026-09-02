from typing import Dict, List
import numpy as np

from position_inference.data.schemas import TrackSummary
from position_inference.geometry.footpoints import compute_footpoint


def detect_id_switches(
    track_summaries: Dict[int, TrackSummary],
    max_jump_pixels: float = 120.0,
    max_scale_change_ratio: float = 2.0,
) -> List[Dict[str, float]]:
    """
    Detects suspected MOT ID switches in track trajectories.
    Returns list of dicts describing suspected switches (track_id, frame, confidence, reason).
    """
    suspected_switches: List[Dict[str, float]] = []

    for track_id, summary in track_summaries.items():
        if summary.label != "player" or len(summary.detections) < 3:
            continue

        dets = sorted(summary.detections, key=lambda d: d.frame)

        for i in range(1, len(dets)):
            prev_d = dets[i - 1]
            curr_d = dets[i]

            frame_gap = curr_d.frame - prev_d.frame
            if frame_gap > 10:
                continue

            prev_fp = compute_footpoint(prev_d.bbox_xywh)
            curr_fp = compute_footpoint(curr_d.bbox_xywh)

            dist = np.sqrt((curr_fp[0] - prev_fp[0]) ** 2 + (curr_fp[1] - prev_fp[1]) ** 2)

            prev_h = prev_d.bbox_xywh[3]
            curr_h = curr_d.bbox_xywh[3]
            scale_ratio = max(curr_h / max(prev_h, 1.0), prev_h / max(curr_h, 1.0))

            # Trigger 1: Sudden spatial jump over short frame gap
            if frame_gap == 1 and dist > max_jump_pixels:
                suspected_switches.append(
                    {
                        "track_id": track_id,
                        "frame": curr_d.frame,
                        "distance_jump": float(dist),
                        "confidence": 0.85,
                        "reason": f"Spatial jump of {dist:.1f}px in consecutive frames",
                    }
                )
            elif scale_ratio > max_scale_change_ratio and frame_gap <= 2:
                suspected_switches.append(
                    {
                        "track_id": track_id,
                        "frame": curr_d.frame,
                        "scale_ratio": float(scale_ratio),
                        "confidence": 0.75,
                        "reason": f"Abrupt bbox height ratio change ({scale_ratio:.2f}x)",
                    }
                )

    return suspected_switches
