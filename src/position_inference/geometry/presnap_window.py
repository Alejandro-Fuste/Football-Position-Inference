from typing import Dict, List, Optional, Tuple
import numpy as np

from position_inference.config import get_scoring_weights
from position_inference.data.schemas import ActionAnnotation, MotDetection, TrackSummary
from position_inference.geometry.footpoints import compute_footpoint


def identify_snap_frame(action_annotations: List[ActionAnnotation]) -> Optional[int]:
    """
    Finds snap frame from Ball Snap or Snap Receive action annotations.
    """
    snap_actions = {"Ball Snap", "Action_BallSnap", "Snap Receive", "Action_SnapReceive", "Snap"}
    for act in action_annotations:
        if act.action in snap_actions and act.start_frame is not None:
            return act.start_frame
    return None


def extract_presnap_footpoints(
    track_summaries: Dict[int, TrackSummary],
    snap_frame: Optional[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Computes robust pre-snap median footpoints for all tracks.
    """
    cfg = get_scoring_weights().get("presnap", {})
    lookback = cfg.get("lookback_frames", 30)
    exclusion = cfg.get("snap_exclusion_frames", 3)
    min_stable = cfg.get("minimum_stable_frames", 8)

    presnap_footpoints: Dict[int, Tuple[float, float]] = {}

    for track_id, summary in track_summaries.items():
        if summary.label != "player" or not summary.detections:
            continue

        dets = summary.detections

        if snap_frame is not None:
            end_f = max(1, snap_frame - exclusion)
            start_f = max(1, end_f - lookback)
            stable_dets = [d for d in dets if start_f <= d.frame <= end_f]
        else:
            # Fallback to early frames
            min_f = min(d.frame for d in dets)
            stable_dets = [d for d in dets if d.frame <= min_f + lookback]

        if len(stable_dets) < min_stable:
            # Fallback to all available detections
            stable_dets = dets

        fps = [compute_footpoint(d.bbox_xywh) for d in stable_dets]
        med_x = float(np.median([fp[0] for fp in fps]))
        med_y = float(np.median([fp[1] for fp in fps]))

        presnap_footpoints[track_id] = (med_x, med_y)
        summary.presnap_median_footpoint = (med_x, med_y)

        # Pre-snap motion magnitude
        if len(fps) > 3:
            xs = [fp[0] for fp in fps]
            ys = [fp[1] for fp in fps]
            summary.presnap_motion = float(np.sqrt(np.var(xs) + np.var(ys)))
        else:
            summary.presnap_motion = 0.0

    return presnap_footpoints
