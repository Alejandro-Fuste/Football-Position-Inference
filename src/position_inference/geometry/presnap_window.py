from typing import Dict, List, Optional, Tuple
import numpy as np

from position_inference.config import get_scoring_weights
from position_inference.data.schemas import ActionAnnotation, TrackSummary
from position_inference.geometry.footpoints import compute_footpoint


def identify_snap_frame(action_annotations: List[ActionAnnotation]) -> Optional[int]:
    """
    Identifies the snap frame from Key Actions annotations.
    """
    snap_actions = {"Ball Snap", "Snap Receive", "Snap"}
    for act in action_annotations:
        if act.action in snap_actions and act.start_frame is not None:
            return act.start_frame
    return None


def compute_preliminary_footpoints(
    track_summaries: Dict[int, TrackSummary],
    max_frames: int = 45,
) -> Dict[int, Tuple[float, float]]:
    """
    Computes preliminary footpoints across early frames before snap identification.
    Populates summary.median_footpoint and preliminary summary.presnap_median_footpoint
    so view classification has actual geometric features available.
    Tracks appearing late in the video are not given early-frame footpoints.
    """
    preliminary_fps: Dict[int, Tuple[float, float]] = {}
    for track_id, summary in track_summaries.items():
        if summary.label != "player" or not summary.detections:
            continue
        dets = summary.detections
        min_f = min(d.frame for d in dets)

        fps = [compute_footpoint(d.bbox_xywh) for d in dets]
        med_x = float(np.median([fp[0] for fp in fps]))
        med_y = float(np.median([fp[1] for fp in fps]))
        summary.median_footpoint = (med_x, med_y)

        # Only assign preliminary presnap footpoint if track is present in early frames
        if min_f <= max_frames:
            early_dets = [d for d in dets if d.frame <= min_f + max_frames]
            if early_dets:
                e_fps = [compute_footpoint(d.bbox_xywh) for d in early_dets]
                e_x = float(np.median([fp[0] for fp in e_fps]))
                e_y = float(np.median([fp[1] for fp in e_fps]))
                preliminary_fps[track_id] = (e_x, e_y)
                summary.presnap_median_footpoint = (e_x, e_y)
        else:
            summary.presnap_median_footpoint = None

    return preliminary_fps


def extract_presnap_footpoints(
    track_summaries: Dict[int, TrackSummary],
    snap_frame: Optional[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Computes robust pre-snap median footpoints for all tracks relative to snap frame.
    Tracks with fewer than min_stable frames in the pre-snap window are set to None
    and are not given post-snap fallback footpoints.
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
            min_f = min(d.frame for d in dets)
            # If no snap frame, only allow early-appearing tracks
            if min_f <= 50:
                stable_dets = [d for d in dets if d.frame <= min_f + lookback]
            else:
                stable_dets = []

        if len(stable_dets) < min_stable:
            # Do NOT fall back to post-snap frames
            summary.presnap_median_footpoint = None
            summary.presnap_motion = 0.0
            continue

        fps = [compute_footpoint(d.bbox_xywh) for d in stable_dets]
        med_x = float(np.median([fp[0] for fp in fps]))
        med_y = float(np.median([fp[1] for fp in fps]))

        presnap_footpoints[track_id] = (med_x, med_y)
        summary.presnap_median_footpoint = (med_x, med_y)

        if len(fps) > 3:
            xs = [fp[0] for fp in fps]
            ys = [fp[1] for fp in fps]
            summary.presnap_motion = float(np.sqrt(np.var(xs) + np.var(ys)))
        else:
            summary.presnap_motion = 0.0

    return presnap_footpoints
