from typing import Dict, List, Optional, Tuple
import numpy as np

from position_inference.config import get_scoring_weights
from position_inference.data.schemas import ActionAnnotation, TrackSummary
from position_inference.geometry.footpoints import compute_footpoint


def identify_snap_frame(action_annotations: List[ActionAnnotation]) -> Optional[int]:
    """Identifies the snap frame from Key Actions annotations."""
    snap_actions = {"Ball Snap", "Snap Receive", "Snap"}
    for act in action_annotations:
        if act.action in snap_actions and act.start_frame is not None:
            return act.start_frame
    return None


def _frame_footpoint(summary: TrackSummary, frame: int) -> Optional[Tuple[float, float]]:
    dets = [d for d in summary.detections if d.frame == frame]
    if not dets:
        return None
    fps = [compute_footpoint(d.bbox_xywh) for d in dets]
    return (
        float(np.median([fp[0] for fp in fps])),
        float(np.median([fp[1] for fp in fps])),
    )


def compute_preliminary_footpoints(
    track_summaries: Dict[int, TrackSummary],
    max_frames: int = 45,
) -> Dict[int, Tuple[float, float]]:
    """Compute early geometry and establish immutable formation anchors.

    Position identity follows the annotation workflow: use the earliest reliable formation
    frame (normally frame 0) for every player visible there. A formation anchor set here is
    never overwritten by later pre-snap observations; later frames are only fallbacks for
    tracks that were absent or occluded at the primary anchor.
    """
    preliminary_fps: Dict[int, Tuple[float, float]] = {}
    player_summaries = [
        s for s in track_summaries.values() if s.label == "player" and s.detections
    ]
    primary_anchor_frame = min(
        (d.frame for s in player_summaries for d in s.detections),
        default=0,
    )

    for track_id, summary in track_summaries.items():
        if summary.label != "player" or not summary.detections:
            continue
        dets = summary.detections
        min_f = min(d.frame for d in dets)

        fps = [compute_footpoint(d.bbox_xywh) for d in dets]
        med_x = float(np.median([fp[0] for fp in fps]))
        med_y = float(np.median([fp[1] for fp in fps]))
        summary.median_footpoint = (med_x, med_y)

        anchor_fp = _frame_footpoint(summary, primary_anchor_frame)
        if anchor_fp is not None:
            summary.formation_anchor_footpoint = anchor_fp
            summary.formation_anchor_frame = primary_anchor_frame

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
    """Compute robust pre-snap statistics and fill missing formation anchors only.

    Existing formation anchors are immutable. For a player absent/occluded at the primary
    formation frame, use the earliest stable pre-snap observation as a fallback anchor.
    This later observation may fill an unresolved position but must not redefine a position
    already established from the primary formation.
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
            end_f = max(0, snap_frame - exclusion)
            start_f = max(0, end_f - lookback)
            stable_dets = [d for d in dets if start_f <= d.frame <= end_f]
        else:
            min_f = min(d.frame for d in dets)
            if min_f <= 50:
                stable_dets = [d for d in dets if d.frame <= min_f + lookback]
            else:
                stable_dets = []

        if len(stable_dets) < min_stable:
            summary.presnap_median_footpoint = None
            summary.presnap_motion = 0.0
            continue

        fps = [compute_footpoint(d.bbox_xywh) for d in stable_dets]
        med_x = float(np.median([fp[0] for fp in fps]))
        med_y = float(np.median([fp[1] for fp in fps]))

        presnap_footpoints[track_id] = (med_x, med_y)
        summary.presnap_median_footpoint = (med_x, med_y)

        if summary.formation_anchor_footpoint is None:
            earliest_frame = min(d.frame for d in stable_dets)
            fallback_dets = [d for d in stable_dets if d.frame <= earliest_frame + 2]
            fallback_fps = [compute_footpoint(d.bbox_xywh) for d in fallback_dets]
            summary.formation_anchor_footpoint = (
                float(np.median([fp[0] for fp in fallback_fps])),
                float(np.median([fp[1] for fp in fallback_fps])),
            )
            summary.formation_anchor_frame = earliest_frame

        if len(fps) > 3:
            xs = [fp[0] for fp in fps]
            ys = [fp[1] for fp in fps]
            summary.presnap_motion = float(np.sqrt(np.var(xs) + np.var(ys)))
        else:
            summary.presnap_motion = 0.0

    return presnap_footpoints
