from typing import Dict, List, Optional, Tuple
import numpy as np

from position_inference.data.schemas import OffenseDirectionPrediction, TrackSummary


def infer_offensive_direction(
    track_summaries: Dict[int, TrackSummary],
    center_track_id: Optional[int] = None,
    qb_track_id: Optional[int] = None,
    view: str = "sideline",
) -> OffenseDirectionPrediction:
    """
    Infers view-relative offensive direction (right, left, up, down) using Center/QB alignment,
    OL formation orientation, and player placement.
    """
    evidence: Dict[str, float] = {}

    # Signal 1: Center to QB relative vector
    if center_track_id and qb_track_id:
        c_summary = track_summaries.get(center_track_id)
        q_summary = track_summaries.get(qb_track_id)

        if c_summary and q_summary and c_summary.presnap_median_footpoint and q_summary.presnap_median_footpoint:
            cx, cy = c_summary.presnap_median_footpoint
            qx, qy = q_summary.presnap_median_footpoint

            dx = qx - cx
            dy = qy - cy

            # QB is behind Center. So offense plays in direction opposite to (QB - Center) vector
            if abs(dx) > abs(dy):
                # Lateral view alignment
                inferred_dir = "left" if dx > 0 else "right"
                evidence["cq_dx"] = float(dx)
                return OffenseDirectionPrediction(
                    direction=inferred_dir,
                    confidence=0.92,
                    evidence=evidence,
                )
            else:
                # Vertical view alignment (Endzone)
                inferred_dir = "up" if dy > 0 else "down"
                evidence["cq_dy"] = float(dy)
                return OffenseDirectionPrediction(
                    direction=inferred_dir,
                    confidence=0.92,
                    evidence=evidence,
                )

    # Signal 2: Footpoint spread distribution fallback
    fps = [t.presnap_median_footpoint for t in track_summaries.values() if t.label == "player" and t.presnap_median_footpoint]
    if fps:
        xs = [fp[0] for fp in fps]
        ys = [fp[1] for fp in fps]

        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)

        if view == "sideline" or x_span > y_span:
            # Default sideline plays left-to-right or right-to-left
            evidence["fallback_span_ratio"] = float(x_span / max(y_span, 1.0))
            return OffenseDirectionPrediction(
                direction="right",
                confidence=0.60,
                evidence=evidence,
            )
        else:
            evidence["fallback_span_ratio"] = float(y_span / max(x_span, 1.0))
            return OffenseDirectionPrediction(
                direction="up",
                confidence=0.60,
                evidence=evidence,
            )

    return OffenseDirectionPrediction(
        direction="right",
        confidence=0.50,
        evidence={"fallback": 1.0},
    )
