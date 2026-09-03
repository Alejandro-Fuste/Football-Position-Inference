from typing import Dict, Optional

from position_inference.data.schemas import OffenseDirectionPrediction, TrackSummary


def _anchor(summary: TrackSummary, view: str):
    # Keep sideline direction inference on the previously validated pre-snap
    # geometry. Use the immutable earliest-formation anchor only for endzone clips.
    if view == "endzone":
        return (
            summary.formation_anchor_footpoint
            or summary.presnap_median_footpoint
            or summary.median_footpoint
        )
    return summary.presnap_median_footpoint or summary.median_footpoint


def infer_offensive_direction(
    track_summaries: Dict[int, TrackSummary],
    center_track_id: Optional[int] = None,
    qb_track_id: Optional[int] = None,
    view: str = "sideline",
) -> OffenseDirectionPrediction:
    """Infer view-relative offensive direction using view-appropriate formation geometry."""
    evidence: Dict[str, float] = {}

    if center_track_id and qb_track_id:
        c_summary = track_summaries.get(center_track_id)
        q_summary = track_summaries.get(qb_track_id)
        c_fp = _anchor(c_summary, view) if c_summary else None
        q_fp = _anchor(q_summary, view) if q_summary else None

        if c_fp and q_fp:
            cx, cy = c_fp
            qx, qy = q_fp
            dx = qx - cx
            dy = qy - cy

            if abs(dx) > abs(dy):
                inferred_dir = "left" if dx > 0 else "right"
                evidence["cq_dx"] = float(dx)
                return OffenseDirectionPrediction(
                    direction=inferred_dir,
                    confidence=0.92,
                    evidence=evidence,
                )
            inferred_dir = "up" if dy > 0 else "down"
            evidence["cq_dy"] = float(dy)
            return OffenseDirectionPrediction(
                direction=inferred_dir,
                confidence=0.92,
                evidence=evidence,
            )

    fps = [
        _anchor(t, view)
        for t in track_summaries.values()
        if t.label == "player" and _anchor(t, view)
    ]
    if fps:
        xs = [fp[0] for fp in fps]
        ys = [fp[1] for fp in fps]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)

        if view == "sideline" or x_span > y_span:
            evidence["fallback_span_ratio"] = float(x_span / max(y_span, 1.0))
            return OffenseDirectionPrediction(
                direction="right",
                confidence=0.60,
                evidence=evidence,
            )

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
