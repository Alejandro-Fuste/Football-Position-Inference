from typing import Dict, Optional
import numpy as np

from position_inference.data.schemas import TrackSummary, VideoMetadata, ViewPrediction


def classify_view(
    track_summaries: Dict[int, TrackSummary],
    video_metadata: Optional[VideoMetadata] = None,
) -> ViewPrediction:
    """
    Classifies clip camera view as 'sideline', 'endzone', or 'unknown'.
    Prefers explicit DatasetSummary metadata when available, falling back to geometric features.
    """
    evidence: Dict[str, float] = {}

    # Signal 1: DatasetSummary metadata
    if video_metadata and video_metadata.view_raw:
        raw = video_metadata.view_raw.strip().lower()
        if raw in ("sideline", "s", "side") or "sideline" in raw:
            evidence["metadata_view"] = 1.0
            return ViewPrediction(view="sideline", confidence=0.98, evidence=evidence)
        elif raw in ("endzone", "e", "end zone", "ez") or "endzone" in raw:
            evidence["metadata_view"] = 1.0
            return ViewPrediction(view="endzone", confidence=0.98, evidence=evidence)

    # Signal 2: Geometric footpoint distribution
    fps = [
        t.presnap_median_footpoint or t.median_footpoint
        for t in track_summaries.values()
        if t.label == "player" and (t.presnap_median_footpoint or t.median_footpoint)
    ]
    if len(fps) >= 5:
        xs = [fp[0] for fp in fps]
        ys = [fp[1] for fp in fps]

        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        span_ratio = x_span / max(y_span, 1.0)
        evidence["footpoint_span_ratio"] = float(span_ratio)

        num_players = len(fps)
        evidence["player_count"] = float(num_players)

        # Sideline has wide lateral spread (high span_ratio > 1.8) and usually 20-22 visible players
        # Endzone has narrower lateral spread (span_ratio < 1.4) and fewer visible players
        if span_ratio >= 1.75 or num_players >= 20:
            conf = min(0.95, 0.70 + 0.15 * min(span_ratio, 3.0))
            return ViewPrediction(view="sideline", confidence=float(conf), evidence=evidence)
        elif span_ratio <= 1.45:
            conf = min(0.95, 0.70 + 0.15 * (1.8 - span_ratio))
            return ViewPrediction(view="endzone", confidence=float(conf), evidence=evidence)

    # Fallback to unknown when evidence is insufficient
    evidence["insufficient_evidence"] = 1.0
    return ViewPrediction(view="unknown", confidence=0.50, evidence=evidence)
