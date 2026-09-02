from typing import Dict, Optional
import numpy as np

from position_inference.data.schemas import TrackSummary, VideoMetadata, ViewPrediction


def classify_view(
    track_summaries: Dict[int, TrackSummary],
    video_metadata: Optional[VideoMetadata] = None,
) -> ViewPrediction:
    """
    Classifies clip camera view as 'sideline' or 'endzone'.
    Prefers explicit DatasetSummary metadata when available, falling back to geometric features.
    """
    evidence: Dict[str, float] = {}

    # Signal 1: DatasetSummary metadata
    if video_metadata and video_metadata.view_raw:
        raw = video_metadata.view_raw.strip().lower()
        if "sideline" in raw:
            evidence["metadata_view"] = 1.0
            return ViewPrediction(view="sideline", confidence=0.98, evidence=evidence)
        elif "endzone" in raw or "end zone" in raw:
            evidence["metadata_view"] = 1.0
            return ViewPrediction(view="endzone", confidence=0.98, evidence=evidence)

    # Signal 2: Geometric pre-snap footpoint distribution
    fps = [t.presnap_median_footpoint for t in track_summaries.values() if t.label == "player" and t.presnap_median_footpoint]
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
        # Endzone has narrower lateral spread (span_ratio < 1.5) and fewer visible players
        if span_ratio >= 1.8 or num_players >= 20:
            conf = min(0.95, 0.70 + 0.15 * min(span_ratio, 3.0))
            return ViewPrediction(view="sideline", confidence=float(conf), evidence=evidence)
        elif span_ratio <= 1.4:
            conf = min(0.95, 0.70 + 0.15 * (1.8 - span_ratio))
            return ViewPrediction(view="endzone", confidence=float(conf), evidence=evidence)

    # Fallback to unknown or default sideline
    evidence["default_fallback"] = 0.5
    return ViewPrediction(view="sideline", confidence=0.55, evidence=evidence)
