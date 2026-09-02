from typing import Dict, List, Set, Tuple
import numpy as np

from position_inference.data.schemas import ActionAnnotation, TrackSummary


def evaluate_player_validity(
    track_summaries: Dict[int, TrackSummary],
    action_annotations: List[ActionAnnotation] = None,
    max_expected_players: int = 30,
) -> Tuple[Dict[int, TrackSummary], List[int]]:
    """
    Evaluates player track validity. Filters obvious false positives, extreme outliers,
    or non-player tracks without mutating source MOT data.
    Returns (updated_summaries, rejected_track_ids).
    """
    if action_annotations is None:
        action_annotations = []

    # Collect track_ids explicitly referenced in key actions
    referenced_tracks: Set[int] = set()
    for act in action_annotations:
        if act.actor_track_id is not None:
            referenced_tracks.add(act.actor_track_id)

    rejected_tracks: List[int] = []

    # Calculate median bbox size across all player tracks
    player_heights = [t.median_bbox_height for t in track_summaries.values() if t.label == "player" and t.num_boxes > 3]
    global_med_height = float(np.median(player_heights)) if player_heights else 100.0

    for track_id, summary in track_summaries.items():
        # Exclude non-player labels (e.g. ball)
        if summary.label != "player":
            summary.validity_score = 0.0
            summary.validity_flags.append("non_player_label")
            rejected_tracks.append(track_id)
            continue

        flags = []
        score = 1.0

        # Check 1: Extremely short track lifetime (unless referenced by action)
        if summary.num_boxes < 5 and track_id not in referenced_tracks:
            score -= 0.6
            flags.append("very_short_lifetime")

        # Check 2: Bbox size anomaly (too tiny or too huge)
        if summary.median_bbox_height < 0.20 * global_med_height and track_id not in referenced_tracks:
            score -= 0.5
            flags.append("tiny_bbox")
        elif summary.median_bbox_height > 4.0 * global_med_height and track_id not in referenced_tracks:
            score -= 0.5
            flags.append("huge_bbox")

        # Check 3: Appears late after play is completed
        if summary.first_frame > 250 and summary.num_boxes < 10 and track_id not in referenced_tracks:
            score -= 0.4
            flags.append("late_appearing_track")

        summary.validity_score = max(0.0, score)
        summary.validity_flags = flags

        if summary.validity_score < 0.30 and track_id not in referenced_tracks:
            rejected_tracks.append(track_id)

    return track_summaries, rejected_tracks
