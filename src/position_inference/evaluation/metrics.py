from typing import Dict, List, Optional, Set, Tuple

from position_inference.data.schemas import GroundTruthRole, PositionAssignment, ViewInferenceResult


def evaluate_predictions(
    ground_truth: List[GroundTruthRole],
    assignments: List[PositionAssignment],
) -> Dict[str, float]:
    """
    Computes position inference evaluation metrics against labeled ground truth.
    Supports allowed_predictions (e.g. SAF -> [FS, SS]) and set-based evaluation.
    """
    if not ground_truth:
        return {}

    # Build GT map by position / track_id
    gt_by_track: Dict[int, str] = {}
    gt_allowed: Dict[int, List[str]] = {}
    gt_by_pos: Dict[str, Set[int]] = {}

    for gt in ground_truth:
        if gt.track_id is not None:
            gt_by_track[gt.track_id] = gt.position
            gt_allowed[gt.track_id] = gt.allowed_predictions or [gt.position]
            gt_by_pos.setdefault(gt.position, set()).add(gt.track_id)

    total_vis = 0
    correct_vis = 0

    total_offense = 0
    correct_offense = 0

    total_defense = 0
    correct_defense = 0

    c_correct = 0
    c_total = 0

    qb_correct = 0
    qb_total = 0

    ol_correct = 0
    ol_total = 0

    high_conf_total = 0
    high_conf_correct = 0

    for a in assignments:
        if a.track_id is not None and a.visibility == "visible":
            total_vis += 1
            expected_pos = gt_by_track.get(a.track_id)
            allowed = gt_allowed.get(a.track_id, [expected_pos] if expected_pos else [])

            is_correct = (a.position in allowed)
            if is_correct:
                correct_vis += 1

            if a.side == "offense":
                total_offense += 1
                if is_correct:
                    correct_offense += 1
            else:
                total_defense += 1
                if is_correct:
                    correct_defense += 1

            if a.position == "C":
                c_total += 1
                if is_correct:
                    c_correct += 1

            if a.position == "QB":
                qb_total += 1
                if is_correct:
                    qb_correct += 1

            if a.position in ("LT", "LG", "C", "RG", "RT"):
                ol_total += 1
                if is_correct:
                    ol_correct += 1

            if a.confidence >= 0.90:
                high_conf_total += 1
                if is_correct:
                    high_conf_correct += 1

    metrics = {
        "visible_accuracy": correct_vis / max(total_vis, 1),
        "offense_accuracy": correct_offense / max(total_offense, 1),
        "defense_accuracy": correct_defense / max(total_defense, 1),
        "center_accuracy": c_correct / max(c_total, 1),
        "qb_accuracy": qb_correct / max(qb_total, 1),
        "ol_accuracy": ol_correct / max(ol_total, 1),
        "high_confidence_precision": high_conf_correct / max(high_conf_total, 1),
        "total_visible": float(total_vis),
        "total_ground_truth": float(len(gt_by_track)),
    }

    return metrics
