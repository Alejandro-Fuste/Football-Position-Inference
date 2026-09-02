from typing import List, Optional

from position_inference.config import get_confidence_config
from position_inference.data.schemas import ViewInferenceResult


def evaluate_result_confidence(
    result: ViewInferenceResult,
    hard_warnings: Optional[List[str]] = None,
) -> ViewInferenceResult:
    """
    Evaluates overall video inference confidence and determines review status.
    Combines:
    - Assignment score margins & ambiguity
    - Semantic anchor status (Center & QB)
    - View and offensive direction certainty
    - Calibration policy (conservative review if calibrated: false)
    """
    if hard_warnings is None:
        hard_warnings = []

    cfg = get_confidence_config().get("confidence", {})
    auto_thresh = cfg.get("auto_accept", 0.90)
    review_thresh = cfg.get("review_recommended", 0.70)
    is_calibrated = cfg.get("calibrated", False)
    result.confidence_calibrated = is_calibrated

    active_assignments = [a for a in result.assignments if a.slot_state != "INACTIVE_SLOT"]
    vis_assignments = [a for a in active_assignments if a.visibility == "visible"]

    if not vis_assignments:
        result.confidence = 0.0
        result.status = "HUMAN_REQUIRED"
        result.warnings.extend(hard_warnings)
        return result

    # 1. Base mean confidence of active assignments
    mean_conf = float(sum(a.confidence for a in active_assignments) / max(len(active_assignments), 1))

    # 2. Critical role check
    has_center = any(a.position == "C" and a.track_id is not None for a in vis_assignments)
    has_qb = any(a.position == "QB" and a.track_id is not None for a in vis_assignments)

    if not has_center:
        hard_warnings.append("missing_center_track")
    if not has_qb:
        hard_warnings.append("missing_qb_track")

    # 3. View & direction certainty
    if result.view == "unknown" or result.view_confidence < 0.65:
        hard_warnings.append("unresolved_camera_view")
    if result.offense_direction_confidence < 0.65:
        hard_warnings.append("unresolved_offensive_direction")

    # 4. Ambiguity penalty (count of visible assignments with small score margins)
    ambiguous_count = sum(1 for a in vis_assignments if a.score_margin < 0.15 and a.position not in ("C", "QB"))
    ambiguity_penalty = 0.04 * ambiguous_count

    view_penalty = 0.15 if (result.view == "unknown" or result.view_confidence < 0.70) else 0.0
    dir_penalty = 0.10 if result.offense_direction_confidence < 0.70 else 0.0

    final_conf = max(0.20, min(1.0, mean_conf - ambiguity_penalty - view_penalty - dir_penalty))
    result.confidence = float(final_conf)
    result.warnings.extend(hard_warnings)

    # 5. Status determination based on calibration and thresholds
    if hard_warnings or result.status == "PAIR_REVIEW_REQUIRED":
        if result.status != "PAIR_REVIEW_REQUIRED":
            result.status = "HUMAN_REQUIRED"
    elif not is_calibrated:
        # Conservative policy: when uncalibrated, do not aggressively auto-accept
        result.status = "REVIEW_RECOMMENDED"
    elif final_conf >= auto_thresh:
        result.status = "AUTO_ACCEPTED"
    elif final_conf >= review_thresh:
        result.status = "REVIEW_RECOMMENDED"
    else:
        result.status = "HUMAN_REQUIRED"

    return result
