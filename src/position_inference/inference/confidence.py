from typing import List, Tuple

from position_inference.config import get_confidence_config
from position_inference.data.schemas import PositionAssignment, ViewInferenceResult


def evaluate_result_confidence(
    result: ViewInferenceResult,
    hard_warnings: List[str] = None,
) -> ViewInferenceResult:
    """
    Evaluates overall video inference confidence and determines final status.
    Uses assignment score margin, anchor presence, view/direction confidence, and hard review triggers.
    """
    if hard_warnings is None:
        hard_warnings = []

    cfg = get_confidence_config().get("confidence", {})
    auto_thresh = cfg.get("auto_accept", 0.90)
    review_thresh = cfg.get("review_recommended", 0.70)

    # Compute mean confidence of visible assignments
    vis_conf = [a.confidence for a in result.assignments if a.visibility == "visible"]
    mean_conf = float(sum(vis_conf) / max(len(vis_conf), 1))

    result.confidence = mean_conf

    # Check for hard review triggers
    has_center = any(a.position == "C" and a.track_id is not None for a in result.assignments)
    has_qb = any(a.position == "QB" and a.track_id is not None for a in result.assignments)

    if not has_center:
        hard_warnings.append("Missing Center track assignment")
    if not has_qb:
        hard_warnings.append("Missing QB track assignment")
    if result.view == "unknown" or result.view_confidence < 0.60:
        hard_warnings.append("Unresolved camera view confidence")
    if result.offense_direction_confidence < 0.60:
        hard_warnings.append("Unresolved offensive direction confidence")

    result.warnings.extend(hard_warnings)

    if hard_warnings or result.status == "PAIR_REVIEW_REQUIRED":
        if result.status != "PAIR_REVIEW_REQUIRED":
            result.status = "HUMAN_REQUIRED"
    elif mean_conf >= auto_thresh:
        result.status = "AUTO_ACCEPTED"
    elif mean_conf >= review_thresh:
        result.status = "REVIEW_RECOMMENDED"
    else:
        result.status = "HUMAN_REQUIRED"

    return result
