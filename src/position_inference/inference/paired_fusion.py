from typing import Dict, List, Optional, Tuple

from position_inference.config import get_confidence_config
from position_inference.data.schemas import PositionAssignment, ViewInferenceResult
from position_inference.semantics.personnel import extract_personnel_hypothesis


def fuse_paired_views(
    sideline_result: ViewInferenceResult,
    endzone_result: ViewInferenceResult,
) -> Tuple[ViewInferenceResult, ViewInferenceResult, List[str]]:
    """
    Fuses role and personnel evidence across paired sideline and endzone view results.
    Sideline provides stronger full-formation evidence; Endzone provides trench/backfield detail.
    Resolves discrepancies using confidence margins or flags for review.
    """
    cfg = get_confidence_config().get("confidence", {})
    resolution_margin = cfg.get("pair_resolution_margin", 0.12)

    warnings: List[str] = []

    # 1. Extract personnel hypotheses from both views
    s_hyp = extract_personnel_hypothesis(sideline_result.assignments)
    e_hyp = extract_personnel_hypothesis(endzone_result.assignments)

    sideline_result.personnel_hypothesis = s_hyp
    endzone_result.personnel_hypothesis = e_hyp

    # 2. Reconcile personnel package
    # Sideline typically sees the entire width of the field, so its skill counts (WR/TE/RB) are prioritized
    shared_personnel = dict(s_hyp) if sideline_result.view_confidence >= endzone_result.view_confidence else dict(e_hyp)

    # 3. Map slot_id to assignment for both views
    s_map = {a.slot_id: a for a in sideline_result.assignments}
    e_map = {a.slot_id: a for a in endzone_result.assignments}

    all_slots = set(s_map.keys()) | set(e_map.keys())

    for slot_id in all_slots:
        s_assign = s_map.get(slot_id)
        e_assign = e_map.get(slot_id)

        if not s_assign or not e_assign:
            continue

        # Check for role mismatch if both are visible
        if s_assign.visibility == "visible" and e_assign.visibility == "visible":
            if s_assign.position != e_assign.position:
                diff = abs(s_assign.confidence - e_assign.confidence)
                if diff < resolution_margin:
                    warnings.append(
                        f"Paired disagreement on slot {slot_id}: Sideline={s_assign.position} ({s_assign.confidence:.2f}) vs Endzone={e_assign.position} ({e_assign.confidence:.2f})"
                    )
                    s_assign.flags.append("pair_disagreement")
                    e_assign.flags.append("pair_disagreement")
                elif s_assign.confidence > e_assign.confidence:
                    e_assign.position = s_assign.position
                    e_assign.evidence["paired_fusion_override"] = s_assign.confidence
                else:
                    s_assign.position = e_assign.position
                    s_assign.evidence["paired_fusion_override"] = e_assign.confidence

        # If sideline establishes complete formation but endzone player is out_of_view
        if s_assign.visibility == "visible" and e_assign.visibility == "out_of_view":
            e_assign.slot_state = "ACTIVE_NOT_VISIBLE"
            e_assign.evidence["sideline_confirmed_slot"] = s_assign.confidence

    if warnings:
        sideline_result.status = "PAIR_REVIEW_REQUIRED"
        endzone_result.status = "PAIR_REVIEW_REQUIRED"
        sideline_result.warnings.extend(warnings)
        endzone_result.warnings.extend(warnings)

    return sideline_result, endzone_result, warnings
