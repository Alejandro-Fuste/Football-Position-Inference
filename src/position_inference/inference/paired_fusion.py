from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from position_inference.config import get_confidence_config
from position_inference.data.schemas import PositionAssignment, VideoMetadata, ViewInferenceResult
from position_inference.semantics.personnel import extract_personnel_hypothesis


def fuse_paired_views(
    sideline_result: ViewInferenceResult,
    endzone_result: ViewInferenceResult,
) -> Tuple[ViewInferenceResult, ViewInferenceResult, List[str]]:
    """Legacy post-hoc paired fusion interface retained for compatibility."""
    cfg = get_confidence_config().get("confidence", {})
    resolution_margin = cfg.get("pair_resolution_margin", 0.12)
    warnings: List[str] = []

    s_hyp = extract_personnel_hypothesis(sideline_result.assignments)
    e_hyp = extract_personnel_hypothesis(endzone_result.assignments)
    sideline_result.personnel_hypothesis = s_hyp
    endzone_result.personnel_hypothesis = e_hyp

    s_map = {a.slot_id: a for a in sideline_result.assignments if a.slot_state != "INACTIVE_SLOT"}
    e_map = {a.slot_id: a for a in endzone_result.assignments if a.slot_state != "INACTIVE_SLOT"}

    for slot_id in set(s_map) | set(e_map):
        s_assign = s_map.get(slot_id)
        e_assign = e_map.get(slot_id)
        if not s_assign or not e_assign:
            continue
        if s_assign.visibility == "visible" and e_assign.visibility == "visible" and s_assign.position != e_assign.position:
            diff = abs(s_assign.confidence - e_assign.confidence)
            if diff < resolution_margin:
                warnings.append(
                    f"Paired disagreement on slot {slot_id}: Sideline={s_assign.position} ({s_assign.confidence:.2f}) "
                    f"vs Endzone={e_assign.position} ({e_assign.confidence:.2f})"
                )
                s_assign.flags.append("pair_disagreement")
                e_assign.flags.append("pair_disagreement")
        if s_assign.visibility == "visible" and e_assign.visibility == "out_of_view":
            e_assign.slot_state = "ACTIVE_NOT_VISIBLE"
            e_assign.evidence["sideline_confirmed_slot"] = float(s_assign.confidence)

    if warnings:
        sideline_result.status = "PAIR_REVIEW_REQUIRED"
        endzone_result.status = "PAIR_REVIEW_REQUIRED"
        sideline_result.warnings.extend(warnings)
        endzone_result.warnings.extend(warnings)

    return sideline_result, endzone_result, warnings


def _personnel_count_disagreements(a: Dict[str, int], b: Dict[str, int]) -> List[str]:
    return [pos for pos in sorted(set(a) | set(b)) if a.get(pos, 0) != b.get(pos, 0)]


def _build_shared_personnel_prior(
    sideline_result: ViewInferenceResult,
    endzone_result: ViewInferenceResult,
    sideline_hyp: Dict[str, int],
    endzone_hyp: Dict[str, int],
) -> Tuple[Dict[str, int], str]:
    """Build a formation-count prior without allowing endzone uncertainty to redefine personnel.

    In this dataset the sideline clip normally shows the complete 22-player formation and is
    therefore the primary source for *personnel counts*. The endzone clip contributes role and
    alignment evidence inside that package, but does not override counts merely because its scalar
    confidence is slightly higher.
    """
    if sideline_result.view == "sideline":
        return dict(sideline_hyp), "sideline"
    if endzone_result.view == "sideline":
        return dict(endzone_hyp), "endzone_argument_was_sideline"

    # If metadata is unavailable/misclassified, use the result with more visible active players;
    # only then use confidence as a tie-breaker.
    def visible_count(result: ViewInferenceResult) -> int:
        return sum(1 for a in result.assignments if a.slot_state == "ACTIVE_VISIBLE" and a.track_id is not None)

    s_vis = visible_count(sideline_result)
    e_vis = visible_count(endzone_result)
    if s_vis != e_vis:
        return (dict(sideline_hyp), "more_visible_players") if s_vis > e_vis else (dict(endzone_hyp), "more_visible_players")
    return (
        (dict(sideline_hyp), "confidence_fallback")
        if sideline_result.confidence >= endzone_result.confidence
        else (dict(endzone_hyp), "confidence_fallback")
    )


def fuse_paired_views_two_pass(
    sideline_mot: Union[str, Path],
    endzone_mot: Union[str, Path],
    action_source: Optional[Union[str, Path]] = None,
    sideline_id: str = "JetSweep_1",
    endzone_id: str = "JetSweep_2",
    dataset_summary: Optional[Union[str, Path, Dict[str, VideoMetadata]]] = None,
    pair_id: str = "pair_001",
) -> Tuple[ViewInferenceResult, ViewInferenceResult, Dict[str, Any]]:
    """Run independent Pass 1 inference, fuse formation evidence, then perform Pass 2 solves."""
    from position_inference.pipeline import infer_video_positions

    cfg = get_confidence_config().get("confidence", {})
    resolution_margin = cfg.get("pair_resolution_margin", 0.12)
    pair_warnings: List[str] = []

    # PASS 1: independent inference.
    s_pass1 = infer_video_positions(
        sideline_mot,
        action_source,
        video_id=sideline_id,
        dataset_summary=dataset_summary,
        solver_pass=1,
    )
    e_pass1 = infer_video_positions(
        endzone_mot,
        action_source,
        video_id=endzone_id,
        dataset_summary=dataset_summary,
        solver_pass=1,
    )

    s_prelim_hyp = dict(s_pass1.personnel_hypothesis)
    e_prelim_hyp = dict(e_pass1.personnel_hypothesis)
    shared_prior, prior_source = _build_shared_personnel_prior(
        s_pass1, e_pass1, s_prelim_hyp, e_prelim_hyp
    )

    disagreements = _personnel_count_disagreements(s_prelim_hyp, e_prelim_hyp)
    conf_diff = abs(s_pass1.confidence - e_pass1.confidence)
    if len(disagreements) >= 3 and conf_diff < resolution_margin:
        pair_warnings.append(
            "Ambiguous preliminary personnel disagreement across paired views; "
            f"using {prior_source} personnel counts and requiring review. "
            f"Differing roles: {', '.join(disagreements)}."
        )

    # PASS 2: solve each independent track space using the shared formation counts.
    s_pass2 = infer_video_positions(
        sideline_mot,
        action_source,
        video_id=sideline_id,
        dataset_summary=dataset_summary,
        personnel_priors=shared_prior,
        solver_pass=2,
    )
    e_pass2 = infer_video_positions(
        endzone_mot,
        action_source,
        video_id=endzone_id,
        dataset_summary=dataset_summary,
        personnel_priors=shared_prior,
        solver_pass=2,
    )

    for result, prelim in ((s_pass2, s_prelim_hyp), (e_pass2, e_prelim_hyp)):
        result.preliminary_personnel_hypothesis = prelim
        result.paired_personnel_prior = shared_prior
        result.pair_resolution_margin = float(conf_diff)

    # Add paired support evidence to missing endzone slots without inventing a confidence floor.
    s_active_map = {a.slot_id: a for a in s_pass2.assignments if a.slot_state != "INACTIVE_SLOT"}
    for a in e_pass2.assignments:
        if a.slot_state != "ACTIVE_NOT_VISIBLE":
            continue
        s_match = s_active_map.get(a.slot_id)
        if s_match and s_match.visibility == "visible":
            a.evidence["sideline_confirmed_slot"] = float(s_match.confidence)
            a.evidence["paired_visibility_support"] = 1.0

    if pair_warnings:
        s_pass2.status = "PAIR_REVIEW_REQUIRED"
        e_pass2.status = "PAIR_REVIEW_REQUIRED"
        s_pass2.warnings.extend(pair_warnings)
        e_pass2.warnings.extend(pair_warnings)

    pair_summary = {
        "pair_id": pair_id,
        "sideline_video_id": sideline_id,
        "endzone_video_id": endzone_id,
        "sideline_view": s_pass2.view,
        "endzone_view": e_pass2.view,
        "preliminary_sideline_personnel": s_prelim_hyp,
        "preliminary_endzone_personnel": e_prelim_hyp,
        "shared_personnel_prior": shared_prior,
        "shared_personnel_source": prior_source,
        "personnel_disagreements": disagreements,
        "final_sideline_personnel": s_pass2.personnel_hypothesis,
        "final_endzone_personnel": e_pass2.personnel_hypothesis,
        "pair_resolution_margin": float(conf_diff),
        "pair_status": "PAIR_REVIEW_REQUIRED" if pair_warnings else s_pass2.status,
        "pair_warnings": pair_warnings,
        "confidence_calibrated": s_pass2.confidence_calibrated,
        "pass_1_diagnostics": {
            "sideline_status": s_pass1.status,
            "endzone_status": e_pass1.status,
            "sideline_confidence": float(s_pass1.confidence),
            "endzone_confidence": float(e_pass1.confidence),
        },
        "pass_2_diagnostics": {
            "sideline_status": s_pass2.status,
            "endzone_status": e_pass2.status,
            "sideline_confidence": float(s_pass2.confidence),
            "endzone_confidence": float(e_pass2.confidence),
        },
    }

    s_pass2.pair_diagnostics = pair_summary
    e_pass2.pair_diagnostics = pair_summary
    return s_pass2, e_pass2, pair_summary
