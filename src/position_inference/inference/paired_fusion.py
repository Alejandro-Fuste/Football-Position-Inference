from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from position_inference.config import get_confidence_config
from position_inference.data.schemas import PositionAssignment, VideoMetadata, ViewInferenceResult
from position_inference.semantics.personnel import extract_personnel_hypothesis


def fuse_paired_views(
    sideline_result: ViewInferenceResult,
    endzone_result: ViewInferenceResult,
) -> Tuple[ViewInferenceResult, ViewInferenceResult, List[str]]:
    """
    Legacy post-hoc paired fusion interface.
    Compares existing Pass 1 or Pass 2 results, reconciles out-of-view slots, and flags discrepancies.
    """
    cfg = get_confidence_config().get("confidence", {})
    resolution_margin = cfg.get("pair_resolution_margin", 0.12)

    warnings: List[str] = []

    s_hyp = extract_personnel_hypothesis(sideline_result.assignments)
    e_hyp = extract_personnel_hypothesis(endzone_result.assignments)

    sideline_result.personnel_hypothesis = s_hyp
    endzone_result.personnel_hypothesis = e_hyp

    shared_personnel = dict(s_hyp) if sideline_result.view_confidence >= endzone_result.view_confidence else dict(e_hyp)

    s_map = {a.slot_id: a for a in sideline_result.assignments if a.slot_state != "INACTIVE_SLOT"}
    e_map = {a.slot_id: a for a in endzone_result.assignments if a.slot_state != "INACTIVE_SLOT"}

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


def fuse_paired_views_two_pass(
    sideline_mot: Union[str, Path],
    endzone_mot: Union[str, Path],
    action_source: Optional[Union[str, Path]] = None,
    sideline_id: str = "JetSweep_1",
    endzone_id: str = "JetSweep_2",
    dataset_summary: Optional[Union[str, Path, Dict[str, VideoMetadata]]] = None,
    pair_id: str = "pair_001",
) -> Tuple[ViewInferenceResult, ViewInferenceResult, Dict[str, Any]]:
    """
    True Two-Pass Paired-View Evidence Fusion:
    PASS 1: Independent preliminary inference on Sideline and Endzone
    FUSION: Combine personnel hypotheses using domain-specific weights
            (Sideline stronger for full personnel & skill width; Endzone for interior box)
    PASS 2: Re-run CP-SAT solve for both views using shared paired priors
    Returns: Final Pass 2 Sideline result, Final Pass 2 Endzone result, and pair_summary metadata.
    """
    from position_inference.pipeline import infer_video_positions

    cfg = get_confidence_config().get("confidence", {})
    resolution_margin = cfg.get("pair_resolution_margin", 0.12)
    pair_warnings: List[str] = []

    # --- PASS 1: Independent Preliminary Inference ---
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

    # --- FUSION: Reconcile Shared Personnel Prior ---
    shared_prior: Dict[str, int] = {}

    # Sideline is authoritative for skill positions and defensive backfield (wider view)
    for pos in ("WR", "TE", "RB", "FB", "CB", "FS", "SS"):
        s_count = s_prelim_hyp.get(pos, 0)
        e_count = e_prelim_hyp.get(pos, 0)
        if s_pass1.view == "sideline":
            shared_prior[pos] = s_count
        elif e_pass1.view == "sideline":
            shared_prior[pos] = e_count
        else:
            # Fallback to higher confidence view
            shared_prior[pos] = s_count if s_pass1.confidence >= e_pass1.confidence else e_count

    # Endzone has direct view down trench for OL spacing & interior front (DE/DT)
    for pos in ("DE", "DT", "LB"):
        s_count = s_prelim_hyp.get(pos, 0)
        e_count = e_prelim_hyp.get(pos, 0)
        if e_pass1.view == "endzone" and e_count > 0:
            shared_prior[pos] = e_count if e_pass1.confidence >= s_pass1.confidence else s_count
        else:
            shared_prior[pos] = s_count

    # Check for personnel conflict margin
    disagreement_count = sum(
        1 for pos in set(s_prelim_hyp.keys()) | set(e_prelim_hyp.keys())
        if s_prelim_hyp.get(pos, 0) != e_prelim_hyp.get(pos, 0)
    )
    conf_diff = abs(s_pass1.confidence - e_pass1.confidence)

    if disagreement_count >= 3 and conf_diff < resolution_margin:
        pair_warnings.append(
            f"Ambiguous personnel disagreement across paired views (diff: {conf_diff:.3f} < threshold {resolution_margin}). Manual review required."
        )

    # --- PASS 2: Final Inference Guided by Shared Priors ---
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

    # Attach diagnostic provenance & preliminary hypotheses to results
    s_pass2.preliminary_personnel_hypothesis = s_prelim_hyp
    s_pass2.paired_personnel_prior = shared_prior
    s_pass2.pair_resolution_margin = float(conf_diff)

    e_pass2.preliminary_personnel_hypothesis = e_prelim_hyp
    e_pass2.paired_personnel_prior = shared_prior
    e_pass2.pair_resolution_margin = float(conf_diff)

    # Reconcile slot states for endzone out-of-view slots
    s_active_map = {a.slot_id: a for a in s_pass2.assignments if a.slot_state != "INACTIVE_SLOT"}
    for a in e_pass2.assignments:
        if a.slot_state == "ACTIVE_NOT_VISIBLE":
            s_match = s_active_map.get(a.slot_id)
            if s_match and s_match.visibility == "visible":
                a.evidence["sideline_confirmed_slot"] = float(s_match.confidence)
                a.confidence = max(a.confidence, 0.82)

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
