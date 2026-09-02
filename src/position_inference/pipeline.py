import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from position_inference.data import (
    ActionAnnotation,
    MotDetection,
    VideoMetadata,
    ViewInferenceResult,
    load_action_annotations,
    load_mot_detections,
    resolve_video_metadata,
)
from position_inference.data.action_loader import filter_actions_for_video
from position_inference.geometry import (
    compute_preliminary_footpoints,
    compute_spatial_features,
    extract_presnap_footpoints,
    identify_snap_frame,
    infer_offensive_direction,
    partition_teams,
)
from position_inference.inference import (
    compute_candidate_role_scores,
    evaluate_result_confidence,
    solve_global_assignments,
)
from position_inference.pairing import classify_view
from position_inference.quality import (
    detect_id_switches,
    evaluate_player_validity,
    summarize_tracks,
)
from position_inference.semantics import extract_semantic_anchors
from position_inference.semantics.personnel import extract_personnel_hypothesis

logger = logging.getLogger(__name__)


def infer_video_positions(
    mot_source: Union[str, Path],
    action_source: Optional[Union[str, Path]] = None,
    video_id: str = "video_001",
    video_metadata: Optional[VideoMetadata] = None,
    dataset_summary: Optional[Union[str, Path, Dict[str, VideoMetadata]]] = None,
    learned_model=None,
    allow_missing_actions: bool = False,
    personnel_priors: Optional[Dict[str, int]] = None,
    solver_pass: int = 1,
) -> ViewInferenceResult:
    """
    Main single-video position inference pipeline.
    Executes end-to-end V1 structured inference flow in the corrected order:
    1. Load MOT & summarize tracks
    2. Compute preliminary geometry
    3. Resolve VideoMetadata from DatasetSummary & classify camera view
    4. Load & match video-specific KeyActions safely
    5. Evaluate track validity & filter noise
    6. Identify snap frame & compute pre-snap stable window
    7. Extract action semantic anchors & side evidence
    8. Infer view-relative offensive direction
    9. Compute normalized spatial features in offensive perspective
    10. Partition teams into offense/defense
    11. Candidate role probability scoring
    12. Global CP-SAT constrained optimization solver (flexible personnel)
    13. Calibrate confidence and evaluate review triggers
    """
    hard_warnings: List[str] = []
    meta_src = None

    # Resolve VideoMetadata from DatasetSummary if not already provided
    if video_metadata is None:
        if dataset_summary is not None:
            video_metadata = resolve_video_metadata(dataset_summary, video_id)
            meta_src = str(dataset_summary)
            if video_metadata is None:
                hard_warnings.append("dataset_summary_video_not_found")
        else:
            default_summary_path = Path("data/dataset_summary/DatasetSummary.csv")
            if default_summary_path.exists():
                video_metadata = resolve_video_metadata(default_summary_path, video_id)
                if video_metadata is not None:
                    meta_src = str(default_summary_path)

    # 1. Load MOT detections & summarize tracks
    detections = load_mot_detections(mot_source)
    track_summaries = summarize_tracks(detections)

    # 2. Compute preliminary geometry (before view classification and snap detection)
    compute_preliminary_footpoints(track_summaries)

    # 3. Classify camera view (prioritizing metadata view when present)
    view_pred = classify_view(track_summaries, video_metadata)

    # 4. Load Key Actions annotations safely (no unsafe fallback)
    actions: List[ActionAnnotation] = []
    if action_source:
        all_actions = load_action_annotations(action_source)
        actions = filter_actions_for_video(
            all_actions,
            video_id=video_id,
            action_source=action_source,
            allow_missing_actions=allow_missing_actions,
        )
        if not actions and allow_missing_actions:
            hard_warnings.append("missing_action_annotations")

    # 5. Evaluate track validity & filter non-player false positives
    track_summaries, rejected_tids = evaluate_player_validity(track_summaries, actions)

    # 6. Detect suspected ID switches
    id_switches = detect_id_switches(track_summaries)

    # 7. Identify snap frame & extract pre-snap stable window
    snap_frame = identify_snap_frame(actions)
    extract_presnap_footpoints(track_summaries, snap_frame)

    # 8. Extract action semantic anchors and side evidence
    play_type = video_metadata.play_type if video_metadata else None
    center_tid, qb_tid, action_role_scores, track_side_scores = extract_semantic_anchors(actions, play_type)

    # 9. Infer view-relative offensive direction
    dir_pred = infer_offensive_direction(
        track_summaries,
        center_track_id=center_tid,
        qb_track_id=qb_tid,
        view=view_pred.view,
    )

    # 10. Compute normalized spatial features in offensive perspective
    spatial_feats = compute_spatial_features(
        track_summaries,
        center_track_id=center_tid,
        qb_track_id=qb_tid,
        direction=dir_pred.direction,
    )

    # 11. Partition player tracks into Offense vs Defense candidate pools
    off_tids, def_tids = partition_teams(
        track_summaries,
        action_annotations=actions,
        track_side_scores=track_side_scores,
        center_track_id=center_tid,
        qb_track_id=qb_tid,
        direction=dir_pred.direction,
    )

    # 12. Candidate role probability scoring
    cand_scores = compute_candidate_role_scores(
        track_summaries,
        spatial_feats,
        action_role_scores,
        view=view_pred.view,
        learned_model=learned_model,
    )

    # 13. Global CP-SAT constrained optimization solver (flexible personnel)
    all_assignments = solve_global_assignments(
        track_summaries,
        spatial_feats,
        cand_scores,
        off_tids,
        def_tids,
        center_track_id=center_tid,
        qb_track_id=qb_tid,
        direction=dir_pred.direction,
        view=view_pred.view,
        personnel_priors=personnel_priors,
        solver_pass=solver_pass,
    )

    # 14. Extract active personnel counts
    personnel_hyp = extract_personnel_hypothesis(all_assignments)

    # Construct result object
    result = ViewInferenceResult(
        video_id=video_id,
        view=view_pred.view,
        view_confidence=view_pred.confidence,
        offense_direction=dir_pred.direction,
        offense_direction_confidence=dir_pred.confidence,
        assignments=all_assignments,
        rejected_track_ids=rejected_tids,
        suspected_id_switches=id_switches,
        personnel_hypothesis=personnel_hyp,
        solver_pass=solver_pass,
        metadata_source=meta_src,
        status="AUTO_ACCEPTED",
    )

    # 15. Evaluate result confidence & hard review triggers
    result = evaluate_result_confidence(result, hard_warnings=hard_warnings)

    return result
