from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from position_inference.data import (
    ActionAnnotation,
    MotDetection,
    VideoMetadata,
    ViewInferenceResult,
    load_action_annotations,
    load_mot_detections,
)
from position_inference.geometry import (
    compute_spatial_features,
    extract_presnap_footpoints,
    identify_snap_frame,
    infer_offensive_direction,
    partition_teams,
)
from position_inference.inference import (
    complete_missing_slots,
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
from position_inference.semantics import extract_semantic_anchors, get_canonical_slots


def infer_video_positions(
    mot_source: Union[str, Path],
    action_source: Optional[Union[str, Path]] = None,
    video_id: str = "video_001",
    video_metadata: Optional[VideoMetadata] = None,
    learned_model=None,
) -> ViewInferenceResult:
    """
    Main single-video position inference pipeline.
    Executes end-to-end V1 structured inference flow.
    """
    # 1. Load MOT detections
    detections = load_mot_detections(mot_source)

    # 2. Separate player tracks & summarize statistics
    track_summaries = summarize_tracks(detections)

    # 3. Load Key Actions annotations if provided
    actions: List[ActionAnnotation] = []
    if action_source:
        all_actions = load_action_annotations(action_source)
        actions = [a for a in all_actions if a.video_id == video_id or a.video_id.endswith(video_id)]
        if not actions and all_actions:

            actions = all_actions

    # 4. Evaluate track validity & filter non-player false positives
    track_summaries, rejected_tids = evaluate_player_validity(track_summaries, actions)

    # 5. Detect suspected ID switches
    id_switches = detect_id_switches(track_summaries)

    # 6. Classify camera view (sideline vs endzone)
    view_pred = classify_view(track_summaries, video_metadata)

    # 7. Identify snap frame & extract pre-snap median footpoints
    snap_frame = identify_snap_frame(actions)
    extract_presnap_footpoints(track_summaries, snap_frame)

    # 8. Extract action semantic anchors (Ball Snap -> C, Snap Receive -> QB)
    play_type = video_metadata.play_type if video_metadata else None
    center_tid, qb_tid, action_role_scores = extract_semantic_anchors(actions, play_type)

    # 9. Infer view-relative offensive direction
    dir_pred = infer_offensive_direction(
        track_summaries,
        center_track_id=center_tid,
        qb_track_id=qb_tid,
        view=view_pred.view,
    )

    # 10. Compute Center-relative spatial features
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

    # 13. Global assignment solver
    raw_assignments = solve_global_assignments(
        track_summaries,
        spatial_feats,
        cand_scores,
        off_tids,
        def_tids,
        center_track_id=center_tid,
        qb_track_id=qb_tid,
        direction=dir_pred.direction,
        view=view_pred.view,
    )

    # 14. Ensure complete canonical slots (fill missing slots with not_visible)
    expected_off = get_canonical_slots("offense")
    expected_def = get_canonical_slots("defense")

    off_assigned = [a for a in raw_assignments if a.side == "offense"]
    def_assigned = [a for a in raw_assignments if a.side == "defense"]

    off_completed = complete_missing_slots(off_assigned, expected_off, "offense")
    def_completed = complete_missing_slots(def_assigned, expected_def, "defense")

    all_assignments = off_completed + def_completed

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
        status="AUTO_ACCEPTED",
    )

    # 15. Evaluate result confidence & hard review triggers
    result = evaluate_result_confidence(result)

    return result
