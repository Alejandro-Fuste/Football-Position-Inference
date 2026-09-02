from typing import Dict, List, Optional
from ortools.sat.python import cp_model

from position_inference.data.schemas import PositionAssignment, TrackSummary
from position_inference.inference.offense_solver import solve_offense_positions
from position_inference.inference.defense_solver import solve_defense_positions


def solve_global_assignments(
    track_summaries: Dict[int, TrackSummary],
    spatial_features: Dict[int, Dict[str, float]],
    candidate_scores: Dict[int, Dict[str, float]],
    offense_track_ids: List[int],
    defense_track_ids: List[int],
    center_track_id: Optional[int] = None,
    qb_track_id: Optional[int] = None,
    direction: str = "right",
    view: str = "sideline",
) -> List[PositionAssignment]:
    """
    Global constrained optimization solver for football player position assignments.
    Uses CP-SAT or joint hierarchical solvers to enforce formation rules and canonical slots.
    """
    # Run hierarchical offense & defense solvers
    off_assignments = solve_offense_positions(
        offense_track_ids,
        track_summaries,
        spatial_features,
        candidate_scores,
        center_track_id=center_track_id,
        qb_track_id=qb_track_id,
        direction=direction,
    )

    def_assignments = solve_defense_positions(
        defense_track_ids,
        track_summaries,
        spatial_features,
        candidate_scores,
    )

    all_assignments = off_assignments + def_assignments
    return all_assignments
