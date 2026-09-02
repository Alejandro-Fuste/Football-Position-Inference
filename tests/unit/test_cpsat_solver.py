import pytest
from position_inference.data.schemas import TrackSummary
from position_inference.inference.assignment_solver import solve_global_assignments


def test_cpsat_solver_ol_ordering():
    # 5 offensive linemen with distinct lateral_offense coordinates:
    # LT (lat=2.0) > LG (lat=1.0) > C (lat=0.0) > RG (lat=-1.0) > RT (lat=-2.0)
    summaries = {
        1: TrackSummary(1, "player", [1], [], 1, 1, 1, 1.0, 100, 50, (1000, 500), (1000, 500)),
        2: TrackSummary(2, "player", [1], [], 1, 1, 1, 1.0, 100, 50, (1000, 500), (1000, 500)),
        3: TrackSummary(3, "player", [1], [], 1, 1, 1, 1.0, 100, 50, (1000, 500), (1000, 500)),
        4: TrackSummary(4, "player", [1], [], 1, 1, 1, 1.0, 100, 50, (1000, 500), (1000, 500)),
        5: TrackSummary(5, "player", [1], [], 1, 1, 1, 1.0, 100, 50, (1000, 500), (1000, 500)),
        6: TrackSummary(6, "player", [1], [], 1, 1, 1, 1.0, 100, 50, (1000, 500), (1000, 500)),
    }

    spatial = {
        1: {"lateral_offense": 2.0, "depth_los": 0.0, "depth_offense": 0.0, "dist_center": 2.0}, # LT
        2: {"lateral_offense": 1.0, "depth_los": 0.0, "depth_offense": 0.0, "dist_center": 1.0}, # LG
        3: {"lateral_offense": 0.0, "depth_los": 0.0, "depth_offense": 0.0, "dist_center": 0.0}, # C
        4: {"lateral_offense": -1.0, "depth_los": 0.0, "depth_offense": 0.0, "dist_center": 1.0}, # RG
        5: {"lateral_offense": -2.0, "depth_los": 0.0, "depth_offense": 0.0, "dist_center": 2.0}, # RT
        6: {"lateral_offense": 0.0, "depth_los": -1.0, "depth_offense": 1.0, "dist_center": 1.0}, # QB
    }

    candidate_scores = {
        1: {"LT": 0.9, "LG": 0.8, "C": 0.1, "RG": 0.1, "RT": 0.1},
        2: {"LT": 0.8, "LG": 0.9, "C": 0.1, "RG": 0.1, "RT": 0.1},
        3: {"C": 1.0},
        4: {"RG": 0.9, "RT": 0.8, "C": 0.1},
        5: {"RT": 0.9, "RG": 0.8, "C": 0.1},
        6: {"QB": 1.0},
    }

    off_tids = [1, 2, 3, 4, 5, 6]
    def_tids = []

    assignments = solve_global_assignments(
        summaries,
        spatial,
        candidate_scores,
        off_tids,
        def_tids,
        center_track_id=3,
        qb_track_id=6,
    )

    assign_map = {a.slot_id: a.track_id for a in assignments if a.track_id is not None}

    # Verify OL strictly ordered
    assert assign_map.get("offense.LT_1") == 1
    assert assign_map.get("offense.LG_1") == 2
    assert assign_map.get("offense.C_1") == 3
    assert assign_map.get("offense.RG_1") == 4
    assert assign_map.get("offense.RT_1") == 5
    assert assign_map.get("offense.QB_1") == 6
