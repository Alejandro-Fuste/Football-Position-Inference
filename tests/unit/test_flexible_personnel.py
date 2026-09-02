import pytest
from position_inference.data.schemas import TrackSummary
from position_inference.inference.assignment_solver import solve_global_assignments
from position_inference.semantics.personnel import extract_personnel_hypothesis


def make_dummy_summaries(track_ids):
    return {
        t: TrackSummary(t, "player", [1], [], 1, 1, 1, 1.0, 100, 50, (1000.0 + t * 10, 500.0), (1000.0 + t * 10, 500.0))
        for t in track_ids
    }


def make_dummy_spatial(track_ids):
    return {
        t: {"lateral_offense": float(t), "depth_los": 0.0, "depth_offense": 0.0, "dist_center": float(t)}
        for t in track_ids
    }


def test_solver_11_personnel_package():
    # 11 offense tracks: C(1), QB(2), LT(3), LG(4), RG(5), RT(6), RB(7), TE(8), WR(9, 10, 11)
    off_tids = list(range(1, 12))
    def_tids = list(range(12, 23))
    all_tids = off_tids + def_tids

    summaries = make_dummy_summaries(all_tids)
    spatial = make_dummy_spatial(all_tids)

    candidate_scores = {
        1: {"C": 1.0},
        2: {"QB": 1.0},
        3: {"LT": 1.0},
        4: {"LG": 1.0},
        5: {"RG": 1.0},
        6: {"RT": 1.0},
        7: {"RB": 0.95},
        8: {"TE": 0.95},
        9: {"WR": 0.95},
        10: {"WR": 0.95},
        11: {"WR": 0.95},
    }
    for t in def_tids:
        candidate_scores[t] = {"CB": 0.5, "LB": 0.5, "DE": 0.5, "DT": 0.5, "FS": 0.5, "SS": 0.5}

    assignments = solve_global_assignments(
        summaries, spatial, candidate_scores, off_tids, def_tids, center_track_id=1, qb_track_id=2
    )

    hyp = extract_personnel_hypothesis(assignments)
    assert hyp.get("C") == 1
    assert hyp.get("QB") == 1
    assert hyp.get("LT") == 1
    assert hyp.get("LG") == 1
    assert hyp.get("RG") == 1
    assert hyp.get("RT") == 1
    assert hyp.get("RB") == 1
    assert hyp.get("TE") == 1
    assert hyp.get("WR") == 3

    # Total offense must be exactly 11
    off_active = sum(hyp.get(pos, 0) for pos in ("C", "QB", "LT", "LG", "RG", "RT", "RB", "FB", "TE", "WR"))
    assert off_active == 11


def test_solver_12_personnel_package():
    # 12 personnel: 1 RB, 2 TE, 2 WR
    off_tids = list(range(1, 12))
    def_tids = list(range(12, 23))
    all_tids = off_tids + def_tids

    summaries = make_dummy_summaries(all_tids)
    spatial = make_dummy_spatial(all_tids)

    candidate_scores = {
        1: {"C": 1.0}, 2: {"QB": 1.0}, 3: {"LT": 1.0}, 4: {"LG": 1.0}, 5: {"RG": 1.0}, 6: {"RT": 1.0},
        7: {"RB": 0.95},
        8: {"TE": 0.95},
        9: {"TE": 0.95},  # 2nd TE
        10: {"WR": 0.95},
        11: {"WR": 0.95},
    }
    for t in def_tids:
        candidate_scores[t] = {"CB": 0.5, "LB": 0.5, "DE": 0.5, "DT": 0.5, "FS": 0.5, "SS": 0.5}

    assignments = solve_global_assignments(
        summaries, spatial, candidate_scores, off_tids, def_tids, center_track_id=1, qb_track_id=2
    )

    hyp = extract_personnel_hypothesis(assignments)
    assert hyp.get("RB") == 1
    assert hyp.get("TE") == 2
    assert hyp.get("WR") == 2

    off_active = sum(hyp.get(pos, 0) for pos in ("C", "QB", "LT", "LG", "RG", "RT", "RB", "FB", "TE", "WR"))
    assert off_active == 11


def test_solver_10_personnel_package():
    # 10 personnel: 1 RB, 0 TE, 4 WR
    off_tids = list(range(1, 12))
    def_tids = list(range(12, 23))
    all_tids = off_tids + def_tids

    summaries = make_dummy_summaries(all_tids)
    spatial = make_dummy_spatial(all_tids)

    candidate_scores = {
        1: {"C": 1.0}, 2: {"QB": 1.0}, 3: {"LT": 1.0}, 4: {"LG": 1.0}, 5: {"RG": 1.0}, 6: {"RT": 1.0},
        7: {"RB": 0.95},
        8: {"WR": 0.95},
        9: {"WR": 0.95},
        10: {"WR": 0.95},
        11: {"WR": 0.95},
    }
    for t in def_tids:
        candidate_scores[t] = {"CB": 0.5, "LB": 0.5, "DE": 0.5, "DT": 0.5, "FS": 0.5, "SS": 0.5}

    assignments = solve_global_assignments(
        summaries, spatial, candidate_scores, off_tids, def_tids, center_track_id=1, qb_track_id=2
    )

    hyp = extract_personnel_hypothesis(assignments)
    assert hyp.get("RB") == 1
    assert hyp.get("TE", 0) == 0
    assert hyp.get("WR") == 4

    off_active = sum(hyp.get(pos, 0) for pos in ("C", "QB", "LT", "LG", "RG", "RT", "RB", "FB", "TE", "WR"))
    assert off_active == 11


def test_solver_defensive_personnel_packages():
    # Test 3 distinct defensive packages:
    # Package A: 4 DL (2 DE, 2 DT), 3 LB, 4 DB (2 CB, 1 FS, 1 SS) -> sum = 11
    off_tids = list(range(1, 12))
    def_tids = list(range(12, 23))
    all_tids = off_tids + def_tids

    summaries = make_dummy_summaries(all_tids)
    spatial = make_dummy_spatial(all_tids)

    candidate_scores = {
        1: {"C": 1.0}, 2: {"QB": 1.0}, 3: {"LT": 1.0}, 4: {"LG": 1.0}, 5: {"RG": 1.0}, 6: {"RT": 1.0},
        7: {"RB": 0.9}, 8: {"TE": 0.9}, 9: {"WR": 0.9}, 10: {"WR": 0.9}, 11: {"WR": 0.9},
        # Def 4-3: 12-13 DE, 14-15 DT, 16-18 LB, 19-20 CB, 21 FS, 22 SS
        12: {"DE": 0.95}, 13: {"DE": 0.95},
        14: {"DT": 0.95}, 15: {"DT": 0.95},
        16: {"LB": 0.95}, 17: {"LB": 0.95}, 18: {"LB": 0.95},
        19: {"CB": 0.95}, 20: {"CB": 0.95},
        21: {"FS": 0.95},
        22: {"SS": 0.95},
    }

    assignments = solve_global_assignments(
        summaries, spatial, candidate_scores, off_tids, def_tids, center_track_id=1, qb_track_id=2
    )

    hyp = extract_personnel_hypothesis(assignments)
    assert hyp.get("DE") == 2
    assert hyp.get("DT") == 2
    assert hyp.get("LB") == 3
    assert hyp.get("CB") == 2
    assert hyp.get("FS") == 1
    assert hyp.get("SS") == 1

    def_active = sum(hyp.get(pos, 0) for pos in ("DE", "DT", "LB", "CB", "FS", "SS"))
    assert def_active == 11
