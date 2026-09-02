from position_inference.inference.candidate_scores import _endzone_geometry_scores


def test_endzone_interior_ol_scores_are_stronger_than_wr_for_line_player():
    scores = _endzone_geometry_scores(
        depth_los=0.10,
        depth_off=0.10,
        lat_off=0.55,
        dist_c=0.55,
    )
    assert scores["LT"] > scores["WR"]
    assert scores["LG"] > scores["WR"]


def test_endzone_te_edge_alignment_is_not_treated_like_perimeter_wr():
    scores = _endzone_geometry_scores(
        depth_los=0.15,
        depth_off=0.10,
        lat_off=1.35,
        dist_c=1.35,
    )
    assert scores["TE"] > scores["WR"]


def test_endzone_second_level_player_prefers_lb_over_dt():
    scores = _endzone_geometry_scores(
        depth_los=2.5,
        depth_off=-2.5,
        lat_off=0.8,
        dist_c=1.0,
    )
    assert scores["LB"] > scores["DT"]
