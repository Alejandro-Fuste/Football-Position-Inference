from position_inference.geometry import compute_footpoint


def test_compute_footpoint():
    bbox = (100.0, 200.0, 50.0, 100.0) # x, y, w, h
    fp = compute_footpoint(bbox)
    assert fp == (125.0, 300.0) # x + w/2, y + h
