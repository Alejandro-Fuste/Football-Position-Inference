from typing import Tuple


def compute_footpoint(bbox_xywh: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Computes bottom-center footpoint (foot_x, foot_y) for bounding box [x, y, w, h].
    """
    x, y, w, h = bbox_xywh
    foot_x = x + w / 2.0
    foot_y = y + h
    return (foot_x, foot_y)
