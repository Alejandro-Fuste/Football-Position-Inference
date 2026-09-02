from typing import Dict, Tuple


def calculate_pair_confidence(
    sideline_id: str,
    endzone_id: str,
    view_s_conf: float,
    view_e_conf: float,
    action_compatibility: float = 1.0,
) -> Tuple[float, str]:
    """
    Computes overall paired-view confidence and status string.
    """
    base_score = 0.5 * (view_s_conf + view_e_conf)
    final_score = base_score * action_compatibility

    if final_score >= 0.85:
        status = "PAIR_CONFIRMED"
    elif final_score >= 0.70:
        status = "PAIR_INFERRED_HIGH_CONFIDENCE"
    else:
        status = "PAIR_REVIEW_REQUIRED"

    return float(final_score), status
