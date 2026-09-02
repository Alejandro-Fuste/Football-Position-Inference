from typing import Dict, List
import numpy as np


def calibrate_probabilities(prob_dict: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """
    Applies temperature scaling to raw probability dictionary.
    """
    if not prob_dict:
        return {}

    keys = list(prob_dict.keys())
    probs = np.array([prob_dict[k] for k in keys], dtype=np.float64)

    # Temperature scaling
    scaled_logits = np.log(np.clip(probs, 1e-7, 1.0)) / max(temperature, 0.1)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    norm_probs = exp_logits / np.sum(exp_logits)

    return {k: float(p) for k, p in zip(keys, norm_probs)}
