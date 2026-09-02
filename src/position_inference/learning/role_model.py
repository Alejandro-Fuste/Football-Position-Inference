from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from position_inference.learning.feature_matrix import FEATURE_NAMES


class ViewSpecificRoleModel:
    """
    Learned tabular role model predicting position probabilities P(position | track_features).
    Supports LogisticRegression and RandomForest baselines.
    """

    def __init__(self, model_type: str = "rf", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.classes_: List[str] = []

        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: List[str]):
        if len(X) == 0 or len(y) == 0:
            return
        self.classes_ = sorted(list(set(y)))
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> List[Dict[str, float]]:
        if not self.is_fitted:
            return [{} for _ in range(len(X))]

        probs_matrix = self.model.predict_proba(X)
        results = []

        for row in probs_matrix:
            prob_dict = {cls_name: float(p) for cls_name, p in zip(self.model.classes_, row)}
            results.append(prob_dict)

        return results
