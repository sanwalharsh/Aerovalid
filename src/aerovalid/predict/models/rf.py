"""Random Forest wrapper (FR-PRD-005): uncertainty from the spread of individual
trees' predictions rather than a separate quantile model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from aerovalid.predict.models import numeric_feature_columns


class RFModel:
    """`sklearn.ensemble.RandomForestRegressor` wrapper with per-tree-spread intervals."""

    def __init__(self, n_estimators: int = 200, seed: int = 0) -> None:
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
        self._feature_columns: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
        groups: NDArray[np.str_] | None = None,
    ) -> None:
        del groups
        self._feature_columns = numeric_feature_columns(features)
        x = features[self._feature_columns].to_numpy(dtype=np.float64)
        self.model.fit(x, target)

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        x = features[self._feature_columns].to_numpy(dtype=np.float64)
        return np.asarray(self.model.predict(x), dtype=np.float64)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = features[self._feature_columns].to_numpy(dtype=np.float64)
        per_tree = np.stack([tree.predict(x) for tree in self.model.estimators_], axis=0)
        lower = np.percentile(per_tree, 100 * alpha / 2.0, axis=0)
        upper = np.percentile(per_tree, 100 * (1.0 - alpha / 2.0), axis=0)
        return lower, upper
