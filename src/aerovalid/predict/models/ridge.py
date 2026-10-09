"""Ridge/ElasticNet wrapper (FR-PRD-005): the classical baseline model. Uncertainty is
a Gaussian prediction interval from the training residual std (not calibrated;
`predict/uncertainty.py`'s split-conformal wrapper is the calibrated option).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.linear_model import ElasticNet, Ridge

from aerovalid.predict.models import numeric_feature_columns


class RidgeModel:
    """`sklearn.linear_model.Ridge` (or `ElasticNet` if `l1_ratio > 0`) wrapper."""

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.0) -> None:
        self.model = (
            Ridge(alpha=alpha) if l1_ratio == 0.0 else ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        )
        self._feature_columns: list[str] = []
        self._residual_std: float = 1.0

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
        residuals = target - self.model.predict(x)
        self._residual_std = float(np.std(residuals)) or 1e-9

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        x = features[self._feature_columns].to_numpy(dtype=np.float64)
        return np.asarray(self.model.predict(x), dtype=np.float64)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        point = self.predict(features)
        half_width = norm.ppf(1.0 - alpha / 2.0) * self._residual_std
        return point - half_width, point + half_width
