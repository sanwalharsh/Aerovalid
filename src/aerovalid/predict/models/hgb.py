"""HistGradientBoosting wrapper (FR-PRD-005): point prediction via squared-error loss,
uncertainty via two auxiliary quantile-loss models (FR-PRD-007's quantile-GBM option).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor

from aerovalid.predict.models import numeric_feature_columns


class HGBModel:
    """`sklearn.ensemble.HistGradientBoostingRegressor` wrapper with quantile intervals."""

    def __init__(self, max_iter: int = 100, seed: int = 0) -> None:
        self.max_iter = max_iter
        self.seed = seed
        self.model = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)
        self._lower_model: HistGradientBoostingRegressor | None = None
        self._upper_model: HistGradientBoostingRegressor | None = None
        self._fitted_alpha: float | None = None
        self._feature_columns: list[str] = []
        self._train_x: NDArray[np.float64] = np.empty((0, 0))
        self._train_y: NDArray[np.float64] = np.empty(0)

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
        self._train_x, self._train_y = x, target  # kept only to lazily fit quantile models

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        x = features[self._feature_columns].to_numpy(dtype=np.float64)
        return np.asarray(self.model.predict(x), dtype=np.float64)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._fitted_alpha != alpha:
            self._lower_model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=alpha / 2.0,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            self._upper_model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=1.0 - alpha / 2.0,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            self._lower_model.fit(self._train_x, self._train_y)
            self._upper_model.fit(self._train_x, self._train_y)
            self._fitted_alpha = alpha

        x = features[self._feature_columns].to_numpy(dtype=np.float64)
        assert self._lower_model is not None and self._upper_model is not None
        lower = np.asarray(self._lower_model.predict(x), dtype=np.float64)
        upper = np.asarray(self._upper_model.predict(x), dtype=np.float64)
        return lower, upper
