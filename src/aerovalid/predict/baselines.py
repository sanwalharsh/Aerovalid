"""Baseline predictors (FR-PRD-002/004): persistence, constant-velocity, sim-only
(zero-residual), and mean-by-condition-bin. Every baseline implements the same
`ResidualModel` protocol as the real models, so comparisons are apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class PersistenceBaseline:
    """Task A: `predicted_residual(t+h) = residual(t)` -- `build_windows`'s
    `"current_residual"` feature column."""

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
        groups: NDArray[np.str_] | None = None,
    ) -> None:
        del features, target, groups  # nothing to learn

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        return features["current_residual"].to_numpy(dtype=np.float64)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        point = self.predict(features)
        return point, point  # a degenerate (zero-width) interval; has no uncertainty model


class SimOnlyBaseline:
    """Task B's reference point: trust the simulator as-is, i.e. `predicted_residual = 0`."""

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
        groups: NDArray[np.str_] | None = None,
    ) -> None:
        del features, target, groups

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        return np.zeros(len(features), dtype=np.float64)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        zeros = self.predict(features)
        return zeros, zeros


class ConstantVelocityBaseline:
    """Task C: `y(t+h) = y(t) + h * slope(t)`, using `build_windows`'s
    `"<signal>_last"` / `"<signal>_slope"` feature columns for `target_signal`."""

    def __init__(self, target_signal: str, horizon_s: float) -> None:
        self.target_signal = target_signal
        self.horizon_s = horizon_s

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
        groups: NDArray[np.str_] | None = None,
    ) -> None:
        del features, target, groups

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        last = features[f"{self.target_signal}_last"].to_numpy(dtype=np.float64)
        slope = features[f"{self.target_signal}_slope"].to_numpy(dtype=np.float64)
        return last + slope * self.horizon_s

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        point = self.predict(features)
        return point, point


class MeanByBinBaseline:
    """Predicts the training-fold mean target within each value of a condition-bin
    column (falls back to the global training mean for unseen bin values)."""

    def __init__(self, bin_column: str) -> None:
        self.bin_column = bin_column
        self._bin_means: dict[str, float] = {}
        self._global_mean: float = 0.0

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
        groups: NDArray[np.str_] | None = None,
    ) -> None:
        del groups
        self._global_mean = float(np.mean(target))
        if self.bin_column not in features.columns:
            self._bin_means = {}
            return
        frame = pd.DataFrame({"bin": features[self.bin_column].to_numpy(), "target": target})
        means = frame.groupby("bin")["target"].mean()
        self._bin_means = {str(k): float(v) for k, v in means.to_dict().items()}

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        if self.bin_column not in features.columns:
            return np.full(len(features), self._global_mean)
        mapped = features[self.bin_column].map(self._bin_means).fillna(self._global_mean)
        return np.asarray(mapped.to_numpy(), dtype=np.float64)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        point = self.predict(features)
        return point, point
