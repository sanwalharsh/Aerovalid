"""Uncertainty quantification (FR-PRD-007, analysis methods §7.4): split-conformal
prediction intervals (model-agnostic, finite-sample coverage guarantee) and the
reliability metrics used to validate them (coverage, mean width, Winkler score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aerovalid.predict.models import ResidualModel


class SplitConformalWrapper:
    """
    Wraps any `ResidualModel` with split-conformal prediction intervals: the base
    model is fit on a training split, then a *held-out calibration split* (never seen
    during training) supplies the nonconformity quantile used to size the interval.

    Parameters
    ----------
    base_model : ResidualModel
    alpha : float, default 0.1
        Target miscoverage rate (`1 - alpha` nominal coverage).
    """

    def __init__(self, base_model: ResidualModel, alpha: float = 0.1) -> None:
        self.base_model = base_model
        self.alpha = alpha
        self._quantile: float = 0.0

    def fit(
        self,
        train_features: pd.DataFrame,
        train_target: NDArray[np.float64],
        calib_features: pd.DataFrame,
        calib_target: NDArray[np.float64],
        groups: NDArray[np.str_] | None = None,
    ) -> None:
        """
        Fit the base model on `(train_features, train_target)`, then compute the
        conformal quantile from `(calib_features, calib_target)`.

        Parameters
        ----------
        train_features, train_target : training split.
        calib_features, calib_target : calibration split, disjoint from training
            (and ideally from different flights than the eventual test split too).
        groups : ndarray, optional
            Flight IDs for the training split, forwarded to `base_model.fit`.
        """
        self.base_model.fit(train_features, train_target, groups)
        calib_prediction = self.base_model.predict(calib_features)
        nonconformity = np.abs(calib_target - calib_prediction)
        n = len(nonconformity)
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self._quantile = float(np.quantile(nonconformity, q_level))

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        return self.base_model.predict(features)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """`alpha` is accepted for protocol compatibility but ignored: the interval
        width is fixed by the calibration quantile from `fit`."""
        del alpha
        point = self.predict(features)
        return point - self._quantile, point + self._quantile


def evaluate_coverage(
    lower: NDArray[np.float64], upper: NDArray[np.float64], actual: NDArray[np.float64]
) -> tuple[float, float]:
    """
    Empirical coverage and mean interval width.

    Returns
    -------
    tuple[float, float]
        `(coverage, mean_width)`; coverage in `[0, 1]`.
    """
    covered = (actual >= lower) & (actual <= upper)
    return float(np.mean(covered)), float(np.mean(upper - lower))


def winkler_score(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    actual: NDArray[np.float64],
    alpha: float,
) -> float:
    """
    Winkler (interval) score: width, plus a penalty proportional to `2/alpha` for each
    sample outside the interval -- rewards narrow intervals that still cover.

    Parameters
    ----------
    lower, upper, actual : ndarray, shape (N,)
    alpha : float
        The interval's nominal miscoverage rate (matches the `alpha` used to build it).

    Returns
    -------
    float
        Mean Winkler score (lower is better).
    """
    width = upper - lower
    below = actual < lower
    above = actual > upper
    score = np.where(below, width + 2.0 / alpha * (lower - actual), width)
    score = np.where(above, width + 2.0 / alpha * (actual - upper), score)
    return float(np.mean(score))
