"""Predictive early warning (FR-DET-005, analysis methods §8.2): reuse a Task-A/C
residual predictor's split-conformal interval to estimate `P(|res(t+h)| > tol)` and
alarm before the tolerance is actually exceeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from aerovalid.predict.uncertainty import SplitConformalWrapper


class EarlyWarningDetector:
    """Wraps a fitted `SplitConformalWrapper` to turn its prediction interval into an
    exceedance probability: the interval half-width implies a Gaussian-tail `sigma`
    centred on the point prediction, from which `P(|res| > tol)` follows directly.
    """

    def __init__(self, wrapper: SplitConformalWrapper, tol: float, p_star: float = 0.8) -> None:
        self.wrapper = wrapper
        self.tol = tol
        self.p_star = p_star
        self._z_alpha = norm.ppf(1.0 - wrapper.alpha / 2.0)

    def exceedance_probability(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """`P(|res(t+h)| > tol)` per row, from a Normal(point, sigma) approximation
        where `sigma` is backed out of the conformal interval's half-width."""
        point = self.wrapper.predict(features)
        lower, upper = self.wrapper.predict_interval(features)
        half_width = (upper - lower) / 2.0
        sigma = np.where(half_width > 0, half_width / self._z_alpha, 1e-9)
        upper_tail = 1.0 - norm.cdf((self.tol - point) / sigma)
        lower_tail = norm.cdf((-self.tol - point) / sigma)
        return np.asarray(upper_tail + lower_tail, dtype=np.float64)

    def alarm(self, features: pd.DataFrame) -> NDArray[np.bool_]:
        return self.exceedance_probability(features) >= self.p_star
