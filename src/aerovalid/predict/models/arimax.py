"""ARIMAX-style wrapper (FR-PRD-005) via `statsmodels.GLSAR`: linear regression on the
exogenous features with an autoregressive error structure. `SARIMAX`/`ARIMA` proper
need a *contiguous* endogenous series to forecast forward from, which doesn't fit this
project's windowed, flight-shuffled evaluation framework (arbitrary held-out rows, not
"the next N steps after training"); `GLSAR` keeps the AR-error idea while still
supporting `predict(exog=...)` on arbitrary new rows. See DECISIONS.md for the ADR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy.stats import norm
from statsmodels.regression.linear_model import GLSAR

from aerovalid.predict.models import numeric_feature_columns


class ARIMAXModel:
    """Regression-with-AR-errors wrapper standing in for ARIMAX (see module docstring)."""

    def __init__(self, ar_order: int = 1) -> None:
        self.ar_order = ar_order
        self._feature_columns: list[str] = []
        self._params: NDArray[np.float64] = np.empty(0)
        self._residual_std: float = 1.0

    def _design_matrix(self, features: pd.DataFrame) -> NDArray[np.float64]:
        x = features[self._feature_columns].to_numpy(dtype=np.float64)
        return np.asarray(sm.add_constant(x, has_constant="add"), dtype=np.float64)

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
        groups: NDArray[np.str_] | None = None,
    ) -> None:
        del groups
        self._feature_columns = numeric_feature_columns(features)
        design = self._design_matrix(features)
        model = GLSAR(target, design, rho=self.ar_order)
        results = model.iterative_fit(maxiter=10)
        self._params = np.asarray(results.params, dtype=np.float64)
        residuals = target - design @ self._params
        self._residual_std = float(np.std(residuals)) or 1e-9

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        design = self._design_matrix(features)
        return np.asarray(design @ self._params, dtype=np.float64)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        point = self.predict(features)
        half_width = norm.ppf(1.0 - alpha / 2.0) * self._residual_std
        return point - half_width, point + half_width
