"""Shared `ResidualModel` protocol every baseline and model wrapper implements
(architecture spec §3), so `predict/tasks.py` and `predict/cv.py` can treat baselines
and real models uniformly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@runtime_checkable
class ResidualModel(Protocol):
    """Common interface for every baseline and model wrapper in `predict/`."""

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
        groups: NDArray[np.str_] | None = None,
    ) -> None:
        """Fit on training rows; `groups` (flight IDs) is available for models that
        want it (e.g. `MeanByBinBaseline`'s per-bin means are still computed from
        training data only, no leakage, regardless of `groups`)."""
        ...

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Point prediction, shape `(len(features),)`."""
        ...

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """`(1 - alpha)` prediction interval: `(lower, upper)`, each shape `(len(features),)`."""
        ...


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    """Numeric feature columns only (drops `phase`/`cond_*` string columns); shared by
    every wrapper that doesn't do its own categorical encoding (Ridge, RF, HGB)."""
    return [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
