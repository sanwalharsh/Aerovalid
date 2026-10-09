"""ML detector (FR-DET-003): Isolation Forest anomaly score on windowed residual
features (mean, std, slope per signal); `score(window) -> float`, higher = more
anomalous, so every ML detector this project adds later can share the same interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest


def extract_window_features(
    z_df: pd.DataFrame, window_samples: int, stride_samples: int
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """
    Rolling mean/std/slope per `z_*` column over non-overlapping-by-`stride` windows.

    Parameters
    ----------
    z_df : pandas.DataFrame
        One flight's `z_<signal>` columns (`ResidualBaseline.z_scores` output).
    window_samples : int
    stride_samples : int

    Returns
    -------
    tuple[ndarray, ndarray]
        `(features, window_end_idx)`: `features` has shape `(n_windows, 3 * n_signals)`;
        `window_end_idx` is each window's last row index into `z_df` (for alignment).
    """
    n = len(z_df)
    if n < window_samples:
        return np.empty((0, 3 * len(z_df.columns))), np.empty(0, dtype=np.intp)

    starts = np.arange(0, n - window_samples + 1, stride_samples)
    rows = []
    for start in starts:
        window = z_df.iloc[start : start + window_samples]
        values = window.to_numpy(dtype=np.float64)
        slope = (values[-1] - values[0]) / max(window_samples - 1, 1)
        rows.append(np.concatenate([values.mean(axis=0), values.std(axis=0), slope]))
    return np.asarray(rows, dtype=np.float64), (starts + window_samples - 1).astype(np.intp)


class IsolationForestDetector:
    """`sklearn.ensemble.IsolationForest` over windowed residual features."""

    def __init__(self, n_estimators: int = 100, contamination: float = 0.01, seed: int = 0) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=seed
        )

    def fit(self, features: NDArray[np.float64]) -> None:
        self.model.fit(features)

    def score(self, window: NDArray[np.float64]) -> float:
        """Anomaly score for one window's feature vector; higher = more anomalous."""
        return float(-self.model.score_samples(window.reshape(1, -1))[0])

    def score_batch(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(-self.model.score_samples(features), dtype=np.float64)
