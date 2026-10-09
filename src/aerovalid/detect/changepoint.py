"""Change-point onset estimation (FR-DET-004, analysis methods §8.2): `ruptures` PELT
on a residual (or z-score) stream to flag where its mean/variance shifted, used to
estimate a fault's onset time independently of any threshold-based detector.
"""

from __future__ import annotations

import numpy as np
import ruptures as rpt
from numpy.typing import NDArray


def pelt_onsets(
    t_s: NDArray[np.float64],
    signal: NDArray[np.float64],
    penalty: float = 10.0,
    min_size: int = 5,
) -> list[float]:
    """
    PELT change-point times on `signal` (FR-DET-004).

    Parameters
    ----------
    t_s : ndarray, shape (N,)
        Sample times, s (assumed sorted, regular grid).
    signal : ndarray, shape (N,)
        The stream to segment (e.g. a `z_<signal>` or combined detector statistic).
    penalty : float, default 10.0
        PELT's BIC-style segmentation penalty; higher = fewer, more confident
        breakpoints.
    min_size : int, default 5
        Minimum segment length, samples.

    Returns
    -------
    list[float]
        Change-point times, s (excludes the trailing "end of signal" breakpoint
        `ruptures` always returns).
    """
    values = np.asarray(signal, dtype=np.float64).reshape(-1, 1)
    algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1).fit(values)
    breakpoints = algo.predict(pen=penalty)
    return [float(t_s[idx]) for idx in breakpoints[:-1] if idx < len(t_s)]


def nearest_onset_error(estimated_onsets: list[float], true_onset_s: float) -> float | None:
    """Absolute time error of the estimated onset closest to `true_onset_s`, or `None`
    if no change-points were detected (counted as a miss, not zero error)."""
    if not estimated_onsets:
        return None
    return float(min(abs(onset - true_onset_s) for onset in estimated_onsets))
