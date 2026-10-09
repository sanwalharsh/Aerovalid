"""Statistical deviation detectors (FR-DET-002, analysis methods §8.2): each is
streaming-compatible (`update(z) -> bool`, one sample at a time) and `detect_batch`
replays the exact same `update` path over an array, so batch and streaming outputs are
identical by construction (FR-DET-008).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class StreamingDetector(Protocol):
    """`update` consumes one new (already-normalised) sample and returns whether this
    sample triggers an alarm; `reset` clears all running state."""

    def update(self, z: float) -> bool: ...
    def reset(self) -> None: ...


def detect_batch(detector: StreamingDetector, z: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Run `detector.update` over every sample of `z` in order (FR-DET-008: this is the
    "batch" path, built from the same `update` calls a live stream would make)."""
    detector.reset()
    return np.array([detector.update(float(value)) for value in z], dtype=bool)


class ZScoreDetector:
    """Alarm when `|z| > h` holds for `>= m` consecutive samples."""

    def __init__(self, h: float = 4.0, m: int = 5) -> None:
        self.h = h
        self.m = m
        self._run_length = 0

    def reset(self) -> None:
        self._run_length = 0

    def update(self, z: float) -> bool:
        self._run_length = self._run_length + 1 if abs(z) > self.h else 0
        return self._run_length >= self.m


class EWMADetector:
    """`E_t = lam*z_t + (1-lam)*E_{t-1}`; alarm if `|E_t| > L * sigma_E`, where
    `sigma_E` is the EWMA's own steady-state standard deviation under unit-variance
    input noise (analysis methods §8.2)."""

    def __init__(self, lam: float = 0.2, upper_limit: float = 3.5) -> None:
        self.lam = lam
        self.upper_limit = upper_limit
        self._sigma_e = float(np.sqrt(lam / (2.0 - lam)))
        self._ewma = 0.0

    def reset(self) -> None:
        self._ewma = 0.0

    def update(self, z: float) -> bool:
        self._ewma = self.lam * z + (1.0 - self.lam) * self._ewma
        return abs(self._ewma) > self.upper_limit * self._sigma_e


class CUSUMDetector:
    """Two-sided CUSUM: `S+ = max(0, S+ + z - k)`, `S- = max(0, S- - z - k)`; alarm if
    either exceeds `H`."""

    def __init__(self, k: float = 0.5, h: float = 5.0) -> None:
        self.k = k
        self.h = h
        self._s_pos = 0.0
        self._s_neg = 0.0

    def reset(self) -> None:
        self._s_pos = 0.0
        self._s_neg = 0.0

    def update(self, z: float) -> bool:
        self._s_pos = max(0.0, self._s_pos + z - self.k)
        self._s_neg = max(0.0, self._s_neg - z - self.k)
        return self._s_pos > self.h or self._s_neg > self.h


class PageHinkleyDetector:
    """Cumulative deviation of each sample from the running mean, relative to its
    running minimum (or maximum); alarm when that gap exceeds `lam`."""

    def __init__(self, delta: float = 0.5, lam: float = 10.0) -> None:
        self.delta = delta
        self.lam = lam
        self._n = 0
        self._mean = 0.0
        self._cum_pos = 0.0
        self._cum_neg = 0.0
        self._min_cum_pos = 0.0
        self._max_cum_neg = 0.0

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._cum_pos = 0.0
        self._cum_neg = 0.0
        self._min_cum_pos = 0.0
        self._max_cum_neg = 0.0

    def update(self, z: float) -> bool:
        self._n += 1
        self._mean += (z - self._mean) / self._n
        self._cum_pos += z - self._mean - self.delta
        self._cum_neg += z - self._mean + self.delta
        self._min_cum_pos = min(self._min_cum_pos, self._cum_pos)
        self._max_cum_neg = max(self._max_cum_neg, self._cum_neg)
        return (self._cum_pos - self._min_cum_pos > self.lam) or (
            self._max_cum_neg - self._cum_neg > self.lam
        )


class MahalanobisDetector:
    """Multivariate `z' Sigma^-1 z` vs a `chi2_d` threshold on the whitened residual
    vector; `Sigma` is a robust covariance (Minimum Covariance Determinant) fit on
    nominal data. Unlike the univariate detectors, `update` takes the full `z` vector
    for one sample (shape `(d,)`), not a scalar."""

    def __init__(self, inv_cov: NDArray[np.float64], threshold: float) -> None:
        self.inv_cov = inv_cov
        self.threshold = threshold

    def reset(self) -> None:
        pass  # memoryless: each sample's statistic depends only on that sample

    def update(self, z: NDArray[np.float64]) -> bool:
        statistic = float(z @ self.inv_cov @ z)
        return statistic > self.threshold

    def statistic(self, z: NDArray[np.float64]) -> float:
        return float(z @ self.inv_cov @ z)


def fit_mahalanobis(nominal_z: NDArray[np.float64], alpha: float = 0.01) -> MahalanobisDetector:
    """
    Fit a `MahalanobisDetector` from nominal (fault-free) whitened residual vectors.

    Parameters
    ----------
    nominal_z : ndarray, shape (N, d)
        Per-sample z-score vectors (`ResidualBaseline.z_scores` output) from
        fault-free flights.
    alpha : float, default 0.01
        Chi-squared tail probability for the alarm threshold (`chi2(d).ppf(1 - alpha)`).

    Returns
    -------
    MahalanobisDetector
    """
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet

    d = nominal_z.shape[1]
    robust_cov = MinCovDet().fit(nominal_z)
    inv_cov = np.asarray(np.linalg.pinv(robust_cov.covariance_), dtype=np.float64)
    threshold = float(chi2(d).ppf(1.0 - alpha))
    return MahalanobisDetector(inv_cov=inv_cov, threshold=threshold)
