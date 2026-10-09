"""Tolerance compliance (FR-VAL-004, analysis methods §4): fraction of time a residual
stays within its engineering tolerance, and time-to-first-sustained-exceedance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def tolerance_compliance(residual: NDArray[np.float64], tol: float) -> float:
    """
    Fraction of valid samples with `|residual| <= tol`.

    Parameters
    ----------
    residual : ndarray, shape (N,)
    tol : float
        Per-signal engineering tolerance (same units as `residual`).

    Returns
    -------
    float
        `P(|e| <= tol)`, in `[0, 1]`; `nan` if there are no valid samples.
    """
    res = np.asarray(residual, dtype=np.float64)
    valid = res[np.isfinite(res)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(np.abs(valid) <= tol))


def time_to_first_exceedance(
    t_s: NDArray[np.float64], residual: NDArray[np.float64], tol: float, min_run_s: float = 1.0
) -> float | None:
    """
    First time `|residual| > tol` holds for a sustained run of >= `min_run_s`.

    Parameters
    ----------
    t_s : ndarray, shape (N,)
        Sample times, s (assumed sorted).
    residual : ndarray, shape (N,)
    tol : float
    min_run_s : float, default 1.0
        Minimum sustained-exceedance duration to count (ignores single-sample blips).

    Returns
    -------
    float or None
        Time of the first qualifying exceedance, s; `None` if it never occurs.
    """
    t = np.asarray(t_s, dtype=np.float64)
    res = np.asarray(residual, dtype=np.float64)
    exceeds = np.abs(res) > tol
    n = len(t)
    i = 0
    while i < n:
        if not exceeds[i]:
            i += 1
            continue
        j = i
        while j < n and exceeds[j]:
            j += 1
        run_end_t = t[min(j, n - 1)] if j < n else t[-1]
        if run_end_t - t[i] >= min_run_s:
            return float(t[i])
        i = j
    return None
