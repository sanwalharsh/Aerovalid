"""Residual diagnostics (FR-VAL-008): autocorrelation, spectrum, normality,
stationarity (ADF). Flags non-white, non-stationary, or non-normal residuals rather
than asserting a pass/fail verdict -- the caller decides what to do with the flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from scipy.stats import normaltest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller


@dataclass(frozen=True)
class DiagnosticsResult:
    """Residual diagnostic summary over one (flight, variant, signal) residual series."""

    n_samples: int
    acf_values: NDArray[np.float64]
    ljung_box_p: float
    is_white_noise: bool
    normality_p: float
    is_normal: bool
    adf_stat: float
    adf_p: float
    is_stationary: bool


def residual_diagnostics(
    residual: NDArray[np.float64], max_lag: int = 20, alpha: float = 0.05
) -> DiagnosticsResult:
    """
    Compute ACF, Ljung-Box (whiteness), D'Agostino-Pearson (normality), and
    ADF (stationarity) diagnostics for a residual series (FR-VAL-008).

    Parameters
    ----------
    residual : ndarray, shape (N,)
        A single (flight, variant, signal) residual time series, no NaNs.
    max_lag : int, default 20
        Maximum lag for the ACF and Ljung-Box test.
    alpha : float, default 0.05
        Significance level for the whiteness/normality/stationarity flags.

    Returns
    -------
    DiagnosticsResult
    """
    res = np.asarray(residual, dtype=np.float64)
    res = res[np.isfinite(res)]
    n = res.shape[0]
    lag = min(max_lag, max(1, n // 3))
    if n < 8:
        return DiagnosticsResult(
            n,
            np.array([]),
            float("nan"),
            False,
            float("nan"),
            False,
            float("nan"),
            float("nan"),
            False,
        )

    acf_values = acf(res, nlags=lag, fft=True)
    ljung_box_p = float(acorr_ljungbox(res, lags=[lag], return_df=True)["lb_pvalue"].iloc[0])
    _stat_norm, normality_p = normaltest(res)
    adf_stat, adf_p = adfuller(res, result_object=False)[:2]

    return DiagnosticsResult(
        n_samples=n,
        acf_values=acf_values,
        ljung_box_p=ljung_box_p,
        is_white_noise=ljung_box_p > alpha,
        normality_p=float(normality_p),
        is_normal=float(normality_p) > alpha,
        adf_stat=float(adf_stat),
        adf_p=float(adf_p),
        is_stationary=float(adf_p) < alpha,
    )


def power_spectrum(
    residual: NDArray[np.float64], sample_rate_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Welch power spectral density of a residual series.

    Parameters
    ----------
    residual : ndarray, shape (N,)
    sample_rate_hz : float

    Returns
    -------
    tuple[ndarray, ndarray]
        `(frequencies_hz, psd)`.
    """
    res = np.asarray(residual, dtype=np.float64)
    res = res[np.isfinite(res)]
    freqs, psd = welch(res, fs=sample_rate_hz, nperseg=min(256, len(res)))
    return freqs, psd
