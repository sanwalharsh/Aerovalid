"""Regime-wise identification (FR-IDN-004, analysis methods §10): repeat an
output-error fit per condition bin (e.g. Mach band) to reveal regime-dependent
parameter drift, and a Cochran's Q heterogeneity test flagging missing model structure.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2

from aerovalid.identify.fit import FitResult, fit_parameters
from aerovalid.sim.protocols import StructuralSwitches, VehicleParams


def fit_by_regime(
    param_names: list[str],
    bounds: dict[str, tuple[float, float]],
    nominal: VehicleParams,
    switches: StructuralSwitches,
    dt_s: float,
    telemetry_by_regime: dict[str, pd.DataFrame],
    signals: list[str],
    weights: dict[str, float],
    horizon_s: float = 30.0,
    stride_s: float = 15.0,
    min_samples: int = 50,
) -> dict[str, FitResult]:
    """
    One `fit_parameters` call per regime bin (FR-IDN-004).

    Parameters
    ----------
    telemetry_by_regime : dict[str, pandas.DataFrame]
        Each regime bin's (e.g. Mach-band's) telemetry rows, already filtered by the
        caller.
    min_samples : int, default 50
        Bins with fewer rows than this are skipped (too little data for a stable fit).

    Returns
    -------
    dict[str, FitResult]
        Keyed by regime bin label; bins skipped for insufficient data are omitted.
    """
    return {
        regime: fit_parameters(
            param_names,
            bounds,
            nominal,
            switches,
            dt_s,
            telemetry,
            signals,
            weights,
            horizon_s,
            stride_s,
        )
        for regime, telemetry in telemetry_by_regime.items()
        if len(telemetry) >= min_samples
    }


def cochrans_q_heterogeneity(
    estimates: NDArray[np.float64], variances: NDArray[np.float64]
) -> tuple[float, float]:
    """
    Cochran's Q test for heterogeneity across regime-wise parameter estimates
    (meta-analysis form: weighted by inverse variance), analysis methods §10.

    Parameters
    ----------
    estimates : ndarray, shape (k,)
        One parameter's fitted value per regime.
    variances : ndarray, shape (k,)
        That estimate's variance per regime (e.g. the diagonal of
        `IdentifiabilityResult.covariance` for this parameter).

    Returns
    -------
    tuple[float, float]
        `(Q, p_value)`; `Q ~ chi2(k - 1)` under the null of homogeneity. A low
        p-value flags regime-dependent drift -- evidence of missing model structure.
    """
    weights = 1.0 / variances
    weighted_mean = float(np.sum(weights * estimates) / np.sum(weights))
    q_statistic = float(np.sum(weights * (estimates - weighted_mean) ** 2))
    k = len(estimates)
    p_value = float(1.0 - chi2(k - 1).cdf(q_statistic)) if k > 1 else float("nan")
    return q_statistic, p_value


def bin_telemetry_by_regime(
    telemetry: pd.DataFrame,
    regime_column: Callable[[pd.DataFrame], pd.Series],
    bin_edges: list[float],
) -> dict[str, pd.DataFrame]:
    """
    Split `telemetry` into regime bins via `pandas.cut` on a derived column (e.g.
    Mach number computed from `vel_air_mps`/altitude).

    Parameters
    ----------
    telemetry : pandas.DataFrame
    regime_column : callable
        `telemetry -> Series` (e.g. a Mach-number column computed from telemetry).
    bin_edges : list[float]
        `pandas.cut` bin edges (`identify.regimes.mach_bins` in config).

    Returns
    -------
    dict[str, pandas.DataFrame]
    """
    labels = pd.cut(regime_column(telemetry), bins=bin_edges)
    return {str(label): telemetry[labels == label] for label in labels.unique() if pd.notna(label)}
