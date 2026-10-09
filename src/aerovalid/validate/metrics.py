"""Validation metrics (FR-VAL-001, analysis methods §4): bias/MAE/RMSE/NRMSE/MAPE,
R²/Pearson r, Theil's U2/TIC/decomposition, tolerance compliance, and circular stats
for angle residuals. All vectorised numpy; every formula has a unit test against a
hand-computed or analytic reference case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_EPS_FRACTION = 0.01  # MAPE's epsilon is 1% of the signal's scale (analysis methods §4)


@dataclass(frozen=True)
class MetricSet:
    """One signal's error metrics over a sample of (obs, sim) pairs."""

    n_samples: int
    me: float
    mae: float
    rmse: float
    nrmse_range: float
    nrmse_std: float
    mape_pct: float
    r2: float
    pearson_r: float
    theil_u2: float
    tic: float
    theil_u_m: float
    theil_u_s: float
    theil_u_c: float


def _finite_pairs(
    obs: NDArray[np.float64], sim: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    valid = np.isfinite(obs) & np.isfinite(sim)
    return obs[valid], sim[valid]


def compute_metrics(obs: NDArray[np.float64], sim: NDArray[np.float64]) -> MetricSet:
    """
    Compute the full `MetricSet` for one signal (analysis methods §4).

    Parameters
    ----------
    obs, sim : ndarray, shape (N,)
        Observed and simulated values (already angle-wrapped by the caller if this
        is an angular residual pair: pass `obs` and `sim` such that `obs - sim`
        makes sense directly, i.e. use `core.angles.angle_diff`-wrapped residuals
        via `compute_metrics_from_residual` for angles instead).

    Returns
    -------
    MetricSet

    Notes
    -----
    Masked/non-finite samples are dropped before any statistic is computed.
    """
    obs_v, sim_v = _finite_pairs(
        np.asarray(obs, dtype=np.float64), np.asarray(sim, dtype=np.float64)
    )
    residual = obs_v - sim_v
    return _metrics_from_residual(obs_v, sim_v, residual)


def compute_metrics_from_residual(
    obs: NDArray[np.float64], sim: NDArray[np.float64], residual: NDArray[np.float64]
) -> MetricSet:
    """
    Like `compute_metrics`, but `residual` is supplied directly (e.g. already
    angle-wrapped via `core.angles.angle_diff(obs, sim)`), instead of `obs - sim`.
    """
    obs_a = np.asarray(obs, dtype=np.float64)
    sim_a = np.asarray(sim, dtype=np.float64)
    res_a = np.asarray(residual, dtype=np.float64)
    valid = np.isfinite(obs_a) & np.isfinite(sim_a) & np.isfinite(res_a)
    return _metrics_from_residual(obs_a[valid], sim_a[valid], res_a[valid])


def _metrics_from_residual(
    obs: NDArray[np.float64], sim: NDArray[np.float64], residual: NDArray[np.float64]
) -> MetricSet:
    n = residual.shape[0]
    if n == 0:
        return MetricSet(0, *([float("nan")] * 13))

    me = float(np.mean(residual))
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))

    obs_range = float(np.max(obs) - np.min(obs)) if n > 1 else 0.0
    obs_std = float(np.std(obs)) if n > 1 else 0.0
    nrmse_range = rmse / obs_range if obs_range > 0 else float("nan")
    nrmse_std = rmse / obs_std if obs_std > 0 else float("nan")

    eps = _EPS_FRACTION * max(obs_range, 1e-12)
    mape_pct = float(np.mean(np.abs(residual) / np.maximum(np.abs(obs), eps)) * 100.0)

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2)) if n > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if n > 1 and np.std(obs) > 0 and np.std(sim) > 0:
        pearson_r = float(np.corrcoef(obs, sim)[0, 1])
    else:
        pearson_r = float("nan")

    # U2's numerator and denominator must cover the same index range: persistence has
    # no prediction for the first sample, so the model's error there is excluded too.
    if n > 1:
        rmse_aligned = float(np.sqrt(np.mean(residual[1:] ** 2)))
        rmse_persistence = float(np.sqrt(np.mean(np.diff(obs) ** 2)))
        theil_u2 = rmse_aligned / rmse_persistence if rmse_persistence > 0 else float("nan")
    else:
        theil_u2 = float("nan")

    tic_denom = np.sqrt(np.mean(sim**2)) + np.sqrt(np.mean(obs**2))
    tic = rmse / tic_denom if tic_denom > 0 else float("nan")

    mean_obs, mean_sim = float(np.mean(obs)), float(np.mean(sim))
    std_obs, std_sim = float(np.std(obs)), float(np.std(sim))
    if mse > 0:
        u_m = (mean_sim - mean_obs) ** 2 / mse
        u_s = (std_sim - std_obs) ** 2 / mse
        r_for_decomp = pearson_r if np.isfinite(pearson_r) else 0.0
        u_c = 2.0 * (1.0 - r_for_decomp) * std_sim * std_obs / mse
    else:
        u_m = u_s = u_c = float("nan")

    return MetricSet(
        n_samples=n,
        me=me,
        mae=mae,
        rmse=rmse,
        nrmse_range=nrmse_range,
        nrmse_std=nrmse_std,
        mape_pct=mape_pct,
        r2=r2,
        pearson_r=pearson_r,
        theil_u2=theil_u2,
        tic=tic,
        theil_u_m=u_m,
        theil_u_s=u_s,
        theil_u_c=u_c,
    )


def error_growth_exponent(
    horizons_s: NDArray[np.float64], rmse_by_horizon: NDArray[np.float64]
) -> float:
    """
    Power-law exponent of RMSE growth with horizon: fits `log(RMSE) = a + b*log(h)`.

    Parameters
    ----------
    horizons_s : ndarray, shape (H,)
        Positive horizon values, s.
    rmse_by_horizon : ndarray, shape (H,)
        RMSE at each horizon (windowed-mode validation).

    Returns
    -------
    float
        The fitted exponent `b` (log-log slope); `nan` if fewer than 2 valid points.
    """
    h = np.asarray(horizons_s, dtype=np.float64)
    r = np.asarray(rmse_by_horizon, dtype=np.float64)
    valid = (h > 0) & (r > 0) & np.isfinite(h) & np.isfinite(r)
    if valid.sum() < 2:
        return float("nan")
    slope, _intercept = np.polyfit(np.log(h[valid]), np.log(r[valid]), 1)
    return float(slope)
