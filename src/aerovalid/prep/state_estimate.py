"""Sensor-fusion-free state estimate for windowed re-initialisation (FR-PRP-011): a
Savitzky-Golay smoother over telemetry's state columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aerovalid.prep.features import savgol_derivative

_STATE_COLUMNS = (
    "x_m",
    "y_m",
    "alt_m",
    "vel_air_mps",
    "gamma_rad",
    "chi_rad",
    "mass_kg",
    "alpha_rad",
)


def estimate_state(
    telemetry: pd.DataFrame, dt: float, sg_window: int = 11, sg_order: int = 3
) -> pd.DataFrame:
    """
    Smooth telemetry's state columns for use as a windowed re-init state estimate.

    Parameters
    ----------
    telemetry : pandas.DataFrame
        Resampled (regular-grid) telemetry; must have `t_s` and at least some of
        `_STATE_COLUMNS`.
    dt : float
        Grid spacing, s.
    sg_window, sg_order : Savitzky-Golay parameters (`preprocess.sg` config).

    Returns
    -------
    pandas.DataFrame
        `t_s` plus a smoothed (deriv=0) copy of each present state column; angle
        columns (`gamma_rad`, `chi_rad`, `alpha_rad`) are unwrapped before smoothing
        and wrapped back to `(-pi, pi]` afterward.
    """
    out = pd.DataFrame({"t_s": telemetry["t_s"]})
    angle_columns = {"gamma_rad", "chi_rad", "alpha_rad"}

    for name in _STATE_COLUMNS:
        if name not in telemetry.columns:
            continue
        series = telemetry[name].interpolate(limit_direction="both")
        if name in angle_columns:
            unwrapped = pd.Series(np.unwrap(series.to_numpy(dtype=np.float64)))
            smoothed = savgol_derivative(unwrapped, dt, sg_window, sg_order, deriv=0)
            smoothed = (smoothed + np.pi) % (2.0 * np.pi) - np.pi
        else:
            smoothed = savgol_derivative(series, dt, sg_window, sg_order, deriv=0)
        out[name] = smoothed

    return out
