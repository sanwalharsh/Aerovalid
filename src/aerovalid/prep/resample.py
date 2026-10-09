"""Resampling to a common grid (FR-PRP-004): anti-alias low-pass (when downsampling)
then linear interpolation onto a regular grid; original-sample membership is flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from aerovalid.core.signals import known_signals


def anti_alias_filter(series: pd.Series, source_rate_hz: float, target_rate_hz: float) -> pd.Series:
    """
    Zero-phase low-pass filter a series before downsampling, cutoff at 0.45x the
    target Nyquist frequency.

    Parameters
    ----------
    series : pandas.Series
        No NaNs (interpolate/fill before calling).
    source_rate_hz : float
        The series' current sample rate, Hz.
    target_rate_hz : float
        The rate it will be resampled to, Hz.

    Returns
    -------
    pandas.Series
        Filtered series, same length. Unfiltered (returned unchanged) if
        `target_rate_hz >= source_rate_hz` (no anti-aliasing needed, allowing a small
        relative tolerance for floating-point rate estimates that are nominally
        equal) or there are too few samples for a stable filter.
    """
    if target_rate_hz >= source_rate_hz * (1.0 - 1e-6) or len(series) < 15:
        return series
    nyquist = source_rate_hz / 2.0
    cutoff = 0.45 * target_rate_hz / 2.0
    normalised_cutoff = min(cutoff / nyquist, 0.99)
    b, a = butter(4, normalised_cutoff, btype="low")
    filtered = filtfilt(b, a, series.to_numpy(dtype=np.float64))
    return pd.Series(filtered, index=series.index)


def resample(
    df: pd.DataFrame,
    dt: float,
    source_rate_hz: float | None = None,
    t_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Resample a cleaned table onto a regular grid with spacing `dt` (FR-PRP-004).

    Parameters
    ----------
    df : pandas.DataFrame
        Canonical-schema time series with `t_s`, sorted, no duplicate timestamps
        (`prep/clean.py`'s output).
    dt : float
        Target grid spacing, s (`1 / preprocess.grid_hz`).
    source_rate_hz : float, optional
        The table's current sample rate, Hz; estimated from `df["t_s"]` if omitted.
        Anti-aliasing is applied only when this exceeds the target rate.
    t_grid : ndarray, optional
        Use this exact grid instead of building one from `df["t_s"]` -- e.g. to
        resample a variant's trajectory onto telemetry's grid so `prep/align.py` can
        line them up sample-for-sample (points outside `df`'s own range extrapolate
        via `numpy.interp`'s edge-hold behaviour).

    Returns
    -------
    pandas.DataFrame
        One row per grid point; non-numeric columns (`flight_id`, `source`) are
        forward-filled; an `is_original_sample` column flags grid points within half
        a source sample of an original timestamp.
    """
    t_source = df["t_s"].to_numpy(dtype=np.float64)
    if len(t_source) < 2:
        return df.copy()
    source_dt = (
        float(np.median(np.diff(t_source))) if source_rate_hz is None else 1.0 / source_rate_hz
    )
    source_rate = 1.0 / source_dt if source_dt > 0 else 1.0 / dt
    target_rate = 1.0 / dt

    if t_grid is None:
        t_grid = np.arange(t_source[0], t_source[-1] + 1e-9, dt)
    out = pd.DataFrame({"t_s": t_grid})

    for name in known_signals():
        if name not in df.columns:
            continue
        series = df[name].interpolate(limit_direction="both")
        filtered = anti_alias_filter(series, source_rate, target_rate)
        out[name] = np.interp(t_grid, t_source, filtered.to_numpy(dtype=np.float64))

    for name in ("flight_id", "source"):
        if name in df.columns:
            out[name] = df[name].iloc[0]

    original_idx = np.searchsorted(t_source, t_grid)
    original_idx = np.clip(original_idx, 0, len(t_source) - 1)
    out["is_original_sample"] = np.abs(t_source[original_idx] - t_grid) <= source_dt / 2.0

    return out
