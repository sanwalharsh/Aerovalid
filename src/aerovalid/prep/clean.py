"""Cleaning (FR-PRP-002): sort/dedupe timestamps, Hampel outlier filter, and a gap
policy (interpolate short gaps, mask long ones rather than silently dropping them).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aerovalid.core.signals import known_signals


def hampel_filter(
    series: pd.Series, window: int = 7, n_sigma: float = 4.0
) -> tuple[pd.Series, pd.Series]:
    """
    Rolling-median Hampel filter: replace samples more than `n_sigma` scaled-MAD from
    the local median with that median. Fully vectorised (pandas rolling), NaNs pass
    through untouched.

    Parameters
    ----------
    series : pandas.Series
    window : int, default 7
        Rolling window size, samples.
    n_sigma : float, default 4.0
        Outlier threshold in scaled-MAD units.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        Cleaned series and a boolean outlier mask (True where replaced).
    """
    rolling_median = series.rolling(window, center=True, min_periods=1).median()
    abs_dev = (series - rolling_median).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    threshold = n_sigma * 1.4826 * mad
    outlier_mask = abs_dev > threshold
    cleaned = series.where(~outlier_mask, rolling_median)
    return cleaned, outlier_mask.fillna(False)


def _interpolate_short_gaps(series: pd.Series, max_gap_samples: int) -> pd.Series:
    """Linearly interpolate only NaN *runs* no longer than `max_gap_samples`; longer
    runs are left as NaN entirely (unlike `Series.interpolate(limit=...)`, which
    partially fills the first `limit` samples of an arbitrarily long gap instead of
    leaving the whole thing alone)."""
    is_na = series.isna()
    if not is_na.any():
        return series
    run_id = (is_na != is_na.shift()).cumsum()
    run_length = is_na.groupby(run_id).transform("sum")
    fully_interpolated = series.interpolate(method="linear", limit_direction="both")
    short_gap = is_na & (run_length <= max_gap_samples)
    return series.where(~short_gap, fully_interpolated)


@dataclass
class CleanResult:
    """Output of `clean`: the cleaned table, an exclusion mask, and counters."""

    df: pd.DataFrame
    mask: pd.DataFrame
    n_spikes_removed: dict[str, int] = field(default_factory=dict)
    n_gaps_filled: dict[str, int] = field(default_factory=dict)
    n_gaps_masked: dict[str, int] = field(default_factory=dict)


def clean(
    df: pd.DataFrame, max_gap_s: float = 0.5, hampel_window: int = 7, hampel_n_sigma: float = 4.0
) -> CleanResult:
    """
    Sort/dedupe timestamps, interpolate short per-signal gaps, mask long ones, and
    apply a Hampel filter to remove spikes (FR-PRP-002).

    Parameters
    ----------
    df : pandas.DataFrame
        Canonical-schema time series with `t_s`.
    max_gap_s : float, default 0.5
        Gaps up to this duration are linearly interpolated; longer gaps are masked
        (`mask_<sig>` -- caller decides the final column name) rather than filled.
    hampel_window, hampel_n_sigma : passed to `hampel_filter`.

    Returns
    -------
    CleanResult
        `t_s` is rebased to start at 0 (data spec §4.1: `t_s` is "seconds from flight
        start" *after* cleaning; raw telemetry can legitimately start at a nonzero --
        even negative -- `t_s` due to an injected constant clock-offset defect).
        Spike counts measure ``>= 95%`` removal of injected spikes on synthetic test
        data (FR-PRP-002's acceptance criterion).
    """
    sorted_df = (
        df.sort_values("t_s").drop_duplicates(subset="t_s", keep="first").reset_index(drop=True)
    )
    sorted_df["t_s"] = sorted_df["t_s"] - sorted_df["t_s"].iloc[0]
    dt = float(np.median(np.diff(sorted_df["t_s"].to_numpy()))) if len(sorted_df) > 1 else 0.0
    max_gap_samples = max(1, round(max_gap_s / dt)) if dt > 0 else 1

    cleaned = sorted_df.copy()
    mask = pd.DataFrame(False, index=sorted_df.index, columns=[])
    n_spikes: dict[str, int] = {}
    n_filled: dict[str, int] = {}
    n_masked: dict[str, int] = {}

    for name in known_signals():
        if name not in sorted_df.columns:
            continue
        series = sorted_df[name]
        was_missing = series.isna()

        interpolated = _interpolate_short_gaps(series, max_gap_samples)
        still_missing = interpolated.isna()
        n_filled[name] = int((was_missing & ~still_missing).sum())
        n_masked[name] = int(still_missing.sum())

        hampel_cleaned, outlier_mask = hampel_filter(interpolated, hampel_window, hampel_n_sigma)
        n_spikes[name] = int(outlier_mask.sum())

        cleaned[name] = hampel_cleaned
        mask[name] = still_missing

    return CleanResult(
        df=cleaned,
        mask=mask,
        n_spikes_removed=n_spikes,
        n_gaps_filled=n_filled,
        n_gaps_masked=n_masked,
    )
