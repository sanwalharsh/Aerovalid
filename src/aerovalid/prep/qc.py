"""QC report per flight (FR-PRP-001): missing %, duplicate/non-monotonic timestamps,
range violations, stuck-sensor/flatline detection, spike count, sample-rate stats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from aerovalid.core.errors import SignalRegistryError
from aerovalid.core.signals import get_signal, known_signals

_FLATLINE_MIN_RUN = 20  # consecutive identical samples to call a signal "stuck"
_QUARANTINE_NON_MONOTONIC_FRACTION = 0.05  # data spec §8


class QCReport(BaseModel):
    """Per-flight data-quality summary (FR-PRP-001)."""

    model_config = ConfigDict(extra="forbid")

    flight_id: str
    n_samples: int
    sample_rate_hz: float
    sample_rate_std_hz: float
    duplicate_timestamps: int
    non_monotonic_count: int
    missing_fraction: dict[str, float] = Field(default_factory=dict)
    range_violations: dict[str, int] = Field(default_factory=dict)
    stuck_signals: list[str] = Field(default_factory=list)
    spike_counts: dict[str, int] = Field(default_factory=dict)
    severity: str = "ok"


def _spike_count(series: pd.Series, n_sigma: float = 4.0) -> int:
    """A quick z-score-based spike count for the QC summary (not a replacement for
    `prep/clean.py`'s Hampel filter, which does the actual cleaning)."""
    values = series.to_numpy(dtype=np.float64)
    if np.all(np.isnan(values)):
        return 0
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median)) * 1.4826
    if mad == 0:
        return 0
    return int(np.nansum(np.abs(values - median) > n_sigma * mad))


def _is_stuck(series: pd.Series, min_run: int = _FLATLINE_MIN_RUN) -> bool:
    values = series.to_numpy(dtype=np.float64)
    if len(values) < min_run:
        return False
    is_repeat = np.concatenate([[False], np.diff(values) == 0.0])
    run_length = 0
    for repeat in is_repeat:
        run_length = run_length + 1 if repeat else 0
        if run_length >= min_run - 1:
            return True
    return False


def qc_report(df: pd.DataFrame, flight_id: str) -> QCReport:
    """
    Build a `QCReport` for one flight's telemetry (or truth/variant) table.

    Parameters
    ----------
    df : pandas.DataFrame
        Canonical-schema time series, must have `t_s`.
    flight_id : str

    Returns
    -------
    QCReport
    """
    t_s = df["t_s"].to_numpy(dtype=np.float64)
    n = len(t_s)
    dt = np.diff(t_s)
    duplicate_timestamps = int(np.sum(dt == 0.0))
    non_monotonic = int(np.sum(dt < 0.0))
    sample_rate_hz = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else 0.0
    sample_rate_std_hz = float(np.std(1.0 / dt[dt > 0])) if np.any(dt > 0) else 0.0

    missing_fraction: dict[str, float] = {}
    range_violations: dict[str, int] = {}
    stuck_signals: list[str] = []
    spike_counts: dict[str, int] = {}

    for name in known_signals():
        if name not in df.columns:
            continue
        series = df[name]
        missing_fraction[name] = float(series.isna().mean())
        try:
            spec = get_signal(name)
        except SignalRegistryError:
            continue
        if spec.plausible_range is not None:
            low, high = spec.plausible_range
            range_violations[name] = int(((series < low) | (series > high)).sum())
        if _is_stuck(series):
            stuck_signals.append(name)
        spike_counts[name] = _spike_count(series)

    non_monotonic_fraction = non_monotonic / max(n - 1, 1)
    severity = "quarantine" if non_monotonic_fraction > _QUARANTINE_NON_MONOTONIC_FRACTION else "ok"

    return QCReport(
        flight_id=flight_id,
        n_samples=n,
        sample_rate_hz=sample_rate_hz,
        sample_rate_std_hz=sample_rate_std_hz,
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_count=non_monotonic,
        missing_fraction=missing_fraction,
        range_violations=range_violations,
        stuck_signals=stuck_signals,
        spike_counts=spike_counts,
        severity=severity,
    )
