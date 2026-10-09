"""Time synchronisation (FR-PRP-003): cross-correlation lag (+ optional linear drift)
estimation between a reference channel set in two tables (typically telemetry vs a
variant's simulated output), with sub-sample parabolic refinement and a bootstrap CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import correlate as scipy_correlate

_MIN_OVERLAP_SAMPLES = 16
_DRIFT_WINDOW_FRACTION = 0.2  # each bookend window is this fraction of the overlap
_LOW_CORRELATION_THRESHOLD = 0.6  # data spec §8: below this, flag + fall back to zero-lag
_UPSAMPLE_FACTOR = 5  # interpolation grid is this much finer than the native sample rate


@dataclass
class SyncEstimate:
    """Estimated time alignment between a reference and another table.

    Attributes
    ----------
    lag_s : float
        Estimated constant lag: `other`'s clock is ahead of `ref`'s by this much, s.
        Positive means `other` should be shifted *later* to align with `ref`.
    drift_ppm : float
        Estimated linear clock drift, parts per million.
    peak_correlation : float
        Normalised cross-correlation at the estimated lag, in [-1, 1].
    lag_ci_s : tuple[float, float]
        Bootstrap 95% CI for `lag_s`.
    low_confidence : bool
        True if `peak_correlation < 0.6` (data spec §8): the estimate fell back to
        zero-lag and should be treated with suspicion.
    """

    lag_s: float
    drift_ppm: float
    peak_correlation: float
    lag_ci_s: tuple[float, float]
    low_confidence: bool


def _combined_zscore_diff(
    df: pd.DataFrame, t_grid: np.ndarray, ref_signals: list[str]
) -> np.ndarray:
    combined = np.zeros_like(t_grid, dtype=np.float64)
    n_used = 0
    for name in ref_signals:
        if name not in df.columns:
            continue
        values = np.interp(
            t_grid, df["t_s"].to_numpy(), df[name].to_numpy(), left=np.nan, right=np.nan
        )
        # Centred difference (np.gradient), not a forward diff: a forward
        # difference is phase-biased by half a sample, which alone can blow the
        # +-0.5-sample lag-recovery tolerance (FR-PRP-003).
        diff = np.gradient(values) if len(values) > 1 else np.zeros_like(values)
        std = np.nanstd(diff)
        if std <= 0 or np.all(np.isnan(diff)):
            continue
        combined += np.nan_to_num((diff - np.nanmean(diff)) / std, nan=0.0)
        n_used += 1
    return combined / max(n_used, 1)


def _parabolic_peak(corr: np.ndarray, peak_idx: int) -> float:
    if 0 < peak_idx < len(corr) - 1:
        left, center, right = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = left - 2 * center + right
        if denom != 0:
            return float(peak_idx + 0.5 * (left - right) / denom)
    return float(peak_idx)


def _xcorr_lag(
    ref_signal: np.ndarray, other_signal: np.ndarray, max_lag_samples: int
) -> tuple[float, float]:
    """Returns (lag_in_samples, peak_normalised_correlation)."""
    # FFT-based correlation (O(n log n)): `np.correlate`'s direct O(n*m) algorithm is
    # far too slow once the interpolation grid is upsampled (see _UPSAMPLE_FACTOR).
    full = scipy_correlate(other_signal, ref_signal, mode="full", method="fft")
    center = len(ref_signal) - 1
    lo, hi = max(0, center - max_lag_samples), min(len(full), center + max_lag_samples + 1)
    window = full[lo:hi]
    if window.size == 0 or not np.any(window):
        return 0.0, 0.0
    local_peak_idx = int(np.argmax(np.abs(window)))
    peak_idx = lo + local_peak_idx
    lag_samples = _parabolic_peak(full, peak_idx) - center

    norm = np.sqrt(np.sum(ref_signal**2) * np.sum(other_signal**2))
    peak_corr = float(full[peak_idx] / norm) if norm > 0 else 0.0
    return float(lag_samples), peak_corr


def estimate_sync(
    ref: pd.DataFrame,
    other: pd.DataFrame,
    ref_signals: list[str],
    max_lag_s: float,
    dt: float,
    *,
    estimate_drift: bool = True,
    n_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
) -> SyncEstimate:
    """
    Estimate constant lag (and optional linear drift) of `other` relative to `ref`.

    Parameters
    ----------
    ref : pandas.DataFrame
        Reference table (typically telemetry), must have `t_s` and `ref_signals`.
    other : pandas.DataFrame
        Table to align (typically a variant's simulated output).
    ref_signals : list[str]
        Signal names to cross-correlate (z-scored first differences, averaged).
    max_lag_s : float
        Maximum lag magnitude to search, s.
    dt : float
        Grid spacing for the common interpolation grid, s.
    estimate_drift : bool, default True
        If True, split the overlap into windows and fit a line (lag vs window centre
        time) for the drift estimate; otherwise `drift_ppm` is 0.
    n_bootstrap : int, default 200
        Block-bootstrap resamples for the lag CI.
    rng : numpy.random.Generator, optional
        Seeded generator for the bootstrap (FR-COR-006); a fresh one is created if
        omitted (acceptable here since this doesn't feed back into generated data).

    Returns
    -------
    SyncEstimate
    """
    rng = rng if rng is not None else np.random.default_rng()
    t_min = max(float(ref["t_s"].min()), float(other["t_s"].min()))
    t_max = min(float(ref["t_s"].max()), float(other["t_s"].max()))
    if t_max - t_min < _MIN_OVERLAP_SAMPLES * dt:
        return SyncEstimate(0.0, 0.0, 0.0, (0.0, 0.0), low_confidence=True)

    # Interpolate onto a finer grid than the native sample spacing: the parabolic
    # peak refinement is only as precise as the grid it's fit on, and the native
    # telemetry rate alone isn't fine enough to reliably hit the +-0.5-sample
    # (FR-PRP-003) tolerance.
    fine_dt = dt / _UPSAMPLE_FACTOR
    t_grid = np.arange(t_min, t_max, fine_dt)
    ref_signal = _combined_zscore_diff(ref, t_grid, ref_signals)
    other_signal = _combined_zscore_diff(other, t_grid, ref_signals)
    max_lag_samples = max(1, int(round(max_lag_s / fine_dt)))

    lag_samples, peak_corr = _xcorr_lag(ref_signal, other_signal, max_lag_samples)
    lag_s = lag_samples * fine_dt
    low_confidence = abs(peak_corr) < _LOW_CORRELATION_THRESHOLD
    if low_confidence:
        lag_s = 0.0

    drift_ppm = 0.0
    if estimate_drift and not low_confidence and len(t_grid) >= 2 * _MIN_OVERLAP_SAMPLES:
        # Two bookend windows (start/end of the overlap) rather than several evenly
        # spaced ones: drift is a slope (delta lag / delta t), and maximising the time
        # baseline between the two lag measurements minimises how much per-window
        # xcorr noise gets amplified into the ppm estimate.
        window_len = max(_MIN_OVERLAP_SAMPLES, int(len(t_grid) * _DRIFT_WINDOW_FRACTION))
        start_sl, end_sl = slice(0, window_len), slice(len(t_grid) - window_len, len(t_grid))
        start_lag, start_corr = _xcorr_lag(
            ref_signal[start_sl], other_signal[start_sl], max_lag_samples
        )
        end_lag, end_corr = _xcorr_lag(ref_signal[end_sl], other_signal[end_sl], max_lag_samples)
        confident = (
            abs(start_corr) >= _LOW_CORRELATION_THRESHOLD
            and abs(end_corr) >= _LOW_CORRELATION_THRESHOLD
        )
        if confident:
            t_start = t_grid[start_sl][window_len // 2]
            t_end = t_grid[end_sl][window_len // 2]
            delta_t = t_end - t_start
            if delta_t > 0:
                slope = (end_lag - start_lag) * fine_dt / delta_t
                drift_ppm = float(slope * 1e6)

    bootstrap_lags = []
    block = max(1, len(t_grid) // 10)
    n_blocks = len(t_grid) // block
    if n_blocks >= 2:
        for _ in range(n_bootstrap):
            block_ids = rng.integers(0, n_blocks, size=n_blocks)
            idx = np.concatenate([np.arange(b * block, (b + 1) * block) for b in block_ids])
            boot_lag, _ = _xcorr_lag(ref_signal[idx], other_signal[idx], max_lag_samples)
            bootstrap_lags.append(boot_lag * fine_dt)
    ci = (
        (float(np.percentile(bootstrap_lags, 2.5)), float(np.percentile(bootstrap_lags, 97.5)))
        if bootstrap_lags
        else (lag_s, lag_s)
    )

    return SyncEstimate(
        lag_s=lag_s,
        drift_ppm=drift_ppm,
        peak_correlation=peak_corr,
        lag_ci_s=ci,
        low_confidence=low_confidence,
    )


def apply_sync(df: pd.DataFrame, estimate: SyncEstimate) -> pd.DataFrame:
    """
    Shift `df`'s `t_s` to correct the estimated lag/drift: `t_corrected = (t - lag) /
    (1 + drift_ppm * 1e-6)`.

    Parameters
    ----------
    df : pandas.DataFrame
    estimate : SyncEstimate

    Returns
    -------
    pandas.DataFrame
        `df` with `t_s` corrected (a copy; other columns untouched).
    """
    out = df.copy()
    out["t_s"] = (out["t_s"] - estimate.lag_s) / (1.0 + estimate.drift_ppm * 1e-6)
    return out
