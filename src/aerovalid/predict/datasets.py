"""Leakage-safe windowed dataset builder and flight-aware splitters (FR-PRD-001).

Every window's lagged features and its target come from the *same* flight -- windows
are built one flight at a time and never straddle a flight boundary by construction.
Splits are always by `flight_id` (leave-one-flight-out, grouped K-fold, or
leave-condition-out), never by random row (CLAUDE.md §2).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold


@dataclass
class WindowedDataset:
    """One window per row: lagged summary-stat features, a target, and the flight/
    time/condition metadata needed for leakage-safe splitting and condition analysis."""

    features: pd.DataFrame
    target: NDArray[np.float64]
    flight_id: NDArray[np.str_]
    t_s: NDArray[np.float64]
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.target)


def _lagged_feature_columns(
    group: pd.DataFrame,
    feature_signals: list[str],
    centers: NDArray[np.intp],
    lag_samples: int,
    lag_duration_s: float,
) -> dict[str, NDArray[np.float64]]:
    """Last value, rolling mean/std, and slope over the lag window, per feature signal."""
    columns: dict[str, NDArray[np.float64]] = {}
    for signal in feature_signals:
        obs_col = f"obs_{signal}"
        if obs_col not in group.columns:
            continue
        series = group[obs_col]
        values = series.to_numpy(dtype=np.float64)
        rolling_mean = series.rolling(lag_samples + 1, min_periods=1).mean().to_numpy()
        rolling_std = series.rolling(lag_samples + 1, min_periods=1).std().to_numpy()
        columns[f"{signal}_last"] = values[centers]
        columns[f"{signal}_mean"] = rolling_mean[centers]
        columns[f"{signal}_std"] = np.nan_to_num(rolling_std[centers], nan=0.0)
        columns[f"{signal}_slope"] = (
            values[centers] - values[centers - lag_samples]
        ) / lag_duration_s
    return columns


def _categorical_context_columns(
    group: pd.DataFrame, centers: NDArray[np.intp]
) -> dict[str, NDArray[np.str_]]:
    """Flight-phase and condition-bin labels at each window's reference time."""
    columns: dict[str, NDArray[np.str_]] = {}
    if "phase" in group.columns:
        columns["phase"] = group["phase"].astype(str).to_numpy(dtype=np.str_)[centers]
    for cond_col in (c for c in group.columns if c.startswith("cond_")):
        columns[cond_col] = group[cond_col].astype(str).to_numpy(dtype=np.str_)[centers]
    return columns


def _target_and_current_columns(
    group: pd.DataFrame,
    target_signal: str,
    centers: NDArray[np.intp],
    target_idx: NDArray[np.intp],
    target_kind: str,
    include_future_sim: bool,
) -> dict[str, NDArray[np.float64]] | None:
    """The window's target plus "today's" residual/sim value (every baseline and
    several models need these, not just the generic feature_signals' summary stats).
    Returns None if the target column isn't present in `group`.
    """
    target_col = f"res_{target_signal}" if target_kind == "residual" else f"obs_{target_signal}"
    if target_col not in group.columns:
        return None

    columns: dict[str, NDArray[np.float64]] = {
        "target": group[target_col].to_numpy(dtype=np.float64)[target_idx]
    }
    res_target_col = f"res_{target_signal}"
    if res_target_col in group.columns:
        columns["current_residual"] = group[res_target_col].to_numpy(dtype=np.float64)[centers]
    sim_target_col = f"sim_{target_signal}"
    if sim_target_col in group.columns:
        columns["current_sim"] = group[sim_target_col].to_numpy(dtype=np.float64)[centers]
        if include_future_sim:
            columns["sim_future"] = group[sim_target_col].to_numpy(dtype=np.float64)[target_idx]
    return columns


def _flight_windows(
    group: pd.DataFrame,
    feature_signals: list[str],
    target_signal: str,
    dt: float,
    lag_samples: int,
    horizon_samples: int,
    stride_samples: int,
    target_kind: str,
    include_future_sim: bool,
) -> pd.DataFrame:
    n = len(group)
    if n <= lag_samples + horizon_samples:
        return pd.DataFrame()
    centers = np.arange(lag_samples, n - horizon_samples, stride_samples)
    if centers.size == 0:
        return pd.DataFrame()
    target_idx = centers + horizon_samples

    target_columns = _target_and_current_columns(
        group, target_signal, centers, target_idx, target_kind, include_future_sim
    )
    if target_columns is None:
        return pd.DataFrame()

    columns: dict[str, NDArray[np.float64] | NDArray[np.str_]] = {
        **_lagged_feature_columns(group, feature_signals, centers, lag_samples, lag_samples * dt),
        **_categorical_context_columns(group, centers),
        **target_columns,
    }

    out = pd.DataFrame(columns)
    out["flight_id"] = group["flight_id"].iloc[0]
    out["t_s"] = group["t_s"].to_numpy(dtype=np.float64)[centers]
    return out


def build_windows(
    aligned: pd.DataFrame,
    feature_signals: list[str],
    target_signal: str,
    horizon_s: float,
    dt: float,
    lag_s: float = 20.0,
    stride_s: float = 1.0,
    target_kind: str = "residual",
    include_future_sim: bool = False,
) -> WindowedDataset:
    """
    Build a leakage-safe windowed dataset (FR-PRD-001, analysis methods §7.2).

    Parameters
    ----------
    aligned : pandas.DataFrame
        One or more concatenated `prep/align.py` aligned tables (must have
        `flight_id`, `t_s`, and `obs_/sim_/res_` columns for `feature_signals` and
        `target_signal`).
    feature_signals : list[str]
        Signals to derive lagged features from: last value, rolling mean/std over the
        lag window, and slope.
    target_signal : str
        Signal the target is drawn from.
    horizon_s : float
        How far ahead (s) the target is relative to each window's reference time.
    dt : float
        The aligned table's (regular) grid spacing, s.
    lag_s : float, default 20.0
        Lookback window for features, s.
    stride_s : float, default 1.0
        Spacing between successive window reference times, s.
    target_kind : str, default "residual"
        `"residual"` (Task A: predict `res_<target_signal>(t+h)`) or `"observed"`
        (Task C: predict `obs_<target_signal>(t+h)`).
    include_future_sim : bool, default False
        If True, add `sim_<target_signal>(t+h)` as a feature column `"sim_future"`
        (Task B's hybrid correction needs this; Task C "with sim as exogenous" too).

    Returns
    -------
    WindowedDataset
    """
    lag_samples = max(1, round(lag_s / dt))
    horizon_samples = max(1, round(horizon_s / dt))
    stride_samples = max(1, round(stride_s / dt))

    frames = [
        _flight_windows(
            group,
            feature_signals,
            target_signal,
            dt,
            lag_samples,
            horizon_samples,
            stride_samples,
            target_kind,
            include_future_sim,
        )
        for _flight_id, group in aligned.groupby("flight_id", sort=False)
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("No windows could be built (flights too short for lag_s + horizon_s)")

    combined = pd.concat(frames, ignore_index=True).dropna()
    feature_names = [c for c in combined.columns if c not in ("flight_id", "t_s", "target")]

    return WindowedDataset(
        features=combined[feature_names].reset_index(drop=True),
        target=combined["target"].to_numpy(dtype=np.float64),
        flight_id=combined["flight_id"].to_numpy(dtype=str),
        t_s=combined["t_s"].to_numpy(dtype=np.float64),
        feature_names=feature_names,
    )


def leave_one_flight_out(
    flight_id: NDArray[np.str_],
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Yield `(train_idx, test_idx)` holding out one flight at a time."""
    for held_out in pd.unique(flight_id):
        test_mask = flight_id == held_out
        yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def grouped_kfold(
    flight_id: NDArray[np.str_], n_splits: int = 5
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Grouped K-fold by `flight_id` (every sample from one flight stays in one fold)."""
    n_splits = min(n_splits, len(pd.unique(flight_id)))
    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(flight_id), 1))
    yield from splitter.split(dummy_x, groups=flight_id)


def leave_condition_out(
    condition: NDArray[np.str_], held_out_values: list[str]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Single train/test split holding out specific condition-bin values entirely, for
    extrapolation testing (e.g. the highest altitude or Mach band).

    Parameters
    ----------
    condition : ndarray, shape (N,)
        A condition-bin column (e.g. `cond_alt_band`) aligned with the dataset.
    held_out_values : list[str]
        Bin values to exclude from training (test set).

    Returns
    -------
    tuple[ndarray, ndarray]
        `(train_idx, test_idx)`.
    """
    test_mask = np.isin(condition, held_out_values)
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
