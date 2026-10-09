"""Predictive tasks (FR-PRD-002/003/004, analysis methods §7.1): Task A (residual
prediction), Task B (hybrid correction = sim + predicted residual, vs. sim-only and
persistence-of-residual), Task C (forecast, optionally with sim as exogenous input).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aerovalid.predict.datasets import WindowedDataset
from aerovalid.predict.models import ResidualModel
from aerovalid.validate.metrics import MetricSet, compute_metrics_from_residual


@dataclass(frozen=True)
class TaskAResult:
    """Task A: residual prediction."""

    predicted_residual: NDArray[np.float64]
    actual_residual: NDArray[np.float64]
    metrics: MetricSet


def task_a_residual_prediction(
    model: ResidualModel,
    dataset: WindowedDataset,
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
) -> TaskAResult:
    """
    Task A: fit `model` to predict `res_signal(t+h)` on the training rows, evaluate on
    the held-out rows (FR-PRD-002).

    Parameters
    ----------
    model : ResidualModel
    dataset : WindowedDataset
        Built with `target_kind="residual"`.
    train_idx, test_idx : ndarray
        Row indices into `dataset` (e.g. from `predict/datasets.py`'s splitters).

    Returns
    -------
    TaskAResult
    """
    model.fit(
        dataset.features.iloc[train_idx], dataset.target[train_idx], dataset.flight_id[train_idx]
    )
    predicted = model.predict(dataset.features.iloc[test_idx])
    actual = dataset.target[test_idx]
    metrics = compute_metrics_from_residual(actual, predicted, actual - predicted)
    return TaskAResult(predicted_residual=predicted, actual_residual=actual, metrics=metrics)


@dataclass(frozen=True)
class TaskBResult:
    """Task B: hybrid correction vs. sim-only and persistence-of-residual baselines."""

    rmse_sim_only: float
    rmse_persistence: float
    rmse_hybrid: float
    rmse_reduction_vs_sim_only_pct: float
    rmse_reduction_vs_persistence_pct: float


def task_b_hybrid_correction(
    model: ResidualModel,
    dataset: WindowedDataset,
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
) -> TaskBResult:
    """
    Task B: `hybrid(t+h) = sim(t+h) + predicted_residual(t+h)`, compared against
    sim-only (`predicted_residual = 0`) and persistence-of-residual
    (`predicted_residual = residual(t)`) (FR-PRD-003).

    Parameters
    ----------
    model : ResidualModel
    dataset : WindowedDataset
        Must be built with `target_kind="residual"` and `include_future_sim=True`
        (needs the `"sim_future"` and `"current_residual"` feature columns).
    train_idx, test_idx : ndarray

    Returns
    -------
    TaskBResult
        `rmse_reduction_vs_sim_only_pct`: FR-PRD-003's S4 acceptance criterion
        (`>= 25%` on the benchmark).
    """
    if "sim_future" not in dataset.features.columns:
        raise ValueError("Task B requires a dataset built with include_future_sim=True")

    model.fit(
        dataset.features.iloc[train_idx], dataset.target[train_idx], dataset.flight_id[train_idx]
    )
    predicted_residual = model.predict(dataset.features.iloc[test_idx])

    sim_future = dataset.features["sim_future"].to_numpy(dtype=np.float64)[test_idx]
    current_residual = dataset.features["current_residual"].to_numpy(dtype=np.float64)[test_idx]
    actual_residual = dataset.target[test_idx]
    obs_future = actual_residual + sim_future

    hybrid = sim_future + predicted_residual
    persistence = sim_future + current_residual

    rmse_sim_only = float(np.sqrt(np.mean((obs_future - sim_future) ** 2)))
    rmse_persistence = float(np.sqrt(np.mean((obs_future - persistence) ** 2)))
    rmse_hybrid = float(np.sqrt(np.mean((obs_future - hybrid) ** 2)))

    reduction_vs_sim = (
        100.0 * (rmse_sim_only - rmse_hybrid) / rmse_sim_only if rmse_sim_only > 0 else float("nan")
    )
    reduction_vs_persist = (
        100.0 * (rmse_persistence - rmse_hybrid) / rmse_persistence
        if rmse_persistence > 0
        else float("nan")
    )

    return TaskBResult(
        rmse_sim_only=rmse_sim_only,
        rmse_persistence=rmse_persistence,
        rmse_hybrid=rmse_hybrid,
        rmse_reduction_vs_sim_only_pct=reduction_vs_sim,
        rmse_reduction_vs_persistence_pct=reduction_vs_persist,
    )


@dataclass(frozen=True)
class TaskCResult:
    """Task C: forecast selected signals from history (+ optionally sim as exogenous)."""

    predicted: NDArray[np.float64]
    actual: NDArray[np.float64]
    metrics: MetricSet
    skill_vs_baseline: float


def task_c_forecast(
    model: ResidualModel,
    dataset: WindowedDataset,
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
    baseline_rmse: float,
) -> TaskCResult:
    """
    Task C: forecast `y_signal(t+h)` (built with `target_kind="observed"`), optionally
    using `sim_future` as an exogenous feature (FR-PRD-004).

    Parameters
    ----------
    model : ResidualModel
    dataset : WindowedDataset
        Built with `target_kind="observed"`.
    train_idx, test_idx : ndarray
    baseline_rmse : float
        A baseline's RMSE on the same `test_idx` (e.g. persistence or sim-only), used
        for the skill score `1 - RMSE / RMSE_baseline`.

    Returns
    -------
    TaskCResult
    """
    model.fit(
        dataset.features.iloc[train_idx], dataset.target[train_idx], dataset.flight_id[train_idx]
    )
    predicted = model.predict(dataset.features.iloc[test_idx])
    actual = dataset.target[test_idx]
    metrics = compute_metrics_from_residual(actual, predicted, actual - predicted)
    skill = 1.0 - metrics.rmse / baseline_rmse if baseline_rmse > 0 else float("nan")
    return TaskCResult(predicted=predicted, actual=actual, metrics=metrics, skill_vs_baseline=skill)
