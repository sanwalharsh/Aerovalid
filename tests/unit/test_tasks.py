from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.predict.datasets import build_windows, leave_one_flight_out
from aerovalid.predict.models.ridge import RidgeModel
from aerovalid.predict.tasks import (
    TaskAResult,
    TaskBResult,
    TaskCResult,
    task_a_residual_prediction,
    task_b_hybrid_correction,
    task_c_forecast,
)

_WINDOW_KWARGS = dict(
    feature_signals=["alt_m", "vel_air_mps"],
    target_signal="alt_m",
    horizon_s=5.0,
    dt=1.0,
    lag_s=10.0,
    stride_s=2.0,
)


def _one_split(flight_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return next(iter(leave_one_flight_out(flight_id)))


@pytest.mark.req("FR-PRD-002")
def test_task_a_residual_prediction_shapes_and_metrics(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)
    train_idx, test_idx = _one_split(dataset.flight_id)

    result = task_a_residual_prediction(RidgeModel(), dataset, train_idx, test_idx)
    assert isinstance(result, TaskAResult)
    assert result.predicted_residual.shape == result.actual_residual.shape == test_idx.shape
    assert result.metrics.rmse >= 0.0


@pytest.mark.req("FR-PRD-003")
def test_task_b_hybrid_correction_requires_future_sim_feature(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)  # no include_future_sim
    train_idx, test_idx = _one_split(dataset.flight_id)
    with pytest.raises(ValueError, match="include_future_sim"):
        task_b_hybrid_correction(RidgeModel(), dataset, train_idx, test_idx)


@pytest.mark.req("FR-PRD-003")
def test_task_b_hybrid_correction_beats_sim_only(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    """The synthetic residual is a smooth, flight-specific sinusoid the model can
    extrapolate from recent lag features -- the hybrid correction must substantially
    beat trusting the simulator as-is (predicted_residual = 0)."""
    dataset = build_windows(
        synthetic_predict_aligned, **{**_WINDOW_KWARGS, "include_future_sim": True}
    )
    train_idx, test_idx = _one_split(dataset.flight_id)

    result = task_b_hybrid_correction(RidgeModel(), dataset, train_idx, test_idx)
    assert isinstance(result, TaskBResult)
    assert result.rmse_hybrid < result.rmse_sim_only
    assert result.rmse_reduction_vs_sim_only_pct > 0.0


@pytest.mark.req("FR-PRD-004")
def test_task_c_forecast_computes_finite_skill_score(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    dataset = build_windows(
        synthetic_predict_aligned, **{**_WINDOW_KWARGS, "target_kind": "observed"}
    )
    train_idx, test_idx = _one_split(dataset.flight_id)

    # Persistence-of-observation forecast: y(t+h) = y(t), i.e. the window's last
    # observed value ("alt_m_last"), as the reference baseline for the skill score.
    baseline_pred = dataset.features["alt_m_last"].to_numpy(dtype=np.float64)[test_idx]
    baseline_rmse = float(np.sqrt(np.mean((dataset.target[test_idx] - baseline_pred) ** 2)))

    result = task_c_forecast(RidgeModel(), dataset, train_idx, test_idx, baseline_rmse)
    assert isinstance(result, TaskCResult)
    assert np.isfinite(result.skill_vs_baseline)
    assert result.predicted.shape == result.actual.shape == test_idx.shape
