from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.predict.cv import cross_validate, nested_cv_search
from aerovalid.predict.datasets import build_windows, grouped_kfold, leave_one_flight_out
from aerovalid.predict.models.ridge import RidgeModel
from aerovalid.validate.metrics import MetricSet

_WINDOW_KWARGS = dict(
    feature_signals=["alt_m", "vel_air_mps"],
    target_signal="alt_m",
    horizon_s=5.0,
    dt=1.0,
    lag_s=10.0,
    stride_s=2.0,
)


@pytest.mark.req("FR-PRD-008")
def test_cross_validate_pools_predictions_across_grouped_folds(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)
    splits = list(grouped_kfold(dataset.flight_id, n_splits=3))
    result = cross_validate(lambda: RidgeModel(), dataset, iter(splits))

    assert len(result.fold_metrics) == 3
    assert isinstance(result.overall_metrics, MetricSet)
    assert len(result.predictions) == len(result.actuals) == sum(len(t) for _, t in splits)
    assert set(result.fold_id) == {0, 1, 2}


@pytest.mark.req("FR-PRD-008")
def test_cross_validate_leave_one_flight_out_covers_every_row_once(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)
    result = cross_validate(lambda: RidgeModel(), dataset, leave_one_flight_out(dataset.flight_id))
    assert len(result.predictions) == len(dataset)
    assert len(result.fold_metrics) == len(set(dataset.flight_id))


@pytest.mark.req("FR-PRD-008")
def test_nested_cv_search_no_leakage_one_trial_per_outer_fold(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)
    param_grid = [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}]
    outer_splits = list(grouped_kfold(dataset.flight_id, n_splits=2))

    result = nested_cv_search(
        lambda params: RidgeModel(alpha=params["alpha"]),
        param_grid,
        dataset,
        iter(outer_splits),
        inner_n_splits=2,
        rng=np.random.default_rng(0),
    )

    assert len(result.trials) == len(outer_splits)
    for trial in result.trials:
        assert trial.best_params in param_grid
        assert isinstance(trial.outer_metrics, MetricSet)
        assert np.isfinite(trial.inner_rmse)


@pytest.mark.req("FR-PRD-008")
def test_nested_cv_search_bounds_trials_to_max_trials(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)
    big_grid = [{"alpha": a} for a in np.linspace(0.01, 50.0, 40)]
    outer_splits = list(grouped_kfold(dataset.flight_id, n_splits=2))

    result = nested_cv_search(
        lambda params: RidgeModel(alpha=params["alpha"]),
        big_grid,
        dataset,
        iter(outer_splits),
        inner_n_splits=2,
        rng=np.random.default_rng(0),
        max_trials=5,
    )
    assert len(result.trials) == len(outer_splits)
