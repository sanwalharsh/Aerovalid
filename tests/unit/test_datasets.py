from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.predict.datasets import (
    build_windows,
    grouped_kfold,
    leave_condition_out,
    leave_one_flight_out,
)

_WINDOW_KWARGS = dict(
    feature_signals=["alt_m", "vel_air_mps"],
    target_signal="alt_m",
    horizon_s=5.0,
    dt=1.0,
    lag_s=10.0,
    stride_s=2.0,
)


@pytest.mark.req("FR-PRD-001")
def test_build_windows_basic_shape(synthetic_predict_aligned: pd.DataFrame) -> None:
    dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)
    assert len(dataset) == len(dataset.target) == len(dataset.features)
    assert len(dataset.flight_id) == len(dataset.t_s) == len(dataset)
    assert set(dataset.feature_names) == set(dataset.features.columns)
    assert "current_residual" in dataset.feature_names
    assert "current_sim" in dataset.feature_names
    assert "alt_m_slope" in dataset.feature_names
    assert set(dataset.flight_id) == set(synthetic_predict_aligned["flight_id"].unique())


@pytest.mark.req("FR-PRD-001")
def test_build_windows_no_window_straddles_flight_boundary(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    """Every window's features/target must come from a single flight: building windows
    one flight at a time must reproduce exactly the rows the multi-flight call produces
    for that flight (same values, not just the same count)."""
    combined = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)

    for flight_id, group in synthetic_predict_aligned.groupby("flight_id", sort=False):
        solo = build_windows(group.reset_index(drop=True), **_WINDOW_KWARGS)
        combined_rows = combined.features[combined.flight_id == flight_id].reset_index(drop=True)
        assert len(combined_rows) == len(solo.features)
        for col in solo.feature_names:
            if pd.api.types.is_numeric_dtype(solo.features[col]):
                np.testing.assert_allclose(
                    combined_rows[col].to_numpy(dtype=np.float64),
                    solo.features[col].to_numpy(dtype=np.float64),
                )
            else:
                assert (combined_rows[col].astype(str) == solo.features[col].astype(str)).all()
        combined_target = combined.target[combined.flight_id == flight_id]
        np.testing.assert_allclose(combined_target, solo.target)


@pytest.mark.req("FR-PRD-001")
def test_build_windows_target_kind_observed(synthetic_predict_aligned: pd.DataFrame) -> None:
    residual_dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)
    observed_dataset = build_windows(
        synthetic_predict_aligned, **{**_WINDOW_KWARGS, "target_kind": "observed"}
    )
    # obs_alt_m is a steadily increasing ramp (500 + 0.5*t), unlike the oscillating
    # residual target -- the two target kinds must differ substantially.
    assert observed_dataset.target.min() > 500.0
    assert not np.allclose(observed_dataset.target, residual_dataset.target)


@pytest.mark.req("FR-PRD-001")
def test_build_windows_include_future_sim_adds_column(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    kwargs = {**_WINDOW_KWARGS, "include_future_sim": True}
    dataset = build_windows(synthetic_predict_aligned, **kwargs)
    assert "sim_future" in dataset.feature_names


@pytest.mark.req("FR-PRD-001")
def test_build_windows_raises_when_flights_too_short() -> None:
    short = pd.DataFrame(
        {
            "flight_id": "F0001",
            "t_s": np.arange(5, dtype=float),
            "obs_alt_m": np.arange(5, dtype=float),
            "sim_alt_m": np.arange(5, dtype=float),
            "res_alt_m": np.zeros(5),
        }
    )
    with pytest.raises(ValueError, match="too short"):
        build_windows(short, **_WINDOW_KWARGS)


@pytest.mark.req("FR-PRD-001")
def test_leave_one_flight_out_covers_all_flights_disjointly(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)
    held_out_flights = []
    for train_idx, test_idx in leave_one_flight_out(dataset.flight_id):
        assert set(train_idx).isdisjoint(set(test_idx))
        test_flights = set(dataset.flight_id[test_idx])
        assert len(test_flights) == 1
        assert test_flights.isdisjoint(set(dataset.flight_id[train_idx]))
        held_out_flights.append(next(iter(test_flights)))
    assert set(held_out_flights) == set(dataset.flight_id)


@pytest.mark.req("FR-PRD-001")
def test_grouped_kfold_keeps_each_flight_in_one_fold(
    synthetic_predict_aligned: pd.DataFrame,
) -> None:
    dataset = build_windows(synthetic_predict_aligned, **_WINDOW_KWARGS)
    for train_idx, test_idx in grouped_kfold(dataset.flight_id, n_splits=3):
        train_flights = set(dataset.flight_id[train_idx])
        test_flights = set(dataset.flight_id[test_idx])
        assert train_flights.isdisjoint(test_flights)


@pytest.mark.req("FR-PRD-001")
def test_leave_condition_out_splits_by_condition_value() -> None:
    condition = np.array(["low", "low", "mid", "high", "high"])
    train_idx, test_idx = leave_condition_out(condition, held_out_values=["high"])
    assert set(test_idx) == {3, 4}
    assert set(train_idx) == {0, 1, 2}
