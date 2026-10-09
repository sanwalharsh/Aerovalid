from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.predict.baselines import (
    ConstantVelocityBaseline,
    MeanByBinBaseline,
    PersistenceBaseline,
    SimOnlyBaseline,
)


@pytest.mark.req("FR-PRD-002")
def test_persistence_baseline_predicts_current_residual() -> None:
    features = pd.DataFrame({"current_residual": [1.0, -2.0, 3.5]})
    model = PersistenceBaseline()
    model.fit(features, np.zeros(3))
    np.testing.assert_allclose(model.predict(features), [1.0, -2.0, 3.5])
    lower, upper = model.predict_interval(features)
    np.testing.assert_allclose(lower, upper)


@pytest.mark.req("FR-PRD-003")
def test_sim_only_baseline_always_predicts_zero() -> None:
    features = pd.DataFrame({"anything": [1.0, 2.0, 3.0]})
    model = SimOnlyBaseline()
    model.fit(features, np.array([5.0, -5.0, 1.0]))
    np.testing.assert_allclose(model.predict(features), 0.0)
    lower, upper = model.predict_interval(features)
    np.testing.assert_allclose(lower, 0.0)
    np.testing.assert_allclose(upper, 0.0)


@pytest.mark.req("FR-PRD-004")
def test_constant_velocity_baseline_extrapolates_slope() -> None:
    features = pd.DataFrame({"alt_m_last": [100.0, 200.0], "alt_m_slope": [2.0, -1.0]})
    model = ConstantVelocityBaseline(target_signal="alt_m", horizon_s=5.0)
    model.fit(features, np.zeros(2))
    np.testing.assert_allclose(model.predict(features), [110.0, 195.0])


@pytest.mark.req("FR-PRD-002")
def test_mean_by_bin_baseline_predicts_training_bin_mean() -> None:
    features = pd.DataFrame({"phase": ["climb", "climb", "cruise", "cruise"]})
    target = np.array([1.0, 3.0, 10.0, 20.0])
    model = MeanByBinBaseline(bin_column="phase")
    model.fit(features, target)
    predictions = model.predict(features)
    np.testing.assert_allclose(predictions, [2.0, 2.0, 15.0, 15.0])


@pytest.mark.req("FR-PRD-002")
def test_mean_by_bin_baseline_falls_back_to_global_mean_for_unseen_bin() -> None:
    train_features = pd.DataFrame({"phase": ["climb", "climb"]})
    model = MeanByBinBaseline(bin_column="phase")
    model.fit(train_features, np.array([1.0, 3.0]))
    test_features = pd.DataFrame({"phase": ["descent"]})
    np.testing.assert_allclose(model.predict(test_features), [2.0])


@pytest.mark.req("FR-PRD-002")
def test_mean_by_bin_baseline_handles_missing_bin_column() -> None:
    model = MeanByBinBaseline(bin_column="cond_mach_band")
    model.fit(pd.DataFrame({"other": [1.0, 2.0]}), np.array([4.0, 6.0]))
    predictions = model.predict(pd.DataFrame({"other": [1.0, 2.0, 3.0]}))
    np.testing.assert_allclose(predictions, [5.0, 5.0, 5.0])
