from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.predict.models.ridge import RidgeModel
from aerovalid.predict.uncertainty import SplitConformalWrapper, evaluate_coverage, winkler_score


def _train_calib_test(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)

    def _make(n: int) -> tuple[pd.DataFrame, np.ndarray]:
        f0 = rng.normal(0.0, 1.0, n)
        f1 = rng.normal(0.0, 1.0, n)
        target = 3.0 * f0 - 2.0 * f1 + rng.normal(0.0, 0.5, n)
        return pd.DataFrame({"f0": f0, "f1": f1}), target

    train_features, train_target = _make(300)
    calib_features, calib_target = _make(300)
    return train_features, train_target, calib_features, calib_target


@pytest.mark.req("FR-PRD-007")
def test_split_conformal_wrapper_achieves_approximate_target_coverage() -> None:
    train_features, train_target, calib_features, calib_target = _train_calib_test()
    wrapper = SplitConformalWrapper(RidgeModel(), alpha=0.1)
    wrapper.fit(train_features, train_target, calib_features, calib_target)

    rng = np.random.default_rng(99)
    test_f0 = rng.normal(0.0, 1.0, 500)
    test_f1 = rng.normal(0.0, 1.0, 500)
    test_target = 3.0 * test_f0 - 2.0 * test_f1 + rng.normal(0.0, 0.5, 500)
    test_features = pd.DataFrame({"f0": test_f0, "f1": test_f1})

    lower, upper = wrapper.predict_interval(test_features)
    coverage, mean_width = evaluate_coverage(lower, upper, test_target)
    assert 0.80 <= coverage <= 0.99
    assert mean_width > 0


@pytest.mark.req("FR-PRD-007")
def test_split_conformal_predict_matches_base_model() -> None:
    train_features, train_target, calib_features, calib_target = _train_calib_test()
    base_model = RidgeModel()
    wrapper = SplitConformalWrapper(RidgeModel(), alpha=0.1)
    wrapper.fit(train_features, train_target, calib_features, calib_target)
    base_model.fit(train_features, train_target)
    np.testing.assert_allclose(wrapper.predict(calib_features), base_model.predict(calib_features))


@pytest.mark.req("FR-PRD-007")
def test_evaluate_coverage_all_covered_and_none_covered() -> None:
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 10.0, 10.0])
    actual_inside = np.array([1.0, 5.0, 9.0])
    coverage, width = evaluate_coverage(lower, upper, actual_inside)
    assert coverage == pytest.approx(1.0)
    assert width == pytest.approx(10.0)

    actual_outside = np.array([-1.0, 11.0, 20.0])
    coverage_zero, _ = evaluate_coverage(lower, upper, actual_outside)
    assert coverage_zero == pytest.approx(0.0)


@pytest.mark.req("FR-PRD-007")
def test_winkler_score_penalizes_miscoverage() -> None:
    lower = np.array([0.0])
    upper = np.array([1.0])
    covered_score = winkler_score(lower, upper, np.array([0.5]), alpha=0.1)
    miscovered_score = winkler_score(lower, upper, np.array([5.0]), alpha=0.1)
    assert covered_score == pytest.approx(1.0)  # just the width
    assert miscovered_score > covered_score
