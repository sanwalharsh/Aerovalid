from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.detect.early_warning import EarlyWarningDetector
from aerovalid.predict.models.ridge import RidgeModel
from aerovalid.predict.uncertainty import SplitConformalWrapper


def _fitted_wrapper(seed: int = 0) -> SplitConformalWrapper:
    rng = np.random.default_rng(seed)

    def _make(n: int) -> tuple[pd.DataFrame, np.ndarray]:
        f0 = rng.normal(0.0, 1.0, n)
        target = 2.0 * f0 + rng.normal(0.0, 0.2, n)
        return pd.DataFrame({"f0": f0}), target

    train_features, train_target = _make(300)
    calib_features, calib_target = _make(300)
    wrapper = SplitConformalWrapper(RidgeModel(), alpha=0.1)
    wrapper.fit(train_features, train_target, calib_features, calib_target)
    return wrapper


@pytest.mark.req("FR-DET-005")
def test_exceedance_probability_is_low_when_prediction_is_well_inside_tolerance() -> None:
    wrapper = _fitted_wrapper()
    detector = EarlyWarningDetector(wrapper, tol=10.0, p_star=0.8)
    features = pd.DataFrame({"f0": [0.0]})  # predicted residual ~0, far from +-10 tol
    probability = detector.exceedance_probability(features)
    assert probability[0] < 0.05


@pytest.mark.req("FR-DET-005")
def test_exceedance_probability_is_high_when_prediction_is_past_tolerance() -> None:
    wrapper = _fitted_wrapper()
    detector = EarlyWarningDetector(wrapper, tol=1.0, p_star=0.8)
    features = pd.DataFrame({"f0": [10.0]})  # predicted residual ~20, way past +-1 tol
    probability = detector.exceedance_probability(features)
    assert probability[0] > 0.95


@pytest.mark.req("FR-DET-005")
def test_alarm_at_a_stricter_p_star_implies_alarm_at_a_looser_one() -> None:
    wrapper = _fitted_wrapper()
    low_p_star_detector = EarlyWarningDetector(wrapper, tol=1.0, p_star=0.5)
    high_p_star_detector = EarlyWarningDetector(wrapper, tol=1.0, p_star=0.999)
    features = pd.DataFrame({"f0": [2.0]})  # predicted residual ~4: likely past tol=1

    if high_p_star_detector.alarm(features)[0]:
        assert low_p_star_detector.alarm(features)[0]
