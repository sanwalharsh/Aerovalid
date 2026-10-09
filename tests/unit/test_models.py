"""Interface-contract tests for every `predict/models/` wrapper (FR-PRD-005's AC:
"Each model passes interface contract test"), plus a guarded GRU test (FR-PRD-006).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from aerovalid.predict.models import ResidualModel
from aerovalid.predict.models.arimax import ARIMAXModel
from aerovalid.predict.models.gru import HAS_TORCH, GRUModel
from aerovalid.predict.models.hgb import HGBModel
from aerovalid.predict.models.rf import RFModel
from aerovalid.predict.models.ridge import RidgeModel


def _synthetic_regression(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    f0 = rng.normal(0.0, 1.0, n)
    f1 = rng.normal(0.0, 1.0, n)
    f2 = rng.normal(0.0, 1.0, n)
    target = 2.0 * f0 - 1.5 * f1 + 0.5 * f2 + rng.normal(0.0, 0.1, n)
    features = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2, "phase": "cruise"})
    return features, target


_MODEL_FACTORIES = {
    "ridge": lambda: RidgeModel(),
    "elasticnet": lambda: RidgeModel(alpha=0.5, l1_ratio=0.3),
    "hgb": lambda: HGBModel(max_iter=20),
    "rf": lambda: RFModel(n_estimators=10),
    "arimax": lambda: ARIMAXModel(ar_order=1),
}


@pytest.mark.req("FR-PRD-005")
@pytest.mark.parametrize("name", list(_MODEL_FACTORIES))
def test_model_satisfies_residual_model_protocol(name: str) -> None:
    model = _MODEL_FACTORIES[name]()
    assert isinstance(model, ResidualModel)


@pytest.mark.req("FR-PRD-005")
@pytest.mark.parametrize("name", list(_MODEL_FACTORIES))
def test_model_fit_predict_interval_contract(name: str) -> None:
    features, target = _synthetic_regression()
    model: Any = _MODEL_FACTORIES[name]()
    model.fit(features, target)

    point = model.predict(features)
    assert point.shape == target.shape
    assert np.all(np.isfinite(point))

    lower, upper = model.predict_interval(features, alpha=0.1)
    assert lower.shape == upper.shape == target.shape
    assert np.all(lower <= upper + 1e-9)


@pytest.mark.req("FR-PRD-005")
@pytest.mark.parametrize("name", ["ridge", "hgb", "rf", "arimax"])
def test_model_learns_the_dominant_linear_signal(name: str) -> None:
    """A model with real learning capacity should beat predicting the training mean
    on data with a strong, mostly-linear signal-to-noise ratio."""
    features, target = _synthetic_regression(n=200, seed=1)
    model: Any = _MODEL_FACTORIES[name]()
    model.fit(features, target)
    predictions = model.predict(features)

    model_mse = np.mean((predictions - target) ** 2)
    baseline_mse = np.mean((target - target.mean()) ** 2)
    assert model_mse < 0.5 * baseline_mse


@pytest.mark.req("FR-PRD-005")
def test_hgb_predict_interval_caches_quantile_models_per_alpha() -> None:
    features, target = _synthetic_regression()
    model = HGBModel(max_iter=20)
    model.fit(features, target)

    model.predict_interval(features, alpha=0.1)
    lower_model_a = model._lower_model
    model.predict_interval(features, alpha=0.1)
    assert model._lower_model is lower_model_a  # same alpha: no refit

    model.predict_interval(features, alpha=0.2)
    assert model._lower_model is not lower_model_a  # different alpha: refit


@pytest.mark.req("FR-PRD-006")
@pytest.mark.skipif(HAS_TORCH, reason="torch is installed; this tests the guard itself")
def test_gru_model_raises_clear_error_without_torch() -> None:
    with pytest.raises(ImportError, match="\\[ml\\]"):
        GRUModel()


@pytest.mark.req("FR-PRD-006")
@pytest.mark.skipif(not HAS_TORCH, reason="requires the optional '[ml]' torch extra")
def test_gru_model_fit_predict_interval_contract() -> None:
    features, target = _synthetic_regression()
    model = GRUModel(hidden_dim=4, epochs=2)
    model.fit(features, target)
    point = model.predict(features)
    assert point.shape == target.shape
    lower, upper = model.predict_interval(features)
    assert np.all(lower <= upper + 1e-9)
