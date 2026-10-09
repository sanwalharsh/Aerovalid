from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.predict.explain import (
    HAS_SHAP,
    partial_dependence_curve,
    permutation_importance_table,
    shap_summary,
)
from aerovalid.predict.models.ridge import RidgeModel


def _dominant_feature_regression(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 300
    dominant = rng.normal(0.0, 1.0, n)
    noise_feature = rng.normal(0.0, 1.0, n)
    target = 5.0 * dominant + 0.01 * noise_feature + rng.normal(0.0, 0.05, n)
    features = pd.DataFrame({"dominant": dominant, "noise_feature": noise_feature})
    return features, target


@pytest.mark.req("FR-PRD-009")
def test_permutation_importance_ranks_dominant_feature_highest() -> None:
    features, target = _dominant_feature_regression()
    model = RidgeModel()
    model.fit(features, target)

    table = permutation_importance_table(model, features, target, n_repeats=5, seed=0)
    assert list(table.columns) == ["feature", "importance_mean", "importance_std"]
    assert table.iloc[0]["feature"] == "dominant"
    assert table.iloc[0]["importance_mean"] > table.iloc[1]["importance_mean"]


@pytest.mark.req("FR-PRD-009")
def test_partial_dependence_curve_is_monotonic_increasing_for_positive_coefficient() -> None:
    features, target = _dominant_feature_regression()
    model = RidgeModel()
    model.fit(features, target)

    curve = partial_dependence_curve(model, features, "dominant", grid_resolution=15)
    assert list(curve.columns) == ["value", "partial_dependence"]
    assert len(curve) <= 15
    assert np.all(np.diff(curve["partial_dependence"].to_numpy()) >= -1e-9)


@pytest.mark.req("FR-PRD-009")
@pytest.mark.skipif(HAS_SHAP, reason="shap is installed; this tests the guard itself")
def test_shap_summary_returns_none_without_shap() -> None:
    features, target = _dominant_feature_regression()
    model = RidgeModel()
    model.fit(features, target)
    assert shap_summary(model, features) is None


@pytest.mark.req("FR-PRD-009")
@pytest.mark.skipif(not HAS_SHAP, reason="requires the optional '[ml]' shap extra")
def test_shap_summary_ranks_dominant_feature_highest_when_installed() -> None:
    features, target = _dominant_feature_regression()
    model = RidgeModel()
    model.fit(features, target)
    table = shap_summary(model, features, background_samples=20)
    assert table is not None
    assert table.iloc[0]["feature"] == "dominant"
