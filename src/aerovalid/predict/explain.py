"""Explainability (FR-PRD-009): model-agnostic permutation importance and partial
dependence for any fitted `ResidualModel`, plus an optional (guarded) SHAP summary.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import partial_dependence, permutation_importance

from aerovalid.predict.models import ResidualModel

logger = logging.getLogger(__name__)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class _SklearnPredictShim(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """Wraps a fitted `ResidualModel` so sklearn's `permutation_importance` /
    `partial_dependence` (which expect an already-fitted sklearn-style estimator) will
    call back into it. Inherits `BaseEstimator` (for `__sklearn_tags__`, required by
    modern sklearn's `check_is_fitted`) and sets `is_fitted_` to satisfy it; `model` is
    a public attribute (not `_model`) so `BaseEstimator.get_params` can introspect it.
    """

    def __init__(self, model: ResidualModel) -> None:
        self.model = model
        self.is_fitted_ = True

    def fit(self, features: pd.DataFrame, target: np.ndarray | None = None) -> _SklearnPredictShim:
        del features, target
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features)

    def score(self, features: pd.DataFrame, target: np.ndarray) -> float:
        predicted = self.model.predict(features)
        ss_res = float(np.sum((target - predicted) ** 2))
        ss_tot = float(np.sum((target - np.mean(target)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def permutation_importance_table(
    model: ResidualModel,
    features: pd.DataFrame,
    target: np.ndarray,
    n_repeats: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Permutation importance of a fitted model's features (FR-PRD-009).

    Parameters
    ----------
    model : ResidualModel
        Already fitted.
    features : pandas.DataFrame
    target : ndarray
    n_repeats : int, default 10
    seed : int, default 0

    Returns
    -------
    pandas.DataFrame
        `feature, importance_mean, importance_std`, sorted most-important first.
    """
    shim = _SklearnPredictShim(model)
    result = permutation_importance(shim, features, target, n_repeats=n_repeats, random_state=seed)
    return pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def partial_dependence_curve(
    model: ResidualModel, features: pd.DataFrame, feature: str, grid_resolution: int = 20
) -> pd.DataFrame:
    """1-D partial dependence of a fitted model on one feature (FR-PRD-009)."""
    shim = _SklearnPredictShim(model)
    result = partial_dependence(shim, features, [feature], grid_resolution=grid_resolution)
    return pd.DataFrame(
        {"value": result["grid_values"][0], "partial_dependence": result["average"][0]}
    )


def shap_summary(
    model: ResidualModel, features: pd.DataFrame, background_samples: int = 100, seed: int = 0
) -> pd.DataFrame | None:
    """
    Mean absolute SHAP value per feature (FR-PRD-009, optional). Returns `None` (with
    a warning logged, never raising) if the optional `shap` package isn't installed.

    Parameters
    ----------
    model : ResidualModel
        Already fitted.
    features : pandas.DataFrame
    background_samples : int, default 100
        Background sample size for the kernel explainer.
    seed : int, default 0

    Returns
    -------
    pandas.DataFrame or None
        `feature, mean_abs_shap`, sorted most-important first; `None` if `shap` is
        not installed.
    """
    if not HAS_SHAP:
        logger.warning("shap is not installed ('[ml]' extra); skipping SHAP summary.")
        return None

    rng = np.random.default_rng(seed)
    background_idx = rng.choice(
        len(features), size=min(background_samples, len(features)), replace=False
    )
    explainer = shap.KernelExplainer(model.predict, features.iloc[background_idx])
    shap_values = explainer.shap_values(features, nsamples="auto")
    return pd.DataFrame(
        {"feature": features.columns, "mean_abs_shap": np.mean(np.abs(shap_values), axis=0)}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
