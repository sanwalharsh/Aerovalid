"""Condition-dependence analysis (FR-VAL-007, analysis methods §6): binned heatmap
data, a gradient-boosted regressor of normalised residual magnitude with grouped CV
and permutation importance, partial dependence, Kruskal-Wallis/Dunn, and a
mixed-effects (`MixedLM`) model for condition effects adjusted for repeated flights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import statsmodels.formula.api as smf
from scipy.stats import kruskal
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import GroupKFold
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper


def condition_heatmap_data(
    df: pd.DataFrame, cond_cols: list[str], value_col: str, min_n: int = 5
) -> pd.DataFrame:
    """
    Binned (1-D or 2-D) mean/std/count of `value_col` over `cond_cols` (FR-VAL-007).

    Parameters
    ----------
    df : pandas.DataFrame
        Row-level or per-(flight, variant, signal) data with the condition columns
        (e.g. `cond_alt_band`, `cond_mach_band`) and `value_col` (e.g. normalised
        `|residual| / tolerance`).
    cond_cols : list[str]
        One or two condition columns to bin over.
    value_col : str
    min_n : int, default 5
        Cells with fewer than this many samples get `mean = NaN` (hidden, not
        dropped, so the sparse-cell structure is still visible).

    Returns
    -------
    pandas.DataFrame
        `cond_cols + ["mean", "std", "count"]`.
    """
    grouped = (
        df.groupby(cond_cols, observed=True)[value_col].agg(["mean", "std", "count"]).reset_index()
    )
    grouped.loc[grouped["count"] < min_n, "mean"] = np.nan
    return grouped


@dataclass
class ConditionImportanceResult:
    """Fitted condition-importance model and its diagnostics."""

    model: HistGradientBoostingRegressor
    importance: pd.DataFrame
    cv_r2: float


def fit_condition_importance(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    group_column: str = "flight_id",
    n_splits: int = 5,
    seed: int = 0,
) -> ConditionImportanceResult:
    """
    Fit `HistGradientBoostingRegressor(target ~ condition features)` with grouped CV
    by flight, and compute permutation importance (FR-VAL-007).

    Parameters
    ----------
    df : pandas.DataFrame
        Must have `feature_columns` (numeric; encode categoricals upstream),
        `target_column` (e.g. `|residual| / tolerance`), and `group_column`.
    feature_columns : list[str]
    target_column : str
    group_column : str, default "flight_id"
    n_splits : int, default 5
        Capped at the number of distinct groups present.
    seed : int, default 0

    Returns
    -------
    ConditionImportanceResult
    """
    valid = df.dropna(subset=[*feature_columns, target_column])
    features = valid[feature_columns]
    target = valid[target_column]
    groups = valid[group_column]

    n_groups = groups.nunique()
    cv_scores: list[float] = []
    if n_groups >= 2:
        splitter = GroupKFold(n_splits=min(n_splits, n_groups))
        for train_idx, test_idx in splitter.split(features, target, groups):
            fold_model = HistGradientBoostingRegressor(random_state=seed)
            fold_model.fit(features.iloc[train_idx], target.iloc[train_idx])
            cv_scores.append(
                float(fold_model.score(features.iloc[test_idx], target.iloc[test_idx]))
            )

    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(features, target)
    perm = permutation_importance(model, features, target, n_repeats=10, random_state=seed)
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    return ConditionImportanceResult(
        model=model,
        importance=importance,
        cv_r2=float(np.mean(cv_scores)) if cv_scores else float("nan"),
    )


def partial_dependence_curve(
    model: HistGradientBoostingRegressor,
    features: pd.DataFrame,
    feature: str,
    grid_resolution: int = 20,
) -> pd.DataFrame:
    """1-D partial dependence of `model` on `feature` (FR-VAL-007)."""
    result = partial_dependence(model, features, [feature], grid_resolution=grid_resolution)
    return pd.DataFrame(
        {"value": result["grid_values"][0], "partial_dependence": result["average"][0]}
    )


def kruskal_wallis_across_bins(values_by_bin: dict[str, np.ndarray]) -> tuple[float, float]:
    """Kruskal-Wallis test across condition bins (flight-level medians recommended)."""
    stat, p_value = kruskal(*values_by_bin.values())
    return float(stat), float(p_value)


def dunn_posthoc(values_by_bin: dict[str, np.ndarray]) -> pd.DataFrame:
    """Dunn post-hoc test (Holm-corrected) following a significant Kruskal-Wallis test."""
    names = list(values_by_bin)
    result: pd.DataFrame = sp.posthoc_dunn(list(values_by_bin.values()), p_adjust="holm")
    result.index = names
    result.columns = names
    return result


def mixed_effects_condition_model(
    df: pd.DataFrame, formula: str, group_column: str = "flight_id"
) -> MixedLMResultsWrapper:
    """
    Fit a mixed-effects model for condition effects adjusted for repeated flights,
    e.g. `formula="log_abs_res ~ cond_mach_band"` (analysis methods §6).

    Parameters
    ----------
    df : pandas.DataFrame
    formula : str
        A `statsmodels` R-style formula referencing columns in `df`.
    group_column : str, default "flight_id"
        Random-intercept grouping variable.

    Returns
    -------
    statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper
    """
    return smf.mixedlm(formula, df, groups=df[group_column]).fit()
