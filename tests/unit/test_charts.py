"""Smoke tests for `app/charts.py`'s Plotly chart builders: each must return a
non-empty `Figure` without raising, given plausible synthetic data (mirrors
`test_figures.py`'s pattern for the static matplotlib figures)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure

from aerovalid.app import charts


def _assert_is_figure(fig: Figure) -> None:
    assert isinstance(fig, Figure)
    assert len(fig.data) >= 1


@pytest.mark.req("FR-RPT-004")
def test_overlay_timeseries() -> None:
    t_s = np.linspace(0, 10, 50)
    obs = np.sin(t_s)
    sims = {"V0": np.sin(t_s) + 0.1, "V1": np.sin(t_s) - 0.1}
    _assert_is_figure(charts.overlay_timeseries(t_s, obs, sims))


@pytest.mark.req("FR-RPT-004")
def test_residual_with_tolerance() -> None:
    t_s = np.linspace(0, 10, 50)
    residual = np.random.default_rng(0).normal(0, 1, 50)
    _assert_is_figure(charts.residual_with_tolerance(t_s, residual, tol=2.0))


@pytest.mark.req("FR-RPT-004")
def test_ranking_forest() -> None:
    variants = ["V0", "V1", "V2"]
    mean_rank = np.array([2.1, 1.0, 3.0])
    ci_low = np.array([1.5, 0.8, 2.5])
    ci_high = np.array([2.8, 1.3, 3.5])
    _assert_is_figure(charts.ranking_forest(variants, mean_rank, ci_low, ci_high))


@pytest.mark.req("FR-RPT-004")
def test_ranking_forest_tolerates_a_ci_bound_crossing_the_point_estimate() -> None:
    """Same real-world edge case ADR-023 fixed in report/figures.py: a bootstrap CI
    bound can land on the "wrong" side of the point estimate."""
    variants = ["V0", "V1"]
    mean_rank = np.array([1.0, 2.0])
    ci_low = np.array([1.2, 1.5])  # ci_low > mean_rank for V0
    ci_high = np.array([1.8, 2.5])
    _assert_is_figure(charts.ranking_forest(variants, mean_rank, ci_low, ci_high))


@pytest.mark.req("FR-RPT-004")
def test_condition_heatmap() -> None:
    pivot = pd.DataFrame(
        {"V0": [1.0, 2.0, 3.0], "V1": [0.5, 1.5, 2.5]}, index=["low", "mid", "high"]
    )
    _assert_is_figure(charts.condition_heatmap(pivot))


@pytest.mark.req("FR-RPT-004")
def test_parameter_ci() -> None:
    param_names = ["CL0", "CD0"]
    fitted_values = np.array([0.25, 0.03])
    ci_95 = {"CL0": (0.22, 0.28), "CD0": (0.025, 0.035)}
    _assert_is_figure(charts.parameter_ci(param_names, fitted_values, ci_95))


@pytest.mark.req("FR-RPT-004")
def test_rmse_reduction_bar() -> None:
    _assert_is_figure(
        charts.rmse_reduction_bar(["V0", "V1"], ["alt_m", "alt_m"], np.array([25.0, 40.0]))
    )


@pytest.mark.req("FR-RPT-004")
def test_rmse_reduction_bar_handles_empty_input() -> None:
    _assert_is_figure(charts.rmse_reduction_bar([], [], np.array([])))


@pytest.mark.req("FR-RPT-004")
def test_recall_bar() -> None:
    _assert_is_figure(charts.recall_bar(["zscore", "cusum"], np.array([0.9, 0.8])))


@pytest.mark.req("FR-RPT-004")
def test_far_bar() -> None:
    _assert_is_figure(charts.far_bar(["zscore", "cusum"], np.array([0.5, 1.2]), target=1.0))
