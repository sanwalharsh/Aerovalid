"""Smoke tests for every `report/figures.py` function (FR-RPT-002's AC): each just
needs to return a non-empty `Figure` without raising, given plausible synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from aerovalid.report import figures


def _assert_is_figure(fig: Figure) -> None:
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


@pytest.mark.req("FR-RPT-002")
def test_overlay_timeseries() -> None:
    t_s = np.linspace(0, 10, 50)
    obs = np.sin(t_s)
    sims = {"V0": np.sin(t_s) + 0.1, "V1": np.sin(t_s) - 0.1}
    _assert_is_figure(figures.overlay_timeseries(t_s, obs, sims))


@pytest.mark.req("FR-RPT-002")
def test_residual_timeseries_with_tolerance() -> None:
    t_s = np.linspace(0, 10, 50)
    residual = np.random.default_rng(0).normal(0, 1, 50)
    _assert_is_figure(figures.residual_timeseries_with_tolerance(t_s, residual, tol=2.0))


@pytest.mark.req("FR-RPT-002")
def test_error_vs_horizon() -> None:
    horizons = np.array([1.0, 5.0, 10.0, 30.0])
    rmse = np.array([0.5, 1.2, 2.0, 3.5])
    _assert_is_figure(figures.error_vs_horizon(horizons, rmse))


@pytest.mark.req("FR-RPT-002")
def test_condition_heatmap() -> None:
    pivot = pd.DataFrame(
        {"V0": [1.0, 2.0, 3.0], "V1": [0.5, 1.5, 2.5]}, index=["low", "mid", "high"]
    )
    _assert_is_figure(figures.condition_heatmap(pivot))


@pytest.mark.req("FR-RPT-002")
def test_bland_altman() -> None:
    rng = np.random.default_rng(0)
    obs = rng.normal(100, 5, 200)
    sim = obs + rng.normal(0, 2, 200)
    _assert_is_figure(figures.bland_altman(obs, sim))


@pytest.mark.req("FR-RPT-002")
def test_qq_plot() -> None:
    residual = np.random.default_rng(0).normal(0, 1, 200)
    _assert_is_figure(figures.qq_plot(residual))


@pytest.mark.req("FR-RPT-002")
def test_ranking_forest_plot() -> None:
    variants = ["V0", "V1", "V2"]
    mean_rank = np.array([2.1, 1.0, 3.0])
    ci_low = np.array([1.5, 0.8, 2.5])
    ci_high = np.array([2.8, 1.3, 3.5])
    _assert_is_figure(figures.ranking_forest_plot(variants, mean_rank, ci_low, ci_high))


@pytest.mark.req("FR-RPT-002")
def test_ranking_forest_plot_tolerates_a_ci_bound_crossing_the_point_estimate() -> None:
    """Bootstrap CI bounds are empirical quantiles, not derived from `mean_rank`
    itself, so one can land on the "wrong" side of it with few resamples or ties --
    a real case hit on a full-scale demo run, not a synthetic edge case."""
    variants = ["V0", "V1"]
    mean_rank = np.array([1.0, 2.0])
    ci_low = np.array([1.2, 1.5])  # ci_low > mean_rank for V0: a naive diff is negative
    ci_high = np.array([1.8, 2.5])
    _assert_is_figure(figures.ranking_forest_plot(variants, mean_rank, ci_low, ci_high))


@pytest.mark.req("FR-RPT-002")
def test_calibration_plot() -> None:
    nominal = np.array([0.5, 0.8, 0.9])
    empirical = np.array([0.48, 0.82, 0.88])
    _assert_is_figure(figures.calibration_plot(nominal, empirical))


@pytest.mark.req("FR-RPT-002")
def test_roc_pr_curve() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 200)
    labels = scores + rng.normal(0, 0.5, 200) > 0.0
    fig = figures.roc_pr_curve(scores, labels)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


@pytest.mark.req("FR-RPT-002")
def test_detection_timeline() -> None:
    t_s = np.linspace(0, 100, 200)
    statistic = np.abs(np.sin(t_s / 10.0)) * 5.0
    _assert_is_figure(
        figures.detection_timeline(
            t_s, statistic, threshold=4.0, t_on_s=40.0, t_alarm_s=42.0, t_exc_s=45.0
        )
    )


@pytest.mark.req("FR-RPT-002")
def test_detection_timeline_with_no_markers() -> None:
    t_s = np.linspace(0, 100, 200)
    statistic = np.zeros(200)
    _assert_is_figure(
        figures.detection_timeline(
            t_s, statistic, threshold=4.0, t_on_s=None, t_alarm_s=None, t_exc_s=None
        )
    )


@pytest.mark.req("FR-RPT-002")
def test_parameter_ci_plot() -> None:
    param_names = ["CL0", "CD0"]
    fitted_values = np.array([0.25, 0.03])
    ci_95 = {"CL0": (0.22, 0.28), "CD0": (0.025, 0.035)}
    _assert_is_figure(figures.parameter_ci_plot(param_names, fitted_values, ci_95))


@pytest.mark.req("FR-RPT-002")
def test_trajectory_3d_with_error() -> None:
    rng = np.random.default_rng(0)
    n = 100
    x_m = np.cumsum(rng.normal(0, 10, n))
    y_m = np.cumsum(rng.normal(0, 10, n))
    alt_m = 1500.0 + np.cumsum(rng.normal(0, 1, n))
    error = np.abs(rng.normal(0, 1, n))
    fig = figures.trajectory_3d_with_error(x_m, y_m, alt_m, error)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
