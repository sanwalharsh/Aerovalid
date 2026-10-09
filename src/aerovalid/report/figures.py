"""Report figure library (FR-RPT-002): every function takes data only (no global
state) and returns a `matplotlib.figure.Figure`; saving is the caller's job
(`report/builder.py` or the CLI). Each has a smoke test in `test_figures.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


def _new_figure(figsize: tuple[float, float] = (8.0, 4.5)) -> tuple[Figure, Axes]:
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    return fig, ax


def overlay_timeseries(
    t_s: NDArray[np.float64],
    obs: NDArray[np.float64],
    sim_by_variant: dict[str, NDArray[np.float64]],
) -> Figure:
    """Observed vs every variant's simulated trace for one signal (FR-RPT-002)."""
    fig, ax = _new_figure()
    ax.plot(t_s, obs, color="black", linewidth=1.5, label="observed")
    for name, sim in sim_by_variant.items():
        ax.plot(t_s, sim, linewidth=1.0, alpha=0.8, label=name)
    ax.set_xlabel("t [s]")
    ax.legend(fontsize="small")
    return fig


def residual_timeseries_with_tolerance(
    t_s: NDArray[np.float64], residual: NDArray[np.float64], tol: float
) -> Figure:
    """Residual trace with its +-tolerance band shaded (FR-RPT-002)."""
    fig, ax = _new_figure()
    ax.plot(t_s, residual, color="tab:blue", linewidth=1.0)
    ax.axhspan(-tol, tol, color="tab:green", alpha=0.15, label=f"tol=+-{tol:g}")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("residual")
    ax.legend(fontsize="small")
    return fig


def error_vs_horizon(
    horizons_s: NDArray[np.float64], rmse_by_horizon: NDArray[np.float64]
) -> Figure:
    """RMSE growth with forecast/replay horizon (FR-RPT-002)."""
    fig, ax = _new_figure()
    ax.plot(horizons_s, rmse_by_horizon, marker="o")
    ax.set_xlabel("horizon [s]")
    ax.set_ylabel("RMSE")
    return fig


def condition_heatmap(pivot: pd.DataFrame) -> Figure:
    """2-D condition-bin heatmap (e.g. altitude x Mach band) of a metric value
    (FR-RPT-002); `pivot` is already a `row x column` numeric table."""
    fig, ax = _new_figure((6.0, 5.0))
    image = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)), labels=[str(r) for r in pivot.index])
    fig.colorbar(image, ax=ax)
    return fig


def bland_altman(obs: NDArray[np.float64], sim: NDArray[np.float64]) -> Figure:
    """Bland-Altman agreement plot: mean vs difference, with +-1.96 SD limits
    (FR-RPT-002)."""
    mean = (obs + sim) / 2.0
    diff = obs - sim
    bias, sd = float(np.mean(diff)), float(np.std(diff))
    fig, ax = _new_figure()
    ax.scatter(mean, diff, s=8, alpha=0.5)
    ax.axhline(bias, color="tab:red", linewidth=1.0, label=f"bias={bias:.3g}")
    for sign in (1, -1):
        ax.axhline(bias + sign * 1.96 * sd, color="tab:red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("mean(obs, sim)")
    ax.set_ylabel("obs - sim")
    ax.legend(fontsize="small")
    return fig


def qq_plot(residual: NDArray[np.float64]) -> Figure:
    """Quantile-quantile plot of `residual` against the standard normal (FR-RPT-002)."""
    from scipy.stats import probplot

    fig, ax = _new_figure((5.0, 5.0))
    probplot(residual, dist="norm", plot=ax)
    ax.set_title("")
    return fig


def ranking_forest_plot(
    variants: list[str],
    mean_rank: NDArray[np.float64],
    rank_ci_low: NDArray[np.float64],
    rank_ci_high: NDArray[np.float64],
) -> Figure:
    """Variant ranking forest plot: mean rank with bootstrap CIs (FR-RPT-002)."""
    fig, ax = _new_figure((6.0, max(2.5, 0.5 * len(variants))))
    y = np.arange(len(variants))
    lower_err = mean_rank - rank_ci_low
    upper_err = rank_ci_high - mean_rank
    ax.errorbar(mean_rank, y, xerr=[lower_err, upper_err], fmt="o", capsize=3)
    ax.set_yticks(y, labels=variants)
    ax.invert_yaxis()
    ax.invert_xaxis()  # rank 1 (best) on the right
    ax.set_xlabel("mean rank (1 = best)")
    return fig


def calibration_plot(
    nominal_coverage: NDArray[np.float64], empirical_coverage: NDArray[np.float64]
) -> Figure:
    """Reliability curve: nominal vs empirical prediction-interval coverage
    (FR-RPT-002); a perfectly calibrated model lies on the diagonal."""
    fig, ax = _new_figure((5.0, 5.0))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8, label="perfect")
    ax.plot(nominal_coverage, empirical_coverage, marker="o")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small")
    return fig


def roc_pr_curve(scores: NDArray[np.float64], labels: NDArray[np.bool_]) -> Figure:
    """Side-by-side ROC and precision-recall curves (FR-RPT-002)."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig = Figure(figsize=(9.0, 4.0))
    roc_ax, pr_ax = fig.subplots(1, 2)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_ax.plot(fpr, tpr)
    roc_ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    roc_ax.set_xlabel("FPR")
    roc_ax.set_ylabel("TPR")
    roc_ax.set_title("ROC")

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_ax.plot(recall, precision)
    pr_ax.set_xlabel("recall")
    pr_ax.set_ylabel("precision")
    pr_ax.set_title("PR")
    return fig


def detection_timeline(
    t_s: NDArray[np.float64],
    statistic: NDArray[np.float64],
    threshold: float,
    t_on_s: float | None,
    t_alarm_s: float | None,
    t_exc_s: float | None,
) -> Figure:
    """Detector statistic over time with onset/alarm/exceedance markers (FR-RPT-002)."""
    fig, ax = _new_figure()
    ax.plot(t_s, statistic, color="tab:blue", linewidth=1.0)
    ax.axhline(threshold, color="grey", linestyle="--", linewidth=0.8, label="threshold")
    markers = {"onset": t_on_s, "alarm": t_alarm_s, "exceedance": t_exc_s}
    colors = {"onset": "tab:orange", "alarm": "tab:red", "exceedance": "tab:purple"}
    for label, t in markers.items():
        if t is not None:
            ax.axvline(t, color=colors[label], linestyle=":", linewidth=1.2, label=label)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("detector statistic")
    ax.legend(fontsize="small")
    return fig


def parameter_ci_plot(
    param_names: list[str],
    fitted_values: NDArray[np.float64],
    ci_95: dict[str, tuple[float, float]],
) -> Figure:
    """Fitted parameter point estimates with 95% Wald CIs (FR-RPT-002)."""
    fig, ax = _new_figure((6.0, max(2.5, 0.5 * len(param_names))))
    y = np.arange(len(param_names))
    lower = np.array([fitted_values[i] - ci_95[name][0] for i, name in enumerate(param_names)])
    upper = np.array([ci_95[name][1] - fitted_values[i] for i, name in enumerate(param_names)])
    ax.errorbar(fitted_values, y, xerr=[lower, upper], fmt="o", capsize=3)
    ax.set_yticks(y, labels=param_names)
    ax.invert_yaxis()
    ax.set_xlabel("fitted value")
    return fig


def trajectory_3d_with_error(
    x_m: NDArray[np.float64],
    y_m: NDArray[np.float64],
    alt_m: NDArray[np.float64],
    error: NDArray[np.float64],
) -> Figure:
    """3-D ground-track + altitude trajectory, coloured by (e.g. absolute) error
    (FR-RPT-002)."""
    fig = Figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(x_m, y_m, alt_m, c=error, cmap="inferno", s=6)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("alt [m]")
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="error")
    return fig
