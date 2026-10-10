"""Interactive Plotly charts for the dashboard (FR-RPT-004), following the
`dataviz` skill's method: form picked by the data's job, color assigned by that
job (categorical for identity, sequential for magnitude), hover as a first-class
part of the design, non-negative error bars, a legend whenever there's >= 2 series.

These are dashboard-only, separate from `report/figures.py`'s static matplotlib
figures -- the static HTML report benefits from self-contained images; the live
dashboard benefits from Plotly's hover/zoom/pan. Same underlying data, two
renderers for two different jobs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray

from aerovalid.app.theme import (
    CATEGORICAL,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    SEQUENTIAL_BLUE,
    STATUS_CRITICAL,
    STATUS_GOOD,
    STATUS_SERIOUS,
    variant_color,
)

_LAYOUT_DEFAULTS: dict[str, object] = {
    "plot_bgcolor": "#fcfcfb",
    "paper_bgcolor": "#fcfcfb",
    "font": {"color": INK_PRIMARY, "family": "system-ui, -apple-system, sans-serif"},
    "margin": {"l": 50, "r": 20, "t": 30, "b": 40},
}


def _apply_layout(fig: go.Figure, **overrides: object) -> go.Figure:
    """Apply the shared layout defaults, then any per-chart overrides (an override
    with the same key, e.g. `margin`, replaces the default outright rather than
    raising -- a plain `**dict, **dict` double-unpack errors on any overlapping key)."""
    fig.update_layout(**{**_LAYOUT_DEFAULTS, **overrides})
    fig.update_xaxes(gridcolor=GRIDLINE, zeroline=False)
    fig.update_yaxes(gridcolor=GRIDLINE, zeroline=False)
    return fig


def overlay_timeseries(
    t_s: NDArray[np.float64],
    obs: NDArray[np.float64],
    sim_by_variant: dict[str, NDArray[np.float64]],
) -> go.Figure:
    """Observed vs every variant's simulated trace -- one series is identity
    (categorical, color follows the variant), one shared tooltip per X."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_s,
            y=obs,
            mode="lines",
            name="observed",
            line={"width": 2, "color": INK_PRIMARY},
        )
    )
    for name, sim in sim_by_variant.items():
        fig.add_trace(
            go.Scatter(
                x=t_s,
                y=sim,
                mode="lines",
                name=name,
                line={"width": 2, "color": variant_color(name)},
            )
        )
    _apply_layout(fig, hovermode="x unified", xaxis_title="t [s]", legend_title="")
    return fig


def residual_with_tolerance(
    t_s: NDArray[np.float64], residual: NDArray[np.float64], tol: float
) -> go.Figure:
    """Residual trace with its +-tolerance band -- a single series, no legend
    needed (the title/axis already say what's plotted)."""
    fig = go.Figure()
    fig.add_hrect(y0=-tol, y1=tol, fillcolor=STATUS_GOOD, opacity=0.1, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=t_s,
            y=residual,
            mode="lines",
            name="residual",
            line={"width": 2, "color": CATEGORICAL[0]},
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_color=INK_MUTED, line_width=1)
    _apply_layout(fig, hovermode="x", xaxis_title="t [s]", yaxis_title="residual")
    return fig


def ranking_forest(
    variants: list[str],
    mean_rank: NDArray[np.float64],
    rank_ci_low: NDArray[np.float64],
    rank_ci_high: NDArray[np.float64],
) -> go.Figure:
    """Variant ranking with bootstrap CIs. Error magnitudes are clipped to
    non-negative: `rank_ci_low`/`rank_ci_high` are empirical bootstrap quantiles,
    not derived from `mean_rank` itself, so a bound can land on the "wrong" side of
    the point estimate with few resamples or ties (hit on a real full-scale run,
    not a contrived case -- see DECISIONS.md ADR-023's sibling fix in
    `report/figures.py`)."""
    lower = np.clip(mean_rank - rank_ci_low, 0.0, None)
    upper = np.clip(rank_ci_high - mean_rank, 0.0, None)
    colors = [variant_color(v) for v in variants]
    fig = go.Figure(
        go.Scatter(
            x=mean_rank,
            y=variants,
            mode="markers",
            marker={"size": 10, "color": colors, "line": {"width": 2, "color": "#fcfcfb"}},
            error_x={"type": "data", "symmetric": False, "array": upper, "arrayminus": lower},
            showlegend=False,
        )
    )
    _apply_layout(fig, xaxis_title="mean rank (1 = best)", height=max(220, 60 * len(variants)))
    fig.update_xaxes(autorange="reversed")
    return fig


def condition_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """2-D condition-bin heatmap of a magnitude metric -- sequential (one hue,
    light -> dark), never a rainbow."""
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(dtype=np.float64),
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
            hoverongaps=False,
            colorbar={"title": ""},
        )
    )
    _apply_layout(fig)
    return fig


def parameter_ci(
    param_names: list[str],
    fitted_values: NDArray[np.float64],
    ci_95: dict[str, tuple[float, float]],
) -> go.Figure:
    """Fitted parameter point estimates with 95% Wald CIs; non-negative error
    clipping as in `ranking_forest` (defensive, Wald CIs are symmetric by
    construction, but a shared contract is one less thing to get wrong twice)."""
    lower = np.clip(
        [fitted_values[i] - ci_95[name][0] for i, name in enumerate(param_names)], 0.0, None
    )
    upper = np.clip(
        [ci_95[name][1] - fitted_values[i] for i, name in enumerate(param_names)], 0.0, None
    )
    fig = go.Figure(
        go.Scatter(
            x=fitted_values,
            y=param_names,
            mode="markers",
            marker={"size": 10, "color": CATEGORICAL[0], "line": {"width": 2, "color": "#fcfcfb"}},
            error_x={"type": "data", "symmetric": False, "array": upper, "arrayminus": lower},
            showlegend=False,
        )
    )
    _apply_layout(fig, xaxis_title="fitted value", height=max(220, 60 * len(param_names)))
    return fig


def rmse_reduction_bar(
    variants: list[str], targets: list[str], reduction_pct: NDArray[np.float64]
) -> go.Figure:
    """Hybrid-correction RMSE reduction vs sim-only, one bar per `(variant,
    target)` -- magnitude compared across named categories, so each bar is its
    own color (color follows the variant, the entity the story is about)."""
    labels = [f"{v} / {t}" for v, t in zip(variants, targets, strict=True)]
    colors = [variant_color(v) for v in variants]
    fig = go.Figure(
        go.Bar(
            x=reduction_pct,
            y=labels,
            orientation="h",
            marker={"color": colors},
            text=[f"{v:+.0f}%" for v in reduction_pct],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color=INK_MUTED, line_width=1)
    _apply_layout(
        fig,
        xaxis_title="RMSE reduction vs sim-only [%]",
        height=max(220, 36 * len(labels)),
        showlegend=False,
        margin={"l": 50, "r": 60, "t": 30, "b": 40},  # headroom for outside bar-end labels
    )
    axis_max = float(np.max(reduction_pct)) if len(reduction_pct) else 0.0
    axis_min = float(np.min(reduction_pct)) if len(reduction_pct) else 0.0
    fig.update_xaxes(range=[min(0.0, axis_min * 1.15), max(axis_max * 1.2, 1.0)])
    return fig


def recall_bar(detectors: list[str], recall: NDArray[np.float64]) -> go.Figure:
    """Per-detector recall. Paired with `far_bar` as two side-by-side charts, not
    one dual-axis chart -- recall (a fraction) and FAR (a rate) are different
    scales, and combining them on two y-axes would invent a correlation that
    isn't in the data."""
    fig = go.Figure(go.Bar(x=detectors, y=recall, marker_color=STATUS_GOOD))
    _apply_layout(fig, yaxis_title="recall", yaxis_range=[0, 1], showlegend=False)
    return fig


def far_bar(detectors: list[str], far: NDArray[np.float64], target: float = 1.0) -> go.Figure:
    colors = [STATUS_GOOD if f <= target else STATUS_SERIOUS for f in far]
    fig = go.Figure(go.Bar(x=detectors, y=far, marker_color=colors))
    fig.add_hline(
        y=target,
        line_color=STATUS_CRITICAL,
        line_dash="dot",
        annotation_text=f"target <= {target:g}/hr",
    )
    _apply_layout(fig, yaxis_title="false alarms / flight-hour", showlegend=False)
    return fig
