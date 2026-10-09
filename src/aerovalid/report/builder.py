"""Report builder (FR-RPT-001): reads whichever phase artifacts exist for a run,
builds the figures/tables those phases support, and renders the Jinja2 HTML report.
A phase that didn't run for this study (e.g. no `fsa detect`) simply leaves its
section showing "no data for this run" rather than failing -- the report always
reflects what's actually on disk, never an assumed full pipeline (CLAUDE.md §2).

**Scope note:** the figure library (`report/figures.py`) has 12 functions, one per
FR-RPT-002 figure type, each independently smoke-tested; this builder wires in the 6
that a typical run's artifacts directly support (ranking forest, overlay + residual
time series, condition heatmap, detection timeline, parameter CI). Bland-Altman,
Q-Q, error-vs-horizon, calibration, ROC/PR, and the 3-D trajectory plot are available
to call directly (e.g. from a notebook or a future report extension) -- see
DECISIONS.md ADR-021.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jinja2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from aerovalid.core.manifest import read_manifest
from aerovalid.core.signals import get_signal
from aerovalid.report import figures
from aerovalid.report.summary import SummaryFacts, build_executive_summary, build_summary_json
from aerovalid.report.tables import write_table

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _fig_to_base64(fig: Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _read_json_or_none(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _read_json_list_or_none(path: Path) -> list[dict[str, object]] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _to_html(df: pd.DataFrame, fmt: str = "%.4g") -> str:
    """`df.to_html` with a numeric format string (pandas-stubs types `float_format`
    as `Callable[[float], str]` only, even though pandas itself also accepts a
    `%`-style string at runtime)."""
    html: str = df.to_html(index=False, float_format=lambda x: fmt % x)
    return html


@dataclass(frozen=True)
class ReportResult:
    html_path: Path
    summary_json_path: Path
    figures_dir: Path
    tables_dir: Path


def _variant_comparison_section(
    run_dir: Path, index: dict[str, list[str]], figures_dir: Path, tables_dir: Path
) -> dict[str, object]:
    ranking = _read_parquet_or_empty(run_dir / "validation" / "variant_ranking.parquet")
    metrics_signal = _read_parquet_or_empty(run_dir / "validation" / "metrics_signal.parquet")
    section: dict[str, object] = {
        "ranking_table": None,
        "ranking_figure": None,
        "metrics_table": None,
    }

    if not ranking.empty:
        fig = figures.ranking_forest_plot(
            ranking["variant"].tolist(),
            ranking["mean_rank"].to_numpy(),
            ranking["rank_ci_low"].to_numpy(),
            ranking["rank_ci_high"].to_numpy(),
        )
        section["ranking_figure"] = _fig_to_base64(fig)
        write_table(ranking, tables_dir, "variant_ranking")
        section["ranking_table"] = _to_html(ranking, "%.3g")

    if not metrics_signal.empty:
        write_table(metrics_signal, tables_dir, "metrics_signal")
        section["metrics_table"] = _to_html(metrics_signal, "%.3g")

    signal = "alt_m"
    sims: dict[str, np.ndarray] = {}
    obs, t_s = None, None
    if index.get("flights") and index.get("variants"):
        flight_id = index["flights"][0]
        for variant in index["variants"]:
            path = run_dir / "aligned" / variant / f"{flight_id}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if f"obs_{signal}" not in df.columns:
                continue
            t_s = df["t_s"].to_numpy()
            obs = df[f"obs_{signal}"].to_numpy()
            sims[variant] = df[f"sim_{signal}"].to_numpy()

    if obs is not None and t_s is not None and sims:
        overlay_fig = figures.overlay_timeseries(t_s, obs, sims)
        section["overlay_figure"] = _fig_to_base64(overlay_fig)
        first_variant = next(iter(sims))
        tol = get_signal(signal).default_tolerance or 1.0
        residual_fig = figures.residual_timeseries_with_tolerance(
            t_s, obs - sims[first_variant], tol
        )
        section["residual_figure"] = _fig_to_base64(residual_fig)
        section["overlay_signal"] = signal
        section["overlay_variant"] = first_variant

    return section


def _condition_section(run_dir: Path, tables_dir: Path) -> dict[str, object]:
    heatmaps = _read_parquet_or_empty(run_dir / "validation" / "condition_heatmaps.parquet")
    section: dict[str, object] = {"heatmap_figure": None}
    if heatmaps.empty:
        return section

    write_table(heatmaps, tables_dir, "condition_heatmaps")
    first_column = heatmaps["cond_column"].iloc[0]
    subset = heatmaps[heatmaps["cond_column"] == first_column]
    pivot = subset.pivot_table(index="bin", columns="variant", values="mean")
    if not pivot.empty:
        section["heatmap_figure"] = _fig_to_base64(figures.condition_heatmap(pivot))
        section["heatmap_column"] = str(first_column)
    return section


def _prediction_section(run_dir: Path, tables_dir: Path) -> dict[str, object]:
    cv_results = _read_parquet_or_empty(run_dir / "predict" / "cv_results.parquet")
    intervals = _read_parquet_or_empty(run_dir / "predict" / "intervals.parquet")
    section: dict[str, object] = {"cv_table": None, "intervals_table": None}
    if not cv_results.empty:
        write_table(cv_results, tables_dir, "predict_cv_results")
        section["cv_table"] = _to_html(cv_results, "%.4g")
    if not intervals.empty:
        write_table(intervals, tables_dir, "predict_intervals")
        section["intervals_table"] = _to_html(intervals, "%.4g")
    return section


def _detection_section(run_dir: Path, tables_dir: Path) -> dict[str, object]:
    metrics = _read_parquet_or_empty(run_dir / "detect" / "detection_metrics.parquet")
    section: dict[str, object] = {"metrics_table": None}
    if not metrics.empty:
        write_table(metrics, tables_dir, "detection_metrics")
        section["metrics_table"] = _to_html(metrics, "%.4g")
    return section


def _identification_section(
    run_dir: Path, tables_dir: Path, figures_dir: Path
) -> dict[str, object]:
    fit_results = _read_json_or_none(run_dir / "identify" / "fit_results.json")
    identifiability = _read_json_or_none(run_dir / "identify" / "identifiability.json")
    recommendations = _read_json_list_or_none(run_dir / "identify" / "recommendations.json")
    revalidation = _read_parquet_or_empty(run_dir / "identify" / "revalidation.parquet")

    section: dict[str, object] = {
        "fit_results": fit_results,
        "parameter_ci_figure": None,
        "recommendations": recommendations,
        "revalidation_table": None,
    }
    if fit_results is not None and identifiability is not None:
        params = cast("dict[str, float]", fit_results["params"])
        param_names = list(params)
        fitted_values = np.array([params[name] for name in param_names])
        raw_ci_95 = cast("dict[str, list[float]]", identifiability["ci_95"])
        ci_95 = {name: (bounds[0], bounds[1]) for name, bounds in raw_ci_95.items()}
        section["parameter_ci_figure"] = _fig_to_base64(
            figures.parameter_ci_plot(param_names, fitted_values, ci_95)
        )
        section["identifiability"] = identifiability
    if not revalidation.empty:
        write_table(revalidation, tables_dir, "revalidation")
        section["revalidation_table"] = _to_html(revalidation, "%.4g")
    return section


def _gather_summary_facts(
    run_id: str,
    index: dict[str, list[str]],
    run_dir: Path,
) -> SummaryFacts:
    ranking = _read_parquet_or_empty(run_dir / "validation" / "variant_ranking.parquet")
    best_variant = str(ranking.iloc[0]["variant"]) if not ranking.empty else None
    best_mean_rank = float(ranking.iloc[0]["mean_rank"]) if not ranking.empty else None

    cv_results = _read_parquet_or_empty(run_dir / "predict" / "cv_results.parquet")
    hybrid_rows = (
        cv_results[cv_results.get("task") == "hybrid"] if not cv_results.empty else cv_results
    )
    predict_reduction = (
        float(hybrid_rows["rmse_reduction_vs_sim_only_pct"].mean())
        if not hybrid_rows.empty and "rmse_reduction_vs_sim_only_pct" in hybrid_rows.columns
        else None
    )

    intervals = _read_parquet_or_empty(run_dir / "predict" / "intervals.parquet")
    predict_coverage = float(intervals["coverage"].mean()) if not intervals.empty else None

    detect_metrics = _read_parquet_or_empty(run_dir / "detect" / "detection_metrics.parquet")
    overall_detect = (
        detect_metrics[detect_metrics["fault_type"].isna()]
        if not detect_metrics.empty
        else detect_metrics
    )
    detect_recall = float(overall_detect["recall"].mean()) if not overall_detect.empty else None
    detect_far = float(overall_detect["far_per_hour"].mean()) if not overall_detect.empty else None
    detect_delay = (
        float(overall_detect["median_delay_s"].dropna().mean())
        if not overall_detect.empty and overall_detect["median_delay_s"].notna().any()
        else None
    )

    recommendations = _read_json_list_or_none(run_dir / "identify" / "recommendations.json")
    top_param: str | None = None
    top_change: float | None = None
    if recommendations:
        top = recommendations[0]
        top_param = cast(str, top["parameter"])
        top_change = cast(float, top["estimated_change_pct"])

    revalidation = _read_parquet_or_empty(run_dir / "identify" / "revalidation.parquet")
    revalidation_reduction = (
        float(revalidation["rmse_reduction_pct"].mean()) if not revalidation.empty else None
    )

    return SummaryFacts(
        run_id=run_id,
        n_flights=len(index.get("flights", [])),
        variants=index.get("variants", []),
        best_variant=best_variant,
        best_variant_mean_rank=best_mean_rank,
        predict_rmse_reduction_vs_sim_only_pct=predict_reduction,
        predict_interval_coverage=predict_coverage,
        detect_recall=detect_recall,
        detect_far_per_hour=detect_far,
        detect_median_delay_s=detect_delay,
        identify_top_parameter=top_param,
        identify_top_parameter_change_pct=top_change,
        identify_revalidation_rmse_reduction_pct=revalidation_reduction,
    )


def build_report(run_dir: Path, run_id: str) -> ReportResult:
    """
    Build the full HTML report for a run (FR-RPT-001): executive summary, data & QC,
    alignment, variant comparison, condition analysis, prediction, detection,
    identification & recommendations, and an appendix (config/manifest/methods).

    Parameters
    ----------
    run_dir : Path
        `runs/<run_id>/`.
    run_id : str

    Returns
    -------
    ReportResult
    """
    reports_dir = run_dir / "reports"
    figures_dir = run_dir / "figures"
    tables_dir = reports_dir / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    index_path = run_dir / "aligned" / "index.json"
    index: dict[str, list[str]] = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {"flights": [], "variants": []}
    )
    qc_summary = _read_parquet_or_empty(run_dir / "qc" / "qc_summary.parquet")
    sync_estimates = _read_parquet_or_empty(run_dir / "sync" / "sync_estimates.parquet")
    manifest = read_manifest(run_dir) if (run_dir / "run_manifest.json").exists() else None

    facts = _gather_summary_facts(run_id, index, run_dir)
    summary_json_path = reports_dir / "summary.json"
    summary_json_path.write_text(json.dumps(build_summary_json(facts), indent=2))
    executive_summary = build_executive_summary(facts)

    context = {
        "run_id": run_id,
        "executive_summary": executive_summary,
        "qc_table": (_to_html(qc_summary, "%.4g") if not qc_summary.empty else None),
        "sync_table": (_to_html(sync_estimates, "%.4g") if not sync_estimates.empty else None),
        "variant_comparison": _variant_comparison_section(run_dir, index, figures_dir, tables_dir),
        "condition_analysis": _condition_section(run_dir, tables_dir),
        "prediction": _prediction_section(run_dir, tables_dir),
        "detection": _detection_section(run_dir, tables_dir),
        "identification": _identification_section(run_dir, tables_dir, figures_dir),
        "manifest": manifest.model_dump() if manifest is not None else None,
    }

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
    html = env.get_template("report.html.jinja2").render(**context)
    html_path = reports_dir / "report.html"
    html_path.write_text(html)

    return ReportResult(
        html_path=html_path,
        summary_json_path=summary_json_path,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
