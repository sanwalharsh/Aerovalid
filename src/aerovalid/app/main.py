"""Streamlit dashboard (FR-RPT-004): an Overview scorecard plus one page per
pipeline phase (Variant comparison, Condition analysis, Prediction, Detection,
Identification & recommendations, Compare runs), navigable from the sidebar.
Reads whichever artifacts exist for the selected run(s); a phase that didn't run
shows "no data", never a crash. Shares its headline numbers with the static HTML
report (`report/summary.py::gather_summary_facts`), so the two always agree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st

from aerovalid.app import charts
from aerovalid.app.theme import dashboard_css
from aerovalid.report.summary import SummaryFacts, build_executive_summary, gather_summary_facts

_RUNS_DIR = Path("runs")


def _list_runs() -> list[str]:
    if not _RUNS_DIR.exists():
        return []
    return sorted(p.name for p in _RUNS_DIR.iterdir() if p.is_dir())


def _read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _read_json_dict_or_none(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _read_json_list_or_none(path: Path) -> list[dict[str, object]] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _current_run_dir() -> Path:
    return cast(Path, st.session_state["run_dir"])


def _current_index() -> dict[str, list[str]]:
    return cast(dict[str, list[str]], st.session_state["index"])


def _current_facts() -> SummaryFacts:
    return cast(SummaryFacts, st.session_state["facts"])


def _kpi_tile(label: str, value: str) -> str:
    return (
        f'<div class="av-kpi-tile"><div class="av-kpi-label">{label}</div>'
        f'<div class="av-kpi-value">{value}</div></div>'
    )


def _page_overview() -> None:
    facts = _current_facts()
    run_dir = _current_run_dir()

    st.header("Overview")
    st.caption(
        "AeroValidate compares flight-simulation variants against telemetry, "
        "predicts and detects deviations, and recommends model refinements. "
        "This page is the scorecard; the pages in the sidebar are the evidence."
    )

    tiles = [
        _kpi_tile("Flights", str(facts.n_flights)),
        _kpi_tile("Variants", str(len(facts.variants))),
    ]
    if facts.best_variant is not None and facts.best_variant_mean_rank is not None:
        tiles.append(_kpi_tile("Best variant", facts.best_variant))
        tiles.append(_kpi_tile("Best variant's mean rank", f"{facts.best_variant_mean_rank:.2f}"))
    if facts.predict_rmse_reduction_vs_sim_only_pct is not None:
        tiles.append(
            _kpi_tile(
                "Hybrid correction vs. sim-only",
                f"{facts.predict_rmse_reduction_vs_sim_only_pct:+.0f}% RMSE",
            )
        )
    if facts.predict_interval_coverage is not None:
        coverage_pct = 100 * facts.predict_interval_coverage
        tiles.append(_kpi_tile("Prediction interval coverage", f"{coverage_pct:.0f}%"))
    if facts.detect_recall is not None:
        tiles.append(_kpi_tile("Detection recall", f"{100 * facts.detect_recall:.0f}%"))
    if facts.detect_far_per_hour is not None:
        tiles.append(_kpi_tile("False alarms / flight-hour", f"{facts.detect_far_per_hour:.2f}"))

    st.markdown(f'<div class="av-kpi-row">{"".join(tiles)}</div>', unsafe_allow_html=True)

    st.subheader("Executive summary")
    statements = build_executive_summary(facts)
    for statement in statements:
        st.markdown(f"- {statement.text}  \n  `{statement.evidence_path}`")

    with st.expander("About this tool"):
        st.markdown(
            "AeroValidate is a research pipeline, not a certified V&V tool (see "
            "`docs/00_vision.md`): it screens simulation-variant fidelity against "
            "telemetry, using a synthetic truth model with known, injected flaws so "
            "every method here can be checked against ground truth. Every number on "
            "this dashboard traces back to an artifact under "
            f"`{run_dir}` -- the path is shown under each executive-summary line."
        )


def _page_variant_comparison() -> None:
    run_dir, index = _current_run_dir(), _current_index()
    st.header("Variant comparison")

    ranking = _read_parquet_or_empty(run_dir / "validation" / "variant_ranking.parquet")
    if not ranking.empty:
        st.plotly_chart(
            charts.ranking_forest(
                ranking["variant"].tolist(),
                ranking["mean_rank"].to_numpy(dtype=np.float64),
                ranking["rank_ci_low"].to_numpy(dtype=np.float64),
                ranking["rank_ci_high"].to_numpy(dtype=np.float64),
            ),
            use_container_width=True,
        )
        st.dataframe(ranking, use_container_width=True)
    else:
        st.info("No variant ranking for this run.")

    flights, variants = index.get("flights", []), index.get("variants", [])
    if not flights or not variants:
        return
    st.subheader("Overlay & residual explorer")
    flight_id = st.selectbox("Flight", flights)
    signal = st.selectbox("Signal", ["alt_m", "vel_air_mps", "gamma_rad", "chi_rad"])

    sims: dict[str, np.ndarray] = {}
    obs, t_s = None, None
    for variant in variants:
        path = run_dir / "aligned" / variant / f"{flight_id}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if f"obs_{signal}" not in df.columns:
            continue
        t_s = df["t_s"].to_numpy()
        obs = df[f"obs_{signal}"].to_numpy()
        sims[variant] = df[f"sim_{signal}"].to_numpy()

    if obs is None or t_s is None or not sims:
        st.info(f"No `{signal}` data for flight {flight_id}.")
        return

    st.plotly_chart(charts.overlay_timeseries(t_s, obs, sims), use_container_width=True)
    residual_variant = st.selectbox("Residual variant", list(sims))
    tol = st.number_input("Tolerance", value=5.0, min_value=0.0)
    st.plotly_chart(
        charts.residual_with_tolerance(t_s, obs - sims[residual_variant], tol),
        use_container_width=True,
    )


def _page_condition_analysis() -> None:
    run_dir = _current_run_dir()
    st.header("Condition analysis")
    heatmaps = _read_parquet_or_empty(run_dir / "validation" / "condition_heatmaps.parquet")
    if heatmaps.empty:
        st.info("No condition analysis for this run.")
        return
    cond_column = st.selectbox("Condition column", sorted(heatmaps["cond_column"].unique()))
    subset = heatmaps[heatmaps["cond_column"] == cond_column]
    pivot = subset.pivot_table(index="bin", columns="variant", values="mean")
    if not pivot.empty:
        st.plotly_chart(charts.condition_heatmap(pivot), use_container_width=True)
    st.dataframe(subset, use_container_width=True)


def _page_prediction() -> None:
    run_dir = _current_run_dir()
    st.header("Prediction")
    cv_results = _read_parquet_or_empty(run_dir / "predict" / "cv_results.parquet")
    if cv_results.empty:
        st.info("No prediction results for this run.")
        return

    hybrid = cv_results[cv_results.get("task") == "hybrid"]
    if not hybrid.empty and "rmse_reduction_vs_sim_only_pct" in hybrid.columns:
        st.subheader("Hybrid correction vs. sim-only")
        st.plotly_chart(
            charts.rmse_reduction_bar(
                hybrid["variant"].tolist(),
                hybrid["target"].tolist(),
                hybrid["rmse_reduction_vs_sim_only_pct"].to_numpy(dtype=np.float64),
            ),
            use_container_width=True,
        )

    st.subheader("Cross-validation results")
    st.dataframe(cv_results, use_container_width=True)

    intervals = _read_parquet_or_empty(run_dir / "predict" / "intervals.parquet")
    if not intervals.empty:
        st.subheader("Prediction interval coverage")
        st.dataframe(intervals, use_container_width=True)


def _page_detection() -> None:
    run_dir = _current_run_dir()
    st.header("Detection")
    metrics = _read_parquet_or_empty(run_dir / "detect" / "detection_metrics.parquet")
    alarms = _read_parquet_or_empty(run_dir / "detect" / "alarms.parquet")
    if metrics.empty:
        st.info("No detection results for this run.")
        return

    overall = metrics[metrics["fault_type"].isna()]
    if not overall.empty:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                charts.recall_bar(
                    overall["detector"].tolist(), overall["recall"].to_numpy(dtype=np.float64)
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                charts.far_bar(
                    overall["detector"].tolist(), overall["far_per_hour"].to_numpy(dtype=np.float64)
                ),
                use_container_width=True,
            )

    st.subheader("Metrics")
    st.dataframe(metrics, use_container_width=True)
    if not alarms.empty:
        st.subheader("Alarms")
        st.dataframe(alarms, use_container_width=True)


def _page_identification() -> None:
    run_dir = _current_run_dir()
    st.header("Identification & recommendations")

    fit_results = _read_json_dict_or_none(run_dir / "identify" / "fit_results.json")
    identifiability = _read_json_dict_or_none(run_dir / "identify" / "identifiability.json")
    if fit_results is not None and identifiability is not None:
        params = cast("dict[str, float]", fit_results["params"])
        param_names = list(params)
        fitted_values = np.array([params[name] for name in param_names])
        raw_ci_95 = cast("dict[str, list[float]]", identifiability["ci_95"])
        ci_95 = {name: (bounds[0], bounds[1]) for name, bounds in raw_ci_95.items()}
        st.plotly_chart(
            charts.parameter_ci(param_names, fitted_values, ci_95), use_container_width=True
        )

    recommendations = _read_json_list_or_none(run_dir / "identify" / "recommendations.json")
    if recommendations:
        st.subheader("Recommendations")
        for rec in recommendations:
            css_class = "av-rec-card hypothesis" if rec["is_hypothesis"] else "av-rec-card"
            badge_class = "av-badge-warning" if rec["is_hypothesis"] else "av-badge-good"
            badge_text = "hypothesis" if rec["is_hypothesis"] else "supported"
            evidence = cast(list[str], rec["evidence"])
            caveat_html = (
                f'<div class="av-rec-caveat">{rec["caveats"]}</div>' if rec["caveats"] else ""
            )
            st.markdown(
                f'<div class="{css_class}">'
                f'<span class="av-badge {badge_class}">{badge_text}</span>'
                f"<strong>{rec['parameter']}</strong>: {rec['estimated_change_pct']:+.1f}% "
                f"-- evidence: {', '.join(evidence) or 'none'}"
                f"{caveat_html}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No recommendations for this run.")

    revalidation = _read_parquet_or_empty(run_dir / "identify" / "revalidation.parquet")
    if not revalidation.empty:
        st.subheader("Re-validation on held-out flights")
        st.dataframe(revalidation, use_container_width=True)


def _page_compare_runs() -> None:
    st.header("Compare runs")
    runs = _list_runs()
    if len(runs) < 2:
        st.info("Need at least two runs to compare.")
        return
    left, right = st.columns(2)
    run_a = left.selectbox("Run A", runs, index=0)
    run_b = right.selectbox("Run B", runs, index=min(1, len(runs) - 1))

    rows = []
    for label, run_id in (("A", run_a), ("B", run_b)):
        summary_path = _RUNS_DIR / run_id / "reports" / "summary.json"
        summary_facts = _read_json_dict_or_none(summary_path)
        if summary_facts is not None:
            rows.append({"run": f"{label} ({run_id})", **summary_facts})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Neither run has a generated report yet (`fsa report`).")


def _check_password() -> bool:
    """Gate the dashboard behind a shared password when `AEROVALID_DASHBOARD_
    PASSWORD` is set (a deployed, publicly-reachable instance); local use
    (`fsa dashboard`/`make dashboard`) leaves that env var unset and skips the gate
    entirely, so this never affects local/test behaviour."""
    required = os.environ.get("AEROVALID_DASHBOARD_PASSWORD")
    if not required:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("AeroValidate")
    entered = st.text_input("Password", type="password")
    if st.button("Log in"):
        if entered == required:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def main() -> None:
    st.set_page_config(page_title="AeroValidate dashboard", layout="wide", page_icon="✈️")
    if not _check_password():
        return
    st.markdown(dashboard_css(), unsafe_allow_html=True)

    runs = _list_runs()
    if not runs:
        st.title("AeroValidate")
        st.warning("No runs found under `runs/`. Run `fsa generate` first.")
        return

    preselected = os.environ.get("AEROVALID_DASHBOARD_RUN_ID")
    default_index = runs.index(preselected) if preselected in runs else 0
    run_id = st.sidebar.selectbox("Run", runs, index=default_index)
    run_dir = _RUNS_DIR / run_id
    index_path = run_dir / "aligned" / "index.json"
    index: dict[str, list[str]] = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {"flights": [], "variants": []}
    )

    st.session_state["run_id"] = run_id
    st.session_state["run_dir"] = run_dir
    st.session_state["index"] = index
    st.session_state["facts"] = gather_summary_facts(run_id, index, run_dir)

    pages = [
        st.Page(_page_overview, title="Overview", icon="\U0001f3e0", default=True),
        st.Page(_page_variant_comparison, title="Variant comparison", icon="\U0001f4ca"),
        st.Page(_page_condition_analysis, title="Condition analysis", icon="\U0001f321️"),
        st.Page(_page_prediction, title="Prediction", icon="\U0001f52e"),
        st.Page(_page_detection, title="Detection", icon="\U0001f6a8"),
        st.Page(_page_identification, title="Identification", icon="\U0001f9ea"),
        st.Page(_page_compare_runs, title="Compare runs", icon="\U0001f500"),
    ]
    st.navigation(pages).run()


main()
