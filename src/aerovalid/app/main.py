"""Streamlit dashboard (FR-RPT-004, `make dashboard` / `fsa dashboard`): run picker,
flight/variant selector, overlay, residual explorer, condition heatmaps, detector
timeline, recommendation list, and run-to-run comparison. Reads whichever artifacts
exist for the selected run(s); a phase that didn't run shows "no data", never a crash.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st

from aerovalid.report import figures

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


def _overlay_and_residual_tab(run_dir: Path, index: dict[str, list[str]]) -> None:
    flights, variants = index.get("flights", []), index.get("variants", [])
    if not flights or not variants:
        st.info("No aligned data for this run.")
        return

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

    st.pyplot(figures.overlay_timeseries(t_s, obs, sims))

    residual_variant = st.selectbox("Residual variant", list(sims))
    tol = st.number_input("Tolerance", value=5.0, min_value=0.0)
    st.pyplot(figures.residual_timeseries_with_tolerance(t_s, obs - sims[residual_variant], tol))


def _condition_heatmap_tab(run_dir: Path) -> None:
    heatmaps = _read_parquet_or_empty(run_dir / "validation" / "condition_heatmaps.parquet")
    if heatmaps.empty:
        st.info("No condition analysis for this run.")
        return
    cond_column = st.selectbox("Condition column", sorted(heatmaps["cond_column"].unique()))
    subset = heatmaps[heatmaps["cond_column"] == cond_column]
    pivot = subset.pivot_table(index="bin", columns="variant", values="mean")
    if not pivot.empty:
        st.pyplot(figures.condition_heatmap(pivot))


def _detector_timeline_tab(run_dir: Path) -> None:
    alarms = _read_parquet_or_empty(run_dir / "detect" / "alarms.parquet")
    metrics = _read_parquet_or_empty(run_dir / "detect" / "detection_metrics.parquet")
    if alarms.empty:
        st.info("No detection results for this run.")
        return
    st.dataframe(alarms)
    if not metrics.empty:
        st.dataframe(metrics)


def _recommendations_tab(run_dir: Path) -> None:
    recommendations = _read_json_list_or_none(run_dir / "identify" / "recommendations.json")
    if not recommendations:
        st.info("No recommendations for this run.")
        return
    for rec in recommendations:
        label = "hypothesis" if rec["is_hypothesis"] else "recommendation"
        evidence = cast(list[str], rec["evidence"])
        st.markdown(
            f"**{rec['parameter']}** ({label}): {rec['estimated_change_pct']:+.1f}% "
            f"-- evidence: {', '.join(evidence) or 'none'}"
        )
        if rec["caveats"]:
            st.caption(cast(str, rec["caveats"]))


def _run_comparison_tab() -> None:
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
        facts = _read_json_dict_or_none(summary_path)
        if facts is not None:
            rows.append({"run": f"{label} ({run_id})", **facts})
    if rows:
        st.dataframe(pd.DataFrame(rows))
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
    st.set_page_config(page_title="AeroValidate dashboard", layout="wide")
    if not _check_password():
        return
    st.title("AeroValidate")

    runs = _list_runs()
    if not runs:
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

    tabs = st.tabs(
        [
            "Overlay / residual",
            "Condition heatmaps",
            "Detector timeline",
            "Recommendations",
            "Compare runs",
        ]
    )
    with tabs[0]:
        _overlay_and_residual_tab(run_dir, index)
    with tabs[1]:
        _condition_heatmap_tab(run_dir)
    with tabs[2]:
        _detector_timeline_tab(run_dir)
    with tabs[3]:
        _recommendations_tab(run_dir)
    with tabs[4]:
        _run_comparison_tab()


main()
