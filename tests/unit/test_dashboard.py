"""Dashboard tests (FR-RPT-004). `main.py`'s entrypoint (password gate, run
picker, "no runs" warning) is tested via `AppTest.from_file` against the real
default (Overview) page. Each other page is tested in isolation via
`AppTest.from_function`, since `AppTest.switch_page` only supports file-based
pages and this dashboard uses function-based `st.Page`s (one script, one process,
no `pages/` directory) -- the page functions already read their data from
`st.session_state` rather than parameters, which is exactly what makes this work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

_APP_PATH = Path(__file__).resolve().parents[2] / "src" / "aerovalid" / "app" / "main.py"


def _write_run_fixture(runs_dir: Path, run_id: str) -> Path:
    run_dir = runs_dir / run_id
    n = 30
    t_s = np.arange(n, dtype=float)
    variant_dir = run_dir / "aligned" / "V0"
    variant_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "t_s": t_s,
            "obs_alt_m": 1500.0 + t_s,
            "sim_alt_m": 1500.0 + t_s + 1.0,
        }
    ).to_parquet(variant_dir / "F0001.parquet", index=False)
    (run_dir / "aligned" / "index.json").write_text(
        json.dumps({"flights": ["F0001"], "variants": ["V0"]})
    )
    return run_dir


# ---------------------------------------------------------------------------
# Entrypoint: password gate, run picker, "no runs" warning (AppTest.from_file).
# ---------------------------------------------------------------------------


@pytest.mark.req("FR-RPT-004")
def test_dashboard_loads_with_no_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(_APP_PATH))
    at.run()
    assert not at.exception
    assert any("No runs found" in warning.value for warning in at.warning)


@pytest.mark.req("FR-RPT-004")
def test_dashboard_loads_demo_run_without_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_run_fixture(tmp_path / "runs", "clitest")

    at = AppTest.from_file(str(_APP_PATH))
    at.run()
    assert not at.exception
    assert any(h.value == "Overview" for h in at.header)


@pytest.mark.req("FR-RPT-004")
def test_dashboard_skips_password_gate_when_env_var_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local use (`fsa dashboard`) never sets `AEROVALID_DASHBOARD_PASSWORD`, so the
    gate must not appear -- this is what every other dashboard test already relies on."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AEROVALID_DASHBOARD_PASSWORD", raising=False)
    at = AppTest.from_file(str(_APP_PATH))
    at.run()
    assert not at.exception
    assert len(at.text_input) == 0


@pytest.mark.req("FR-RPT-004")
def test_dashboard_password_gate_blocks_wrong_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AEROVALID_DASHBOARD_PASSWORD", "s3cret")
    at = AppTest.from_file(str(_APP_PATH))
    at.run()
    assert not at.exception
    assert len(at.text_input) == 1  # the password prompt, nothing else rendered

    at.text_input[0].set_value("wrong-password").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Incorrect password" in e.value for e in at.error)
    assert len(at.text_input) == 1  # still gated


@pytest.mark.req("FR-RPT-004")
def test_dashboard_password_gate_admits_correct_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AEROVALID_DASHBOARD_PASSWORD", "s3cret")
    _write_run_fixture(tmp_path / "runs", "clitest")

    at = AppTest.from_file(str(_APP_PATH))
    at.run()
    at.text_input[0].set_value("s3cret").run()
    at.button[0].click().run()
    assert not at.exception
    assert any(h.value == "Overview" for h in at.header)  # past the gate


# ---------------------------------------------------------------------------
# Individual pages, in isolation (AppTest.from_function).
# ---------------------------------------------------------------------------


def _run_page(page_name: str, run_dir: Path, index: dict[str, list[str]]) -> AppTest:
    def script() -> None:
        import streamlit as st

        from aerovalid.app import main as dashboard_main
        from aerovalid.report.summary import gather_summary_facts

        run_dir_ = st.session_state["_test_run_dir"]
        index_ = st.session_state["_test_index"]
        st.session_state["run_dir"] = run_dir_
        st.session_state["index"] = index_
        st.session_state["facts"] = gather_summary_facts("test", index_, run_dir_)
        getattr(dashboard_main, st.session_state["_test_page_name"])()

    at = AppTest.from_function(script)
    at.session_state["_test_run_dir"] = run_dir
    at.session_state["_test_index"] = index
    at.session_state["_test_page_name"] = page_name
    at.run()
    return at


@pytest.mark.req("FR-RPT-004")
def test_page_overview_renders_kpis_and_executive_summary(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path / "runs", "clitest")
    index = json.loads((run_dir / "aligned" / "index.json").read_text())
    at = _run_page("_page_overview", run_dir, index)
    assert not at.exception
    assert any(h.value == "Overview" for h in at.header)
    assert any("Executive summary" in s.value for s in at.subheader)


@pytest.mark.req("FR-RPT-004")
def test_page_variant_comparison_renders_overlay_chart(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path / "runs", "clitest")
    index = json.loads((run_dir / "aligned" / "index.json").read_text())
    at = _run_page("_page_variant_comparison", run_dir, index)
    assert not at.exception


@pytest.mark.req("FR-RPT-004")
def test_page_condition_analysis_handles_no_data(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path / "runs", "clitest")
    index = json.loads((run_dir / "aligned" / "index.json").read_text())
    at = _run_page("_page_condition_analysis", run_dir, index)
    assert not at.exception
    assert any("No condition analysis" in i.value for i in at.info)


@pytest.mark.req("FR-RPT-004")
def test_page_prediction_renders_chart_and_tables(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path / "runs", "clitest")
    index = json.loads((run_dir / "aligned" / "index.json").read_text())
    (run_dir / "predict").mkdir(parents=True)
    pd.DataFrame(
        {
            "variant": ["V0"],
            "target": ["alt_m"],
            "task": ["hybrid"],
            "rmse_reduction_vs_sim_only_pct": [42.0],
        }
    ).to_parquet(run_dir / "predict" / "cv_results.parquet", index=False)

    at = _run_page("_page_prediction", run_dir, index)
    assert not at.exception


@pytest.mark.req("FR-RPT-004")
def test_page_detection_renders_charts_and_tables(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path / "runs", "clitest")
    index = json.loads((run_dir / "aligned" / "index.json").read_text())
    (run_dir / "detect").mkdir(parents=True)
    pd.DataFrame(
        {
            "variant": ["V0"],
            "detector": ["zscore"],
            "fault_type": [None],
            "recall": [0.9],
            "precision": [0.8],
            "far_per_hour": [0.5],
            "median_delay_s": [2.5],
            "median_lead_time_s": [1.0],
            "n_detected": [1],
            "n_faulty": [1],
            "n_false_alarms": [0],
        }
    ).to_parquet(run_dir / "detect" / "detection_metrics.parquet", index=False)

    at = _run_page("_page_detection", run_dir, index)
    assert not at.exception


@pytest.mark.req("FR-RPT-004")
def test_page_identification_renders_recommendation_cards(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path / "runs", "clitest")
    index = json.loads((run_dir / "aligned" / "index.json").read_text())
    (run_dir / "identify").mkdir(parents=True)
    (run_dir / "identify" / "recommendations.json").write_text(
        json.dumps(
            [
                {
                    "parameter": "CD0",
                    "estimated_change_pct": 14.0,
                    "evidence": ["fit", "sensitivity"],
                    "caveats": "",
                    "is_hypothesis": False,
                }
            ]
        )
    )

    at = _run_page("_page_identification", run_dir, index)
    assert not at.exception


@pytest.mark.req("FR-RPT-004")
def test_page_compare_runs_needs_at_least_two_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_run_fixture(tmp_path / "runs", "clitest")

    def script() -> None:
        from aerovalid.app.main import _page_compare_runs

        _page_compare_runs()

    at = AppTest.from_function(script)
    at.run()
    assert not at.exception
    assert any("Need at least two runs" in i.value for i in at.info)
