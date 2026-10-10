from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

_APP_PATH = Path(__file__).resolve().parents[2] / "src" / "aerovalid" / "app" / "main.py"


def _write_run_fixture(runs_dir: Path, run_id: str) -> None:
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
    assert len(at.tabs) == 5  # past the gate: the real dashboard rendered
