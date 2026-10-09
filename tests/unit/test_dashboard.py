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
