from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerovalid.report.builder import build_report


def _write_aligned_fixture(run_dir: Path) -> None:
    n = 50
    t_s = np.arange(n, dtype=float)
    for variant in ("V0", "V1"):
        df = pd.DataFrame(
            {
                "t_s": t_s,
                "flight_id": "F0001",
                "obs_alt_m": 1500.0 + t_s,
                "sim_alt_m": 1500.0 + t_s + (1.0 if variant == "V0" else 0.5),
                "res_alt_m": -1.0 if variant == "V0" else -0.5,
            }
        )
        variant_dir = run_dir / "aligned" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(variant_dir / "F0001.parquet", index=False)

    index = {"flights": ["F0001"], "variants": ["V0", "V1"]}
    (run_dir / "aligned").mkdir(parents=True, exist_ok=True)
    (run_dir / "aligned" / "index.json").write_text(json.dumps(index))


def _write_validation_fixture(run_dir: Path) -> None:
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    ranking = pd.DataFrame(
        {
            "variant": ["V1", "V0"],
            "mean_rank": [1.2, 1.8],
            "rank_ci_low": [1.0, 1.5],
            "rank_ci_high": [1.5, 2.0],
            "p_best": [0.8, 0.2],
        }
    )
    ranking.to_parquet(validation_dir / "variant_ranking.parquet", index=False)

    metrics_signal = pd.DataFrame(
        {"variant": ["V0", "V1"], "signal": ["alt_m", "alt_m"], "rmse": [1.0, 0.5]}
    )
    metrics_signal.to_parquet(validation_dir / "metrics_signal.parquet", index=False)

    heatmaps = pd.DataFrame(
        {
            "bin": ["low", "high", "low", "high"],
            "mean": [0.5, 1.5, 0.3, 1.1],
            "std": [0.1, 0.2, 0.1, 0.2],
            "count": [10, 10, 10, 10],
            "variant": ["V0", "V0", "V1", "V1"],
            "cond_column": ["cond_alt_band"] * 4,
        }
    )
    heatmaps.to_parquet(validation_dir / "condition_heatmaps.parquet", index=False)


@pytest.mark.req("FR-RPT-001")
def test_build_report_populates_variant_comparison_and_condition_sections(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "clitest"
    _write_aligned_fixture(run_dir)
    _write_validation_fixture(run_dir)

    result = build_report(run_dir, "clitest")

    assert result.html_path.exists()
    assert result.summary_json_path.exists()
    html = result.html_path.read_text()
    assert "clitest" in html
    assert "No data for this run" not in html.split("Variant comparison")[1].split("Condition")[0]

    summary = json.loads(result.summary_json_path.read_text())
    assert summary["best_variant"] == "V1"
    assert summary["n_flights"] == 1


@pytest.mark.req("FR-RPT-001")
def test_build_report_degrades_gracefully_with_no_phase_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "empty"
    run_dir.mkdir(parents=True)

    result = build_report(run_dir, "empty")

    assert result.html_path.exists()
    html = result.html_path.read_text()
    assert html.count("No data for this run") >= 5  # most sections empty, none crash

    summary = json.loads(result.summary_json_path.read_text())
    assert summary["n_flights"] == 0
    assert summary["best_variant"] is None


@pytest.mark.req("FR-RPT-001")
def test_build_report_writes_csv_and_markdown_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "clitest"
    _write_aligned_fixture(run_dir)
    _write_validation_fixture(run_dir)

    build_report(run_dir, "clitest")

    tables_dir = run_dir / "reports" / "tables"
    assert (tables_dir / "variant_ranking.csv").exists()
    assert (tables_dir / "variant_ranking.md").exists()
    assert (tables_dir / "condition_heatmaps.csv").exists()
