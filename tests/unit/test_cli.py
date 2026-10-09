from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from aerovalid.cli import _predict_splits, app
from aerovalid.core.errors import ConfigError

runner = CliRunner()

_COMMANDS = [
    "generate",
    "ingest",
    "preprocess",
    "validate",
    "analyze",
    "predict",
    "detect",
    "identify",
    "report",
    "run-all",
    "dashboard",
    "benchmark",
]
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.req("FR-OPS-001")
def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in _COMMANDS:
        assert command in result.output


@pytest.mark.req("FR-OPS-001")
@pytest.mark.req("FR-PRP-001")
def test_preprocess_without_run_id_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="--run-id"):
        runner.invoke(app, ["preprocess"], catch_exceptions=False)


@pytest.mark.req("FR-PRD-001")
def test_predict_splits_uses_true_loso_within_the_fold_cap() -> None:
    flight_id = np.array(["F1", "F1", "F2", "F2", "F3", "F3"])
    splits = _predict_splits("loso", folds=6, flight_id=flight_id)
    assert len(splits) == 3  # one fold per flight: true leave-one-flight-out
    for _, test_idx in splits:
        assert len(set(flight_id[test_idx])) == 1


@pytest.mark.req("FR-PRD-001")
def test_predict_splits_caps_loso_at_configured_folds() -> None:
    """More flights than `folds`: falls back to grouped K-fold with exactly `folds`
    folds rather than one per flight (ADR-023 -- this is what kept a 12-flight study's
    `fsa predict` step from scaling its fold count, and so its runtime, with the
    flight count)."""
    flight_id = np.array([f"F{i}" for i in range(10) for _ in range(3)])
    splits = _predict_splits("loso", folds=6, flight_id=flight_id)
    assert len(splits) == 6
    for train_idx, test_idx in splits:
        assert set(flight_id[train_idx]).isdisjoint(set(flight_id[test_idx]))


@pytest.mark.req("FR-PRD-001")
def test_predict_splits_grouped_kfold_scheme_ignores_the_loso_cap() -> None:
    flight_id = np.array(["F1", "F1", "F2", "F2", "F3", "F3"])
    splits = _predict_splits("grouped_kfold", folds=2, flight_id=flight_id)
    assert len(splits) == 2


@pytest.mark.req("FR-TEL-005")
def test_ingest_cli_converts_fixture_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixtures = _REPO_ROOT / "tests" / "fixtures"
    result = runner.invoke(
        app,
        [
            "ingest",
            "--run-id",
            "clitest",
            "--source",
            str(fixtures),
            "--mapping",
            str(fixtures / "sample_external_flight_mapping.yaml"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    ingested = tmp_path / "runs" / "clitest" / "raw" / "telemetry" / "F_EXTERNAL_0001.parquet"
    assert ingested.exists()


@pytest.mark.req("FR-TEL-005")
def test_ingest_then_preprocess_then_validate_works_on_real_data_style_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The J-B user journey (PRD §6): ingest external telemetry plus one external
    simulation variant's output, then run the rest of the pipeline unmodified."""
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    fixtures = _REPO_ROOT / "tests" / "fixtures" / "external_demo"

    telemetry_args = [
        "ingest",
        "--run-id",
        "clitest",
        "--source",
        str(fixtures / "telemetry"),
        "--mapping",
        str(fixtures / "telemetry_mapping.yaml"),
    ]
    assert runner.invoke(app, telemetry_args, catch_exceptions=False).exit_code == 0

    variant_args = [
        "ingest",
        "--run-id",
        "clitest",
        "--source",
        str(fixtures / "variant_v0"),
        "--mapping",
        str(fixtures / "variant_v0_mapping.yaml"),
    ]
    assert runner.invoke(app, variant_args, catch_exceptions=False).exit_code == 0

    run_dir = tmp_path / "runs" / "clitest"
    assert (run_dir / "raw" / "telemetry" / "F_EXT_0001.parquet").exists()
    assert (run_dir / "raw" / "variants" / "V0" / "F_EXT_0001.parquet").exists()

    preprocess_args = ["preprocess", "--config", str(config), "--run-id", "clitest"]
    assert runner.invoke(app, preprocess_args, catch_exceptions=False).exit_code == 0
    assert (run_dir / "aligned" / "V0" / "F_EXT_0001.parquet").exists()

    validate_args = ["validate", "--config", str(config), "--run-id", "clitest"]
    result = runner.invoke(app, validate_args, catch_exceptions=False)
    assert result.exit_code == 0
    metrics = pd.read_parquet(run_dir / "validation" / "metrics_signal.parquet")
    assert "V0" in set(metrics["variant"])


@pytest.mark.slow
@pytest.mark.req("FR-TEL-004")
def test_generate_cli_produces_flights_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    args = [
        "generate",
        "--config",
        str(config),
        "--run-id",
        "clitest",
        "--set",
        "generate.n_flights=1",
        "--set",
        "generate.scenarios=[doublet_series]",
        "--set",
        "variants=[V0]",
    ]

    result = runner.invoke(app, args, catch_exceptions=False)
    assert result.exit_code == 0

    run_dir = tmp_path / "runs" / "clitest"
    assert (run_dir / "raw" / "truth" / "F0001.parquet").exists()
    assert (run_dir / "raw" / "telemetry" / "F0001.parquet").exists()
    assert (run_dir / "raw" / "variants" / "V0" / "F0001.parquet").exists()
    assert (run_dir / "raw" / "meta" / "truth_params.json").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["steps"]["generate"]["status"] == "completed"

    # second invocation with the same run_id and config should skip, not re-run
    result_again = runner.invoke(app, args, catch_exceptions=False)
    assert result_again.exit_code == 0


@pytest.mark.slow
@pytest.mark.req("FR-PRP-009")
def test_preprocess_cli_builds_aligned_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    generate_args = [
        "generate",
        "--config",
        str(config),
        "--run-id",
        "clitest",
        "--set",
        "generate.n_flights=1",
        "--set",
        "generate.scenarios=[doublet_series]",
        "--set",
        "variants=[V0]",
    ]
    assert runner.invoke(app, generate_args, catch_exceptions=False).exit_code == 0

    preprocess_args = ["preprocess", "--config", str(config), "--run-id", "clitest"]
    result = runner.invoke(app, preprocess_args, catch_exceptions=False)
    assert result.exit_code == 0

    run_dir = tmp_path / "runs" / "clitest"
    assert (run_dir / "qc" / "F0001.json").exists()
    assert (run_dir / "qc" / "qc_summary.parquet").exists()
    assert (run_dir / "sync" / "sync_estimates.parquet").exists()
    assert (run_dir / "aligned" / "V0" / "F0001.parquet").exists()
    assert (run_dir / "aligned" / "index.json").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["steps"]["preprocess"]["status"] == "completed"


@pytest.mark.slow
@pytest.mark.req("FR-VAL-006")
def test_validate_and_analyze_cli_produce_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    generate_args = [
        "generate",
        "--config",
        str(config),
        "--run-id",
        "clitest",
        "--set",
        "generate.n_flights=2",
        "--set",
        "generate.scenarios=[doublet_series]",
        "--set",
        "variants=[V0, V1]",
    ]
    assert runner.invoke(app, generate_args, catch_exceptions=False).exit_code == 0

    preprocess_args = ["preprocess", "--config", str(config), "--run-id", "clitest"]
    assert runner.invoke(app, preprocess_args, catch_exceptions=False).exit_code == 0

    validate_args = ["validate", "--config", str(config), "--run-id", "clitest"]
    result = runner.invoke(app, validate_args, catch_exceptions=False)
    assert result.exit_code == 0

    run_dir = tmp_path / "runs" / "clitest"
    validation_dir = run_dir / "validation"
    assert (validation_dir / "metrics_signal.parquet").exists()
    assert (validation_dir / "variant_ranking.parquet").exists()
    assert (validation_dir / "pairwise_tests.parquet").exists()
    ranking = pd.read_parquet(validation_dir / "variant_ranking.parquet")
    assert set(ranking["variant"]) == {"V0", "V1"}
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["steps"]["validate"]["status"] == "completed"

    analyze_args = ["analyze", "--config", str(config), "--run-id", "clitest"]
    result = runner.invoke(app, analyze_args, catch_exceptions=False)
    assert result.exit_code == 0
    assert (validation_dir / "condition_heatmaps.parquet").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["steps"]["analyze"]["status"] == "completed"


@pytest.mark.slow
@pytest.mark.req("FR-PRD-001")
def test_predict_cli_produces_cv_results_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    generate_args = [
        "generate",
        "--config",
        str(config),
        "--run-id",
        "clitest",
        "--set",
        "generate.n_flights=2",
        "--set",
        "generate.scenarios=[doublet_series]",
        "--set",
        "variants=[V0, V1]",
    ]
    assert runner.invoke(app, generate_args, catch_exceptions=False).exit_code == 0

    preprocess_args = ["preprocess", "--config", str(config), "--run-id", "clitest"]
    assert runner.invoke(app, preprocess_args, catch_exceptions=False).exit_code == 0

    # `predict` has no `--set` option (unlike `generate`), so a small `include:`-based
    # override config keeps this test's model/target/horizon sweep fast. `predict()`
    # never resolves sibling config dirs (vehicles/variants/...) the way generate()
    # does, so this override doesn't need to live next to `configs/default.yaml`.
    predict_config = tmp_path / "predict_test_config.yaml"
    predict_config.write_text(
        f"include: [{config}]\n"
        "predict:\n"
        "  models: [ridge]\n"
        "  targets: [alt_m]\n"
        "  horizons_s: [1.0]\n"
    )
    predict_args = ["predict", "--config", str(predict_config), "--run-id", "clitest"]
    result = runner.invoke(app, predict_args, catch_exceptions=False)
    assert result.exit_code == 0

    run_dir = tmp_path / "runs" / "clitest"
    predict_dir = run_dir / "predict"
    assert (predict_dir / "cv_results.parquet").exists()
    assert (predict_dir / "forecasts.parquet").exists()
    assert (predict_dir / "intervals.parquet").exists()
    assert (predict_dir / "explain.parquet").exists()
    assert (predict_dir / "metrics.json").exists()
    assert (predict_dir / "models" / "residual" / "V0_alt_m_ridge.joblib").exists()
    assert (predict_dir / "models" / "residual" / "V1_alt_m_ridge.joblib").exists()

    cv_results = pd.read_parquet(predict_dir / "cv_results.parquet")
    assert set(cv_results["variant"]) == {"V0", "V1"}
    assert set(cv_results["task"]) == {"residual", "hybrid"}

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["steps"]["predict"]["status"] == "completed"


@pytest.mark.slow
@pytest.mark.req("FR-DET-001")
def test_detect_cli_produces_calibrated_alarms_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    generate_args = [
        "generate",
        "--config",
        str(config),
        "--run-id",
        "clitest",
        "--set",
        "generate.n_flights=6",
        "--set",
        "generate.scenarios=[doublet_series]",
        "--set",
        "variants=[V0]",
    ]
    assert runner.invoke(app, generate_args, catch_exceptions=False).exit_code == 0

    preprocess_args = ["preprocess", "--config", str(config), "--run-id", "clitest"]
    assert runner.invoke(app, preprocess_args, catch_exceptions=False).exit_code == 0

    detect_args = ["detect", "--config", str(config), "--run-id", "clitest"]
    result = runner.invoke(app, detect_args, catch_exceptions=False)
    assert result.exit_code == 0

    run_dir = tmp_path / "runs" / "clitest"
    detect_dir = run_dir / "detect"
    assert (detect_dir / "baselines.json").exists()
    assert (detect_dir / "alarms.parquet").exists()
    assert (detect_dir / "detection_metrics.parquet").exists()
    assert (detect_dir / "roc.parquet").exists()

    faults = json.loads((run_dir / "raw" / "meta" / "faults.json").read_text())
    n_faulty = sum(1 for events in faults.values() if events)
    assert n_faulty >= 1  # this fixture's fixed seed samples at least one fault

    alarms = pd.read_parquet(detect_dir / "alarms.parquet")
    assert set(alarms["variant"]) == {"V0"}
    assert {"zscore", "ewma", "cusum", "mahalanobis"} <= set(alarms["detector"])
    assert alarms["is_fault"].sum() == n_faulty * len(set(alarms["detector"]))

    metrics = pd.read_parquet(detect_dir / "detection_metrics.parquet")
    overall = metrics[metrics["fault_type"].isna()]
    assert overall["recall"].between(0.0, 1.0).all()
    assert (overall["far_per_hour"] >= 0.0).all()

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["steps"]["detect"]["status"] == "completed"


@pytest.mark.slow
@pytest.mark.req("FR-IDN-002")
def test_identify_cli_produces_fit_identifiability_and_recommendations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    generate_args = [
        "generate",
        "--config",
        str(config),
        "--run-id",
        "clitest",
        "--set",
        "generate.n_flights=3",
        "--set",
        "generate.scenarios=[doublet_series]",
        "--set",
        "variants=[V0]",
    ]
    assert runner.invoke(app, generate_args, catch_exceptions=False).exit_code == 0

    identify_args = ["identify", "--config", str(config), "--run-id", "clitest"]
    result = runner.invoke(app, identify_args, catch_exceptions=False)
    assert result.exit_code == 0

    run_dir = tmp_path / "runs" / "clitest"
    identify_dir = run_dir / "identify"
    assert (identify_dir / "fit_results.json").exists()
    assert (identify_dir / "identifiability.json").exists()
    assert (identify_dir / "sensitivity.parquet").exists()
    assert (identify_dir / "recommendations.json").exists()
    assert (identify_dir / "revalidation.parquet").exists()

    fit_results = json.loads((identify_dir / "fit_results.json").read_text())
    assert fit_results["variant"] == "V0"
    assert fit_results["success"] is True
    assert set(fit_results["params"]) == {"CL0", "CLa", "CD0", "k", "thrust_lapse_n"}

    identifiability = json.loads((identify_dir / "identifiability.json").read_text())
    assert set(identifiability["ci_95"]) == set(fit_results["params"])
    assert identifiability["condition_number"] > 0

    recommendations = json.loads((identify_dir / "recommendations.json").read_text())
    assert len(recommendations) == len(fit_results["params"])
    for rec in recommendations:
        assert rec["is_hypothesis"] or len(rec["evidence"]) >= 2

    sensitivity = pd.read_parquet(identify_dir / "sensitivity.parquet")
    assert set(sensitivity["parameter"]) == set(fit_results["params"])

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["steps"]["identify"]["status"] == "completed"


@pytest.mark.slow
@pytest.mark.req("FR-RPT-001")
def test_report_cli_builds_html_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    generate_args = [
        "generate",
        "--config",
        str(config),
        "--run-id",
        "clitest",
        "--set",
        "generate.n_flights=2",
        "--set",
        "generate.scenarios=[doublet_series]",
        "--set",
        "variants=[V0, V1]",
    ]
    assert runner.invoke(app, generate_args, catch_exceptions=False).exit_code == 0

    preprocess_args = ["preprocess", "--config", str(config), "--run-id", "clitest"]
    assert runner.invoke(app, preprocess_args, catch_exceptions=False).exit_code == 0

    validate_args = ["validate", "--config", str(config), "--run-id", "clitest"]
    assert runner.invoke(app, validate_args, catch_exceptions=False).exit_code == 0

    report_args = ["report", "--config", str(config), "--run-id", "clitest"]
    result = runner.invoke(app, report_args, catch_exceptions=False)
    assert result.exit_code == 0

    run_dir = tmp_path / "runs" / "clitest"
    assert (run_dir / "reports" / "report.html").exists()
    assert (run_dir / "reports" / "summary.json").exists()
    html = (run_dir / "reports" / "report.html").read_text()
    assert "Executive summary" in html
    assert "Appendix" in html

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["steps"]["report"]["status"] == "completed"


@pytest.mark.slow
@pytest.mark.req("FR-OPS-002")
def test_run_all_cli_produces_every_phase_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    run_all_args = [
        "run-all",
        "--config",
        str(config),
        "--run-id",
        "clitest",
        "--profile",
        "demo",
        "--set",
        "generate.n_flights=2",
        "--set",
        "generate.scenarios=[doublet_series]",
        "--set",
        "variants=[V0, V1]",
    ]
    result = runner.invoke(app, run_all_args, catch_exceptions=False)
    assert result.exit_code == 0

    run_dir = tmp_path / "runs" / "clitest"
    for artifact in (
        "aligned/index.json",
        "validation/variant_ranking.parquet",
        "predict/cv_results.parquet",
        "detect/detection_metrics.parquet",
        "identify/fit_results.json",
        "reports/report.html",
    ):
        assert (run_dir / artifact).exists(), artifact

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    steps = (
        "generate",
        "preprocess",
        "validate",
        "analyze",
        "predict",
        "detect",
        "identify",
        "report",
    )
    for step in steps:
        assert manifest["steps"][step]["status"] == "completed"


@pytest.mark.slow
@pytest.mark.req("FR-OPS-004")
def test_benchmark_cli_writes_summary_with_s1_through_s9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _REPO_ROOT / "configs" / "default.yaml"
    result = runner.invoke(
        app, ["benchmark", "--config", str(config), "--seeds", "1"], catch_exceptions=False
    )
    assert result.exit_code == 0

    summary_path = tmp_path / "runs" / "benchmark" / "benchmark_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["seeds"] == [1]
    assert set(summary["results"]) >= {
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6_delay",
        "S6_far",
        "S6_recall",
        "S7",
        "S8",
        "S9",
    }
    # S1 is always actually evaluated (it only needs `generate`, never skipped).
    assert summary["results"]["S1"]["passed"] is not None
    # S3/S9 are explicitly out of this reduced scope -- never a fabricated pass/fail.
    assert summary["results"]["S3"]["passed"] is None
    assert summary["results"]["S9"]["passed"] is None
