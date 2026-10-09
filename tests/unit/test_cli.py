from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aerovalid.cli import app
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
@pytest.mark.parametrize("command", ["validate", "predict", "detect", "identify", "report"])
def test_unimplemented_commands_raise_not_implemented(command: str) -> None:
    with pytest.raises(NotImplementedError):
        runner.invoke(app, [command], catch_exceptions=False)


@pytest.mark.req("FR-PRP-001")
def test_preprocess_without_run_id_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="--run-id"):
        runner.invoke(app, ["preprocess"], catch_exceptions=False)


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
    ingested = tmp_path / "runs" / "clitest" / "ingested" / "sample_external_flight.parquet"
    assert ingested.exists()


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
