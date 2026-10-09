from __future__ import annotations

import pytest
from typer.testing import CliRunner

from aerovalid.cli import app

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


@pytest.mark.req("FR-OPS-001")
def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in _COMMANDS:
        assert command in result.output


@pytest.mark.req("FR-OPS-001")
@pytest.mark.parametrize(
    "command", ["generate", "preprocess", "validate", "predict", "detect", "identify", "report"]
)
def test_unimplemented_commands_raise_not_implemented(command: str) -> None:
    with pytest.raises(NotImplementedError):
        runner.invoke(app, [command], catch_exceptions=False)
