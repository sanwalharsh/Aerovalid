"""`fsa` CLI entrypoint (FR-OPS-001). Phase 0 wires up the command surface; each
subcommand raises `NotImplementedError` until its owning phase implements it -- never a
silent `pass` (CLAUDE.md §2).
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="fsa",
    help="AeroValidate: flight-simulation validation and predictive analysis pipeline.",
    no_args_is_help=True,
)

_CONFIG_OPTION = typer.Option(
    Path("configs/default.yaml"), "--config", help="Path to the study YAML config."
)
_RUN_ID_OPTION = typer.Option(
    None, "--run-id", help="Run identifier; auto-generated (uuid4) if omitted."
)
_SEED_OPTION = typer.Option(None, "--seed", help="Override study.seed.")
_SET_OPTION = typer.Option(
    [], "--set", help="Override a config key: section.key=value (repeatable)."
)
_FORCE_OPTION = typer.Option(
    False, "--force", help="Re-run even if artifacts and hashes already match."
)
_PROFILE_OPTION = typer.Option("demo", "--profile", help="'demo' (few flights) or 'full'.")
_SOURCE_OPTION = typer.Option(..., "--source", help="Directory of external flight data.")
_MAPPING_OPTION = typer.Option(..., "--mapping", help="Column-mapping YAML for the adapter.")
_DASHBOARD_RUN_ID_OPTION = typer.Option(..., "--run-id", help="Run to open in the dashboard.")
_SEEDS_OPTION = typer.Option("1,2,3,4,5", "--seeds", help="Comma-separated seed list.")


@app.command()
def generate(
    config: Path = _CONFIG_OPTION,
    run_id: str | None = _RUN_ID_OPTION,
    seed: int | None = _SEED_OPTION,
    profile: str = _PROFILE_OPTION,
    set_: list[str] = _SET_OPTION,
) -> None:
    """Generate synthetic truth/telemetry/variant flights (FR-TEL-001...007)."""
    raise NotImplementedError("`fsa generate` is implemented in Phase 2.")


@app.command()
def ingest(
    config: Path = _CONFIG_OPTION,
    run_id: str | None = _RUN_ID_OPTION,
    source: Path = _SOURCE_OPTION,
    mapping: Path = _MAPPING_OPTION,
) -> None:
    """Ingest external telemetry/variant data via an adapter (FR-TEL-005...007)."""
    raise NotImplementedError("`fsa ingest` is implemented in Phase 2.")


@app.command()
def preprocess(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """QC, clean, sync, resample, and align raw data (FR-PRP-001...011)."""
    raise NotImplementedError("`fsa preprocess` is implemented in Phase 3.")


@app.command()
def validate(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Compute per-variant error metrics and statistical comparisons (FR-VAL-001...010)."""
    raise NotImplementedError("`fsa validate` is implemented in Phase 4.")


@app.command()
def analyze(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Condition-dependent error analysis (heatmaps, importance, PDP)."""
    raise NotImplementedError("`fsa analyze` is implemented in Phase 4.")


@app.command()
def predict(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Residual/hybrid/forecast predictive models (FR-PRD-001...010)."""
    raise NotImplementedError("`fsa predict` is implemented in Phase 5.")


@app.command()
def detect(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Deviation detection and early warning (FR-DET-001...008)."""
    raise NotImplementedError("`fsa detect` is implemented in Phase 6.")


@app.command()
def identify(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Parameter sensitivity, identification, and recommendations (FR-IDN-001...008)."""
    raise NotImplementedError("`fsa identify` is implemented in Phase 7.")


@app.command()
def report(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Build the HTML report and summary.json (FR-RPT-001...006)."""
    raise NotImplementedError("`fsa report` is implemented in Phase 8.")


@app.command(name="run-all")
def run_all(
    config: Path = _CONFIG_OPTION,
    run_id: str | None = _RUN_ID_OPTION,
    profile: str = _PROFILE_OPTION,
    force: bool = _FORCE_OPTION,
) -> None:
    """Orchestrate the full pipeline end to end, resumably (FR-OPS-002)."""
    raise NotImplementedError("`fsa run-all` is implemented in Phase 8.")


@app.command()
def dashboard(run_id: str = _DASHBOARD_RUN_ID_OPTION) -> None:
    """Launch the Streamlit dashboard for a completed run."""
    raise NotImplementedError("`fsa dashboard` is implemented in Phase 8.")


@app.command()
def benchmark(config: Path = _CONFIG_OPTION, seeds: str = _SEEDS_OPTION) -> None:
    """Run the synthetic benchmark suite and report S1-S9 pass/fail (FR-OPS-004)."""
    raise NotImplementedError("`fsa benchmark` is implemented in Phase 8.")


if __name__ == "__main__":
    app()
