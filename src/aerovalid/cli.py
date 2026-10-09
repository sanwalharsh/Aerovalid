"""`fsa` CLI entrypoint (FR-OPS-001). `generate`/`ingest` are implemented (Phase 2);
later subcommands still raise `NotImplementedError` until their owning phase lands --
never a silent `pass` (CLAUDE.md §2).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import typer
import yaml

from aerovalid.core.config import load_config
from aerovalid.core.logging import setup_logging
from aerovalid.core.manifest import (
    create_manifest,
    hash_config,
    read_manifest,
    update_step,
    write_manifest,
)
from aerovalid.sim.protocols import load_vehicle_params
from aerovalid.sim.truth import TruthModel
from aerovalid.sim.variants import load_all_variants, make_variant_v5
from aerovalid.telemetry.adapters.base import ColumnMapping
from aerovalid.telemetry.adapters.csv_parquet import CsvParquetAdapter
from aerovalid.telemetry.defects import TimingDefectParams, load_timing_defect_params
from aerovalid.telemetry.faults import load_fault_library
from aerovalid.telemetry.generator import FlightArtifacts, generate_all_flights
from aerovalid.telemetry.sensors import SensorModel, load_sensor_suite

app = typer.Typer(
    name="fsa",
    help="AeroValidate: flight-simulation validation and predictive analysis pipeline.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)

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


def _resolve_run_id(run_id: str | None) -> str:
    return run_id or uuid.uuid4().hex[:12]


def _configs_dir(config_path: Path) -> Path:
    """Sibling config directories (`vehicles/`, `variants/`, ...) live next to the
    resolved study config, i.e. `config_path.parent` (e.g. `configs/` for the repo's
    own `configs/default.yaml`)."""
    return config_path.resolve().parent


@app.command()
def generate(
    config: Path = _CONFIG_OPTION,
    run_id: str | None = _RUN_ID_OPTION,
    seed: int | None = _SEED_OPTION,
    profile: str = _PROFILE_OPTION,
    set_: list[str] = _SET_OPTION,
    force: bool = _FORCE_OPTION,
) -> None:
    """Generate synthetic truth/telemetry/variant flights (FR-TEL-001...007)."""
    resolved_run_id = _resolve_run_id(run_id)
    overrides = list(set_)
    if seed is not None:
        overrides.append(f"study.seed={seed}")
    # `profile` is recorded in the manifest only; `generate.n_flights` comes from the
    # config (or an explicit --set), so demo-vs-full scale is a config choice, not a
    # hidden CLI side effect.
    study_config = load_config(config, overrides=overrides)

    run_dir = Path("runs") / resolved_run_id
    setup_logging(run_id=resolved_run_id, log_file=run_dir / "log.txt")
    resolved_yaml = yaml.safe_dump(study_config.model_dump(mode="json", by_alias=True))
    config_hash = hash_config(resolved_yaml)

    if not force and (run_dir / "run_manifest.json").exists():
        existing = read_manifest(run_dir)
        if existing.config_hash == config_hash and "generate" in existing.steps:
            logger.info("generate: artifacts up to date for run %s, skipping", resolved_run_id)
            return

    manifest = create_manifest(resolved_run_id, config_hash, study_config.study.seed)
    (run_dir / "config.resolved.yaml").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(resolved_yaml)

    configs_dir = _configs_dir(config)
    nominal = load_vehicle_params(study_config.vehicle.ref, configs_dir / "vehicles")
    truth = TruthModel.from_nominal(nominal, master_seed=study_config.study.seed)

    variant_names = [name for name in study_config.variants if name != "V5"]
    variants = load_all_variants(variant_names, configs_dir / "variants", nominal)
    if "V5" in study_config.variants:
        v5_rng = np.random.Generator(np.random.PCG64(study_config.study.seed + 1))
        variants["V5"] = make_variant_v5(truth.vehicle, v5_rng)

    sensor_models, timing_params = load_timing_and_sensors(configs_dir)
    fault_library = load_fault_library(configs_dir / "faults")

    started_at = datetime.now(UTC).isoformat()
    start_time = time.monotonic()
    flights = generate_all_flights(
        study_config,
        nominal,
        truth,
        variants,
        sensor_models,
        timing_params,
        fault_library,
        configs_dir / "scenarios",
        n_jobs=1,
    )
    output_hashes = _write_generated_flights(run_dir, flights, truth)
    duration_s = time.monotonic() - start_time

    manifest = update_step(
        manifest,
        "generate",
        "completed",
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=duration_s,
        output_hashes=output_hashes,
        extra={"n_flights": len(flights), "profile": profile},
    )
    write_manifest(manifest, run_dir)
    logger.info("generate: wrote %d flights to %s", len(flights), run_dir)


def load_timing_and_sensors(configs_dir: Path) -> tuple[dict[str, SensorModel], TimingDefectParams]:
    """Load both the per-signal sensor suite and the timing-defect params from the
    same `configs/sensors/default.yaml` file (two distinct top-level sections)."""
    sensors_path = configs_dir / "sensors" / "default.yaml"
    return load_sensor_suite(sensors_path), load_timing_defect_params(sensors_path)


def _write_generated_flights(
    run_dir: Path, flights: list[FlightArtifacts], truth: TruthModel
) -> dict[str, str]:
    from aerovalid.core.manifest import hash_file

    raw_dir = run_dir / "raw"
    (raw_dir / "truth").mkdir(parents=True, exist_ok=True)
    (raw_dir / "telemetry").mkdir(parents=True, exist_ok=True)
    (raw_dir / "meta").mkdir(parents=True, exist_ok=True)

    output_hashes: dict[str, str] = {}
    all_faults: dict[str, list[dict[str, object]]] = {}

    for flight in flights:
        truth_path = raw_dir / "truth" / f"{flight.flight_id}.parquet"
        tel_path = raw_dir / "telemetry" / f"{flight.flight_id}.parquet"
        flight.truth_df.to_parquet(truth_path, index=False)
        flight.telemetry_df.to_parquet(tel_path, index=False)
        output_hashes[str(truth_path.relative_to(run_dir))] = hash_file(truth_path)
        output_hashes[str(tel_path.relative_to(run_dir))] = hash_file(tel_path)

        for variant_name, df in flight.variant_dfs.items():
            variant_dir = raw_dir / "variants" / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            variant_path = variant_dir / f"{flight.flight_id}.parquet"
            df.to_parquet(variant_path, index=False)
            output_hashes[str(variant_path.relative_to(run_dir))] = hash_file(variant_path)

        fault_dicts = [event.model_dump() for event in flight.fault_events]
        meta = {
            "flight_id": flight.flight_id,
            "scenario": flight.scenario_name,
            "seed": flight.seed,
            "conditions": flight.conditions,
            "status": flight.status,
            "faults": fault_dicts,
            "defects": {
                "time_offset_s": flight.timing_log.offset_s,
                "drift_ppm": flight.timing_log.drift_ppm,
                "gaps": flight.timing_log.gaps_s,
                "n_duplicated": flight.timing_log.n_duplicated,
                "n_dropped": flight.timing_log.n_dropped,
            },
        }
        meta_path = raw_dir / "meta" / f"{flight.flight_id}.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        output_hashes[str(meta_path.relative_to(run_dir))] = hash_file(meta_path)
        all_faults[flight.flight_id] = fault_dicts

    truth_params_path = raw_dir / "meta" / "truth_params.json"
    truth_params_path.write_text(json.dumps(truth.perturbations, indent=2))
    output_hashes[str(truth_params_path.relative_to(run_dir))] = hash_file(truth_params_path)

    faults_path = raw_dir / "meta" / "faults.json"
    faults_path.write_text(json.dumps(all_faults, indent=2))
    output_hashes[str(faults_path.relative_to(run_dir))] = hash_file(faults_path)

    return output_hashes


@app.command()
def ingest(
    config: Path = _CONFIG_OPTION,
    run_id: str | None = _RUN_ID_OPTION,
    source: Path = _SOURCE_OPTION,
    mapping: Path = _MAPPING_OPTION,
) -> None:
    """Ingest external telemetry/variant data via an adapter (FR-TEL-005...007)."""
    resolved_run_id = _resolve_run_id(run_id)
    run_dir = Path("runs") / resolved_run_id
    setup_logging(run_id=resolved_run_id, log_file=run_dir / "log.txt")

    mapping_raw = yaml.safe_load(Path(mapping).read_text()) or {}
    column_mapping = ColumnMapping.model_validate(mapping_raw)
    adapter = CsvParquetAdapter()

    ingested_dir = run_dir / "ingested"
    ingested_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(p for p in Path(source).iterdir() if p.suffix in (".csv", ".parquet"))
    for source_file in source_files:
        canonical = adapter.read(source_file, column_mapping)
        canonical.to_parquet(ingested_dir / f"{source_file.stem}.parquet", index=False)
        logger.info("ingest: %s -> %d rows", source_file.name, len(canonical))

    logger.info("ingest: wrote %d file(s) to %s", len(source_files), ingested_dir)


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
