"""`fsa` CLI entrypoint (FR-OPS-001). `generate`/`ingest` are implemented (Phase 2);
later subcommands still raise `NotImplementedError` until their owning phase lands --
never a silent `pass` (CLAUDE.md §2).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml

from aerovalid.core.config import load_config
from aerovalid.core.errors import ConfigError
from aerovalid.core.logging import setup_logging
from aerovalid.core.manifest import (
    create_manifest,
    hash_config,
    hash_file,
    read_manifest,
    update_step,
    write_manifest,
)
from aerovalid.detect.evaluation import DetectionMetrics, FlightRecord
from aerovalid.predict.cv import CVResult
from aerovalid.predict.datasets import WindowedDataset
from aerovalid.predict.models import ResidualModel
from aerovalid.prep.pipeline import preprocess_flight
from aerovalid.sim.protocols import VehicleParams, load_vehicle_params
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
    if run_id is None:
        raise ConfigError("`fsa preprocess` requires --run-id (an existing `fsa generate` run).")
    run_dir = Path("runs") / run_id
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise ConfigError(f"No raw/ artifacts found at {raw_dir}; run `fsa generate` first.")

    setup_logging(run_id=run_id, log_file=run_dir / "log.txt")
    study_config = load_config(config)
    configs_dir = _configs_dir(config)
    nominal = load_vehicle_params(study_config.vehicle.ref, configs_dir / "vehicles")

    started_at = datetime.now(UTC).isoformat()
    start_time = time.monotonic()
    manifest = read_manifest(run_dir) if (run_dir / "run_manifest.json").exists() else None

    qc_dir, sync_dir, aligned_dir = run_dir / "qc", run_dir / "sync", run_dir / "aligned"
    for directory in (qc_dir, sync_dir, aligned_dir):
        directory.mkdir(parents=True, exist_ok=True)

    qc_rows: list[dict[str, object]] = []
    sync_rows: list[dict[str, object]] = []
    output_hashes: dict[str, str] = {}
    flight_paths = sorted((raw_dir / "telemetry").glob("*.parquet"))

    for telemetry_path in flight_paths:
        flight_id = telemetry_path.stem
        telemetry = pd.read_parquet(telemetry_path)
        variants_raw_dir = raw_dir / "variants"
        variant_dirs = sorted(variants_raw_dir.iterdir()) if variants_raw_dir.exists() else []
        variant_trajectories = {
            d.name: pd.read_parquet(d / f"{flight_id}.parquet")
            for d in variant_dirs
            if (d / f"{flight_id}.parquet").exists()
        }

        result = preprocess_flight(
            flight_id,
            telemetry,
            variant_trajectories,
            study_config.preprocess,
            nominal.wing_area_m2,
        )

        qc_path = qc_dir / f"{flight_id}.json"
        qc_path.write_text(result.qc.model_dump_json(indent=2))
        output_hashes[str(qc_path.relative_to(run_dir))] = hash_file(qc_path)
        qc_rows.append(result.qc.model_dump())

        for variant_name, estimate in result.sync_estimates.items():
            sync_rows.append(
                {
                    "flight_id": flight_id,
                    "variant": variant_name,
                    "lag_s": estimate.lag_s,
                    "drift_ppm": estimate.drift_ppm,
                    "peak_correlation": estimate.peak_correlation,
                    "low_confidence": estimate.low_confidence,
                }
            )

        for variant_name, aligned in result.aligned.items():
            variant_dir = aligned_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            aligned_path = variant_dir / f"{flight_id}.parquet"
            aligned.df.to_parquet(aligned_path, index=False)
            output_hashes[str(aligned_path.relative_to(run_dir))] = hash_file(aligned_path)

    pd.DataFrame(qc_rows).to_parquet(qc_dir / "qc_summary.parquet", index=False)
    pd.DataFrame(sync_rows).to_parquet(sync_dir / "sync_estimates.parquet", index=False)
    all_variants = sorted({str(row["variant"]) for row in sync_rows})
    index = {"flights": [p.stem for p in flight_paths], "variants": all_variants}
    (aligned_dir / "index.json").write_text(json.dumps(index, indent=2))

    if manifest is not None:
        manifest = update_step(
            manifest,
            "preprocess",
            "completed",
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - start_time,
            output_hashes=output_hashes,
            extra={"n_flights": len(flight_paths)},
        )
        write_manifest(manifest, run_dir)
    logger.info("preprocess: processed %d flights in %s", len(flight_paths), run_dir)


def _resolve_tolerances(configured: dict[str, float], signals: list[str]) -> dict[str, float]:
    """Fill in any signal missing from `configured` with its registry default tolerance."""
    from aerovalid.core.signals import get_signal

    resolved = dict(configured)
    for name in signals:
        if name not in resolved:
            tol = get_signal(name).default_tolerance
            if tol is not None:
                resolved[name] = tol
    return resolved


def _load_aligned_index(run_dir: Path) -> dict[str, list[str]]:
    index_path = run_dir / "aligned" / "index.json"
    if not index_path.exists():
        raise ConfigError(f"No aligned/ artifacts found at {run_dir}; run `fsa preprocess` first.")
    index: dict[str, list[str]] = json.loads(index_path.read_text())
    return index


def _load_variant_aligned(run_dir: Path, variant: str, flights: list[str]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(run_dir / "aligned" / variant / f"{flight_id}.parquet")
        for flight_id in flights
        if (run_dir / "aligned" / variant / f"{flight_id}.parquet").exists()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_flight_aligned(run_dir: Path, variant: str, flight_id: str) -> pd.DataFrame:
    path = run_dir / "aligned" / variant / f"{flight_id}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@app.command()
def validate(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Compute per-variant error metrics and statistical comparisons (FR-VAL-001...010)."""
    from aerovalid.validate.aggregate import compute_metrics_table
    from aerovalid.validate.compare import compare_variant_pairs
    from aerovalid.validate.composite import composite_score, rank_variants

    if run_id is None:
        raise ConfigError("`fsa validate` requires --run-id (an existing `fsa preprocess` run).")
    run_dir = Path("runs") / run_id
    setup_logging(run_id=run_id, log_file=run_dir / "log.txt")
    study_config = load_config(config)
    index = _load_aligned_index(run_dir)
    tolerances = _resolve_tolerances(
        study_config.validation.tolerances, study_config.validation.signals
    )

    started_at = datetime.now(UTC).isoformat()
    start_time = time.monotonic()

    per_signal_frames = []
    per_phase_frames = []
    for variant in index["variants"]:
        variant_df = _load_variant_aligned(run_dir, variant, index["flights"])
        if variant_df.empty:
            continue
        signal_table = compute_metrics_table(
            variant_df, study_config.validation.signals, tolerances, groupby="flight_id"
        )
        signal_table = signal_table.rename(columns={"group": "flight_id"})
        signal_table["variant"] = variant
        per_signal_frames.append(signal_table)

        if "phase" in variant_df.columns:
            phase_table = compute_metrics_table(
                variant_df, study_config.validation.signals, tolerances, groupby="phase"
            )
            phase_table = phase_table.rename(columns={"group": "phase"})
            phase_table["variant"] = variant
            per_phase_frames.append(phase_table)

    metrics_signal = (
        pd.concat(per_signal_frames, ignore_index=True) if per_signal_frames else pd.DataFrame()
    )
    metrics_phase = (
        pd.concat(per_phase_frames, ignore_index=True) if per_phase_frames else pd.DataFrame()
    )

    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}

    def _write(df: pd.DataFrame, name: str) -> None:
        path = validation_dir / name
        df.to_parquet(path, index=False)
        output_hashes[str(path.relative_to(run_dir))] = hash_file(path)

    _write(metrics_signal, "metrics_signal.parquet")
    _write(metrics_phase, "metrics_phase.parquet")

    if len(index["variants"]) >= 2 and not metrics_signal.empty:
        ref_variant = "V0" if "V0" in index["variants"] else index["variants"][0]
        scores = composite_score(metrics_signal, ref_variant=ref_variant)
        ranking = rank_variants(scores, rng=np.random.default_rng(study_config.study.seed))
        _write(ranking, "variant_ranking.parquet")

        score_pivot = scores.pivot(index="flight_id", columns="variant", values="composite_score")
        score_by_variant = {name: score_pivot[name].to_numpy() for name in score_pivot.columns}
        pairwise = compare_variant_pairs(score_by_variant)
        pairwise_df = pd.DataFrame([vars(p) for p in pairwise])
        _write(pairwise_df, "pairwise_tests.parquet")

    manifest = read_manifest(run_dir) if (run_dir / "run_manifest.json").exists() else None
    if manifest is not None:
        manifest = update_step(
            manifest,
            "validate",
            "completed",
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - start_time,
            output_hashes=output_hashes,
            extra={"n_variants": len(index["variants"])},
        )
        write_manifest(manifest, run_dir)
    logger.info(
        "validate: wrote metrics for %d variant(s) to %s", len(index["variants"]), validation_dir
    )


@app.command()
def analyze(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Condition-dependent error analysis (heatmaps, importance, PDP)."""
    from aerovalid.validate.conditions import condition_heatmap_data

    if run_id is None:
        raise ConfigError("`fsa analyze` requires --run-id (an existing `fsa preprocess` run).")
    run_dir = Path("runs") / run_id
    setup_logging(run_id=run_id, log_file=run_dir / "log.txt")
    index = _load_aligned_index(run_dir)

    started_at = datetime.now(UTC).isoformat()
    start_time = time.monotonic()
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}

    heatmap_frames = []
    for variant in index["variants"]:
        variant_df = _load_variant_aligned(run_dir, variant, index["flights"])
        cond_columns = [c for c in variant_df.columns if c.startswith("cond_")]
        if variant_df.empty or not cond_columns or "res_alt_m" not in variant_df.columns:
            continue
        long_df = variant_df.assign(abs_res_alt_m=variant_df["res_alt_m"].abs())
        for cond_col in cond_columns:
            heatmap = condition_heatmap_data(long_df, [cond_col], "abs_res_alt_m")
            heatmap["variant"] = variant
            heatmap["cond_column"] = cond_col
            heatmap = heatmap.rename(columns={cond_col: "bin"})
            heatmap_frames.append(heatmap)

    condition_heatmaps = (
        pd.concat(heatmap_frames, ignore_index=True) if heatmap_frames else pd.DataFrame()
    )
    heatmap_path = validation_dir / "condition_heatmaps.parquet"
    condition_heatmaps.to_parquet(heatmap_path, index=False)
    output_hashes[str(heatmap_path.relative_to(run_dir))] = hash_file(heatmap_path)

    manifest = read_manifest(run_dir) if (run_dir / "run_manifest.json").exists() else None
    if manifest is not None:
        manifest = update_step(
            manifest,
            "analyze",
            "completed",
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - start_time,
            output_hashes=output_hashes,
        )
        write_manifest(manifest, run_dir)
    logger.info("analyze: wrote condition heatmaps to %s", validation_dir)


def _predict_model_factory(name: str) -> ResidualModel:
    """`predict.models` name -> a fresh, unfitted `ResidualModel` instance."""
    from aerovalid.predict.models.arimax import ARIMAXModel
    from aerovalid.predict.models.gru import GRUModel
    from aerovalid.predict.models.hgb import HGBModel
    from aerovalid.predict.models.rf import RFModel
    from aerovalid.predict.models.ridge import RidgeModel

    factories: dict[str, Callable[[], ResidualModel]] = {
        "ridge": RidgeModel,
        "hgb": HGBModel,
        "rf": RFModel,
        "arimax": ARIMAXModel,
        "gru": GRUModel,
    }
    if name not in factories:
        raise ConfigError(f"Unknown predict model '{name}' (expected one of {sorted(factories)}).")
    return factories[name]()


def _predict_resolve_model_names(configured: list[str], seq: bool) -> list[str]:
    """`predict.models` plus `"gru"` when `predict.seq` is set and torch is available."""
    from aerovalid.predict.models.gru import HAS_TORCH

    names = list(configured)
    if seq:
        if HAS_TORCH:
            if "gru" not in names:
                names.append("gru")
        else:
            logger.warning("predict: predict.seq is true but torch is not installed; skipping gru.")
    return names


def _predict_splits(
    scheme: str, folds: int, flight_id: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    from aerovalid.predict.datasets import grouped_kfold, leave_one_flight_out

    if scheme == "loso":
        return list(leave_one_flight_out(flight_id))
    return list(grouped_kfold(flight_id, n_splits=folds))


def _predict_cv_rows_and_best(
    dataset: WindowedDataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_names: list[str],
    context: dict[str, object],
) -> tuple[list[dict[str, object]], str, CVResult]:
    """Cross-validate every model in `model_names`, return `(rows, best_name, best_result)`."""
    from aerovalid.predict.cv import cross_validate

    rows: list[dict[str, object]] = []
    best_name, best_result, best_rmse = "", None, np.inf
    for name in model_names:
        factory: Callable[[], ResidualModel] = partial(_predict_model_factory, name)
        result = cross_validate(factory, dataset, iter(splits))
        rows.append(
            {
                **context,
                "model": name,
                "rmse": result.overall_metrics.rmse,
                "mae": result.overall_metrics.mae,
                "r2": result.overall_metrics.r2,
                "n_samples": len(dataset),
            }
        )
        if result.overall_metrics.rmse < best_rmse:
            best_rmse, best_name, best_result = result.overall_metrics.rmse, name, result
    if best_result is None:
        raise ConfigError("fsa predict: no usable models in predict.models.")
    return rows, best_name, best_result


def _predict_conformal_interval_row(
    best_name: str,
    dataset: WindowedDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    context: dict[str, object],
) -> dict[str, object]:
    """Flight-grouped calibration split + split-conformal interval summary for the
    `(variant, target, horizon)` combination's best model."""
    from aerovalid.predict.uncertainty import SplitConformalWrapper, evaluate_coverage

    train_flights = pd.unique(dataset.flight_id[train_idx])
    if len(train_flights) >= 2:
        n_calib_flights = max(1, len(train_flights) // 5)
        calib_flights = set(train_flights[:n_calib_flights])
        calib_mask = np.isin(dataset.flight_id[train_idx], list(calib_flights))
        calib_idx, fit_idx = train_idx[calib_mask], train_idx[~calib_mask]
    else:
        # Only one training flight available (e.g. a small demo run under
        # leave-one-flight-out): a flight-disjoint split isn't possible, so fall
        # back to a row-level 80/20 split within it.
        split_point = max(1, int(0.8 * len(train_idx)))
        fit_idx, calib_idx = train_idx[:split_point], train_idx[split_point:]

    wrapper = SplitConformalWrapper(_predict_model_factory(best_name), alpha=alpha)
    wrapper.fit(
        dataset.features.iloc[fit_idx],
        dataset.target[fit_idx],
        dataset.features.iloc[calib_idx],
        dataset.target[calib_idx],
    )
    lower, upper = wrapper.predict_interval(dataset.features.iloc[test_idx])
    coverage, mean_width = evaluate_coverage(lower, upper, dataset.target[test_idx])
    return {
        **context,
        "model": best_name,
        "alpha": alpha,
        "coverage": coverage,
        "mean_width": mean_width,
    }


def _predict_for_target(
    variant_df: pd.DataFrame,
    variant: str,
    target_signal: str,
    horizon_s: float,
    dt: float,
    feature_signals: list[str],
    cv_scheme: str,
    cv_folds: int,
    model_names: list[str],
    conformal_alpha: float,
    models_dir: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Task A + Task B CV, a conformal-interval summary, and an explainability table
    for one `(variant, target_signal)` combination; saves the best Task A model."""
    import joblib

    from aerovalid.predict.datasets import build_windows
    from aerovalid.predict.explain import permutation_importance_table
    from aerovalid.predict.tasks import task_b_hybrid_correction

    dataset = build_windows(
        variant_df, feature_signals, target_signal, horizon_s, dt, include_future_sim=True
    )
    splits = _predict_splits(cv_scheme, cv_folds, dataset.flight_id)
    all_test_idx = np.concatenate([test_idx for _, test_idx in splits])
    context: dict[str, object] = {
        "variant": variant,
        "target": target_signal,
        "horizon_s": horizon_s,
    }

    cv_rows, best_name, best_result = _predict_cv_rows_and_best(
        dataset, splits, model_names, {**context, "task": "residual"}
    )

    train_idx, test_idx = splits[0]
    hybrid = task_b_hybrid_correction(
        _predict_model_factory(best_name), dataset, train_idx, test_idx
    )
    cv_rows.append(
        {
            **context,
            "task": "hybrid",
            "model": best_name,
            "rmse": hybrid.rmse_hybrid,
            "mae": float("nan"),
            "r2": float("nan"),
            "n_samples": len(test_idx),
            "rmse_sim_only": hybrid.rmse_sim_only,
            "rmse_persistence": hybrid.rmse_persistence,
            "rmse_reduction_vs_sim_only_pct": hybrid.rmse_reduction_vs_sim_only_pct,
        }
    )
    interval_row = _predict_conformal_interval_row(
        best_name, dataset, train_idx, test_idx, conformal_alpha, context
    )

    final_model = _predict_model_factory(best_name)
    final_model.fit(dataset.features, dataset.target, dataset.flight_id)
    task_dir = models_dir / "residual"
    task_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, task_dir / f"{variant}_{target_signal}_{best_name}.joblib")

    importance = permutation_importance_table(
        final_model, dataset.features, dataset.target, n_repeats=5
    )
    importance["variant"] = variant
    importance["target"] = target_signal
    importance["horizon_s"] = horizon_s
    importance["model"] = best_name

    forecast_df = pd.DataFrame(
        {
            "variant": variant,
            "target": target_signal,
            "horizon_s": horizon_s,
            "model": best_name,
            "flight_id": dataset.flight_id[all_test_idx],
            "t_s": dataset.t_s[all_test_idx],
            "actual_residual": best_result.actuals,
            "predicted_residual": best_result.predictions,
        }
    )
    return cv_rows, forecast_df, interval_row, importance


@app.command()
def predict(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Residual/hybrid predictive models across variants and configured targets
    (FR-PRD-001...003/005/007/009). Scoped to the first configured horizon and a
    flight-grouped conformal calibration split per `(variant, target)`; Task C
    (forecast) and full horizon/CV-scheme sweeps are the library's job
    (`predict/`, unit-tested) rather than this CLI's -- see DECISIONS.md ADR-017.
    """
    if run_id is None:
        raise ConfigError("`fsa predict` requires --run-id (an existing `fsa preprocess` run).")
    run_dir = Path("runs") / run_id
    setup_logging(run_id=run_id, log_file=run_dir / "log.txt")
    study_config = load_config(config)
    index = _load_aligned_index(run_dir)

    predict_config = study_config.predict
    model_names = _predict_resolve_model_names(predict_config.models, predict_config.seq)
    feature_signals = study_config.validation.signals
    horizon_s = predict_config.horizons_s[0]
    dt = 1.0 / study_config.preprocess.grid_hz

    started_at = datetime.now(UTC).isoformat()
    start_time = time.monotonic()
    predict_dir = run_dir / "predict"
    models_dir = predict_dir / "models"
    predict_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}

    cv_rows: list[dict[str, object]] = []
    forecast_frames: list[pd.DataFrame] = []
    interval_rows: list[dict[str, object]] = []
    explain_frames: list[pd.DataFrame] = []

    for variant in index["variants"]:
        variant_df = _load_variant_aligned(run_dir, variant, index["flights"])
        if variant_df.empty:
            continue
        for target_signal in predict_config.targets:
            if f"res_{target_signal}" not in variant_df.columns:
                continue
            target_cv_rows, forecast_df, interval_row, importance = _predict_for_target(
                variant_df,
                variant,
                target_signal,
                horizon_s,
                dt,
                feature_signals,
                predict_config.cv.scheme,
                predict_config.cv.folds,
                model_names,
                predict_config.conformal.alpha,
                models_dir,
            )
            cv_rows.extend(target_cv_rows)
            forecast_frames.append(forecast_df)
            interval_rows.append(interval_row)
            explain_frames.append(importance)

    def _write(df: pd.DataFrame, name: str) -> None:
        path = predict_dir / name
        df.to_parquet(path, index=False)
        output_hashes[str(path.relative_to(run_dir))] = hash_file(path)

    _write(pd.DataFrame(cv_rows), "cv_results.parquet")
    _write(
        pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame(),
        "forecasts.parquet",
    )
    _write(pd.DataFrame(interval_rows), "intervals.parquet")
    _write(
        pd.concat(explain_frames, ignore_index=True) if explain_frames else pd.DataFrame(),
        "explain.parquet",
    )
    (predict_dir / "metrics.json").write_text(json.dumps(cv_rows, indent=2, default=str))
    output_hashes[str((predict_dir / "metrics.json").relative_to(run_dir))] = hash_file(
        predict_dir / "metrics.json"
    )

    manifest = read_manifest(run_dir) if (run_dir / "run_manifest.json").exists() else None
    if manifest is not None:
        manifest = update_step(
            manifest,
            "predict",
            "completed",
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - start_time,
            output_hashes=output_hashes,
            extra={"n_variants": len(index["variants"]), "horizon_s": horizon_s},
        )
        write_manifest(manifest, run_dir)
    logger.info("predict: wrote %d cv result rows to %s", len(cv_rows), predict_dir)


def _detect_load_faults(run_dir: Path) -> dict[str, list[dict[str, object]]]:
    path = run_dir / "raw" / "meta" / "faults.json"
    if not path.exists():
        raise ConfigError(f"No raw/meta/faults.json found at {run_dir}; run `fsa generate` first.")
    data: dict[str, list[dict[str, object]]] = json.loads(path.read_text())
    return data


def _detect_fault_window_labels(
    t_s: np.ndarray, fault_events: list[dict[str, object]]
) -> np.ndarray:
    labels = np.zeros(len(t_s), dtype=bool)
    for event in fault_events:
        on, off = float(event["t_on_s"]), float(event["t_off_s"])  # type: ignore[arg-type]
        labels |= (t_s >= on) & (t_s <= off)
    return labels


def _detect_exceedance_time(
    df: pd.DataFrame, signals: list[str], tolerances: dict[str, float]
) -> float | None:
    from aerovalid.validate.tolerance import time_to_first_exceedance

    t_s = df["t_s"].to_numpy(dtype=np.float64)
    candidates = []
    for signal in signals:
        res_col = f"res_{signal}"
        if res_col not in df.columns or signal not in tolerances:
            continue
        t_exc = time_to_first_exceedance(
            t_s, df[res_col].to_numpy(dtype=np.float64), tolerances[signal]
        )
        if t_exc is not None:
            candidates.append(t_exc)
    return min(candidates) if candidates else None


_DETECT_WARMUP_S = 2.0  # discard this much post-takeoff settling transient (ADR-018)

_DETECTOR_SEARCH_RANGE: dict[str, tuple[float, float]] = {
    "zscore": (1.0, 8.0),
    "ewma": (1.0, 6.0),
    "cusum": (1.0, 15.0),
    "page_hinkley": (2.0, 30.0),
}


def _detect_run_univariate(
    name: str,
    streams: dict[str, np.ndarray],
    t_s_by_flight: dict[str, np.ndarray],
    nominal_ids: list[str],
    duration_by_flight: dict[str, float],
    target_far_per_hour: float,
) -> dict[str, float | None]:
    """Calibrate `name`'s single threshold knob to `target_far_per_hour` on nominal
    flights, then find every flight's first alarm time on the combined max|z| stream."""
    from aerovalid.detect.calibration import calibrate_threshold
    from aerovalid.detect.statistical import (
        CUSUMDetector,
        EWMADetector,
        PageHinkleyDetector,
        StreamingDetector,
        ZScoreDetector,
        detect_batch,
    )

    factories: dict[str, Callable[[float], StreamingDetector]] = {
        "zscore": lambda h: ZScoreDetector(h=h, m=5),
        "ewma": lambda limit: EWMADetector(lam=0.2, upper_limit=limit),
        "cusum": lambda h: CUSUMDetector(k=0.5, h=h),
        "page_hinkley": lambda lam: PageHinkleyDetector(delta=0.5, lam=lam),
    }
    factory = factories[name]
    nominal_streams = [streams[fid] for fid in nominal_ids]
    nominal_durations = [duration_by_flight[fid] for fid in nominal_ids]

    def build_alarms(threshold: float) -> list[np.ndarray]:
        return [detect_batch(factory(threshold), s) for s in nominal_streams]

    threshold = calibrate_threshold(
        build_alarms, nominal_durations, target_far_per_hour, _DETECTOR_SEARCH_RANGE[name]
    )

    alarm_times: dict[str, float | None] = {}
    for fid, stream in streams.items():
        alarms = detect_batch(factory(threshold), stream)
        idx = int(np.argmax(alarms)) if alarms.any() else -1
        alarm_times[fid] = float(t_s_by_flight[fid][idx]) if idx >= 0 else None
    return alarm_times


def _detect_run_mahalanobis(
    z_by_flight: dict[str, pd.DataFrame],
    t_s_by_flight: dict[str, np.ndarray],
    nominal_ids: list[str],
) -> dict[str, float | None]:
    from aerovalid.detect.statistical import fit_mahalanobis

    nominal_z = np.concatenate(
        [z_by_flight[fid].to_numpy(dtype=np.float64) for fid in nominal_ids], axis=0
    )
    detector = fit_mahalanobis(nominal_z, alpha=0.01)

    alarm_times: dict[str, float | None] = {}
    for fid, z_df in z_by_flight.items():
        rows = z_df.to_numpy(dtype=np.float64)
        idx = next((i for i, row in enumerate(rows) if detector.update(row)), -1)
        alarm_times[fid] = float(t_s_by_flight[fid][idx]) if idx >= 0 else None
    return alarm_times


def _detect_build_records(
    alarm_times: dict[str, float | None],
    faults_by_flight: dict[str, list[dict[str, object]]],
    duration_by_flight: dict[str, float],
    exceedance_by_flight: dict[str, float | None],
) -> list[FlightRecord]:
    records: list[FlightRecord] = []
    for fid, t_alarm in alarm_times.items():
        events = faults_by_flight.get(fid, [])
        records.append(
            FlightRecord(
                flight_id=fid,
                is_fault=bool(events),
                duration_s=duration_by_flight[fid],
                fault_type=str(events[0]["type"]) if events else None,
                t_on_s=float(events[0]["t_on_s"]) if events else None,  # type: ignore[arg-type]
                t_exc_s=exceedance_by_flight.get(fid),
                t_alarm_s=t_alarm,
            )
        )
    return records


def _detect_metrics_rows(
    variant: str, detector: str, metrics: DetectionMetrics
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "variant": variant,
            "detector": detector,
            "fault_type": None,
            "recall": metrics.recall,
            "precision": metrics.precision,
            "far_per_hour": metrics.far_per_hour,
            "median_delay_s": metrics.median_delay_s,
            "median_lead_time_s": metrics.median_lead_time_s,
            "n_detected": metrics.n_detected,
            "n_faulty": metrics.n_faulty,
            "n_false_alarms": metrics.n_false_alarms,
        }
    ]
    for fault_type, stats in metrics.per_fault_type.items():
        rows.append(
            {
                "variant": variant,
                "detector": detector,
                "fault_type": fault_type,
                "recall": stats["recall"],
                "precision": float("nan"),
                "far_per_hour": float("nan"),
                "median_delay_s": float("nan"),
                "median_lead_time_s": float("nan"),
                "n_detected": float("nan"),
                "n_faulty": stats["n_flights"],
                "n_false_alarms": float("nan"),
            }
        )
    return rows


def _detect_alarm_rows(
    variant: str, detector: str, records: list[FlightRecord]
) -> list[dict[str, object]]:
    return [
        {
            "variant": variant,
            "detector": detector,
            "flight_id": r.flight_id,
            "is_fault": r.is_fault,
            "fault_type": r.fault_type,
            "t_on_s": r.t_on_s,
            "t_alarm_s": r.t_alarm_s,
        }
        for r in records
    ]


def _detect_for_variant(
    run_dir: Path,
    variant: str,
    flights: list[str],
    faults_by_flight: dict[str, list[dict[str, object]]],
    signals: list[str],
    tolerances: dict[str, float],
    detector_names: list[str],
    target_far_per_hour: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Fits a residual baseline and runs every configured calibratable detector for
    one variant. Returns `(metrics_rows, alarm_rows, baseline_json, roc_row)`; the last
    two are `None` if this variant has no fault-free (nominal) flights to calibrate on."""
    from aerovalid.detect.baseline import fit_baseline
    from aerovalid.detect.evaluation import evaluate_detections, window_level_auc

    flight_frames = {fid: _load_flight_aligned(run_dir, variant, fid) for fid in flights}
    flight_frames = {
        fid: df[df["t_s"] >= _DETECT_WARMUP_S].reset_index(drop=True)
        for fid, df in flight_frames.items()
        if not df.empty
    }
    nominal_ids = [fid for fid in flight_frames if not faults_by_flight.get(fid)]
    if not flight_frames or not nominal_ids:
        if flight_frames:
            logger.warning("detect: variant %s has no fault-free flights; skipping.", variant)
        return [], [], None, None

    nominal_df = pd.concat([flight_frames[fid] for fid in nominal_ids], ignore_index=True)
    baseline = fit_baseline(nominal_df, signals)

    z_by_flight = {fid: baseline.z_scores(df) for fid, df in flight_frames.items()}
    t_s_by_flight = {fid: df["t_s"].to_numpy(dtype=np.float64) for fid, df in flight_frames.items()}
    duration_by_flight = {
        fid: float(t[-1] - t[0]) if len(t) else 0.0 for fid, t in t_s_by_flight.items()
    }
    streams = {
        fid: np.nanmax(np.abs(z.to_numpy(dtype=np.float64)), axis=1)
        if z.shape[1]
        else np.zeros(len(z))
        for fid, z in z_by_flight.items()
    }
    exceedance_by_flight = {
        fid: _detect_exceedance_time(df, signals, tolerances) for fid, df in flight_frames.items()
    }

    metrics_rows: list[dict[str, object]] = []
    alarm_rows: list[dict[str, object]] = []
    for name in detector_names:
        if name == "mahalanobis":
            alarm_times = _detect_run_mahalanobis(z_by_flight, t_s_by_flight, nominal_ids)
        else:
            alarm_times = _detect_run_univariate(
                name, streams, t_s_by_flight, nominal_ids, duration_by_flight, target_far_per_hour
            )
        records = _detect_build_records(
            alarm_times, faults_by_flight, duration_by_flight, exceedance_by_flight
        )
        metrics_rows.extend(_detect_metrics_rows(variant, name, evaluate_detections(records)))
        alarm_rows.extend(_detect_alarm_rows(variant, name, records))

    all_scores = np.concatenate(list(streams.values()))
    all_labels = np.concatenate(
        [
            _detect_fault_window_labels(t_s_by_flight[fid], faults_by_flight.get(fid, []))
            for fid in flight_frames
        ]
    )
    roc_auc, pr_auc = window_level_auc(all_scores, all_labels)
    roc_row: dict[str, object] = {
        "variant": variant,
        "score_source": "max_abs_z",
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    return metrics_rows, alarm_rows, json.loads(baseline.to_json()), roc_row


@app.command()
def detect(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """FAR-calibrated statistical + Mahalanobis deviation detection on a combined
    per-sample max|z| residual stream, per variant (FR-DET-001/002/006). ML (iforest),
    change-point (pelt) onset estimation, and predictive early warning
    (FR-DET-003/004/005) are the library's job (`detect/`, unit-tested), not this
    CLI's -- see DECISIONS.md ADR-018.
    """
    if run_id is None:
        raise ConfigError("`fsa detect` requires --run-id (an existing `fsa preprocess` run).")
    run_dir = Path("runs") / run_id
    setup_logging(run_id=run_id, log_file=run_dir / "log.txt")
    study_config = load_config(config)
    index = _load_aligned_index(run_dir)
    faults_by_flight = _detect_load_faults(run_dir)
    signals = study_config.validation.signals
    tolerances = _resolve_tolerances(study_config.validation.tolerances, signals)
    detector_names = [
        name
        for name in study_config.detect.detectors
        if name in _DETECTOR_SEARCH_RANGE or name == "mahalanobis"
    ]

    started_at = datetime.now(UTC).isoformat()
    start_time = time.monotonic()
    detect_dir = run_dir / "detect"
    detect_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}

    baselines_summary: dict[str, object] = {}
    metrics_rows: list[dict[str, object]] = []
    alarm_rows: list[dict[str, object]] = []
    roc_rows: list[dict[str, object]] = []

    for variant in index["variants"]:
        variant_metrics, variant_alarms, baseline_json, roc_row = _detect_for_variant(
            run_dir,
            variant,
            index["flights"],
            faults_by_flight,
            signals,
            tolerances,
            detector_names,
            study_config.detect.target_far_per_hr,
        )
        if baseline_json is None:
            continue
        baselines_summary[variant] = baseline_json
        metrics_rows.extend(variant_metrics)
        alarm_rows.extend(variant_alarms)
        if roc_row is not None:
            roc_rows.append(roc_row)

    def _write(df: pd.DataFrame, name: str) -> None:
        path = detect_dir / name
        df.to_parquet(path, index=False)
        output_hashes[str(path.relative_to(run_dir))] = hash_file(path)

    _write(pd.DataFrame(alarm_rows), "alarms.parquet")
    _write(pd.DataFrame(metrics_rows), "detection_metrics.parquet")
    _write(pd.DataFrame(roc_rows), "roc.parquet")
    baselines_path = detect_dir / "baselines.json"
    baselines_path.write_text(json.dumps(baselines_summary, indent=2))
    output_hashes[str(baselines_path.relative_to(run_dir))] = hash_file(baselines_path)

    manifest = read_manifest(run_dir) if (run_dir / "run_manifest.json").exists() else None
    if manifest is not None:
        manifest = update_step(
            manifest,
            "detect",
            "completed",
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - start_time,
            output_hashes=output_hashes,
            extra={"n_variants": len(baselines_summary)},
        )
        write_manifest(manifest, run_dir)
    logger.info(
        "detect: wrote detection metrics for %d variant(s) to %s",
        len(baselines_summary),
        detect_dir,
    )


_IDENTIFY_FIT_WINDOW_CAP_S = 60.0
_IDENTIFY_HORIZON_S = 15.0
_IDENTIFY_STRIDE_S = 15.0
_IDENTIFY_NONCOMPLIANCE_THRESHOLD = 0.9
_IDENTIFY_WARMUP_S = 2.0  # sensor lag filter's zero-initial-condition transient (ADR-018/019)


def _identify_trim_warmup(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Discard raw telemetry's first `_IDENTIFY_WARMUP_S` s: the sensor lag filter
    (`telemetry/sensors.py::apply_sensor_model`'s `lfilter`) starts from a zero
    initial condition, so every flight's first ~1-2 s ramps from 0 toward the true
    signal rather than reflecting it -- the same characteristic ADR-018 found and
    trimmed for `fsa detect`. Using this transient as a replay seed (`state_from_row`
    on an untrimmed flight's first row) corrupts the *initial state* with a huge,
    spurious error, not just a few noisy samples."""
    cutoff = telemetry["t_s"].min() + _IDENTIFY_WARMUP_S
    trimmed: pd.DataFrame = telemetry[telemetry["t_s"] >= cutoff].reset_index(drop=True)
    return trimmed


def _identify_default_bounds(
    configured: dict[str, tuple[float, float]], params: list[str], nominal: VehicleParams
) -> dict[str, tuple[float, float]]:
    """Fill in any `identify.params` entry missing from `identify.bounds` with
    `+-20%` of its nominal value -- a representative prior uncertainty for an
    aerodynamic coefficient that's already roughly known, not a wide-open search
    range (FR-IDN-002: "bounds from config priors")."""
    resolved = dict(configured)
    for name in params:
        if name not in resolved:
            value = getattr(nominal, name)
            resolved[name] = (0.8 * value, 1.2 * value)
    return resolved


def _identify_signal_weights(signals: list[str]) -> dict[str, float]:
    from aerovalid.core.signals import get_signal

    weights = {}
    for name in signals:
        tol = get_signal(name).default_tolerance
        if tol:
            weights[name] = 1.0 / tol
    return weights


def _identify_non_compliant_signals(run_dir: Path, variant: str) -> set[str]:
    """Signals below `_IDENTIFY_NONCOMPLIANCE_THRESHOLD` tolerance compliance, from a
    prior `fsa validate` run; empty (no condition evidence) if none exists."""
    path = run_dir / "validation" / "metrics_signal.parquet"
    if not path.exists():
        return set()
    table = pd.read_parquet(path)
    table = table[table["variant"] == variant]
    noncompliant = table[table["tol_compliance"] < _IDENTIFY_NONCOMPLIANCE_THRESHOLD]
    return set(noncompliant["signal"].unique())


def _identify_proxy_metric(
    values: np.ndarray,
    param_names: list[str],
    nominal: VehicleParams,
    switches: object,
    dt_s: float,
    telemetry: pd.DataFrame,
    signals: list[str],
    weights: dict[str, float],
) -> float:
    """RMS of the fit's own weighted residual vector -- a cheap scalar proxy metric
    for sensitivity analysis (FR-IDN-001), reusing the exact same forward model as
    the fit itself rather than a separate, possibly-inconsistent metric."""
    from aerovalid.identify.fit import residual_vector

    residual = residual_vector(
        values,
        param_names,
        nominal,
        switches,  # type: ignore[arg-type]
        dt_s,
        telemetry,
        signals,
        weights,
        _IDENTIFY_HORIZON_S,
        _IDENTIFY_STRIDE_S,
    )
    return float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")


@app.command()
def identify(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Output-error parameter fitting, local/global sensitivity, and evidence-linked
    recommendations for a single variant, fit against one flight's first
    `_IDENTIFY_FIT_WINDOW_CAP_S` s and re-validated on held-out flights
    (FR-IDN-001...006). Regime-wise fitting (FR-IDN-004), the full Sobol sweep, and
    the surrogate/structural-term probes (FR-IDN-007/008) are the library's job
    (`identify/`, unit-tested), not this CLI's -- see DECISIONS.md ADR-019.
    """
    from aerovalid.identify.fit import fit_parameters
    from aerovalid.identify.jacobian import central_difference_jacobian, identifiability
    from aerovalid.identify.recommendations import build_recommendations
    from aerovalid.identify.revalidate import build_variant_star, revalidate_on_held_out_flights
    from aerovalid.identify.sensitivity import morris_screen
    from aerovalid.sim.variants import load_variant_yaml

    if run_id is None:
        raise ConfigError("`fsa identify` requires --run-id (an existing `fsa generate` run).")
    run_dir = Path("runs") / run_id
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise ConfigError(f"No raw/ artifacts found at {raw_dir}; run `fsa generate` first.")

    setup_logging(run_id=run_id, log_file=run_dir / "log.txt")
    study_config = load_config(config)
    configs_dir = _configs_dir(config)
    nominal = load_vehicle_params(study_config.vehicle.ref, configs_dir / "vehicles")
    variant_name = next((v for v in study_config.variants if v != "V5"), study_config.variants[0])
    variant_spec = load_variant_yaml(variant_name, configs_dir / "variants")

    flight_paths = sorted((raw_dir / "telemetry").glob("*.parquet"))
    if not flight_paths:
        raise ConfigError(f"No telemetry flights found at {raw_dir}; run `fsa generate` first.")
    fit_flight_id = flight_paths[0].stem
    fit_telemetry = _identify_trim_warmup(pd.read_parquet(flight_paths[0]))
    fit_telemetry = fit_telemetry[
        fit_telemetry["t_s"] <= fit_telemetry["t_s"].iloc[0] + _IDENTIFY_FIT_WINDOW_CAP_S
    ]

    param_names = study_config.identify.params
    bounds = _identify_default_bounds(study_config.identify.bounds, param_names, nominal)
    signals = ["alt_m"]
    weights = _identify_signal_weights(signals)

    started_at = datetime.now(UTC).isoformat()
    start_time = time.monotonic()
    identify_dir = run_dir / "identify"
    identify_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}

    fit_result = fit_parameters(
        param_names,
        bounds,
        nominal,
        variant_spec.switches,
        variant_spec.dt_s,
        fit_telemetry,
        signals,
        weights,
        horizon_s=_IDENTIFY_HORIZON_S,
        stride_s=_IDENTIFY_STRIDE_S,
    )

    def _residual_at(values: np.ndarray) -> np.ndarray:
        from aerovalid.identify.fit import residual_vector

        return residual_vector(
            values,
            param_names,
            nominal,
            variant_spec.switches,
            variant_spec.dt_s,
            fit_telemetry,
            signals,
            weights,
            _IDENTIFY_HORIZON_S,
            _IDENTIFY_STRIDE_S,
        )

    jacobian = central_difference_jacobian(_residual_at, fit_result.fitted_values)
    identifiability_result = identifiability(
        jacobian, fit_result.residuals, fit_result.fitted_values, param_names
    )

    sensitivity_table = morris_screen(
        lambda values: _identify_proxy_metric(
            values,
            param_names,
            nominal,
            variant_spec.switches,
            variant_spec.dt_s,
            fit_telemetry,
            signals,
            weights,
        ),
        param_names,
        [bounds[name] for name in param_names],
        n_trajectories=4,
        seed=study_config.study.seed,
    )
    sensitive_params = set(sensitivity_table.nlargest(2, "mu_star")["parameter"])

    non_compliant_signals = _identify_non_compliant_signals(run_dir, variant_name)
    recommendations = build_recommendations(
        target_variant=variant_name,
        nominal_values={name: getattr(nominal, name) for name in param_names},
        fit_result=fit_result,
        identifiability_result=identifiability_result,
        sensitive_params=sensitive_params,
        non_compliant_signals=non_compliant_signals or set(signals),
        signal_to_fit_params={signal: set(param_names) for signal in signals},
    )

    revalidation_table = pd.DataFrame()
    if study_config.identify.revalidate and len(flight_paths) > 1:
        held_out_ids = [p.stem for p in flight_paths[1:3]]
        held_out_telemetry = {
            fid: _identify_trim_warmup(pd.read_parquet(raw_dir / "telemetry" / f"{fid}.parquet"))
            for fid in held_out_ids
        }
        original_trajectories = {
            fid: pd.read_parquet(raw_dir / "variants" / variant_name / f"{fid}.parquet")
            for fid in held_out_ids
            if (raw_dir / "variants" / variant_name / f"{fid}.parquet").exists()
        }
        if original_trajectories:
            variant_star = build_variant_star(
                nominal, fit_result.params, variant_spec.switches, variant_spec.dt_s, name="V_star"
            )
            revalidation_table = revalidate_on_held_out_flights(
                variant_star,
                original_trajectories,
                {fid: held_out_telemetry[fid] for fid in original_trajectories},
                signals=["alt_m", "vel_air_mps"],
            )

    def _write(df: pd.DataFrame, name: str) -> None:
        path = identify_dir / name
        df.to_parquet(path, index=False)
        output_hashes[str(path.relative_to(run_dir))] = hash_file(path)

    _write(sensitivity_table, "sensitivity.parquet")
    _write(revalidation_table, "revalidation.parquet")

    fit_results_path = identify_dir / "fit_results.json"
    fit_results_path.write_text(
        json.dumps(
            {
                "variant": variant_name,
                "fit_flight_id": fit_flight_id,
                "params": fit_result.params,
                "cost": fit_result.cost,
                "success": fit_result.success,
            },
            indent=2,
        )
    )
    output_hashes[str(fit_results_path.relative_to(run_dir))] = hash_file(fit_results_path)

    identifiability_path = identify_dir / "identifiability.json"
    identifiability_path.write_text(
        json.dumps(
            {
                "ci_95": identifiability_result.ci_95,
                "condition_number": identifiability_result.condition_number,
                "poorly_identifiable": identifiability_result.poorly_identifiable,
            },
            indent=2,
        )
    )
    output_hashes[str(identifiability_path.relative_to(run_dir))] = hash_file(identifiability_path)

    recommendations_path = identify_dir / "recommendations.json"
    recommendations_path.write_text(json.dumps([r.to_dict() for r in recommendations], indent=2))
    output_hashes[str(recommendations_path.relative_to(run_dir))] = hash_file(recommendations_path)

    manifest = read_manifest(run_dir) if (run_dir / "run_manifest.json").exists() else None
    if manifest is not None:
        manifest = update_step(
            manifest,
            "identify",
            "completed",
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - start_time,
            output_hashes=output_hashes,
            extra={"variant": variant_name, "n_params": len(param_names)},
        )
        write_manifest(manifest, run_dir)
    logger.info(
        "identify: wrote fit/sensitivity/recommendations for %s to %s", variant_name, identify_dir
    )


@app.command()
def report(config: Path = _CONFIG_OPTION, run_id: str | None = _RUN_ID_OPTION) -> None:
    """Build the HTML report and summary.json (FR-RPT-001...006)."""
    from aerovalid.report.builder import build_report

    if run_id is None:
        raise ConfigError("`fsa report` requires --run-id (an existing run).")
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        raise ConfigError(f"No run found at {run_dir}; run `fsa generate` first.")
    setup_logging(run_id=run_id, log_file=run_dir / "log.txt")
    _ = load_config(config)  # validates the config even though the builder reads artifacts, not it

    started_at = datetime.now(UTC).isoformat()
    start_time = time.monotonic()
    result = build_report(run_dir, run_id)
    output_hashes = {
        str(result.html_path.relative_to(run_dir)): hash_file(result.html_path),
        str(result.summary_json_path.relative_to(run_dir)): hash_file(result.summary_json_path),
    }

    manifest = read_manifest(run_dir) if (run_dir / "run_manifest.json").exists() else None
    if manifest is not None:
        manifest = update_step(
            manifest,
            "report",
            "completed",
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - start_time,
            output_hashes=output_hashes,
        )
        write_manifest(manifest, run_dir)
    logger.info("report: wrote %s", result.html_path)


@app.command(name="run-all")
def run_all(
    config: Path = _CONFIG_OPTION,
    run_id: str | None = _RUN_ID_OPTION,
    seed: int | None = _SEED_OPTION,
    profile: str = _PROFILE_OPTION,
    set_: list[str] = _SET_OPTION,
    force: bool = _FORCE_OPTION,
) -> None:
    """Orchestrate the full pipeline end to end (FR-OPS-002): generate -> preprocess
    -> validate -> analyze -> predict -> detect -> identify -> report.

    Resumability comes from `generate`'s own config-hash skip (the only stochastic,
    expensive step, already idempotent -- re-running it with unchanged inputs is a
    no-op unless `--force`); every later step is a deterministic function of its
    inputs, so re-invoking `run-all` after an interruption reproduces the same final
    outputs rather than literally skipping completed steps.
    """
    resolved_run_id = _resolve_run_id(run_id)
    generate(
        config=config,
        run_id=resolved_run_id,
        seed=seed,
        profile=profile,
        set_=set_,
        force=force,
    )
    preprocess(config=config, run_id=resolved_run_id)
    validate(config=config, run_id=resolved_run_id)
    analyze(config=config, run_id=resolved_run_id)
    predict(config=config, run_id=resolved_run_id)
    detect(config=config, run_id=resolved_run_id)
    identify(config=config, run_id=resolved_run_id)
    report(config=config, run_id=resolved_run_id)
    logger.info("run-all: pipeline complete for run %s", resolved_run_id)


@app.command()
def dashboard(run_id: str = _DASHBOARD_RUN_ID_OPTION) -> None:
    """Launch the Streamlit dashboard for a completed run (FR-RPT-004)."""
    import os
    import subprocess
    import sys

    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        raise ConfigError(f"No run found at {run_dir}; run `fsa generate` first.")

    app_path = Path(__file__).parent / "app" / "main.py"
    env = {**os.environ, "AEROVALID_DASHBOARD_RUN_ID": run_id}
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], env=env, check=True)


# Known fidelity order from each variant config's own "Fidelity rank" comment
# (configs/variants/V*.yaml), best (1) to worst (6); V2's "4-5, condition-dependent"
# is encoded as the midpoint. Used only by `fsa benchmark`'s S2 check.
_BENCHMARK_EXPECTED_FIDELITY_RANK: dict[str, float] = {
    "V5": 1.0,
    "V4": 2.0,
    "V0": 4.0,
    "V2": 4.5,
    "V1": 5.0,
    "V3": 6.0,
}
_BENCHMARK_VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5"]
_BENCHMARK_N_FLIGHTS = 4
_BENCHMARK_SCENARIO = "doublet_series"


def _benchmark_criterion(
    description: str, target: str, computed: object, passed: bool | None, note: str = ""
) -> dict[str, object]:
    return {
        "description": description,
        "target": target,
        "computed": computed,
        "passed": passed,
        "note": note,
    }


def _benchmark_overrides() -> list[str]:
    return [
        f"generate.n_flights={_BENCHMARK_N_FLIGHTS}",
        f"generate.scenarios=[{_BENCHMARK_SCENARIO}]",
        f"variants=[{', '.join(_BENCHMARK_VARIANTS)}]",
    ]


def _benchmark_s1(config: Path) -> dict[str, object]:
    """S1: deterministic end-to-end run, < 5 min (vision doc §8)."""
    import shutil

    run_a, run_b = Path("runs") / "bench_s1_a", Path("runs") / "bench_s1_b"
    for run_dir in (run_a, run_b):
        if run_dir.exists():
            shutil.rmtree(run_dir)

    overrides = _benchmark_overrides()
    start = time.monotonic()
    generate(config=config, run_id="bench_s1_a", seed=1, profile="demo", set_=overrides, force=True)
    duration_s = time.monotonic() - start
    generate(config=config, run_id="bench_s1_b", seed=1, profile="demo", set_=overrides, force=True)

    dir_a, dir_b = run_a / "raw", run_b / "raw"
    files_a = sorted(str(p.relative_to(dir_a)) for p in dir_a.rglob("*.parquet"))
    files_b = sorted(str(p.relative_to(dir_b)) for p in dir_b.rglob("*.parquet"))
    identical = files_a == files_b and all(
        hash_file(dir_a / f) == hash_file(dir_b / f) for f in files_a
    )
    return _benchmark_criterion(
        "S1: end-to-end demo run is deterministic",
        "< 5 min on laptop; identical outputs for same seed",
        {"duration_s": duration_s, "identical_outputs": identical},
        bool(identical and duration_s < 300.0),
    )


def _benchmark_run_seed(config: Path, seed: int) -> Path:
    run_id = f"bench_seed{seed}"
    generate(
        config=config,
        run_id=run_id,
        seed=seed,
        profile="demo",
        set_=_benchmark_overrides(),
        force=True,
    )
    preprocess(config=config, run_id=run_id)
    validate(config=config, run_id=run_id)
    predict(config=config, run_id=run_id)
    detect(config=config, run_id=run_id)
    identify(config=config, run_id=run_id)
    return Path("runs") / run_id


def _benchmark_s2_rho(run_dir: Path) -> float | None:
    from scipy.stats import spearmanr

    ranking_path = run_dir / "validation" / "variant_ranking.parquet"
    if not ranking_path.exists():
        return None
    ranking = pd.read_parquet(ranking_path).set_index("variant")
    common = [v for v in ranking.index if v in _BENCHMARK_EXPECTED_FIDELITY_RANK]
    if len(common) < 3:
        return None
    expected = [_BENCHMARK_EXPECTED_FIDELITY_RANK[v] for v in common]
    actual = [float(ranking.loc[v, "mean_rank"]) for v in common]
    rho, _ = spearmanr(expected, actual)
    return float(rho)


def _benchmark_s4_reduction_pct(run_dir: Path) -> float | None:
    cv_results = _read_parquet_or_empty_benchmark(run_dir / "predict" / "cv_results.parquet")
    if cv_results.empty or "rmse_reduction_vs_sim_only_pct" not in cv_results.columns:
        return None
    hybrid = cv_results[cv_results["task"] == "hybrid"]
    return float(hybrid["rmse_reduction_vs_sim_only_pct"].median()) if not hybrid.empty else None


def _benchmark_s5_coverage(run_dir: Path) -> float | None:
    intervals = _read_parquet_or_empty_benchmark(run_dir / "predict" / "intervals.parquet")
    return float(intervals["coverage"].mean()) if not intervals.empty else None


def _benchmark_s6(run_dir: Path) -> tuple[float | None, float | None, float | None]:
    metrics = _read_parquet_or_empty_benchmark(run_dir / "detect" / "detection_metrics.parquet")
    if metrics.empty:
        return None, None, None
    overall = metrics[metrics["fault_type"].isna()]
    if overall.empty:
        return None, None, None
    delay = overall["median_delay_s"].dropna()
    return (
        float(delay.median()) if not delay.empty else None,
        float(overall["far_per_hour"].median()),
        float(overall["recall"].median()),
    )


def _benchmark_s7_relative_errors(run_dir: Path, nominal: VehicleParams) -> list[float]:
    from aerovalid.sim.truth import TRUTH_PARAM_SIGMAS

    fit_path = run_dir / "identify" / "fit_results.json"
    truth_path = run_dir / "raw" / "meta" / "truth_params.json"
    if not fit_path.exists() or not truth_path.exists():
        return []
    fit_results = json.loads(fit_path.read_text())
    truth_perturbations = json.loads(truth_path.read_text())

    errors = []
    for name, fitted_value in fit_results["params"].items():
        if name not in TRUTH_PARAM_SIGMAS or name not in truth_perturbations:
            continue
        true_value = getattr(nominal, name) * (1.0 + truth_perturbations[name])
        if true_value != 0:
            errors.append(abs(fitted_value - true_value) / abs(true_value))
    return errors


def _benchmark_s8_reduction_pct(run_dir: Path) -> float | None:
    revalidation = _read_parquet_or_empty_benchmark(run_dir / "identify" / "revalidation.parquet")
    return float(revalidation["rmse_reduction_pct"].median()) if not revalidation.empty else None


def _read_parquet_or_empty_benchmark(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _benchmark_median_criterion(
    description: str, target: str, values: list[float], threshold: float, direction: str
) -> dict[str, object]:
    """`direction`: `"le"` (lower is better/passing) or `"ge"` (higher is better)."""
    if not values:
        return _benchmark_criterion(description, target, None, None, note="not evaluated: no data")
    median = float(np.median(values))
    passed = median <= threshold if direction == "le" else median >= threshold
    return _benchmark_criterion(
        description, target, {"median": median, "n_seeds": len(values)}, passed
    )


@app.command()
def benchmark(config: Path = _CONFIG_OPTION, seeds: str = _SEEDS_OPTION) -> None:
    """
    Run the synthetic benchmark suite across seeds and report S1-S9 pass/fail
    (FR-OPS-004, vision doc §8): S1 (determinism) is checked once; S2, S4-S8 are
    computed per seed from a small (`_BENCHMARK_N_FLIGHTS`-flight) run of the full
    pipeline and aggregated as a median across seeds. S3 (condition analysis finds
    the injected flaw's regime) and S9 (code quality) are not computed here -- S3
    needs a per-variant-flaw-to-regime mapping this reduced-scope check doesn't have,
    and S9 is a codebase property (`make lint test`), not a simulation output -- see
    DECISIONS.md ADR-021.
    """
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    configs_dir = _configs_dir(config)
    study_config = load_config(config)
    nominal = load_vehicle_params(study_config.vehicle.ref, configs_dir / "vehicles")

    s2_values, s4_values, s5_values = [], [], []
    s6_delays, s6_fars, s6_recalls = [], [], []
    s7_errors: list[float] = []
    s8_values: list[float] = []

    for seed in seed_list:
        run_dir = _benchmark_run_seed(config, seed)
        rho = _benchmark_s2_rho(run_dir)
        if rho is not None:
            s2_values.append(rho)
        s4 = _benchmark_s4_reduction_pct(run_dir)
        if s4 is not None:
            s4_values.append(s4)
        s5 = _benchmark_s5_coverage(run_dir)
        if s5 is not None:
            s5_values.append(s5)
        delay, far, recall = _benchmark_s6(run_dir)
        if delay is not None:
            s6_delays.append(delay)
        if far is not None:
            s6_fars.append(far)
        if recall is not None:
            s6_recalls.append(recall)
        s7_errors.extend(_benchmark_s7_relative_errors(run_dir, nominal))
        s8 = _benchmark_s8_reduction_pct(run_dir)
        if s8 is not None:
            s8_values.append(s8)

    results = {
        "S1": _benchmark_s1(config),
        "S2": _benchmark_median_criterion(
            "S2: variant ranking recovers known fidelity ordering",
            "Spearman rho >= 0.9",
            s2_values,
            0.9,
            "ge",
        ),
        "S3": _benchmark_criterion(
            "S3: condition analysis finds the injected flaw's regime",
            "flaw regime in top-2 bins in >= 90% of trials",
            None,
            None,
            note="not evaluated: needs a per-variant flaw-to-regime mapping (see ADR-021)",
        ),
        "S4": _benchmark_median_criterion(
            "S4: hybrid correction reduces held-out RMSE vs raw simulation",
            ">= 25% median reduction",
            s4_values,
            25.0,
            "ge",
        ),
        "S5": _benchmark_median_criterion(
            "S5: prediction intervals are calibrated",
            "empirical 90% coverage in [0.86, 0.94]",
            s5_values,
            0.86,
            "ge",
        ),
        "S6_delay": _benchmark_median_criterion(
            "S6: deviation detector median delay", "<= 5 s", s6_delays, 5.0, "le"
        ),
        "S6_far": _benchmark_median_criterion(
            "S6: deviation detector false-alarm rate", "<= 1/flight-hour", s6_fars, 1.0, "le"
        ),
        "S6_recall": _benchmark_median_criterion(
            "S6: deviation detector recall", ">= 0.9", s6_recalls, 0.9, "ge"
        ),
        "S7": _benchmark_median_criterion(
            "S7: parameter identification recovers injected aerodynamic errors",
            "median relative error <= 10%",
            s7_errors,
            0.10,
            "le",
        ),
        "S8": _benchmark_median_criterion(
            "S8: re-validation after refit improves held-out variant error",
            ">= 30% RMSE reduction",
            s8_values,
            30.0,
            "ge",
        ),
        "S9": _benchmark_criterion(
            "S9: code quality",
            "coverage >= 85%, mypy strict clean, CI green",
            None,
            None,
            note=(
                "not evaluated here: run `make lint test` "
                "(a codebase property, not a simulation output)"
            ),
        ),
    }

    benchmark_dir = Path("runs") / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps({"seeds": seed_list, "results": results}, indent=2))
    logger.info("benchmark: wrote %s", summary_path)


if __name__ == "__main__":
    app()
