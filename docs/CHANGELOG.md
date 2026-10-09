# Changelog

All notable changes to AeroValidate, grouped by build phase (see `06_implementation_plan.md`).

## Phase 0 — Scaffolding

- Added `pyproject.toml` (hatchling, src layout, extras `ml`/`adapters`/`dev`/`all`), `Makefile`
  (`setup`, `lint`, `test`, `test-fast`, `demo`, `dashboard`, `config-ref`), `.gitignore`,
  `.pre-commit-config.yaml` (ruff, mypy, nbstripout), `.github/workflows/ci.yml` (lint+test on
  Linux, smoke matrix on macOS/Windows).
- Added `src/aerovalid/core/`: `errors.py` (`AeroValidError` hierarchy), `logging.py`
  (structured, run_id-tagged), `rng.py` (`SeedSequence`-based deterministic generators),
  `angles.py` (wrap/circular stats), `signals.py` (canonical signal registry), `config.py`
  (pydantic `StudyConfig` tree + `load_config` with `include:` and `--set` overrides),
  `manifest.py` (`run_manifest.json` writer), `schema.py` (pandera canonical-table schema).
- Added `src/aerovalid/cli.py`: `fsa` typer app with all CLAUDE.md §5 subcommands stubbed
  (`NotImplementedError`, no silent `pass`), each naming the phase that implements it.
- Added `configs/default.yaml` matching `StudyConfig`'s defaults (data spec §7).
- Added `docs/DECISIONS.md` (ADR-001..004), `docs/07_nfr_risks_governance.md`,
  `scripts/gen_config_reference.py` (introspects the pydantic models to regenerate
  `docs/CONFIG_REFERENCE.md`).
- Added `tests/unit/` coverage for config (round-trip, overrides, `ConfigError` messages),
  manifest (write/read/update_step), RNG (determinism, independence), angles (wrap
  idempotence, circular mean), signals (lookup + unknown-signal error), schema (valid/invalid
  tables), and the CLI (`--help` lists every command; each stub raises `NotImplementedError`).

DoD: `pip install -e ".[dev]"` succeeds; `fsa --help` lists all commands; `make lint test`
green; config round-trip passes; manifest written correctly. See `docs/DECISIONS.md` for the
non-obvious calls made while resolving ambiguity.

## Phase 1 — Physics & Simulation

- Added `core/atmosphere.py` (ISA, 0-20km, optional ΔT offset) and `configs/vehicles/generic_uav_v1.yaml`.
- Added `src/aerovalid/sim/`: `protocols.py` (`State`, `ControlHistory`, `SimResult`,
  `Scenario`, `SimulatorProtocol`, `VehicleParams`/`StructuralSwitches` loaders), `aero.py`
  (CL/CD with smooth stall roll-off, Prandtl-Glauert, wave drag), `propulsion.py` (thrust
  lapse/Mach, fuel flow), `environment.py` (steady wind + shear + seeded Dryden-style
  turbulence), `integrators.py` (RK4 + adaptive `solve_ivp` wrapper), `dynamics.py` (point-mass
  3-D equations of motion, quasi-steady and actuator-lag state vectors, derived measurements),
  `controllers.py` (speed-hold PI, altitude-hold PID, rate-limited bank), `maneuvers.py`
  (9 manoeuvre primitives + scenario YAML loader), `truth.py` (`TruthModel`: hidden
  study-fixed parameter perturbations, closed-loop controller-driven flight), `variants.py`
  (`VariantModel`, V0-V5 incl. V5's truth-derived +-2% noise), `runners.py` (`replay_run`,
  `run_windowed`).
- Added `configs/variants/V0.yaml`..`V5.yaml`, `configs/scenarios/*.yaml` (5 scenarios),
  `configs/benchmark_ground_truth.yaml` (fidelity order + regime-flaw tags for `fsa benchmark`).
- Added `notebooks/01_sim_sanity.ipynb` (truth vs V0 replay altitude/airspeed overlay).
- Tests: ISA table values, CL/CD curves, thrust formulas, RK4 convergence order (~4),
  ballistic analytic parabola (<1e-6 rel at dt=0.001), level trim (0.1% over 60s), steady
  turn radius/load factor (0.5%), energy balance, controllers, manoeuvre primitives, truth
  determinism, variant overrides, windowed re-init. Integration (`@pytest.mark.slow`): every
  scenario x every variant (V0-V5) runs to completion with no NaN/inf; V5-vs-truth open-loop
  altitude RMSE measured at 2.17 m over a 5-minute trim flight (see DECISIONS.md ADR-007 for
  why the test asserts the 5 m registry tolerance rather than the spec's nominal "< 1 m").

DoD: all sim-spec §10 tests pass; every scenario x variant runs without NaN; V5-vs-truth RMSE
documented; `notebooks/01_sim_sanity.ipynb` plots a trajectory.

## Phase 2 — Synthetic Telemetry & Ingestion

- Added `telemetry/sensors.py` (per-signal `SensorModel`: white noise, bias, random-walk
  drift/gyro drift, scale, first-order lag via `scipy.signal.lfilter`, quantisation,
  dropout; `apply_sensors` corrupts a whole truth DataFrame into telemetry).
- Added `telemetry/defects.py` (`TimingDefectParams`/`apply_timing_defects`: constant clock
  offset, linear drift, jitter, duplicate timestamps -- collapsed onto their nearest
  preceding non-duplicate sample, not a naive shift -- dropouts, burst gaps).
- Added `telemetry/faults.py` (all 7 fault types from sim spec §7; `sample_fault_schedule`
  draws at most one fault per flight; `build_param_fault_fn`/`build_wind_shear_fn` produce
  the `t_s -> FaultState` / wind-shear closures `sim/truth.py` now accepts natively;
  `apply_sensor_bias_faults` for the sensor-level fault type).
- Added `telemetry/generator.py` (`sample_conditions`: Sobol, balance-preserving
  `random_base2` + truncate; `generate_flight`/`generate_all_flights`: full per-flight
  pipeline -- truth, decimation to telemetry rate, sensor+timing defects, every variant
  replayed from post-defect telemetry controls, V5's wind estimated from the same flight's
  telemetry).
- Added `telemetry/adapters/base.py` (`ColumnMapping`, `Adapter` protocol) and
  `adapters/csv_parquet.py` (unit/time-base conversion into the canonical schema).
- Implemented `fsa generate` (writes `runs/<run_id>/raw/{truth,telemetry,variants/<v>,
  meta}/...` per data spec §6, idempotent on matching config hash, manifest-tracked) and
  `fsa ingest` (adapter + mapping YAML -> `runs/<run_id>/ingested/`).
- Added `configs/sensors/default.yaml`, `configs/faults/*.yaml` (7 files), a fixture CSV +
  mapping YAML for the non-SI-units adapter test.
- `sim/truth.py` and `sim/dynamics.py` extended (`delta_t_k`, `param_fault_fn`,
  `wind_shear_fn`) to receive this phase's time-varying fault effects without `sim/`
  knowing anything about `telemetry/`'s fault schedule representation.
- Manually verified `fsa generate` end to end: two independent runs with the same config
  produce byte-identical Parquet/JSON artifacts (`shasum` diff clean); idempotent re-run
  with the same `--run-id` and no `--force` skips correctly.

DoD: `fsa generate` produces exactly N flights deterministically (hash-identical re-run,
verified manually at small scale and by an automated `@pytest.mark.slow` CLI test);
sensor noise/bias recovery within 5%; fixture CSV with non-SI units ingests to SI schema
correctly; per-flight fault log (`meta/<flight>.json` + aggregate `faults.json`) present.
ULog/DataFlash adapter (FR-TEL-006, priority S) deferred -- see DECISIONS.md ADR-009.
