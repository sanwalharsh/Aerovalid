# CLAUDE.md — AeroValidate

> Read this file first. It is the entry point for every Claude Code session on this repo.
> Full specs live in `docs/`. When this file and a spec disagree, the spec wins; then fix this file.

## 1. What this project is

**AeroValidate** (`aerovalid`) is a Python toolkit and research pipeline for **AI-assisted flight-simulation validation and predictive analysis** for aerospace / defence-class air vehicles.

It does five things:

1. **Generates / ingests** multi-variant flight-simulation outputs and telemetry (synthetic first; real data via adapters).
2. **Prepares & aligns** them (time sync, cleaning, resampling, feature extraction, flight-phase tagging).
3. **Validates** each simulation variant against telemetry (error metrics, statistics, condition-dependent analysis).
4. **Predicts & detects** (residual prediction, hybrid physics+ML correction, uncertainty, early deviation detection).
5. **Explains** (parameter sensitivity/identification, ranked model-refinement recommendations, reports, dashboard).

The research problem and questions (RQ1–RQ4) are in `docs/00_vision.md`. Every feature should trace to an RQ or a requirement ID.

## 2. Hard rules (non-negotiable)

- **Unclassified / synthetic / open data only.** Never add real proprietary, export-controlled, or classified data, parameters, or vehicle specs to the repo. The reference vehicle is a **generic fixed-wing UAV/aircraft** with invented-but-plausible parameters. No weapon systems, no weapon-guidance logic.
- **Reproducibility.** Every stochastic function takes an explicit `seed` / `numpy.random.Generator`. Every pipeline run writes a `run_manifest.json` (config hash, git SHA, package versions, seeds, input hashes).
- **SI units everywhere internally** (m, s, kg, rad, N, Pa, K). Convert only at I/O boundaries. Column names carry the unit suffix convention in `docs/03_architecture_and_data_spec.md §4`.
- **No data leakage.** Splits are by *flight/run*, never by random row. Preprocessing statistics are fit on training folds only.
- **Typed, tested, linted.** Public functions have type hints and docstrings with units. No merge of a phase until its Definition of Done (`docs/06_implementation_plan.md`) passes.
- **Don't invent results.** Findings in reports must be computed from data in the run. No hard-coded conclusions.
- **Large data and outputs are git-ignored** (`data/`, `runs/`, `reports/`). Only configs, code, docs, and tiny fixtures are committed.

## 3. Tech stack

Python ≥ 3.11 · numpy · scipy · pandas · pyarrow (Parquet) · scikit-learn · statsmodels · ruptures · SALib · pydantic v2 · PyYAML · typer (CLI) · matplotlib + plotly · Jinja2 (reports) · streamlit (dashboard) · pytest + hypothesis + pytest-cov · ruff + mypy · optional extras: `torch` (sequence models), `lightgbm`, `pymavlink`/`pyulog` (real-log adapters), `mlflow`.

Packaging: `pyproject.toml` (hatchling), `src/` layout. Extras: `[ml]`, `[adapters]`, `[dev]`, `[all]`.

## 4. Repo layout (target)

```
aerovalid/
├─ CLAUDE.md  README.md  pyproject.toml  Makefile
├─ configs/            # YAML: default.yaml, scenarios/*.yaml, variants/*.yaml, faults/*.yaml
├─ docs/               # specs (this pack)
├─ src/aerovalid/
│  ├─ core/            # units, atmosphere, constants, schema, config, manifest, logging, errors
│  ├─ sim/             # dynamics, aero, propulsion, environment, controllers, integrators, variants, truth
│  ├─ telemetry/       # sensors (noise/fault models), synthetic generator, adapters (csv, parquet, ulog, dataflash)
│  ├─ prep/            # cleaning, sync, resample, features, phases, qc
│  ├─ validate/        # metrics, windowed, compare, stats, conditions, ranking
│  ├─ predict/         # datasets, baselines, residual models, hybrid, seq models, uncertainty, cv
│  ├─ detect/          # statistical detectors, ml detectors, changepoint, evaluation
│  ├─ identify/        # parameter fitting, sensitivity, identifiability, recommendations
│  ├─ report/          # figures, templates, builder
│  ├─ app/             # streamlit dashboard
│  └─ cli.py           # typer entrypoint `fsa`
├─ tests/              # unit/, integration/, e2e/, regression/, fixtures/
├─ notebooks/          # exploratory only; never imported by src
├─ data/ runs/ reports/   # git-ignored
```

## 5. Commands (create these in the Makefile; keep them working)

```
make setup        # create venv, pip install -e ".[all]"
make lint         # ruff check . && ruff format --check . && mypy src
make test         # pytest -q --cov=aerovalid --cov-fail-under=85
make test-fast    # pytest -q -m "not slow"
make demo         # fsa run-all --config configs/default.yaml --profile demo  (small, <5 min)
make dashboard    # streamlit run src/aerovalid/app/main.py
```

CLI (`fsa --help`): `generate`, `ingest`, `preprocess`, `validate`, `analyze`, `predict`, `detect`, `identify`, `report`, `run-all`, `dashboard`. Each step reads/writes versioned artifacts under `runs/<run_id>/` (see data spec §6) and is idempotent.

## 6. How to work (workflow for Claude Code)

1. Read `docs/06_implementation_plan.md` and find the **first unchecked phase**. Work phases in order; do not skip ahead.
2. Before coding a module, read its section in the FRD (`docs/02_FRD.md`) and the relevant spec (`03`, `04`, `05`).
3. Write tests alongside code (TDD where numerical truth is known — see test plan). Commit small, with message `feat(<module>): <summary> [FR-XXX-NNN]`.
4. After each phase: run `make lint test`, tick the phase's DoD checklist in the plan, and update `docs/CHANGELOG.md`.
5. If a requirement is ambiguous, pick the simplest option consistent with the docs, record it in `docs/DECISIONS.md` (ADR-style, one paragraph), and continue. Don't block.
6. Keep modules decoupled: `sim` knows nothing about `predict`; `validate` only consumes aligned datasets (schema §4). Cross-module data moves via the canonical schema, not ad-hoc DataFrames.

## 7. Coding conventions

- `ruff` (line length 100), `mypy --strict` on `src/`. Dataclasses/pydantic models for configs and result objects; no bare dicts across module boundaries.
- Pure functions for numerics; side effects (I/O) only in `cli.py`, adapters, and `report/`.
- Vectorise with numpy; no per-row Python loops on time-series > 10⁴ samples.
- Logging via `logging` (no `print`) with run_id in the record.
- Plots: every figure function returns a `Figure` and takes no global state; saved by `report/` or CLI.
- Docstrings: NumPy style, include units and shapes.
- Errors: raise subclasses of `AeroValidError` with actionable messages.

## 8. Definition of "done" for any task

Code + tests + docstrings + config knobs documented + requirement IDs referenced + `make lint test` green + (if user-visible) an example in `README.md` or the demo run.

## 9. Glossary quick refs

Variant = a simulation model configuration · Truth = the high-fidelity reference model that generates synthetic "observed" flight · Telemetry = sensor-measured (noisy) observations · Residual = observed − simulated · Replay mode = variant driven by recorded control inputs · Windowed (multi-step) prediction = re-initialise variant from telemetry state every H seconds · Tolerance = per-signal engineering error threshold · Lead time = time between detection and tolerance exceedance. Full glossary: `docs/00_vision.md §9`.
