# 06 — Implementation Plan, Definition of Done & Test Plan

Work the phases **in order**. Tick boxes as you go (`[x]`). Each phase ends with `make lint test` green, CHANGELOG entry, and the phase DoD satisfied.

---

## Phase 0 — Scaffolding  `[ ]`
Tasks
- [ ] `pyproject.toml` (src layout, extras `ml`, `adapters`, `dev`, `all`), `Makefile`, `.gitignore` (data/, runs/, reports/), `.pre-commit-config.yaml`, `ruff`/`mypy` config, GitHub Actions workflow.
- [ ] Package skeleton per CLAUDE.md §4 with `__init__.py`, `py.typed`.
- [ ] `core/`: config models (pydantic) + loader with includes/overrides, manifest writer, logging, errors, RNG helper, units/angles, signal registry.
- [ ] `cli.py` with typer app and stub subcommands.
- [ ] `docs/DECISIONS.md`, `docs/CHANGELOG.md`, `docs/CONFIG_REFERENCE.md` generator script.
DoD: `pip install -e ".[dev]"` works; `fsa --help` lists commands; unit tests for config/manifest/RNG/angles pass; CI green. (FR-COR-001,002,004,006,007; FR-OPS-001 partial, 003, 005)

## Phase 1 — Physics & simulation  `[ ]`
- [ ] `core/atmosphere.py` (ISA, ΔT offset) + tests.
- [ ] `sim/aero.py`, `sim/propulsion.py`, `sim/environment.py` (wind, shear, Dryden), `sim/integrators.py` (RK4, adaptive wrapper).
- [ ] `sim/dynamics.py` (state derivative, derived measurements nx/nz/attitude).
- [ ] `sim/controllers.py` + `sim/maneuvers.py` + scenario YAML loader.
- [ ] `sim/truth.py` (α-dynamics, actuator lag, hidden perturbations) and `sim/variants.py` (V0–V6 from YAML), replay & windowed runners.
- [ ] Simulator self-tests (sim spec §10).
DoD: all sim-spec tests pass; every scenario × variant runs w/o NaN; V5-vs-truth RMSE bound documented; notebook `notebooks/01_sim_sanity.ipynb` plots a trajectory. (FR-COR-003, FR-SIM-001…011)

## Phase 2 — Synthetic telemetry & ingestion  `[ ]`
- [ ] `telemetry/sensors.py` (noise, bias, drift, lag, quantisation), `defects.py` (timing), `faults.py` (fault schedule into truth), `generator.py` (Sobol conditions, orchestrates flights, parallel with joblib).
- [ ] Adapters: CSV/Parquet with mapping + unit conversion; optional ULog/DataFlash behind extra.
- [ ] `fsa generate`, `fsa ingest` commands; artifact layout per data spec §6; schema validation.
DoD: `fsa generate --profile demo` produces 12 flights deterministically (hash-identical re-run); statistical sensor tests pass; fixture CSV with non-SI units ingests correctly; faults log present. (FR-TEL-001…007, FR-COR-005)

## Phase 3 — Preprocessing & alignment  `[ ]`
- [ ] QC, cleaning (Hampel, gaps), sync (xcorr + drift + CI), resampling, features, phases, condition tags, state-estimate smoother, aligned dataset builder, windowed builder.
- [ ] `fsa preprocess` writes `qc/`, `sync/`, `aligned/`.
DoD: recovery tests: offset ±0.5 sample, drift within 10 ppm, spikes ≥ 95% removed, phase accuracy ≥ 90%; aligned schema valid; row counts equal across variants. (FR-PRP-001…011)

## Phase 4 — Validation & condition analysis  `[ ]`
- [ ] Metrics library (all FR-VAL-001 metrics), windowed/phase/bin aggregation, trajectory metrics, tolerance compliance.
- [ ] Comparison stats (block bootstrap, Wilcoxon+Holm, Friedman/Nemenyi, composite score, ranking CIs).
- [ ] Condition analysis (heatmaps data, GBM importance, PDP, Kruskal/Dunn, MixedLM), residual diagnostics, DTW.
- [ ] `fsa validate`, `fsa analyze`; E1–E3 scripts.
DoD: metrics validated against reference formulas on fixtures; S2 and S3 pass on benchmark seeds; null-pair false-positive rate within [0.02, 0.08] at α = 0.05. (FR-VAL-001…010)

## Phase 5 — Predictive analysis  `[ ]`
- [ ] Window/dataset builder, splitters (LOFO, grouped K-fold, leave-condition-out), model zoo wrappers (`fit/predict/predict_interval`), baselines, nested CV + search, explainability.
- [ ] Tasks A, B, C; conformal + quantile; extrapolation study; optional torch GRU/TCN.
- [ ] `fsa predict`; E4–E5 scripts.
DoD: leakage tests pass; S4 and S5 pass; skill-score tables generated; artifacts per data spec. (FR-PRD-001…010)

## Phase 6 — Deviation detection  `[ ]`
- [ ] Baseline normalisation, statistical detectors, ML detectors, PELT/BinSeg onset estimation, threshold calibration, early-warning predictor, streaming API, evaluation protocol.
- [ ] `fsa detect`; E6 script.
DoD: calibrated FA within ±25% of target; S6 passes; batch == streaming test passes. (FR-DET-001…008)

## Phase 7 — Identification, sensitivity & recommendations  `[ ]`
- [ ] Local Jacobian + identifiability; output-error fitting with windows; regime-wise fits; Morris/Sobol; structural-term probe; surrogate (optional).
- [ ] Recommendation engine + re-validation; `fsa identify`; E7–E8 scripts.
DoD: S7 and S8 pass; non-identifiable case flagged; every recommendation has ≥ 2 evidence types or is labelled hypothesis. (FR-IDN-001…008)

## Phase 8 — Reporting & dashboard  `[ ]`
- [ ] Figure library (FR-RPT-002), Jinja2 report, summary.json, tables export, executive-summary generator (numbers-only), Streamlit dashboard, optional PDF.
- [ ] `fsa report`, `fsa dashboard`, `fsa run-all`, `fsa benchmark`.
DoD: demo report has all sections populated, zero template placeholders; dashboard loads demo run; `fsa benchmark` writes pass/fail for S1–S9. (FR-RPT-001…006, FR-OPS-002, 004)

## Phase 9 — Hardening & release  `[ ]`
- [ ] Coverage ≥ 85%, mypy strict clean, docs complete (README quick-start, CONFIG_REFERENCE, METHODS, USER_GUIDE for real-data mapping), Dockerfile, performance profiling vs NFR targets, example real-data mapping, tag v1.0.0.
DoD: PRD §9 acceptance criteria all met.

---

## Test plan

| Level | Scope | Tools | Examples |
|---|---|---|---|
| **Unit** | Pure functions | pytest, hypothesis | ISA table values; metric formulas; angle wrap; sensor σ recovery; Hampel; xcorr lag |
| **Property-based** | Invariants | hypothesis | wrapping idempotence; resample∘resample ≈ resample; metrics invariant to permutation of equal-length chunks; RMSE ≥ \|ME\|; Theil U_M+U_S+U_C = 1 |
| **Numerical / physics** | Simulator correctness | pytest + analytic | Level trim, ballistic, turn radius, RK4 order, energy balance |
| **Statistical recovery** | Estimators recover injected truth | pytest (seeded) | Offset/drift recovery, parameter identification, detector FA calibration, conformal coverage |
| **Integration** | Stage-to-stage contracts | pytest tmp_path | generate→preprocess→validate on 3 flights < 60 s |
| **End-to-end** | Whole pipeline | `make demo` in CI | Report exists; no NaN in summary.json |
| **Regression / golden** | Stable outputs | pytest-regressions (tolerant) | Metrics table for fixed seed within 1e-6 |
| **Leakage** | Split hygiene | custom tests | No window across flights; scaler fit on train only; no forbidden columns in features |
| **Performance** | Budgets | pytest-benchmark (marked slow) | Preprocess 10 flights < 60 s |
| **Docs** | Examples run | doctest / nbmake | README snippets |

Marker conventions: `@pytest.mark.slow`, `@pytest.mark.req("FR-…")`. Test files mirror the source tree. Fixtures: tiny synthetic flights (≤ 60 s) generated at test time with fixed seeds; avoid committing big data.

## Suggested first Claude Code prompt

> Read `CLAUDE.md` and everything in `docs/`. Start with Phase 0 in `docs/06_implementation_plan.md`. Create the scaffolding, make `make lint test` pass, update the checklist, and stop for review before Phase 1.

Then iterate phase by phase: *"Proceed with Phase N; follow the DoD; record any ambiguity in DECISIONS.md."*
