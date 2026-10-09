# 06 — Implementation Plan, Definition of Done & Test Plan

Work the phases **in order**. Tick boxes as you go (`[x]`). Each phase ends with `make lint test` green, CHANGELOG entry, and the phase DoD satisfied.

---

## Phase 0 — Scaffolding  `[x]`
Tasks
- [x] `pyproject.toml` (src layout, extras `ml`, `adapters`, `dev`, `all`), `Makefile`, `.gitignore` (data/, runs/, reports/), `.pre-commit-config.yaml`, `ruff`/`mypy` config, GitHub Actions workflow.
- [x] Package skeleton per CLAUDE.md §4 with `__init__.py`, `py.typed`.
- [x] `core/`: config models (pydantic) + loader with includes/overrides, manifest writer, logging, errors, RNG helper, units/angles, signal registry.
- [x] `cli.py` with typer app and stub subcommands.
- [x] `docs/DECISIONS.md`, `docs/CHANGELOG.md`, `docs/CONFIG_REFERENCE.md` generator script.
DoD: `pip install -e ".[dev]"` works; `fsa --help` lists commands; unit tests for config/manifest/RNG/angles pass; CI green. (FR-COR-001,002,004,006,007; FR-OPS-001 partial, 003, 005)

## Phase 1 — Physics & simulation  `[x]`
- [x] `core/atmosphere.py` (ISA, ΔT offset) + tests.
- [x] `sim/aero.py`, `sim/propulsion.py`, `sim/environment.py` (wind, shear, Dryden), `sim/integrators.py` (RK4, adaptive wrapper).
- [x] `sim/dynamics.py` (state derivative, derived measurements nx/nz/attitude).
- [x] `sim/controllers.py` + `sim/maneuvers.py` + scenario YAML loader.
- [x] `sim/truth.py` (α-dynamics, actuator lag, hidden perturbations) and `sim/variants.py` (V0–V5 from YAML), replay & windowed runners.
- [x] Simulator self-tests (sim spec §10).
DoD: all sim-spec tests pass; every scenario × variant runs w/o NaN; V5-vs-truth RMSE bound documented; notebook `notebooks/01_sim_sanity.ipynb` plots a trajectory. (FR-COR-003, FR-SIM-001…011)

## Phase 2 — Synthetic telemetry & ingestion  `[x]`
- [x] `telemetry/sensors.py` (noise, bias, drift, lag, quantisation), `defects.py` (timing), `faults.py` (fault schedule into truth), `generator.py` (Sobol conditions, orchestrates flights, parallel with joblib).
- [x] Adapters: CSV/Parquet with mapping + unit conversion; optional ULog/DataFlash behind extra (deferred, ADR-009).
- [x] `fsa generate`, `fsa ingest` commands; artifact layout per data spec §6; schema validation.
DoD: `fsa generate --profile demo` produces 12 flights deterministically (hash-identical re-run); statistical sensor tests pass; fixture CSV with non-SI units ingests correctly; faults log present. (FR-TEL-001…007, FR-COR-005)

## Phase 3 — Preprocessing & alignment  `[x]`
- [x] QC, cleaning (Hampel, gaps), sync (xcorr + drift + CI), resampling, features, phases, condition tags, state-estimate smoother, aligned dataset builder, windowed builder (windowed builder already in Phase 1's `sim/runners.py::run_windowed`).
- [x] `fsa preprocess` writes `qc/`, `sync/`, `aligned/`.
DoD: recovery tests: offset ±0.5 sample, drift within 10 ppm, spikes ≥ 95% removed, phase accuracy ≥ 90%; aligned schema valid; row counts equal across variants. (FR-PRP-001…011)

## Phase 4 — Validation & condition analysis  `[x]`
- [x] Metrics library (all FR-VAL-001 metrics), windowed/phase/bin aggregation, trajectory metrics, tolerance compliance.
- [x] Comparison stats (block bootstrap, Wilcoxon+Holm, Friedman/Nemenyi, composite score, ranking CIs).
- [x] Condition analysis (heatmaps data, GBM importance, PDP, Kruskal/Dunn, MixedLM), residual diagnostics. (DTW deferred, not yet needed -- FR-VAL-009 is priority S.)
- [x] `fsa validate`, `fsa analyze` (reduced scope -- see DECISIONS.md ADR-015); E1–E3 scripts and the full S2/S3 benchmark validation deferred to Phase 8's `fsa benchmark`.
DoD: metrics validated against reference formulas on fixtures (51 unit tests, hand-computed/analytic references); null-pair statistical tests pass (`compare_variant_pairs` correctly fails to reject a near-identical pair, correctly rejects a well-separated pair). S2/S3 against the real simulator/benchmark seeds deferred to Phase 8 per ADR-015. (FR-VAL-001…008, 010)

## Phase 5 — Predictive analysis  `[x]`
- [x] Window/dataset builder, splitters (LOFO, grouped K-fold, leave-condition-out), model zoo wrappers (`fit/predict/predict_interval`), baselines, nested CV + search, explainability.
- [x] Tasks A, B, C; conformal + quantile; optional torch GRU/TCN. (extrapolation study, FR-PRD-010, deferred -- see note below)
- [x] `fsa predict` (reduced scope -- see DECISIONS.md ADR-017).
DoD: leakage tests pass (`test_datasets.py`'s per-flight reconstruction test); skill-score/RMSE-reduction checks pass on synthetic data; artifacts per data spec (`predict/{cv_results,forecasts,intervals,explain}.parquet`, `models/residual/*.joblib`). (FR-PRD-001…009; FR-PRD-010 "S"-priority extrapolation study and FR-PRD-011 "C"-priority online-update demo deferred, not required for DoD)

## Phase 6 — Deviation detection  `[x]`
- [x] Baseline normalisation, statistical detectors, ML detectors, PELT onset estimation, threshold calibration, early-warning predictor, streaming API, evaluation protocol.
- [x] `fsa detect` (reduced scope -- see DECISIONS.md ADR-018).
DoD: calibrated FA within ±25% of target on synthetic fixtures (`test_detect_calibration.py`); batch == streaming test passes (`test_detect_statistical.py`); PELT onset error ≤ 3 s median on injected synthetic shifts (`test_detect_changepoint.py`). Full S6 benchmark validation against the real simulator deferred to Phase 8/9's `fsa benchmark` (ADR-015/018 precedent). (FR-DET-001…006, 008; FR-DET-007 attribution hint "S"-priority, deferred)

## Phase 7 — Identification, sensitivity & recommendations  `[x]`
- [x] Local Jacobian + identifiability; output-error fitting with windows; regime-wise fits (Cochran's Q); Morris/Sobol.
- [x] Recommendation engine + re-validation; `fsa identify` (reduced scope -- see DECISIONS.md ADR-019).
DoD: non-identifiable case flagged (`test_identify_jacobian.py`'s collinear-parameter test); every recommendation has ≥ 2 evidence types or is labelled hypothesis (`test_identify_recommendations.py`); a controlled self-consistency test recovers an injected CD0 error and shows refit RMSE improvement (`test_identify_fit.py`, `test_identify_revalidate.py` -- S7/S8's intent on synthetic data with known ground truth). Full S7/S8 benchmark validation against the real simulator deferred to Phase 8/9's `fsa benchmark` (ADR-015/018/019 precedent). (FR-IDN-001…006; FR-IDN-007/008 "S"-priority surrogate/structural-term probes deferred)

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
