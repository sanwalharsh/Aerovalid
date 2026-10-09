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

## Phase 3 — Preprocessing & Alignment

- Added `prep/qc.py` (`qc_report`: missing %, duplicate/non-monotonic timestamps,
  registry-driven range violations, stuck/flatline detection, z-score spike counts,
  sample-rate stats; quarantines flights with >5% non-monotonic timestamps per data spec §8).
- Added `prep/clean.py` (vectorised rolling-median Hampel filter; gap policy that
  interpolates NaN *runs* up to `max_gap_s` and masks longer ones entirely, unlike
  `Series.interpolate(limit=...)`'s partial-fill-from-one-side behaviour; rebases `t_s`
  to start at 0 per data spec §4.1 "after cleaning" -- see DECISIONS.md ADR-011).
- Added `prep/sync.py` (cross-correlation lag + linear-drift estimation on z-scored,
  centred-difference reference signals, upsampled 5x with FFT-based correlation for
  sub-sample precision, parabolic peak refinement, block-bootstrap CI; drift uses two
  bookend windows rather than several evenly spaced ones -- see DECISIONS.md ADR-012
  for why, and the accuracy numbers that justify it).
- Added `prep/resample.py` (zero-phase anti-alias low-pass + linear interpolation onto
  a regular or explicit grid -- the latter lets a variant's trajectory land on
  telemetry's exact grid for alignment).
- Added `prep/features.py` (Mach/q, Savitzky-Golay climb/turn-rate derivatives,
  specific energy/power; `cl_eff` only when a wing area is supplied -- never a
  meaningless unitless placeholder).
- Added `prep/phases.py` (rule-based climb/descend/turn/accel/cruise segmentation with
  a minimum-duration debounce; condition-bin tagging for altitude/Mach/bank/load factor).
- Added `prep/state_estimate.py` (Savitzky-Golay smoother for windowed re-init, with
  angle unwrap/rewrap) and `prep/align.py` (`obs_/sim_/res_/mask_/cond_/phase` aligned
  table builder, angle-aware residuals, canonical-schema validated).
- Added `prep/pipeline.py` (`preprocess_flight`: orchestrates QC -> clean -> resample ->
  features -> phases/conditions -> sync -> align for one flight against every variant;
  pure in-memory, per CLAUDE.md §7).
- Implemented `fsa preprocess`: writes `qc/<flight>.json` + `qc_summary.parquet`,
  `sync/sync_estimates.parquet`, `aligned/<variant>/<flight>.parquet` + `index.json`,
  manifest-tracked.

DoD: Hampel filter removes >= 95% of injected spikes (tested); sync recovers injected
offset within +-0.5 sample and drift within +-10 ppm across several injected
offset/drift combinations (`tests/unit/test_sync.py`; see ADR-012 for the windowing
redesign this took); phase segmentation matches a scripted flight >= 90%; aligned tables
pass canonical-schema validation with equal row counts across variants (by construction
-- every variant is resampled onto the same telemetry grid). `fsa generate` ->
`fsa preprocess` verified end to end (`@pytest.mark.slow` CLI test).

## Phase 4 — Validation & Condition Analysis

- Added `validate/metrics.py` (every FR-VAL-001 metric: ME/MAE/RMSE/NRMSE/MAPE, R²/
  Pearson r, Theil's U2/TIC/U_M+U_S+U_C decomposition, error-growth exponent) and
  `validate/tolerance.py` (compliance fraction, sustained time-to-first-exceedance).
- Added `validate/trajectory.py` (3-D/horizontal/vertical RMSE, along/cross-track
  decomposition relative to the observed ground track, CEP50/95).
- Added `validate/aggregate.py` (`compute_metrics_table`: per-signal, optionally
  grouped by phase/condition/flight; `aggregate_across_flights`: mean/median/IQR,
  since flights are the unit of replication for inference).
- Added `validate/compare.py` (moving-block bootstrap CI, Cliff's delta, the paired
  Hodges-Lehmann shift estimator, Wilcoxon signed-rank + Holm-corrected pairwise
  comparison, Friedman + Nemenyi) and `validate/composite.py` (weighted composite
  score, flight-level bootstrap ranking with CIs and P(best), Spearman/Kendall rank
  agreement against a ground-truth ordering).
- Added `validate/conditions.py` (binned heatmap data; `HistGradientBoostingRegressor`
  with grouped-by-flight CV + permutation importance + partial dependence;
  Kruskal-Wallis + Dunn; a `MixedLM` model for condition effects adjusted for repeated
  flights) and `validate/diagnostics.py` (ACF/Ljung-Box, normality, ADF stationarity,
  Welch power spectrum).
- Implemented `fsa validate` (per-flight signal metrics, composite score, bootstrap
  ranking, pairwise comparison -> `validation/{metrics_signal,metrics_phase,
  variant_ranking,pairwise_tests}.parquet`) and `fsa analyze` (condition heatmaps ->
  `validation/condition_heatmaps.parquet`); both manifest-tracked. Scope intentionally
  reduced vs. the full spec -- see DECISIONS.md ADR-015.
- Added `scikit-posthocs` as a main (not dev) dependency -- see ADR-013.

DoD: all metric formulas verified against hand-computed or analytic fixtures (51 unit
tests across 8 test files) -- including an exact-equality Theil's U2 test that caught a
real alignment bug in the first implementation (ADR-014); `compare_variant_pairs`
correctly separates a well-separated synthetic pair and correctly fails to reject a
near-identical one; `fsa generate -> preprocess -> validate -> analyze` verified
end-to-end (`@pytest.mark.slow` CLI test) producing a 2-variant ranking. Full S2/S3
benchmark validation against the real simulator across seeds 1-5 deferred to Phase 8's
`fsa benchmark` (ADR-015).

## Phase 5 — Predictive Analysis

- Added `predict/datasets.py` (`build_windows`: leakage-safe lagged/rolling-stat
  features + a target at horizon `h`, built one flight at a time so no window can
  straddle a flight boundary; `leave_one_flight_out`, `grouped_kfold`,
  `leave_condition_out` splitters) and `predict/baselines.py` (persistence, sim-only,
  constant-velocity, mean-by-condition-bin).
- Added the model zoo: `predict/models/{ridge,hgb,rf,arimax,gru}.py`, all implementing
  the shared `ResidualModel` protocol (`fit`/`predict`/`predict_interval`) from
  `predict/models/__init__.py`. `gru.py` is a guarded optional `torch` import (no
  hard dependency); `arimax.py` uses `statsmodels.GLSAR` rather than true
  `SARIMAX`/`ARIMA` since this project's windowed, flight-shuffled evaluation doesn't
  fit a contiguous-sequence forecaster -- see ADR-016.
- Added `predict/tasks.py` (Task A residual prediction, Task B hybrid correction
  scored against sim-only and persistence-of-residual, Task C forecast with a
  baseline-relative skill score), `predict/uncertainty.py` (split-conformal wrapper,
  coverage/width, Winkler score), `predict/cv.py` (`cross_validate` over any splitter;
  bounded random-search nested CV, `predict/explain.py` (permutation importance,
  partial dependence, guarded optional SHAP).
- Implemented `fsa predict`: Task A (cross-validated across every configured model)
  and Task B (hybrid correction) for every variant and configured target signal, at
  the first configured horizon; the best model per `(variant, target)` is refit on
  the full dataset and persisted (`predict/models/residual/*.joblib`), with a
  flight-grouped split-conformal interval summary and a permutation-importance table.
  Scope intentionally reduced vs. the full spec (no Task C, no full horizon/CV-scheme
  sweep in the CLI) -- see ADR-017.

DoD: `test_datasets.py` proves no window straddles a flight boundary by reconstructing
multi-flight output from independent single-flight builds; every model wrapper passes
a shared interface-contract test (`test_models.py`); Task B's hybrid correction beats
sim-only RMSE on synthetic data with a learnable residual signal (`test_tasks.py`);
split-conformal intervals hit approximate target coverage (`test_uncertainty.py`);
nested CV search never sees outer-test rows (`test_cv.py`); `fsa generate ->
preprocess -> predict` verified end-to-end (`@pytest.mark.slow` CLI test) producing
`cv_results`/`forecasts`/`intervals`/`explain` parquet artifacts and a saved model per
variant. `make lint test` green at 88% coverage.

## Phase 6 — Deviation Detection

- Added `detect/baseline.py` (robust median/MAD residual normalisation, optionally per
  condition bin with a fallback for small bins, JSON-serialisable), `detect/
  statistical.py` (z-score, EWMA, CUSUM, Page-Hinkley -- each a streaming
  `update(z) -> bool`/`reset()` detector, replayed batch-wise by `detect_batch` so
  batch and streaming are identical by construction; Mahalanobis via a robust MCD
  covariance and a chi-squared threshold), `detect/ml.py` (Isolation Forest over
  rolling-window mean/std/slope features, `score(window) -> float`), `detect/
  changepoint.py` (`ruptures` PELT onset estimation), `detect/calibration.py`
  (bisection to a target false-alarm rate per flight-hour, counting alarm episodes
  rather than raw alarming samples), `detect/early_warning.py` (wraps a `predict/`
  split-conformal model to estimate `P(|res(t+h)| > tol)`), and `detect/evaluation.py`
  (per-flight detection delay/lead time/recall/precision/FAR, per-fault-type
  breakdown, window-level ROC/PR AUC).
- Implemented `fsa detect`: per variant, fits a residual baseline on fault-free
  flights, then runs every FAR-calibrated statistical detector configured in
  `detect.detectors` plus Mahalanobis on a combined per-sample `max_s |z_s|` stream
  (matching the z-score detector's own spec definition), writing `detect/
  {baselines.json, alarms.parquet, detection_metrics.parquet, roc.parquet}`. Discovered
  and fixed a real data characteristic while validating this against a demo run: every
  flight's first ~1-2 s shows a sim-vs-telemetry startup transient that swamped every
  detector except z-score's debounce -- `fsa detect` now discards each flight's first
  2 s before fitting or scoring, a standard monitoring-system practice, not a
  results-shaping hack. Isolation Forest, PELT onset estimation, and predictive early
  warning are library-only (not CLI-wired) -- see ADR-018.

DoD: every streaming detector's batch replay matches its live `update`-by-`update`
output exactly (`test_detect_statistical.py`); `calibrate_threshold` hits a target
false-alarm rate within 25% on synthetic nominal data (`test_detect_calibration.py`);
PELT recovers an injected mean-shift onset within 3 s, median across 8 seeded trials
(`test_detect_changepoint.py`); `fsa generate -> preprocess -> detect` verified
end-to-end (`@pytest.mark.slow` CLI test) on a 6-flight fixture with a real sampled
fault, producing calibrated alarms and metrics for every configured detector. Full S6
benchmark validation (median delay ≤ 5 s, FAR ≤ 1/flight-hour, recall ≥ 0.9) against
the real simulator deferred to Phase 8/9's `fsa benchmark` (ADR-015/018 precedent).
`make lint test` green.

## Phase 7 — Identification, Sensitivity & Recommendations

- Added `identify/jacobian.py` (central finite-difference Jacobian; parameter
  covariance/Wald CIs via `s^2 (J^T J)^-1`; condition-number and pairwise-correlation
  identifiability flags), `identify/fit.py` (output-error fitting via `scipy.optimize.
  least_squares` with soft-L1 loss, reusing `sim/runners.py::run_windowed`'s windowed
  re-initialisation to avoid chaotic drift -- exactly the method `docs/05_analysis_
  methods.md` §10 specifies), `identify/sensitivity.py` (Morris screening and Sobol/
  Saltelli indices via SALib, generic over any scalar model function), `identify/
  regimes.py` (per-regime refits; Cochran's Q heterogeneity test, meta-analysis form,
  to flag regime-dependent parameter drift as evidence of missing model structure),
  `identify/recommendations.py` (evidence-linked recommendations; never asserts a
  claim with fewer than 2 independent evidence types, labelling it `hypothesis`
  instead), and `identify/revalidate.py` (builds a refit variant V*, replays it on
  held-out flights, compares RMSE against the original).
- Implemented `fsa identify`: fits `identify.params` for one variant against one
  flight's first 60 s, computes Jacobian/identifiability diagnostics, runs Morris
  screening, builds recommendations (pulling condition evidence from a prior `fsa
  validate` run when available), and re-validates on held-out flights. While
  validating this against a demo run, found and fixed the same raw-telemetry
  sensor-warm-up transient ADR-018 discovered for `fsa detect` -- here it was
  corrupting the *replay seed state*, not just a few samples, causing a
  catastrophically bad (not just noisy) revalidation result; both are now trimmed
  via the same `_WARMUP_S` pattern. Regime-wise fitting, the full Sobol sweep, and
  the surrogate/structural-term probes are library-only (not CLI-wired) -- see
  ADR-019.

DoD: `central_difference_jacobian` reproduces a known linear function's exact
coefficients, and `identifiability` correctly flags a deliberately collinear parameter
pair (`test_identify_jacobian.py`); `sobol_indices`/`morris_screen` rank a dominant
parameter first and assign near-zero sensitivity to an inert one on an analytic test
model (`test_identify_sensitivity.py`); a controlled self-consistency test (same model
structure generates and fits the data) recovers an injected +20% CD0 error
(`test_identify_fit.py`) and the corresponding refit genuinely reduces held-out RMSE
(`test_identify_revalidate.py`); every recommendation has ≥ 2 evidence types or is
labelled `hypothesis` (`test_identify_recommendations.py`); `fsa generate -> identify`
verified end-to-end (`@pytest.mark.slow` CLI test). Full S7/S8 benchmark validation
against the real simulator deferred to Phase 8/9's `fsa benchmark` (ADR-015/018/019
precedent). `make lint test` green.
