# 01 — Product Requirements Document (PRD)

**Product:** AeroValidate · **Version:** 1.0 · **Status:** Ready for build
Related: `00_vision.md` (why), `02_FRD.md` (detailed functional requirements), `03`–`05` (specs).

---

## 1. Product summary

AeroValidate is a CLI + library + dashboard that takes simulation-variant outputs and telemetry, aligns them, quantifies model-vs-observation error, learns to predict and detect deviations, and recommends model refinements. It ships with a synthetic data generator so it is usable and testable without sensitive data.

## 2. Users and key jobs-to-be-done

| # | Job-to-be-done | Persona | Success looks like |
|---|---|---|---|
| J1 | "Which of my sim variants best matches flight data?" | Modelling lead | Ranked table with CIs and per-signal breakdown |
| J2 | "Where (flight conditions) does the model go wrong?" | Aero engineer | Heatmaps of error vs altitude/Mach/bank/phase |
| J3 | "Can I forecast flight parameters / the model's error?" | Data scientist | Forecast + calibrated intervals beating baselines |
| J4 | "Warn me when the model starts diverging." | Test analyst | Detector with known delay and false-alarm rate |
| J5 | "What should we fix in the model?" | Aero engineer | Ranked parameter suspects, refit, before/after |
| J6 | "Give me a report I can hand to my manager." | All | One HTML/PDF report per run |

## 3. Scope

### 3.1 In scope (v1.0)
1. Synthetic generator: truth model, ≥ 5 variants, sensor/noise/fault layers, scenario matrix.
2. Ingestion of external CSV/Parquet telemetry and simulation outputs via column-mapping config; optional ULog/DataFlash adapters.
3. Preprocessing: QC, cleaning, time sync (lag/drift estimation), resampling, features, phase segmentation, condition tagging.
4. Validation: metrics, windowed (multi-step) errors, variant comparison with statistics, condition-aware analysis.
5. Prediction: baselines, residual models, hybrid physics+ML correction, forecasting, uncertainty (conformal/quantile).
6. Deviation detection: statistical + ML detectors, change-point, evaluation on injected faults.
7. Identification: sensitivity analysis, output-error parameter fitting, identifiability, recommendation engine, re-validation.
8. Outputs: artifacts (Parquet/JSON), figures, HTML report (PDF optional), Streamlit dashboard.
9. Reproducibility: configs, seeds, manifests, CI.

### 3.2 Out of scope (v1.0)
Real-time streaming; certified V&V claims; CFD; full nonlinear 6-DoF truth (design leaves a plug-in point); multi-vehicle swarms; any classified/export-controlled data or weapon modelling.

### 3.3 Future (v1.x+)
Streaming/online detection; 6-DoF plug-in with stability derivatives; Bayesian parameter estimation (MCMC); digital-twin integration; FMU/FMI co-simulation import; MLflow experiment tracking by default.

## 4. Product principles

1. **Traceable:** every number in the report links to an artifact and a config.
2. **Honest uncertainty:** report CIs / intervals, never bare point estimates.
3. **Physics first, ML second:** ML augments the physical model; baselines are always included.
4. **Config-driven:** no editing code to run a new study.
5. **Modular & replaceable:** simulators and adapters are plug-ins behind small interfaces.

## 5. Functional requirement summary (by epic)

Detailed, testable requirements are in `02_FRD.md`. Priority uses MoSCoW (M = must, S = should, C = could).

| Epic | ID prefix | Summary | Priority |
|---|---|---|---|
| E-CORE | FR-COR | Config, units, atmosphere, manifest, logging, schema | M |
| E-SIM | FR-SIM | Truth model, variants, scenarios, controllers | M |
| E-TEL | FR-TEL | Sensor models, faults, synthetic telemetry, adapters | M |
| E-PREP | FR-PRP | QC, cleaning, sync, resample, features, phases | M |
| E-VAL | FR-VAL | Metrics, windowed error, stats, conditions, ranking | M |
| E-PRD | FR-PRD | Baselines, residual/hybrid/forecast models, uncertainty, CV | M/S |
| E-DET | FR-DET | Detectors, change-point, evaluation | M/S |
| E-IDN | FR-IDN | Sensitivity, fitting, identifiability, recommendations | M/S |
| E-RPT | FR-RPT | Figures, report, dashboard | M/S |
| E-OPS | FR-OPS | CLI, CI, packaging, docs | M |

## 6. User journeys

**J-A: Quick demo.** `make demo` → generates 12 flights × 5 variants → preprocess → validate → predict → detect → identify → report. Opens `reports/<run>/report.html`.

**J-B: Study on own data.** User writes `configs/my_study.yaml` mapping telemetry columns and simulation outputs, runs `fsa run-all --config configs/my_study.yaml`; inspects dashboard; exports CSV tables.

**J-C: What-if model change.** User edits a variant YAML (e.g., CD0 +10%), reruns `fsa validate`, and compares runs side by side in the dashboard.

## 7. Key outputs (deliverables mapped to project brief)

| Brief output | Product artifact |
|---|---|
| Comparative evaluation of flight-simulation variants | `validation/variant_ranking.parquet`, report §Variant comparison |
| Preprocessed and aligned simulation/telemetry datasets | `aligned/*.parquet` + `aligned/index.json` |
| Error analysis and visualisations | `validation/*.parquet`, `figures/*.png|html` |
| Predictive-analysis results for selected flight parameters | `predict/metrics.json`, `predict/forecasts.parquet`, report §Prediction |
| Technical findings supporting model validation & aero investigation | `identify/recommendations.json`, report §Findings |

## 8. Non-functional requirements (summary)

Performance: demo < 5 min; 100 flights × 10 Hz × 1 h processed in < 30 min on 8-core laptop. Reproducibility: bit-stable for the same seed/config on the same platform. Usability: single-command runs; helpful errors. Portability: Linux/macOS/Windows; Python 3.11+. Quality: ≥ 85% coverage; strict typing. Security/governance: no sensitive data in repo; see `07`.

## 9. Acceptance criteria (release v1.0)

1. All **Must** FRs pass their acceptance tests.
2. Success criteria S1–S9 in `00_vision.md §8` met on the synthetic benchmark suite (`fsa benchmark`).
3. `make lint test` green in CI on Linux; smoke test on Windows/macOS.
4. README quick-start works from a clean clone.
5. Report for the demo run contains all sections in FR-RPT-001 with no placeholder text.

## 10. Dependencies and assumptions

- Python ecosystem listed in `CLAUDE.md §3`.
- No external services required; offline capable.
- Real telemetry (when available) must be exportable to CSV/Parquet or a supported log format and carry a timestamp column.

## 11. Open questions (resolve via `docs/DECISIONS.md`; defaults given)

| # | Question | Default |
|---|---|---|
| Q1 | Reference vehicle type? | Generic fixed-wing UAV, ~25 kg, subsonic |
| Q2 | Do real logs exist? | Assume no; adapters ready |
| Q3 | Required signals? | Altitude, airspeed, ground speed, position, attitude, load factors, α (opt), throttle |
| Q4 | Sample rates? | Truth 100 Hz → telemetry 10 Hz nominal, jitter allowed |
| Q5 | Deep learning needed? | Optional extra; classical ML is the default |
