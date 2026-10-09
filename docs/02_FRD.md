# 02 — Functional Requirements Document (FRD)

**Format:** `ID | Pri (M/S/C) | Requirement | Acceptance criteria (AC)`.
Each requirement must have at least one automated test referencing its ID (`@pytest.mark.req("FR-XXX-NNN")` or in the test docstring).

---

## FR-COR — Core

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-COR-001 | M | Typed YAML config loaded into pydantic models; supports `include:` of sub-configs and `--set key=value` CLI overrides. | Invalid config raises `ConfigError` naming the field; round-trip load→dump equal. |
| FR-COR-002 | M | `run_manifest.json` written per run: run_id, timestamp, git SHA, config hash, package versions, seeds, input file hashes, step timings. | Manifest present after each CLI step; same inputs → same hashes. |
| FR-COR-003 | M | ISA atmosphere (0–20 km): T, p, ρ, a, μ as functions of geometric altitude, with optional ΔT offset and humidity ignored. | Matches ISA table values within 0.1% at 0, 5, 11, 15 km. |
| FR-COR-004 | M | Unit helpers and a signal registry (name → unit, frame, description, default tolerance, plausible range). | Registry used by QC and plotting; unknown signal → clear error. |
| FR-COR-005 | M | Canonical dataset schema (see data spec §4) validated on read/write with pandera or pydantic checks. | Schema violation lists all offending columns. |
| FR-COR-006 | M | Deterministic RNG management: master seed → derived per-flight/per-component seeds (`SeedSequence.spawn`). | Two runs, same seed → identical Parquet bytes (excluding timestamps in manifest). |
| FR-COR-007 | S | Structured logging with run_id and step; log file in run dir. | Log contains step start/end and counts. |

## FR-SIM — Simulation

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-SIM-001 | M | Point-mass 3-D flight dynamics (state x,y,h,V,γ,χ,m + α dynamic state in truth) per `04_simulation_spec.md §3`. | Steady level trim holds altitude/speed within 0.1% for 60 s; energy-consistent ballistic test passes. |
| FR-SIM-002 | M | Aerodynamics: CL(α, M), CD(CL, M) with stall roll-off and wave-drag rise; parameters in config. | Unit tests on CL, CD curves; monotonic CD rise past Mdd. |
| FR-SIM-003 | M | Propulsion: thrust vs throttle, altitude lapse, Mach effect; fuel/mass flow. | Thrust lapse matches formula; mass decreases monotonically. |
| FR-SIM-004 | M | Environment: ISA + optional steady wind, linear shear, Dryden turbulence (seeded), ΔT offset. | Turbulence RMS/ scale-length statistics match targets within 10%. |
| FR-SIM-005 | M | Integrators: fixed-step RK4 (default), selectable adaptive (`solve_ivp`). | RK4 order-4 convergence test on a known analytic case. |
| FR-SIM-006 | M | **Truth model** class: higher-fidelity configuration (α dynamics, actuator lag, compressibility, wind, mass depletion, hidden parameter perturbations drawn from config distributions). | Hidden parameters saved in `truth_params.json` (never given to variants). |
| FR-SIM-007 | M | **Variant models** defined by YAML (`configs/variants/*.yaml`), each with parameter overrides and structural switches (see sim spec §5): at least V0–V5. | Registry lists variants; each runs on every scenario without NaN. |
| FR-SIM-008 | M | Scenario definitions: initial conditions + manoeuvre script (cruise, climb, descent, coordinated turn, S-turns, pitch doublet, speed step, mixed mission). | Scenario YAML validated; maneuver generator produces control histories (throttle, α_cmd or CL_cmd, bank_cmd). |
| FR-SIM-009 | M | **Replay mode**: variants driven by recorded controls + initial state from telemetry-state estimate. | Variant run reproduces input control schedule exactly. |
| FR-SIM-010 | M | **Windowed mode**: variant re-initialised from (smoothed) telemetry state every H seconds (config, default 10 s); outputs per-window predictions with horizon index. | Output includes `window_id`, `horizon_s`; at horizon 0 error equals state-estimate error. |
| FR-SIM-011 | S | Controller stack (speed hold, altitude hold, heading/bank hold) used by scenario generator to create realistic control histories. | Closed-loop tracking error within configured bounds in nominal scenarios. |
| FR-SIM-012 | C | Plug-in `SimulatorProtocol` so external simulators (e.g., FMU wrapper, JSBSim) can supply variant outputs. | Example plug-in in `examples/`. |

## FR-TEL — Telemetry synthesis & ingestion

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-TEL-001 | M | Sensor models per signal: white noise σ, bias, random-walk drift, scale factor, quantisation, first-order lag, saturation. | Statistical tests: noise σ, bias recovered within 5%. |
| FR-TEL-002 | M | Timing defects: constant offset, linear clock drift, timestamp jitter, dropped samples, burst gaps, duplicate timestamps. | Defect injectors parameterised and seeded; ground-truth defect log saved. |
| FR-TEL-003 | M | Fault injection library (`configs/faults/*.yaml`): drag-rise step, lift-curve-slope degradation, thrust loss ramp, control-effectiveness loss, sensor bias step, icing (CD0↑ + CLmax↓), wind shear event. Each with onset time, magnitude, ramp. | Fault log (`faults.json`) with onset/end for every injected fault. |
| FR-TEL-004 | M | Synthetic dataset generator `fsa generate` producing N flights × scenarios × conditions (Latin-hypercube/Sobol sampling over condition ranges) with truth, telemetry, and variant simulations. | Produces `raw/` tree per data spec §6; deterministic. |
| FR-TEL-005 | M | Ingestion adapters for CSV and Parquet with column-mapping + unit conversion + time-base spec (epoch, relative, ms/us). | Mapping YAML test with a fixture file in non-SI units. |
| FR-TEL-006 | S | ULog (PX4) and ArduPilot DataFlash adapters (optional extra). | Fixture log parses to canonical schema. |
| FR-TEL-007 | S | Ingest *simulation* outputs from external sims via the same mapping mechanism. | Variant outputs from CSV validate against schema. |
| FR-TEL-008 | C | Public-dataset helper docs (how to map a public UAV log) — no data bundled. | Doc section exists. |

## FR-PRP — Preprocessing

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-PRP-001 | M | QC report per flight: missing %, gap histogram, duplicate/non-monotonic timestamps, range violations (registry), stuck-sensor detection, flatline, spike count, sample-rate stats. | `qc/<flight>.json` + summary table; flags severity. |
| FR-PRP-002 | M | Cleaning: sort/dedupe timestamps; Hampel outlier filter; physical-range clipping flagged (not silently); gap policy: interpolate ≤ `max_gap_s` (default 0.5 s), else mask (NaN + `mask` column). | Injected spikes removed ≥ 95%; masks correct. |
| FR-PRP-003 | M | Time synchronisation: estimate constant lag (and optional linear drift) between each variant output and telemetry via cross-correlation on a reference channel set; sub-sample refinement (parabolic/cubic); report estimate + CI (bootstrap). Apply shift. | Recover injected offset within ±0.5 sample and drift within 10 ppm on synthetic data. |
| FR-PRP-004 | M | Resampling to common grid (default 10 Hz): anti-alias low-pass then interpolate (linear default, cubic option); keep original-sample flags. | Spectral test: no alias energy above Nyquist; values preserved at on-grid points. |
| FR-PRP-005 | M | Derived signals: Mach, dynamic pressure q, true/indicated airspeed relationships, flight-path angle, climb rate, turn rate, specific energy, effective CL/CD estimates from load factors, specific power. | Formulas tested vs analytic cases. |
| FR-PRP-006 | M | Smoothed derivatives (Savitzky–Golay; window/order config) for rates (e.g., ḣ, V̇) with noise-amplification report. | Derivative error on noisy sinusoid below threshold. |
| FR-PRP-007 | M | Flight-phase segmentation (rule-based: climb/descent/cruise/turn/accel/manoeuvre) with hysteresis; optional change-point refinement. | Phase labels match scenario script ≥ 90% on synthetic data. |
| FR-PRP-008 | M | Operating-condition tagging per sample and per window: altitude band, Mach/q band, bank band, load-factor band, wind magnitude, mass, phase. Bins configurable. | Tag columns present; bin edges in config. |
| FR-PRP-009 | M | Aligned dataset builder: one table per (flight, variant) with columns `t, obs_*, sim_*, res_*, mask_*, cond_*, phase`. | Passes schema validation; row counts equal across variants. |
| FR-PRP-010 | S | Frame/convention harmonisation (NED vs ENU, body vs wind axes, deg/rad, positive-up altitude) via mapping config. | Mapping test across conventions. |
| FR-PRP-011 | S | Sensor-fusion-free state estimate for windowed init: smoothing (Savitzky–Golay or Kalman smoother) of telemetry to estimate V, γ, χ, h, position. | Init-state error below sensor-noise-scaled bound. |

## FR-VAL — Validation

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-VAL-001 | M | Metrics per signal: bias (ME), MAE, RMSE, NRMSE (range & std), MAPE (ε-guarded), max |e|, R², Pearson r, Theil's U2 and TIC with Theil decomposition (U_M, U_S, U_C), normalised cross-correlation peak and lag, error-growth slope vs horizon. | Verified against reference implementations on fixtures. |
| FR-VAL-002 | M | Metrics computed on: full-trajectory replay, windowed horizons (e.g., 1, 2, 5, 10 s), per phase, per condition bin, per flight. Masked samples excluded. | Tables with `n_samples` column. |
| FR-VAL-003 | M | Trajectory metrics: 3-D position error, along-track/cross-track/vertical decomposition, speed-profile error, time-to-reach-altitude error, energy-error. | Tested on constructed offsets. |
| FR-VAL-004 | M | Tolerance compliance: fraction of time |e| ≤ tolerance (per signal config); time-to-first-exceedance. | Matches hand-computed values. |
| FR-VAL-005 | M | Variant comparison: paired bootstrap CIs (block bootstrap respecting autocorrelation) for metric differences; Wilcoxon signed-rank + Holm correction across flights; Friedman + Nemenyi for >2 variants; effect sizes. | Known-better variant significant on synthetic suite; false-positive rate ≈ α on null variant pair. |
| FR-VAL-006 | M | Composite score: weighted normalised metrics across signals (weights in config); rank variants with CI via bootstrap of flights; rank stability (Kendall τ). | Ranking reproduces ground-truth fidelity ordering (S2). |
| FR-VAL-007 | M | Condition-dependence analysis: binned error heatmaps (altitude×Mach, bank×load-factor, phase×variant); gradient-boosted regressor of |residual| on condition features with permutation importance and partial dependence; ANOVA/Kruskal–Wallis across bins. | Injected regime flaw is top-2 ranked (S3). |
| FR-VAL-008 | S | Residual diagnostics: autocorrelation, spectrum, normality, heteroscedasticity vs conditions, stationarity (ADF). | Diagnostics table; flagged non-white residuals. |
| FR-VAL-009 | S | Signal-shape comparison: DTW distance and peak-timing error for manoeuvre responses (e.g., doublet overshoot, settling). | Tested on synthetic second-order responses. |
| FR-VAL-010 | S | Uncertainty-aware validation: compare observation against variant ensemble (Monte-Carlo over parameter priors); coverage of observation by ensemble envelope. | Coverage metric reported. |

## FR-PRD — Predictive analysis

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-PRD-001 | M | Dataset builder for supervised tasks: sliding windows of lagged features → targets at horizon k; leakage-safe splits: leave-one-flight-out, grouped K-fold by flight, leave-condition-out (extrapolation). | Test proves no window straddles splits. |
| FR-PRD-002 | M | **Task A — Residual prediction:** predict `res_signal(t+k)` from recent state, controls, conditions, and sim output. | Metrics vs baselines (zero-residual, persistence, mean-by-bin). |
| FR-PRD-003 | M | **Task B — Hybrid correction:** corrected prediction = sim + predicted residual; evaluated against sim-only and pure-ML forecast. | Reports RMSE reduction (S4). |
| FR-PRD-004 | M | **Task C — Forecast selected parameters** (default: altitude, airspeed, load factor, pitch, ground track) at horizons 1–30 s with baselines: persistence, constant-velocity/acceleration, sim-only. | Skill score = 1 − RMSE/RMSE_baseline. |
| FR-PRD-005 | M | Model zoo (classical): Ridge/ElasticNet, Random Forest, HistGradientBoosting (or LightGBM), ARIMAX/VAR (statsmodels). Config-selectable; consistent `fit/predict` interface. | Each model passes interface contract test. |
| FR-PRD-006 | S | Sequence models (PyTorch, optional extra): GRU/TCN with residual-prediction head; early stopping; seeds. | Trains on demo in < 3 min CPU; beats ridge or documents why not. |
| FR-PRD-007 | M | Uncertainty: quantile regression (GBM) and split-conformal wrappers; coverage + interval-width metrics. | 90% coverage ∈ [86%, 94%] on held-out flights (S5). |
| FR-PRD-008 | M | Hyper-parameter search with nested grouped CV (random search, bounded budget). | Search log saved; no test-fold leakage. |
| FR-PRD-009 | M | Explainability: permutation importance, partial dependence, SHAP (optional) for residual models; link top features to physical quantities. | Importance table in report. |
| FR-PRD-010 | S | Extrapolation study: train on subset of conditions, test on unseen altitude/Mach bands; report degradation. | Table in report. |
| FR-PRD-011 | C | Online/incremental update demo (partial_fit). | Notebook example. |

## FR-DET — Deviation detection

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-DET-001 | M | Residual normalisation: robust baseline (median/MAD) fit on nominal training flights, per condition bin if enough data. | Baseline artifact stored. |
| FR-DET-002 | M | Statistical detectors: threshold on z-score, EWMA, CUSUM, Page–Hinkley, multivariate Mahalanobis / χ² on residual vector. Parameters via config; calibration to target false-alarm rate on nominal flights. | Calibrated FA ≤ target ± 25%. |
| FR-DET-003 | M | ML detectors: Isolation Forest, One-Class SVM or LOF on windowed residual features; optional autoencoder (torch). | Interface `score(window)->float`. |
| FR-DET-004 | M | Change-point detection (ruptures PELT/BinSeg) on residual streams to estimate deviation onset. | Onset error ≤ 3 s median on injected faults. |
| FR-DET-005 | M | **Predictive early warning:** use Task-A/C predictors to forecast tolerance exceedance within horizon; alarm when predicted exceedance probability > p*. | Lead-time metric computed. |
| FR-DET-006 | M | Evaluation protocol on injected faults + nominal flights: detection delay (alarm − onset), lead time (exceedance − alarm), recall, precision, ROC/PR AUC (window-level), false alarms per flight-hour; per-fault-type breakdown. | Metrics table; satisfies S6 on benchmark. |
| FR-DET-007 | S | Fault-type attribution hint: which signals/residual directions drive the alarm; map to candidate parameters via sensitivity matrix (FR-IDN-003). | Top-3 attribution accuracy reported on synthetic faults. |
| FR-DET-008 | S | Streaming-compatible detector API (`update(sample)->alarm`). | Batch == streaming outputs in test. |

## FR-IDN — Aerodynamic analysis & model optimisation

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-IDN-001 | M | Global sensitivity analysis of validation metrics w.r.t. variant parameters (Morris screening then Sobol via SALib) over priors in config. | Sobol indices sum ≈ 1 within tolerance; known dominant param ranks first on test model. |
| FR-IDN-002 | M | Output-error parameter identification: `scipy.optimize.least_squares` fitting selected parameters (CL0, CLα, CD0, k, thrust lapse exponent, Mdd, …) to telemetry within selected windows/regimes; robust loss (soft-L1); bounds from config. | Recovers injected parameter errors (S7). |
| FR-IDN-003 | M | Local sensitivity (Jacobian) matrix of residual signals wrt parameters; parameter covariance and CIs from Jacobian; identifiability diagnostics (condition number, correlation matrix, profile likelihood optional). | Non-identifiable parameter flagged on constructed case. |
| FR-IDN-004 | M | Regime-wise identification: fit per condition bin (e.g., Mach bands) to reveal regime-dependent parameter drift. | Table of parameter vs regime with CIs. |
| FR-IDN-005 | M | **Recommendation engine:** combine condition analysis, sensitivity, identification, and detector attributions into ranked, evidence-linked recommendations (e.g., "CD0 appears ≈ +14% (CI …) above variant V2 for M>0.55; explains 62% of altitude RMSE"). Templated text, numbers computed. | `recommendations.json` with evidence refs; no unsupported claims. |
| FR-IDN-006 | M | Re-validation: apply refit parameters to create variant V*; evaluate on **held-out flights** and compare vs original (FR-VAL-005 stats). | Meets S8 on benchmark. |
| FR-IDN-007 | S | Surrogate-assisted optimisation (GP/GBM surrogate of metric vs parameters) for expensive simulators. | Example run; uses ≤ 30% of direct evals for similar optimum. |
| FR-IDN-008 | S | Structural-error probe: add candidate model terms (e.g., CL2 quadratic term, Mach-dependent CD) and test improvement via AIC/BIC + held-out error. | Report shows accepted/rejected terms. |

## FR-RPT — Reporting & visualisation

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-RPT-001 | M | HTML report (Jinja2) with sections: Executive summary · Data & QC · Alignment · Variant comparison · Condition analysis · Prediction · Detection · Identification & recommendations · Appendix (config, manifest, methods). | All sections populated on demo. |
| FR-RPT-002 | M | Figures: overlays (obs vs variants), residual time-series with tolerance bands, error vs horizon, heatmaps, Bland–Altman, Q-Q, ranking forest plot (CIs), calibration plots, ROC/PR, detection timeline with onset/alarm/exceedance markers, parameter CI plots, 3-D trajectory with error colouring. | Each figure function has a smoke test. |
| FR-RPT-003 | M | Tables exported as CSV + Markdown; machine-readable `summary.json`. | Files present. |
| FR-RPT-004 | S | Streamlit dashboard: run picker, flight/variant selector, overlay, residual explorer, condition heatmaps, detector timeline, recommendation list, run-to-run comparison. | Launches; loads demo run without error. |
| FR-RPT-005 | S | PDF export (via WeasyPrint or Playwright if available). | Optional extra. |
| FR-RPT-006 | M | Executive summary text auto-generated from computed numbers only; each statement carries an evidence link (artifact path). | Lint test: every sentence template references a numeric field. |

## FR-OPS — Operations

| ID | Pri | Requirement | AC |
|---|---|---|---|
| FR-OPS-001 | M | `fsa` CLI with subcommands (CLAUDE.md §5); `--config`, `--run-id`, `--seed`, `--profile {demo,full}`, `--set`. | `fsa --help` documents all; each step idempotent (skips if artifacts + hashes match unless `--force`). |
| FR-OPS-002 | M | `run-all` orchestrates the full pipeline with resumability. | Interrupt+resume yields identical outputs. |
| FR-OPS-003 | M | CI (GitHub Actions): lint, type-check, tests, coverage gate, demo smoke on Linux; matrix smoke on macOS/Windows. | Workflow file present and passing. |
| FR-OPS-004 | M | `fsa benchmark`: runs synthetic suite with multiple seeds and reports success criteria S1–S9 pass/fail. | `benchmark_summary.json`. |
| FR-OPS-005 | S | Pre-commit hooks (ruff, mypy, nbstripout). | `.pre-commit-config.yaml`. |
| FR-OPS-006 | S | Dockerfile for reproducible environment. | Image builds; `make demo` works inside. |
