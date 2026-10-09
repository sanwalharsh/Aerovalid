# AeroValidate — Master Build Prompt for Claude Code
# Version 1.0 · Use this as your operating instruction set from first session to release.

---

## ⚠️ BEFORE YOU WRITE A SINGLE LINE OF CODE — READ THIS ENTIRE FILE

This prompt is self-contained. Every decision is pre-made. Your job is to **build exactly what is specified**, test it, and stop. Do not invent features, do not add imports you haven't verified exist, do not write placeholder functions — if a function body is not ready, raise `NotImplementedError` with a message, never `pass` silently.

---

## 0. Ground rules (absolute, non-negotiable)

| Rule | What it means in practice |
|---|---|
| **No hallucinated APIs** | Before using any third-party function, verify it exists in the installed version. Check with `import inspect; inspect.signature(fn)` or read the installed package's `__init__`. If uncertain, use stdlib or numpy. |
| **No dead code** | Every function written must be called by at least one test or one CLI command. No `# TODO` bodies, no stubs that silently return `None`. |
| **No repetition** | If logic appears twice, extract it. Shared utilities live in `src/aerovalid/core/`. |
| **No global state** | No module-level mutable objects. Config is passed explicitly. RNG is passed explicitly. |
| **No silent failures** | Every function that can fail raises a subclass of `AeroValidError` with an actionable message. No bare `except: pass`. |
| **No data leakage** | Splits are by *flight_id*, never by row. Preprocessing stats (mean, std, quantiles) are computed on training folds only and passed to test folds. This is tested. |
| **No classified data** | Reference vehicle is a generic fixed-wing UAV with invented parameters. No real weapon specs, no export-controlled data, no classified parameters anywhere in the codebase. |
| **No print statements** | Use `logging.getLogger(__name__)` everywhere. |
| **SI units internally** | Metres, seconds, kilograms, radians, Newtons, Pascals, Kelvin. Convert only at I/O boundaries. Column names carry unit suffixes as defined in `docs/03_architecture_and_data_spec.md §4`. |
| **Deterministic** | Every stochastic function takes an explicit seed or `numpy.random.Generator`. Two runs with the same seed and config produce bit-identical Parquet outputs (excluding manifest timestamps). |
| **Test alongside code** | Every public function gets a test in the same PR/commit. No phase is "done" without `make lint test` passing. |

---

## 1. Project identity

**Name:** AeroValidate (`aerovalid`)
**What it does:** CLI + library + dashboard that ingests flight-simulation variant outputs and telemetry, aligns them, measures model-vs-observation error, trains predictive and deviation-detection models, identifies aerodynamic parameter errors, and produces reports — all reproducible from a YAML config.
**Synthetic-first:** Ships with its own truth-model simulator and sensor/fault layer so it works without any external data. Real-data adapters are optional extras.
**Unclassified only:** Generic fixed-wing UAV, ~25 kg, subsonic, invented-but-plausible parameters.

---

## 2. Exact repo layout to create

Create this structure from scratch. Do not deviate.

```
aerovalid/
├── CLAUDE.md                        # already written — do not edit
├── PROMPT_FOR_CLAUDE_CODE.md        # this file — do not edit
├── README.md                        # write after Phase 8
├── Makefile                         # Phase 0
├── pyproject.toml                   # Phase 0
├── .gitignore                       # Phase 0
├── .pre-commit-config.yaml          # Phase 0
├── .github/
│   └── workflows/
│       └── ci.yml                   # Phase 0
├── configs/
│   ├── default.yaml                 # Phase 0
│   ├── vehicles/
│   │   └── generic_uav_v1.yaml      # Phase 1
│   ├── variants/
│   │   ├── V0.yaml  V1.yaml  V2.yaml  V3.yaml  V4.yaml  V5.yaml  # Phase 1
│   ├── scenarios/
│   │   ├── mixed_mission.yaml  climb_cruise.yaml  turn_series.yaml
│   │   ├── doublet_series.yaml  high_speed_dash.yaml              # Phase 1
│   ├── sensors/
│   │   └── default.yaml             # Phase 2
│   └── faults/
│       ├── cd0_step.yaml  cla_degrade.yaml  thrust_loss.yaml
│       ├── icing.yaml  ctrl_eff_loss.yaml  sensor_bias_step.yaml
│       └── wind_shear_event.yaml    # Phase 2
├── docs/
│   ├── 00_vision.md                 # already written
│   ├── 01_PRD.md                    # already written
│   ├── 02_FRD.md                    # already written
│   ├── 03_architecture_and_data_spec.md  # already written
│   ├── 04_simulation_spec.md        # already written
│   ├── 05_analysis_methods.md       # already written
│   ├── 06_implementation_plan.md    # already written — tick boxes as you go
│   ├── 07_nfr_risks_governance.md   # write in Phase 0
│   ├── DECISIONS.md                 # write in Phase 0; add entries whenever you make a non-obvious choice
│   ├── CHANGELOG.md                 # update at end of each phase
│   └── CONFIG_REFERENCE.md          # auto-generated from pydantic models in Phase 8
├── src/
│   └── aerovalid/
│       ├── __init__.py  py.typed
│       ├── cli.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── atmosphere.py        # ISA
│       │   ├── angles.py            # wrap, circular stats
│       │   ├── config.py            # pydantic models + loader
│       │   ├── errors.py            # AeroValidError hierarchy
│       │   ├── logging.py           # structured logging setup
│       │   ├── manifest.py          # run_manifest.json writer
│       │   ├── rng.py               # SeedSequence helpers
│       │   ├── schema.py            # pandera schema for canonical tables
│       │   └── signals.py           # signal registry
│       ├── sim/
│       │   ├── __init__.py
│       │   ├── aero.py              # CL, CD, wave drag, stall
│       │   ├── propulsion.py        # thrust, lapse, mass flow
│       │   ├── environment.py       # wind, shear, Dryden turbulence
│       │   ├── integrators.py       # RK4 + adaptive wrapper
│       │   ├── dynamics.py          # state derivative, derived measurements
│       │   ├── controllers.py       # speed/alt/bank hold
│       │   ├── maneuvers.py         # primitives + scenario loader
│       │   ├── protocols.py         # SimulatorProtocol, ControlHistory, State types
│       │   ├── truth.py             # TruthModel: high-fidelity config + hidden params
│       │   ├── variants.py          # VariantModel: YAML-driven config, V0-V5
│       │   └── runners.py           # replay_run, windowed_run
│       ├── telemetry/
│       │   ├── __init__.py
│       │   ├── sensors.py           # SensorModel, apply_sensors
│       │   ├── defects.py           # timing defect injectors
│       │   ├── faults.py            # FaultSpec, inject_faults, fault schedule
│       │   ├── generator.py         # full dataset generator (Sobol conditions, joblib)
│       │   └── adapters/
│       │       ├── __init__.py
│       │       ├── base.py          # Adapter protocol + ColumnMapping
│       │       ├── csv_parquet.py   # CSV/Parquet adapter
│       │       └── ulog.py          # PX4 ULog adapter (optional extra)
│       ├── prep/
│       │   ├── __init__.py
│       │   ├── qc.py                # QCReport, qc_report()
│       │   ├── clean.py             # Hampel, gap policy, CleanResult
│       │   ├── sync.py              # SyncEstimate, estimate_sync(), apply_sync()
│       │   ├── resample.py          # anti-alias + interpolate
│       │   ├── features.py          # derived signals, SG derivatives
│       │   ├── phases.py            # rule-based segmentation + condition tagging
│       │   ├── state_estimate.py    # smoother for windowed init
│       │   └── align.py             # AlignedDataset builder
│       ├── validate/
│       │   ├── __init__.py
│       │   ├── metrics.py           # all FR-VAL-001 metrics, vectorised
│       │   ├── aggregate.py         # per-signal/phase/bin/horizon aggregation
│       │   ├── trajectory.py        # 3-D error, along/cross-track, CEP
│       │   ├── tolerance.py         # compliance fraction, TTFE
│       │   ├── compare.py           # bootstrap, Wilcoxon, Friedman, effect sizes
│       │   ├── composite.py         # weighted composite score + ranking
│       │   ├── conditions.py        # heatmaps, GBM importance, PDP, MixedLM
│       │   └── diagnostics.py       # ACF, spectrum, normality, stationarity
│       ├── predict/
│       │   ├── __init__.py
│       │   ├── datasets.py          # window builder, splitters, leakage tests
│       │   ├── baselines.py         # persistence, constant-velocity, sim-only, mean-bin
│       │   ├── models/
│       │   │   ├── __init__.py
│       │   │   ├── ridge.py         # Ridge/ElasticNet wrapper
│       │   │   ├── hgb.py           # HistGradientBoosting wrapper
│       │   │   ├── rf.py            # RandomForest wrapper
│       │   │   ├── arimax.py        # statsmodels ARIMAX/VAR wrapper
│       │   │   └── gru.py           # optional torch GRU (guarded import)
│       │   ├── tasks.py             # Task A (residual), B (hybrid), C (forecast)
│       │   ├── uncertainty.py       # split-conformal, quantile GBM wrappers
│       │   ├── cv.py                # nested CV, LOFO, grouped K-fold, leave-cond-out
│       │   └── explain.py           # permutation importance, PDP, SHAP guard
│       ├── detect/
│       │   ├── __init__.py
│       │   ├── baseline.py          # robust normalisation (median/MAD per bin)
│       │   ├── statistical.py       # zscore, EWMA, CUSUM, Page-Hinkley, Mahalanobis
│       │   ├── ml_detectors.py      # IsolationForest, OCSVM, LOF, autoencoder guard
│       │   ├── changepoint.py       # ruptures PELT/BinSeg wrapper
│       │   ├── early_warning.py     # Task-A/C predictor → exceedance probability
│       │   ├── calibrate.py         # threshold calibration to target FA rate
│       │   ├── evaluate.py          # delay, lead time, recall, precision, ROC, PR
│       │   └── streaming.py         # Detector.update() streaming-compatible API
│       ├── identify/
│       │   ├── __init__.py
│       │   ├── sensitivity.py       # Morris screening + Sobol (SALib)
│       │   ├── jacobian.py          # local Jacobian, covariance, identifiability
│       │   ├── fitting.py           # output-error fitting (scipy least_squares + windowed)
│       │   ├── regime.py            # regime-wise fits per condition bin
│       │   ├── structural.py        # structural-term probe (AIC/BIC + held-out)
│       │   └── recommend.py         # Recommendation engine + recommendations.json
│       ├── report/
│       │   ├── __init__.py
│       │   ├── figures.py           # all figure functions → return Figure, no globals
│       │   ├── templates/
│       │   │   └── report.html.j2   # Jinja2 report template
│       │   └── builder.py           # HTML report builder, summary.json, CSV tables
│       └── app/
│           ├── __init__.py
│           └── main.py              # Streamlit dashboard
├── tests/
│   ├── conftest.py                  # fixtures: tiny synthetic flights (≤60s), fixed seeds
│   ├── unit/                        # one file per source module
│   ├── integration/                 # stage-to-stage contract tests
│   ├── e2e/                         # full pipeline on 3 flights
│   ├── regression/                  # golden outputs (tolerant comparison)
│   └── fixtures/                    # tiny committed data fixtures only
├── notebooks/                       # exploratory only, never imported by src
├── experiments/                     # E1–E10 scripts (read artifacts, write results)
│   ├── e01_trajectory_validation.py
│   ├── e02_windowed_validation.py
│   ├── e03_condition_analysis.py
│   ├── e04_residual_prediction.py
│   ├── e05_hybrid_correction.py
│   ├── e06_deviation_detection.py
│   ├── e07_sensitivity_identification.py
│   ├── e08_refit_revalidation.py
│   ├── e09_robustness.py
│   └── e10_real_data_dryrun.py
├── data/      # git-ignored
├── runs/      # git-ignored
└── reports/   # git-ignored
```

---

## 3. Tech stack — exact versions to install

```toml
# pyproject.toml [project.dependencies]
python = ">=3.11"
numpy = ">=1.26"
scipy = ">=1.12"
pandas = ">=2.2"
pyarrow = ">=15.0"
scikit-learn = ">=1.4"
statsmodels = ">=0.14"
ruptures = ">=1.1.9"
SALib = ">=1.5"
pydantic = ">=2.6"
pyyaml = ">=6.0"
typer = ">=0.12"
matplotlib = ">=3.8"
plotly = ">=5.20"
jinja2 = ">=3.1"
streamlit = ">=1.33"
pandera = ">=0.19"
joblib = ">=1.3"

# [project.optional-dependencies]
# ml = ["torch>=2.2", "lightgbm>=4.3", "shap>=0.45"]
# adapters = ["pymavlink>=2.4", "pyulog>=1.0"]
# dev = ["pytest>=8", "pytest-cov>=5", "hypothesis>=6.100",
#         "ruff>=0.4", "mypy>=1.9", "nbstripout>=0.7",
#         "pytest-benchmark>=4.0", "pytest-regressions>=0.5"]
```

**Do not import any package not in this list without adding it to `pyproject.toml` first and noting the reason in `docs/DECISIONS.md`.**

---

## 4. The reference vehicle — exact parameters

All parameters live in `configs/vehicles/generic_uaw_v1.yaml`. These are **invented** for a generic subsonic UAV. Never change them to match a real restricted platform.

```yaml
# configs/vehicles/generic_uav_v1.yaml
name: generic_uav_v1
description: Invented generic fixed-wing UAV for AeroValidate research. Not representative of any real platform.
mass_nom_kg: 25.0
mass_range_kg: [22.0, 28.0]
wing_area_m2: 1.20
wingspan_m: 3.20
aspect_ratio: 8.533        # b²/S
oswald_e: 0.80
k: 0.04696                 # 1/(pi * e * AR)
CL0: 0.25
CLa: 5.00                  # per rad
alpha_stall_rad: 0.2618    # 15 deg
CLmax: 1.45
CD0: 0.028
Mdd: 0.60                  # drag-divergence Mach
wave_drag_coeff: 20.0      # 20*(M-Mdd)^4 for M>Mdd
T0_N: 180.0                # max thrust at SL static
thrust_lapse_n: 0.85       # T ∝ (rho/rho0)^n
thrust_mach_factor: 0.25   # T(M) = T * (1 - 0.25*M)
SFC_kg_per_Ns: 1.5e-5
nz_max_g: 3.5
bank_limit_rad: 1.0472     # 60 deg
tau_thrust_s: 0.4          # actuator lag (truth only)
tau_alpha_s: 0.15          # alpha lag (truth only)
```

---

## 5. Variant definitions — exactly these six

Write `configs/variants/V0.yaml` through `V5.yaml`. The structural switch fields map to boolean flags in `VariantSpec.switches`.

```
V0  nominal_baseline:  params=vehicle_nominal, compressibility=false, wind_rate_terms=false,
                       alpha_dynamics=false, thrust_lapse=true, wave_drag=false, isa_offset_known=false
    known_flaw: "CD0 underestimate above M=0.20 (no wave drag)"

V1  high_drag:         V0 override CD0 *= 1.20
    known_flaw: "CD0 25% high — worst overall error"

V2  lift_curve_error:  V0 override CLa *= 0.88, CL0 *= 0.90
    known_flaw: "Lift underestimate at high load factor (nz > 1.8)"

V3  atmosphere_naive:  V0, rho=constant(rho_at_h0), thrust_lapse=false
    known_flaw: "Fails at large altitude change — density and thrust both wrong"

V4  compressibility_aware:  V0 + prandtl_glauert=true + wave_drag=true, wind=false
    known_flaw: "None above M=0.15; wind-blind"

V5  near_truth:        truth structure minus alpha_dynamics, params=truth*(1±0.02 noise),
                       wind_estimate=airspeed_minus_groundspeed_vector
    known_flaw: "None — should rank first"
```

Ground-truth fidelity order (error ascending): `V5 < V4 < V0 ≈ V2 < V1 < V3`
Store this in `configs/benchmark_ground_truth.yaml` and read it only in `fsa benchmark`.

---

## 6. Canonical data schema — implement exactly this

Every aligned table must pass `core/schema.py` validation on write and read.

**Required columns:**

```
t_s           float64   seconds from flight start, strictly increasing
flight_id     string    e.g. "F0007"
source        category  one of: "truth", "tel", "V0".."V5"
```

**Signal columns** (SI, internal):
```
alt_m         float64   altitude MSL (m), up positive
x_m, y_m     float64   local NE position from start (m)
vel_air_mps   float64   true airspeed (m/s)
vel_gnd_mps   float64   ground speed (m/s)
gamma_rad     float64   flight-path angle (rad)
chi_rad       float64   track angle (rad), 0=North CW
roll_rad      float64   bank (rad)
pitch_rad     float64   pitch attitude (rad)
yaw_rad       float64   heading (rad)
alpha_rad     float64   angle of attack (rad), optional
nx_g          float64   longitudinal load factor (g)
nz_g          float64   normal load factor (g)
throttle      float64   0–1
mach          float64   derived
q_pa          float64   dynamic pressure (Pa), derived
mass_kg       float64   optional
```

**Aligned table prefixes** (one table per flight×variant):
```
obs_<sig>     telemetry observation
sim_<sig>     variant simulation output
res_<sig>     obs - sim  (angles wrapped to (-π, π])
mask_<sig>    bool, True = excluded from metrics
cond_<name>   condition descriptor (alt_band, mach_band, bank_band, nz_band, wind_band, phase)
phase         category
window_id     int (windowed tables only)
horizon_s     float (windowed tables only)
```

---

## 7. Sensor models — exact default parameters

```yaml
# configs/sensors/default.yaml
alt_m:       {sigma: 1.5,  bias_rw: 0.02, lag_s: 0.20, quantise: 0.0}
x_m:         {sigma: 1.2,  bias: 0.0,     lag_s: 0.0,  dropout_p: 0.005}
y_m:         {sigma: 1.2,  bias: 0.0,     lag_s: 0.0,  dropout_p: 0.005}
vel_air_mps: {sigma: 0.40, scale: 1.005,  lag_s: 0.0,  quantise: 0.05}
vel_gnd_mps: {sigma: 0.15, bias: 0.0}
roll_rad:    {sigma_deg: 0.4, bias_deg: 0.0}
pitch_rad:   {sigma_deg: 0.4, bias_deg: 0.0}
yaw_rad:     {sigma_deg: 1.0, drift_deg_per_sqrts: 0.05}
nx_g:        {sigma: 0.02, bias: 0.005}
nz_g:        {sigma: 0.02, bias: 0.005}
alpha_rad:   {sigma_deg: 0.3, bias_deg: 0.2, enabled: false}

timing_defects:
  offset_s:        {distribution: uniform, low: -0.8, high: 0.8}
  drift_ppm:       {distribution: uniform, low: -50, high: 50}
  jitter_sigma_ms: 5.0
  dropout_p_per_sample: 0.005
  burst_gaps:      {max_n: 2, duration_s: [0.5, 3.0]}
```

---

## 8. Fault library — exact fault types and parameter ranges

```yaml
# Each configs/faults/<name>.yaml has this structure:
# fault_type, affected_signals, effect_formula, magnitude_range, onset_ramp_s

cd0_step:        {effect: "CD0 *= (1 + m)",          magnitude: [0.10, 0.30], ramp_s: [0, 5]}
cla_degrade:     {effect: "CLa *= (1 - m)",          magnitude: [0.05, 0.20], ramp_s: [0, 10]}
thrust_loss:     {effect: "T0 *= (1 - m)",           magnitude: [0.10, 0.40], ramp_s: [10, 30]}
icing:           {effect: "CD0*=(1+m), CLmax*=(1-m), CLa*=(1-0.5*m)", magnitude: [0.1,0.3], ramp_s:[30,90]}
ctrl_eff_loss:   {effect: "alpha_response_gain *= (1-m)", magnitude: [0.2, 0.5], ramp_s: [0, 1]}
sensor_bias_step:{effect: "additive bias on one sensor = k*sigma", magnitude: [3, 10], ramp_s: [0, 0]}
wind_shear_event:{effect: "wind_step delta_V_mps",   magnitude: [5, 15], ramp_s: [2, 5]}
```

---

## 9. Build phases — work strictly in this order

**Never start phase N+1 until `make lint test` passes for phase N and the DoD checklist in `docs/06_implementation_plan.md` is ticked.**

### Phase 0 — Scaffolding
Create:
- `pyproject.toml` with hatchling, extras `ml`, `adapters`, `dev`, `all`
- `Makefile` with targets: `setup`, `lint`, `test`, `test-fast`, `demo`, `dashboard`
- `.gitignore`: ignore `data/`, `runs/`, `reports/`, `__pycache__/`, `.venv/`, `*.egg-info/`, `.mypy_cache/`
- `.pre-commit-config.yaml`: ruff, mypy, nbstripout
- `.github/workflows/ci.yml`: lint + type-check + pytest on ubuntu-latest; smoke on windows/macos
- `src/aerovalid/core/`: atmosphere, angles, config (pydantic), errors, logging, manifest, rng, schema, signals
- `src/aerovalid/cli.py`: typer app with stub subcommands (each raises NotImplementedError until implemented)
- `docs/DECISIONS.md`, `docs/CHANGELOG.md`, `docs/07_nfr_risks_governance.md`

**DoD:** `pip install -e ".[dev]"` succeeds; `fsa --help` lists all commands; `make lint test` green; config round-trip test passes; manifest written correctly.

### Phase 1 — Physics & Simulation
- Implement `sim/` in this dependency order: `aero.py` → `propulsion.py` → `environment.py` → `integrators.py` → `dynamics.py` → `controllers.py` → `maneuvers.py` → `protocols.py` → `truth.py` → `variants.py` → `runners.py`
- Load vehicle from YAML, variants from YAML. No hardcoded vehicle numbers in Python.
- ISA: vectorised, handles array inputs.
- RK4: test convergence order (must be ≈4 on a log-log step refinement plot).
- Truth model saves `runs/<run_id>/raw/meta/<flight>.json` with hidden params (never expose to variants).
- Windowed runner: `run_windowed(sim, tel_state_df, controls_df, horizon_s, stride_s) -> DataFrame`

**DoD:** All sim-spec §10 physics tests pass. Every scenario × every variant runs without NaN/inf. V5 vs truth with no sensor noise: altitude RMSE < 1 m over a 5-minute trim flight (document the actual value).

### Phase 2 — Synthetic Telemetry & Ingestion
- `telemetry/sensors.py`: each sensor model is a pydantic class with a `corrupt(signal_array, t_array, rng) -> (corrupted, DefectLog)` method.
- `telemetry/defects.py`: timing defects applied to the full timestamp array.
- `telemetry/faults.py`: fault schedule injected into truth integration (modify `CD0`, `CLa`, etc. as a time-varying parameter inside the truth ODE).
- `telemetry/generator.py`: Sobol sequence over condition ranges → flight × scenario × condition matrix. Parallel with `joblib.Parallel`. Deterministic.
- Adapters: `base.py` defines `Adapter(Protocol)` with `read(path, mapping) -> DataFrame`. `csv_parquet.py` implements it with unit conversion from the mapping YAML.
- `fsa generate` and `fsa ingest` CLI commands, fully implemented.

**DoD:** `fsa generate --profile demo` produces exactly 12 flights with identical hashes on rerun. Sensor sigma recovery test passes (within 5%). Fault log present and correct. Fixture CSV with non-SI units ingests to SI schema correctly.

### Phase 3 — Preprocessing & Alignment
- Implement strictly in this order: `qc.py` → `clean.py` → `sync.py` → `resample.py` → `features.py` → `phases.py` → `state_estimate.py` → `align.py`
- Cross-correlation sync: use `numpy.correlate` on z-scored first-differences of reference signals. Parabolic peak refinement. Bootstrap CI. Drift detection via Huber regression on per-window lags.
- Hampel filter: pure numpy, no per-row loops.
- Aligned dataset: one Parquet per (flight, variant). Validate against `core/schema.py` before writing.
- `fsa preprocess` writes `qc/`, `sync/`, `aligned/`.

**DoD:** Offset recovery ±0.5 sample, drift ±10 ppm on synthetic data (verified by injecting known offsets and recovering). Hampel removes ≥95% of injected spikes. Phase accuracy ≥90% on scenario script. Aligned schema validates.

### Phase 4 — Validation & Condition Analysis
- `metrics.py`: every formula from `docs/05_analysis_methods.md §4`. All vectorised numpy. Zero for-loops over time. Add unit test for every metric against a known analytic case.
- `compare.py`: block bootstrap (using `numpy` strided views, not a loop). Wilcoxon from `scipy.stats`. Holm correction from `statsmodels.stats.multitest`. Friedman + Nemenyi using `scikit-posthocs` — add it to dev deps if needed, note in DECISIONS.md.
- `conditions.py`: `HistGradientBoostingClassifier` is NOT used here; use `HistGradientBoostingRegressor` on `|res| / tol`. PDP via `sklearn.inspection.partial_dependence`.
- `fsa validate` and `fsa analyze` implemented.
- E1, E2, E3 experiment scripts.

**DoD:** All metric formulas verified against analytic fixtures. S2 (Spearman ρ ≥ 0.9 on ranking) and S3 (regime flaw in top-2 bins ≥90% of trials) pass on the benchmark with seeds 1–5. Null-pair false-positive rate 2–8% at α=0.05.

### Phase 5 — Predictive Analysis
- `datasets.py`: `build_windows(aligned_df, cfg) -> WindowedDataset`. Write a test that proves no window straddles a flight boundary. Scalers fitted on train fold only.
- `tasks.py`: Task A (residual), B (hybrid), C (forecast). Each task is a function that takes a model factory, dataset, CV splitter, and returns `CVResult`.
- All model wrappers implement `fit(X, y, groups) / predict(X) / predict_interval(X, alpha)`.
- Conformal: split-conformal wrapper in `uncertainty.py`. Uses calibration flights held out from training.
- GRU: fully guarded behind `try: import torch; HAS_TORCH = True except ImportError: HAS_TORCH = False`. If not available, log a warning and skip; never fail.
- `fsa predict` implemented. E4, E5 scripts.

**DoD:** Leakage tests pass (automated). S4 (≥25% RMSE reduction, Task B) and S5 (90% coverage in [86%,94%]) pass on benchmark.

### Phase 6 — Deviation Detection
- `baseline.py`: `RobustBaseline.fit(nominal_residuals)` stores per-signal `(median, mad)` per condition bin. Serialisable to JSON.
- All detectors in `statistical.py` implement `Detector(Protocol)` with `fit`, `score`, `update`.
- `streaming.py`: wrapper that calls `update()` sample-by-sample; test that batch output == streaming output.
- `calibrate.py`: empirical quantile of per-flight max statistic on calibration set. Target FA per flight-hour from config.
- `evaluate.py`: reads fault logs for ground-truth onset times. Computes all FR-DET-006 metrics. Per-fault-type breakdown.
- `fsa detect` implemented. E6 script.

**DoD:** Calibrated FA within ±25% of target on held-out nominal flights. S6 (delay ≤5 s, FA ≤1/hr, recall ≥0.9) passes on benchmark seeds 1–5. Batch == streaming test passes.

### Phase 7 — Identification & Recommendations
- `sensitivity.py`: `morris_screening(sim_factory, params_prior, metric_fn, r, rng) -> DataFrame` then `sobol_analysis(sim_factory, params_prior, metric_fn, N, rng) -> DataFrame`. Uses `SALib.sample.saltelli` and `SALib.analyze.sobol`. Test: S1 indices sum within 0.05 of 1.0.
- `fitting.py`: `fit_parameters(sim_factory, params0, bounds, data, windows, cfg) -> FitResult`. Uses `scipy.optimize.least_squares` with `loss='soft_l1'`. Windowed re-initialisation: for each window, reset sim state from smoothed telemetry, run forward horizon_s, compare. Jacobian via central finite differences (`rel_step=1e-4`).
- `jacobian.py`: `identifiability_diagnostics(J, param_names) -> dict`. Reports condition number, correlation matrix, flags `|corr| > 0.95` or `cond > 1e6`.
- `recommend.py`: `recommend(cond_result, sens_result, fit_result, det_result, cfg) -> list[Recommendation]`. A Recommendation is a pydantic model with fields: `target_variant, parameter, regime, estimated_change_ci, expected_improvement, evidence, caveats`. Never emits a recommendation without ≥2 independent evidence types.
- E7, E8 scripts. `fsa identify` implemented.

**DoD:** S7 (median relative error ≤10% for CLa, CD0, k when identifiable) and S8 (≥30% RMSE reduction after refit) pass on benchmark. Non-identifiable parameter test case flags correctly.

### Phase 8 — Reporting & Dashboard
- `figures.py`: every figure function signature: `def plot_X(...) -> matplotlib.figure.Figure`. Takes no global state. Returns the Figure object. Smoke test: call every figure function with minimal fixture data, check `isinstance(fig, Figure)`.
- Report template `report.html.j2`: sections — Executive Summary · Data & QC · Alignment · Variant Comparison · Condition Analysis · Prediction · Detection · Identification & Recommendations · Appendix. Every number in Executive Summary references a computed field from `summary.json`, never a hardcoded string.
- `summary.json` lint test: every sentence in the executive summary template contains a `{{ field }}` reference.
- Dashboard `app/main.py`: run picker → flight/variant selector → overlay tab → residual tab → heatmap tab → detector tab → recommendations tab → run comparison tab. No hardcoded run paths.
- `fsa run-all`: orchestrate all steps with resumability (skip step if artifact hash matches).
- `fsa benchmark`: runs E1–E8 on seeds 1–5, writes `benchmark_summary.json` with S1–S9 pass/fail.
- README.md: quick-start that works from a clean clone.
- `docs/CONFIG_REFERENCE.md`: auto-generated from pydantic model `__doc__` strings.

**DoD:** `make demo` runs in < 5 minutes on an 8-core laptop. `fsa benchmark` reports S1–S9 all PASS. Report has zero placeholder text. Dashboard loads without error.

### Phase 9 — Hardening
- Coverage ≥ 85% (`pytest --cov-fail-under=85`)
- `mypy --strict src/` zero errors
- Dockerfile: FROM python:3.11-slim, COPY, pip install, make demo
- Performance profile: preprocess 100 flights in < 30 min on 8-core laptop (document measured time)
- E10 real-data dry-run: map the NASA DASHlink FOQA `.mat` file to the canonical schema using a column-mapping YAML. Do not convert to another format first; write a `scipy.io.loadmat` adapter directly.
- Tag `v1.0.0`.

---

## 10. Datasets — where to find them and how to use them

### Primary (used directly in the pipeline)

**D1 — NASA DASHlink Sample Flight Data (FOQA)**
- URL: `https://c3.ndc.nasa.gov/dashlink/projects/85`
- Format: MATLAB `.mat` files. Read with `scipy.io.loadmat`.
- Signals: 186 engineering parameters including altitude, airspeed, load factors, control surfaces.
- Use for: E10 real-data dry-run adapter; anomaly detection benchmarking; residual prediction on real data.
- Adapter: write `telemetry/adapters/dashlink_mat.py` with a column mapping YAML at `configs/adapters/dashlink.yaml`.
- Registration: may require free registration at the DASHlink site.
- Licence: NASA public.

**D2 — PX4 Flight Review Community Logs**
- URL: `https://logs.px4.io`
- Format: `.ulg` binary. Read with `pyulog` (optional extra).
- Signals: full autopilot state — attitude, position, velocity, rates, IMU, baro, GPS, actuators.
- Use for: ULog adapter testing (FR-TEL-006); real fixed-wing flight telemetry for E10.
- Filter: exclude flights < 120 s (likely bench tests). Select fixed-wing mode logs.
- Adapter: `telemetry/adapters/ulog.py` (already in tree plan, behind `[adapters]` extra).
- Licence: community-uploaded, publicly accessible.

**D3 — Alaska DOT&PF ARROW UAS Dataset (2023–2025)**
- URL: `https://rosap.ntl.bts.gov/view/dot/92371`
- Format: CSV, 1,151 records.
- Use for: CSV adapter testing; operational condition diversity (altitude range, mission duration, battery).
- Adapter: standard `csv_parquet.py` with mapping YAML `configs/adapters/arrow_uas.yaml`.
- Licence: U.S. Public Domain.

**D4 — UIUC Airfoil Database**
- URL: `https://m-selig.ae.illinois.edu/ads/coord_database.html`
- Format: `.dat` polar files.
- Use for: Populating CL(alpha), CD(CL) reference tables for variant validation; not used directly in the pipeline but referenced in `docs/` and tests.
- Do not bundle in repo. Reference the URL in test fixtures.

**D5 — OpenAP Aircraft Performance Model**
- URL: `https://github.com/junzis/openap` (MIT)
- Format: JSON/YAML coefficients, pip-installable.
- Use for: Alternative parameter priors for identification bounds; cross-checking variant parameter assumptions.
- Add as optional dev dep: `openap>=1.3` in `[dev]` extras.

### Secondary (benchmarking and method validation only)

**D6 — NASA CMAPSS Turbofan Degradation Dataset**
- URL: `https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/`
- Format: TXT (space-separated).
- Use for: E6 detector benchmarking — compare your CUSUM/EWMA detection delay against published CMAPSS baselines. Shows your methods work on an independently-benchmarked dataset.
- Write a one-file helper: `experiments/helpers/cmapss_loader.py`. Not part of the main pipeline.

**D7 — OpenSky Network ADS-B**
- URL: `https://opensky-network.org/data/` (CC BY 4.0)
- Format: CSV via REST API or direct download.
- Signals: position (lat/lon/alt), ground speed, heading, vertical rate — 1 Hz typical.
- Use for: E10 large-scale trajectory overlay; condition diversity in altitude/speed for benchmarking condition analysis.
- Limitation: no aerodynamic signals (no AoA, no load factor). Use trajectory-level metrics only.
- Adapter: `telemetry/adapters/opensky.py` — maps lat/lon/alt to x/y/alt_m via local flat-Earth projection.

**D8 — Comprehensive Airfoil Dataset (2,900 airfoils, Agarwal et al. 2024)**
- URL: `https://github.com/kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils`
- DOI: `10.3390/data9050064`
- Format: CSV.
- Use for: Providing a statistical prior distribution over CL/CD parameters for sensitivity analysis bounds (FR-IDN-001). Sample 20 airfoils similar in thickness to your UAV wing.
- Write `experiments/helpers/airfoil_priors.py`. Not part of the main pipeline.

### What NOT to do with datasets
- Do not bundle any dataset file in the repo. Reference URLs and write loaders.
- Do not assume column names match the canonical schema. Always use an adapter + mapping YAML.
- Do not train predictive models on D6/D7/D8 and report as AeroValidate results — these are benchmark/prior reference only.
- If a dataset requires registration or terms acceptance, document that in `docs/07_nfr_risks_governance.md` and provide the URL only, not the data.

---

## 11. Research papers — how to use them (not just cite them)

Map each paper to a specific implementation decision. Every non-obvious algorithm choice in the codebase gets a comment `# Method: [PXX]`.

| Code location | Paper | What to implement from it |
|---|---|---|
| `identify/fitting.py` | **P01** (IEEE Aerospace 2023, `10.1109/AERO55745.2023.10115928`) | Output error method structure; input signal design recommendations |
| `identify/fitting.py` | **P02** (IEEE Access 2024, `10.1109/access.2024.3421579`) | Partial-update estimator as alternative to least_squares; comparison methodology |
| `identify/fitting.py` + `identify/regime.py` | **P03** (AIAA SciTech 2024, `10.2514/6.2024-0248`) | Seven-step procedure maps directly to your E7/E8 pipeline |
| `identify/jacobian.py` | **P04** (Periodica 2023) | Two-step regression-on-error; synthetic data for ID validation |
| `identify/fitting.py` | **P05** (Aeronautical J. 2023, `10.1017/aer.2022.100`) | FEM+PSO as optional alternative optimiser; derivative accuracy targets |
| `detect/ml_detectors.py` | **P07** (IEEE 2024, `10630804`) | MRA-LSTM architecture for multi-rate telemetry |
| `detect/evaluate.py` + literature section | **P08** (IEEE Sensors J. 2024, `10.1109/JSEN.2024.3451648`) | Detector evaluation taxonomy and metric definitions |
| `detect/ml_detectors.py` | **P09** (JCDE 2024) | 1D CNN-LSTM baseline; bias/drift fault classification |
| `detect/ml_detectors.py` | **P10** (STC-LSTM-AE, 2023) | Unsupervised baseline without labelled anomalies |
| `telemetry/generator.py` + `detect/evaluate.py` | **P11** (RflyMAD 2023) | SIL/HIL/real equivalency test; multi-rate clock unification methodology |
| `detect/calibrate.py` | **P12** (LSTM-AE dynamic threshold 2022) | Dynamic thresholding strategy; detection delay definition |
| `predict/tasks.py` | **P13** (PINN UQ, IEEE Aerospace 2023) | UQ framing; comparison baseline for physics-informed approach |
| `predict/tasks.py` | **P15** (Aurora Flight Sciences digital twin) | Physics→predictive twin methodology; interpretability requirements |
| `predict/uncertainty.py` | **P17** (Conformal Safety Monitoring, Stanford 2025, `arXiv:2511.20811`) | Split-conformal implementation; calibration protocol; coverage metric |
| `predict/uncertainty.py` | **P18** (Conformal UQ for surrogates 2024) | Normalised nonconformity score; interval score (Winkler) |
| `prep/sync.py` + `prep/clean.py` | **P20** (PX4 classification, 2024) | Multi-rate preprocessing; data cohesion filtering strategy |
| `prep/sync.py` | **P21** (SAE AeroTech 2024) | Telemetry time alignment; handling low-frequency / dropped data |
| `validate/compare.py` | **P22** (4-DoF VFT aero ID, AIAA 2023) | Model-vs-model comparison methodology; fidelity-vs-accuracy relationship |

Add this comment block at the top of every module that implements a referenced method:
```python
# References:
# [P01] "Missile Aerodynamics Model Identification Using Flight Data"
#       2023 IEEE Aerospace Conference. DOI: 10.1109/AERO55745.2023.10115928
```

---

## 12. CLI specification — implement all commands exactly

```bash
fsa --help                     # shows all subcommands with descriptions

fsa generate \
  --config configs/default.yaml \
  --run-id <auto-uuid4 if not given> \
  --seed 20260101 \
  --profile demo \             # demo: 12 flights; full: use n_flights from config
  --set generate.n_flights=20  # override any config key

fsa ingest \
  --config configs/default.yaml \
  --run-id <id> \
  --source path/to/external/data/ \
  --mapping configs/adapters/my_mapping.yaml

fsa preprocess  --config ... --run-id ...
fsa validate    --config ... --run-id ...
fsa analyze     --config ... --run-id ...  # condition analysis
fsa predict     --config ... --run-id ...
fsa detect      --config ... --run-id ...
fsa identify    --config ... --run-id ...
fsa report      --config ... --run-id ...
fsa run-all     --config ... --run-id ... --profile demo   # orchestrates all above
fsa dashboard   --run-id ...                               # launches streamlit
fsa benchmark   --seeds 1,2,3,4,5 --config configs/default.yaml
```

Every command:
- Is idempotent: if output artifacts exist and input hashes match, skip (unless `--force`).
- Writes to `runs/<run_id>/<step>/` per the artifact layout in `docs/03_architecture_and_data_spec.md §6`.
- Updates `run_manifest.json` with step timing on completion.
- Logs start and end with step name and run_id at INFO level.

---

## 13. Testing requirements — minimum bar

Every test must be runnable in < 2 s unless decorated `@pytest.mark.slow`.

```python
# Mark requirement coverage in test docstrings or with:
@pytest.mark.req("FR-COR-003")
def test_isa_matches_table_at_0km():
    ...

# Every physics test uses analytic ground truth:
def test_level_trim_holds():
    # compute trim throttle analytically, run 60 s, assert alt_err < 0.1%

# Every metric test uses hand-computed values:
def test_theil_decomposition_sums_to_one():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    sim = obs + 0.1
    um, us, uc = theil_decompose(obs, sim)
    assert abs(um + us + uc - 1.0) < 1e-10

# Leakage test example:
def test_no_window_straddles_flight_boundary():
    ds = build_windows(aligned_df, cfg)
    for flight_id in ds.flight_ids:
        windows = ds.get_flight_windows(flight_id)
        for w in windows:
            assert (w["flight_id"] == flight_id).all()
```

**Coverage gate:** `--cov-fail-under=85` in `pyproject.toml [tool.pytest.ini_options]`.
**Marker registration:** All custom markers declared in `pyproject.toml [tool.pytest.ini_options] markers`.

---

## 14. Code quality standards

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E","W","F","I","N","UP","ANN","B","SIM","PIE","C90"]
ignore = ["ANN101","ANN102"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
```

**Docstring style:** NumPy format, always include units and array shapes.

```python
def compute_rmse(obs: np.ndarray, sim: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Root-mean-square error of simulation vs observation.

    Parameters
    ----------
    obs : np.ndarray, shape (N,)
        Observed signal values in SI units.
    sim : np.ndarray, shape (N,)
        Simulated signal values in SI units.
    mask : np.ndarray of bool, shape (N,), optional
        True where samples are excluded. If None, all samples used.

    Returns
    -------
    float
        RMSE in the same unit as obs/sim.

    References
    ----------
    # [P08] IEEE Sensors Journal 2024, DOI: 10.1109/JSEN.2024.3451648
    """
```

---

## 15. Config system — how to implement it

All config lives in pydantic v2 `BaseModel` subclasses in `core/config.py`. Every field has a docstring and a default.

```python
class StudyConfig(BaseModel):
    """Root configuration for an AeroValidate study."""
    study: StudyMeta
    vehicle: str = "generic_uav_v1"   # name of YAML in configs/vehicles/
    generate: GenerateConfig
    variants: list[str] = ["V0","V1","V2","V3","V4","V5"]
    preprocess: PrepConfig
    validate: ValidateConfig
    predict: PredictConfig
    detect: DetectConfig
    identify: IdentifyConfig
    report: ReportConfig

def load_config(path: Path, overrides: list[str] = ()) -> StudyConfig:
    """
    Load YAML config, resolve `include:` directives, apply --set overrides.
    Overrides format: "section.key=value" (dotted path).
    Raises ConfigError with the offending field name on validation failure.
    """
```

The `--set` override system must work for nested keys: `--set validate.bootstrap.n=500`.

---

## 16. Manifest system — implement exactly this

Every CLI step calls `manifest.update_step(run_id, step_name, status, timings, artifact_paths)`.

```json
{
  "run_id": "20260101-abc123",
  "created_at": "2026-01-01T10:00:00Z",
  "git_sha": "a1b2c3d4",
  "config_hash": "sha256:...",
  "python_version": "3.11.9",
  "package_versions": {"aerovalid": "1.0.0", "numpy": "1.26.4", ...},
  "master_seed": 20260101,
  "steps": {
    "generate": {
      "status": "completed",
      "started_at": "...", "completed_at": "...",
      "duration_s": 47.2,
      "input_hashes": {},
      "output_hashes": {"raw/telemetry/F0001.parquet": "sha256:..."},
      "n_flights": 12
    }
  }
}
```

---

## 17. What to do when you are uncertain

1. Check `docs/02_FRD.md` for the requirement.
2. Check `docs/05_analysis_methods.md` for the algorithm.
3. Check `docs/03_architecture_and_data_spec.md` for data shapes and column names.
4. If still ambiguous, pick the simplest option consistent with the docs, write a one-paragraph ADR in `docs/DECISIONS.md` (format: `## ADR-NNN: <title>` / `Context` / `Decision` / `Consequences`), and continue.
5. Never ask a clarifying question mid-build. Make a decision and document it.
6. Never invent a new module not in the repo layout. If you need something new, add it to the layout first in DECISIONS.md.

---

## 18. What you must never do

- Never write `import *`
- Never write `# type: ignore` without a comment explaining why
- Never write a function longer than 60 lines without splitting it
- Never access `runs/`, `data/`, or `reports/` from inside `src/aerovalid/` — only via CLI and tests
- Never hardcode a file path; use `pathlib.Path` and resolve from config
- Never return a bare `dict` across module boundaries; use a pydantic model or dataclass
- Never write a test that depends on the order of other tests
- Never commit a notebook with output cells (nbstripout handles this)
- Never add a dependency without updating `pyproject.toml` and `docs/DECISIONS.md`
- Never modify `docs/00_vision.md` through `docs/06_implementation_plan.md` — they are the spec

---

## 19. How to start — first session instructions

```
1. Clone a fresh empty repo.
2. Copy this file and docs/00-06 into it.
3. Run: git init && git add . && git commit -m "chore: initial spec"
4. Begin Phase 0. Build every file listed under Phase 0 in §9 above.
5. Run: make lint test
6. When green: tick Phase 0 boxes in docs/06_implementation_plan.md, update CHANGELOG.md, commit.
7. Move to Phase 1. Repeat.
8. Never skip ahead.
```

**Suggested session-start message for each phase:**
> "Read `CLAUDE.md` and `PROMPT_FOR_CLAUDE_CODE.md` fully. I am starting Phase N. Build all files specified for Phase N in §9. Follow the DoD checklist. Record any decisions in `docs/DECISIONS.md`. When `make lint test` is green, stop and report: which files were created, which tests pass, and what was the one most non-obvious decision made."

---

## 20. Success criteria — the build is done when all of these pass

Run `fsa benchmark --seeds 1,2,3,4,5` and verify `benchmark_summary.json` shows:

| ID | Criterion | Target |
|---|---|---|
| S1 | `make demo` completes, deterministic | < 5 min, bit-identical rerun |
| S2 | Variant ranking recovers ground-truth fidelity order | Spearman ρ ≥ 0.9 |
| S3 | Condition analysis identifies injected flaw regime in top-2 | ≥ 90% of trials |
| S4 | Hybrid correction reduces held-out RMSE vs raw sim | ≥ 25% median reduction |
| S5 | Prediction intervals calibrated | 90% coverage ∈ [86%, 94%] |
| S6 | Deviation detector on injected faults | Delay ≤ 5 s, FA ≤ 1/hr, Recall ≥ 0.9 |
| S7 | Parameter identification recovers injected errors | Median relative error ≤ 10% for CLa, CD0, k |
| S8 | Re-validation after refit improves held-out error | ≥ 30% RMSE reduction |
| S9 | Code quality | Coverage ≥ 85%, mypy strict clean, CI green |

All nine PASS. Ship.
```
