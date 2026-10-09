# 03 — Architecture & Data Specification

## 1. Architectural principles

- **Pipeline of idempotent stages** communicating through versioned artifacts on disk (Parquet/JSON), orchestrated by CLI.
- **Hexagonal boundaries:** numerical cores (`sim`, `prep`, `validate`, `predict`, `detect`, `identify`) are pure-Python libraries; I/O lives in adapters and `cli.py`.
- **Protocols over inheritance** for plug-ins (simulators, adapters, models, detectors).
- **Config-first:** behaviour determined by pydantic-validated YAML.

## 2. Logical architecture

```
                         ┌─────────────────────────── configs/*.yaml ───────────────────────────┐
                         ▼                                                                      │
 generate ──► raw/ ──► ingest ──► ingested/ ──► preprocess ──► aligned/ ──┬─► validate ──► validation/
 (truth, variants,    (adapters,            (QC, clean, sync,            │                  │
  sensors, faults)     mapping)              resample, features)         ├─► predict  ──► predict/
                                                                         ├─► detect   ──► detect/
                                                                         └─► identify ──► identify/
                                                                                              │
                                                  analyze (condition analysis) ◄──────────────┤
                                                                                              ▼
                                                                                 report/ + dashboard
```

## 3. Module responsibilities and public interfaces

```python
# core/config.py
class StudyConfig(BaseModel): ...           # root config (see §7)
def load_config(path: Path, overrides: list[str] = ()) -> StudyConfig

# sim/
class SimulatorProtocol(Protocol):
    def run(self, scenario: Scenario, controls: ControlHistory | None, init: State | None,
            *, seed: int) -> SimResult: ...          # replay or free-run
class VariantSpec(BaseModel): name: str; params: AeroParams; switches: StructuralSwitches
def make_variant(spec: VariantSpec) -> SimulatorProtocol
def make_truth(params: TruthParams) -> SimulatorProtocol
def run_windowed(sim: SimulatorProtocol, tel_state: DataFrame, controls: DataFrame,
                 horizon_s: float, stride_s: float) -> DataFrame     # window_id, horizon_s, sim_* cols

# telemetry/
class SensorModel(BaseModel): ...            # noise, bias, drift, lag, quantisation
def apply_sensors(truth: DataFrame, cfg: SensorSuite, rng) -> tuple[DataFrame, DefectLog]
def inject_faults(sim_cfg, faults: list[FaultSpec]) -> TruthParams-with-schedule
class Adapter(Protocol):
    def read(self, path: Path, mapping: ColumnMapping) -> DataFrame   # canonical schema

# prep/
def qc_report(df, registry) -> QCReport
def clean(df, cfg) -> CleanResult
def estimate_sync(ref: DataFrame, other: DataFrame, cfg) -> SyncEstimate   # lag, drift, ci
def resample(df, dt, cfg) -> DataFrame
def add_features(df, cfg) -> DataFrame
def segment_phases(df, cfg) -> Series
def build_aligned(obs, sim, cfg) -> AlignedDataset

# validate/
def compute_metrics(aligned, signals, groupby, cfg) -> DataFrame
def compare_variants(metrics_by_flight, cfg) -> ComparisonResult
def condition_analysis(aligned_all, cfg) -> ConditionResult

# predict/
class ResidualModel(Protocol):
    def fit(self, X, y, groups=None): ...
    def predict(self, X) -> ndarray
    def predict_interval(self, X, alpha) -> tuple[ndarray, ndarray]
def build_windows(aligned, cfg) -> WindowedDataset
def cross_validate(model_factory, ds, splitter, cfg) -> CVResult

# detect/
class Detector(Protocol):
    def fit(self, nominal_residuals): ...
    def score(self, residual_window) -> float
    def update(self, residual_sample) -> bool      # streaming
def evaluate_detectors(detectors, flights, fault_logs, cfg) -> DetectionReport

# identify/
def sensitivity(sim_factory, params_prior, metric_fn, cfg) -> SensitivityResult
def fit_parameters(sim_factory, params0, bounds, data, cfg) -> FitResult   # θ̂, cov, jac, diagnostics
def recommend(cond: ConditionResult, sens, fit, det) -> list[Recommendation]
```

## 4. Canonical data schema

### 4.1 Time-series table (one row per time sample)

| Column | Type | Unit | Notes |
|---|---|---|---|
| `t_s` | float64 | s | Seconds from flight start; strictly increasing after cleaning |
| `t_utc` | datetime64[ns] | – | optional |
| `flight_id` | string | – | e.g. `F0007` |
| `source` | category | – | `truth`, `tel`, or variant name (`V0`…) |

**Signal columns** use `<name>` in SI units internally. The registry (`core/signals.py`) is authoritative:

| Signal | Symbol | Unit | Frame / note | Default tolerance |
|---|---|---|---|---|
| `x_m`, `y_m` | position | m | local NE plane from start | 15 m |
| `alt_m` | altitude MSL | m | up positive | 5 m |
| `vel_air_mps` | true airspeed | m/s | | 1.5 m/s |
| `vel_gnd_mps` | ground speed | m/s | | 1.5 m/s |
| `gamma_rad` | flight-path angle | rad | | 0.02 |
| `chi_rad` | track angle | rad | 0 = North, clockwise | 0.03 |
| `roll_rad`, `pitch_rad`, `yaw_rad` | attitude | rad | 3-2-1 Euler | 0.03 |
| `alpha_rad` | angle of attack | rad | optional | 0.01 |
| `nx_g`, `nz_g` | body load factors | g | specific-force based | 0.08 |
| `throttle` | 0–1 | – | control | – |
| `bank_cmd_rad`, `alpha_cmd_rad` | controls | rad | control | – |
| `mass_kg` | mass | kg | optional | – |
| `wind_n_mps`, `wind_e_mps`, `wind_d_mps` | wind | m/s | optional (truth only unless estimated) | – |
| `mach` | derived | – | | 0.005 |
| `q_pa` | dyn. pressure | Pa | derived | 30 Pa |

**Column prefixes in aligned tables:** `obs_<sig>` (telemetry), `sim_<sig>` (variant), `res_<sig> = obs − sim` (wrap angles to (−π, π]), `mask_<sig>` (bool, True = excluded), `cond_<name>` (condition descriptors), `phase` (category).

### 4.2 Windowed-prediction table
`flight_id, variant, window_id, t0_s, horizon_s, t_s, sim_<sig>, obs_<sig>, res_<sig>`

### 4.3 Metadata JSON (per flight)
```json
{
 "flight_id": "F0007", "scenario": "mixed_mission", "seed": 123456,
 "conditions": {"alt0_m": 3200, "v0_mps": 38.2, "mass_kg": 24.6, "wind_mean_mps": 4.1, "turb": "light", "dT_K": 6.0},
 "sample_rate_hz": {"tel": 10.0, "truth": 100.0},
 "faults": [{"type": "cd0_step", "t_on_s": 412.0, "t_off_s": null, "magnitude": 0.18}],
 "defects": {"time_offset_s": 0.37, "drift_ppm": 25, "gaps": [[120.4,121.2]]}
}
```
Ground truth hidden parameters live in `truth_params.json` and are **only** read by the benchmark/evaluation code, never by variants, preprocessing, predictors, or detectors.

## 5. Naming and typing conventions
- Angles in radians internally; attitude wrapping utilities in `core/angles.py`.
- Time grid: `t_s` float seconds; index not used for time.
- Parquet with zstd compression; partitioned by `flight_id` in directories.
- Categorical columns stored as dictionary-encoded.

## 6. Artifact layout (`runs/<run_id>/`)

```
runs/<run_id>/
├─ run_manifest.json  config.resolved.yaml  log.txt
├─ raw/
│  ├─ truth/<flight>.parquet            # high-rate truth (synthetic only)
│  ├─ telemetry/<flight>.parquet        # sensor-corrupted observations
│  ├─ variants/<variant>/<flight>.parquet
│  └─ meta/<flight>.json  truth_params.json  faults.json
├─ ingested/                            # canonical-schema copies (external data path)
├─ qc/<flight>.json  qc_summary.parquet
├─ sync/sync_estimates.parquet          # lag, drift, CI per (flight, variant)
├─ aligned/<variant>/<flight>.parquet   # obs_/sim_/res_/mask_/cond_/phase
├─ aligned/windowed/<variant>/<flight>.parquet
├─ aligned/index.json
├─ validation/
│  ├─ metrics_signal.parquet  metrics_phase.parquet  metrics_condition.parquet  metrics_horizon.parquet
│  ├─ variant_ranking.parquet  pairwise_tests.parquet  condition_importance.parquet
├─ predict/
│  ├─ models/<task>/<model>.joblib  cv_results.parquet  metrics.json
│  ├─ forecasts.parquet  intervals.parquet  explain.parquet
├─ detect/
│  ├─ baselines.json  alarms.parquet  detection_metrics.parquet  roc.parquet
├─ identify/
│  ├─ sensitivity.parquet  fit_results.json  regime_fits.parquet
│  ├─ identifiability.json  recommendations.json  revalidation.parquet
├─ figures/*.png  *.html
└─ reports/report.html  summary.json  tables/*.csv|md
```

## 7. Configuration model (top-level keys)

```yaml
# configs/default.yaml (abridged; see the file in configs/)
study: {name: demo, seed: 20260101, profile: demo}
vehicle: {ref: generic_uav_v1}            # parameters in configs/vehicles/
generate:
  n_flights: 12
  scenarios: [mixed_mission, climb_cruise, turn_series, doublet_series]
  conditions: {alt0_m: [500, 6000], v0_mps: [28, 60], mass_kg: [22, 28], wind_mean_mps: [0, 10], turb: [none, light, moderate], dT_K: [-10, 15]}
  sampling: sobol
  telemetry_rate_hz: 10
  truth_rate_hz: 100
  faults: {probability_per_flight: 0.5, library: configs/faults/}
variants: [V0, V1, V2, V3, V4, V5]
preprocess: {grid_hz: 10, max_gap_s: 0.5, hampel: {window: 7, n_sigma: 4}, sync: {method: xcorr, ref_signals: [alt_m, vel_air_mps], max_lag_s: 3, drift: true}, sg: {window: 11, order: 3}}
validate: {signals: [alt_m, vel_air_mps, gamma_rad, chi_rad, nz_g, pitch_rad, roll_rad, x_m, y_m], horizons_s: [1,2,5,10], window_s: 10, stride_s: 5, weights: {...}, tolerances: {...}, bootstrap: {n: 2000, block_s: 10}}
predict: {tasks: [residual, hybrid, forecast], targets: [alt_m, vel_air_mps], horizons_s: [1,5,10,30], models: [ridge, hgb, rf], seq: false, cv: {scheme: loso, folds: 6}, conformal: {alpha: 0.1}}
detect: {target_far_per_hr: 1.0, detectors: [zscore, ewma, cusum, mahalanobis, iforest, pelt], early_warning: {horizon_s: 30, p_star: 0.8}}
identify: {params: [CL0, CLa, CD0, k, thrust_lapse_n], bounds: {...}, sobol: {n: 1024}, regimes: {mach_bins: [0, 0.12, 0.18, 0.25]}, revalidate: true}
report: {formats: [html], pdf: false}
```
Every key gets a pydantic model with defaults and validators; document all fields in `docs/CONFIG_REFERENCE.md` (auto-generate from the models).

## 8. Error handling & edge cases
- Non-monotonic timestamps → sort + dedupe + warn; if > 5% affected → flight quarantined (excluded, listed in QC summary).
- Variant produces NaN/inf (blow-up) → record `status=diverged`, truncate at divergence time, exclude after; count in report.
- Insufficient sync correlation (peak < 0.6) → flag, fall back to zero-lag with warning.
- Tiny groups in condition bins (n < `min_bin_samples`) → bin merged or marked "insufficient".
- Angle residuals always wrapped; circular statistics for χ, yaw.

## 9. Performance & scaling notes
- Use vectorised numpy; integrators in numba (optional extra) or `scipy` for variants; truth at 100 Hz × ~20 min flights ≈ 1.2×10⁵ steps — fine in pure numpy loops with RK4 if state is small (use numba `@njit` if > 5 s).
- Parallelism: `joblib` across flights; seeds derived per flight to keep determinism.
- Memory: lazy per-flight reading; only concatenate for cross-flight analysis stages.

## 10. Security & governance
See `07_nfr_risks_governance.md`.
