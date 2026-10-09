# Config Reference

Auto-generated from `src/aerovalid/core/config.py` pydantic models by
`scripts/gen_config_reference.py`. Do not hand-edit; run `make config-ref`.

### StudyConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `study` | `StudyMeta` | StudyMeta(name='demo', seed=20260101, profile='demo') |  |
| `vehicle` | `VehicleRef` | VehicleRef(ref='generic_uav_v1') |  |
| `generate` | `GenerateConfig` | GenerateConfig(n_flights=12, scenarios=['mixed_mission', 'climb_cruise', 'turn_series', 'doublet_series'], conditions=GenerateConditions(alt0_m=(500.0, 6000.0), v0_mps=(28.0, 60.0), mass_kg=(22.0, 28.0), wind_mean_mps=(0.0, 10.0), turb=['none', 'light', 'moderate'], dT_K=(-10.0, 15.0)), sampling='sobol', telemetry_rate_hz=10.0, truth_rate_hz=100.0, faults=FaultsConfig(probability_per_flight=0.5, library='configs/faults/')) |  |
| `variants` | `list` | ['V0', 'V1', 'V2', 'V3', 'V4', 'V5'] |  |
| `preprocess` | `PrepConfig` | PrepConfig(grid_hz=10.0, max_gap_s=0.5, hampel=HampelConfig(window=7, n_sigma=4.0), sync=SyncConfig(method='xcorr', ref_signals=['alt_m', 'vel_air_mps'], max_lag_s=3.0, drift=True), sg=SavitzkyGolayConfig(window=11, order=3)) |  |
| `validate` | `ValidateConfig` | ValidateConfig(signals=['alt_m', 'vel_air_mps', 'gamma_rad', 'chi_rad', 'nz_g', 'pitch_rad', 'roll_rad', 'x_m', 'y_m'], horizons_s=[1.0, 2.0, 5.0, 10.0], window_s=10.0, stride_s=5.0, weights={}, tolerances={}, bootstrap=BootstrapConfig(n=2000, block_s=10.0)) |  |
| `predict` | `PredictConfig` | PredictConfig(tasks=['residual', 'hybrid', 'forecast'], targets=['alt_m', 'vel_air_mps'], horizons_s=[1.0, 5.0, 10.0, 30.0], models=['ridge', 'hgb', 'rf'], seq=False, cv=CVConfig(scheme='loso', folds=6), conformal=ConformalConfig(alpha=0.1)) |  |
| `detect` | `DetectConfig` | DetectConfig(target_far_per_hr=1.0, detectors=['zscore', 'ewma', 'cusum', 'mahalanobis', 'iforest', 'pelt'], early_warning=EarlyWarningConfig(horizon_s=30.0, p_star=0.8)) |  |
| `identify` | `IdentifyConfig` | IdentifyConfig(params=['CL0', 'CLa', 'CD0', 'k', 'thrust_lapse_n'], bounds={}, sobol=SobolConfig(n=1024), regimes=RegimesConfig(mach_bins=[0.0, 0.12, 0.18, 0.25]), revalidate=True) |  |
| `report` | `ReportConfig` | ReportConfig(formats=['html'], pdf=False) |  |

### StudyConfig.study

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | 'demo' |  |
| `seed` | `int` | 20260101 |  |
| `profile` | `str` | 'demo' |  |

### StudyConfig.vehicle

| Field | Type | Default | Description |
|---|---|---|---|
| `ref` | `str` | 'generic_uav_v1' |  |

### StudyConfig.generate

| Field | Type | Default | Description |
|---|---|---|---|
| `n_flights` | `int` | 12 |  |
| `scenarios` | `list` | ['mixed_mission', 'climb_cruise', 'turn_series', 'doublet_series'] |  |
| `conditions` | `GenerateConditions` | GenerateConditions(alt0_m=(500.0, 6000.0), v0_mps=(28.0, 60.0), mass_kg=(22.0, 28.0), wind_mean_mps=(0.0, 10.0), turb=['none', 'light', 'moderate'], dT_K=(-10.0, 15.0)) |  |
| `sampling` | `str` | 'sobol' |  |
| `telemetry_rate_hz` | `float` | 10.0 |  |
| `truth_rate_hz` | `float` | 100.0 |  |
| `faults` | `FaultsConfig` | FaultsConfig(probability_per_flight=0.5, library='configs/faults/') |  |

### StudyConfig.generate.conditions

| Field | Type | Default | Description |
|---|---|---|---|
| `alt0_m` | `tuple` | (500.0, 6000.0) |  |
| `v0_mps` | `tuple` | (28.0, 60.0) |  |
| `mass_kg` | `tuple` | (22.0, 28.0) |  |
| `wind_mean_mps` | `tuple` | (0.0, 10.0) |  |
| `turb` | `list` | ['none', 'light', 'moderate'] |  |
| `dT_K` | `tuple` | (-10.0, 15.0) |  |

### StudyConfig.generate.faults

| Field | Type | Default | Description |
|---|---|---|---|
| `probability_per_flight` | `float` | 0.5 |  |
| `library` | `str` | 'configs/faults/' |  |

### StudyConfig.preprocess

| Field | Type | Default | Description |
|---|---|---|---|
| `grid_hz` | `float` | 10.0 |  |
| `max_gap_s` | `float` | 0.5 |  |
| `hampel` | `HampelConfig` | HampelConfig(window=7, n_sigma=4.0) |  |
| `sync` | `SyncConfig` | SyncConfig(method='xcorr', ref_signals=['alt_m', 'vel_air_mps'], max_lag_s=3.0, drift=True) |  |
| `sg` | `SavitzkyGolayConfig` | SavitzkyGolayConfig(window=11, order=3) |  |

### StudyConfig.preprocess.hampel

| Field | Type | Default | Description |
|---|---|---|---|
| `window` | `int` | 7 |  |
| `n_sigma` | `float` | 4.0 |  |

### StudyConfig.preprocess.sync

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | 'xcorr' |  |
| `ref_signals` | `list` | ['alt_m', 'vel_air_mps'] |  |
| `max_lag_s` | `float` | 3.0 |  |
| `drift` | `bool` | True |  |

### StudyConfig.preprocess.sg

| Field | Type | Default | Description |
|---|---|---|---|
| `window` | `int` | 11 |  |
| `order` | `int` | 3 |  |

### StudyConfig.validation

| Field | Type | Default | Description |
|---|---|---|---|
| `signals` | `list` | ['alt_m', 'vel_air_mps', 'gamma_rad', 'chi_rad', 'nz_g', 'pitch_rad', 'roll_rad', 'x_m', 'y_m'] |  |
| `horizons_s` | `list` | [1.0, 2.0, 5.0, 10.0] |  |
| `window_s` | `float` | 10.0 |  |
| `stride_s` | `float` | 5.0 |  |
| `weights` | `dict` | {} |  |
| `tolerances` | `dict` | {} |  |
| `bootstrap` | `BootstrapConfig` | BootstrapConfig(n=2000, block_s=10.0) |  |

### StudyConfig.validation.bootstrap

| Field | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | 2000 |  |
| `block_s` | `float` | 10.0 |  |

### StudyConfig.predict

| Field | Type | Default | Description |
|---|---|---|---|
| `tasks` | `list` | ['residual', 'hybrid', 'forecast'] |  |
| `targets` | `list` | ['alt_m', 'vel_air_mps'] |  |
| `horizons_s` | `list` | [1.0, 5.0, 10.0, 30.0] |  |
| `models` | `list` | ['ridge', 'hgb', 'rf'] |  |
| `seq` | `bool` | False |  |
| `cv` | `CVConfig` | CVConfig(scheme='loso', folds=6) |  |
| `conformal` | `ConformalConfig` | ConformalConfig(alpha=0.1) |  |

### StudyConfig.predict.cv

| Field | Type | Default | Description |
|---|---|---|---|
| `scheme` | `str` | 'loso' |  |
| `folds` | `int` | 6 |  |

### StudyConfig.predict.conformal

| Field | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | 0.1 |  |

### StudyConfig.detect

| Field | Type | Default | Description |
|---|---|---|---|
| `target_far_per_hr` | `float` | 1.0 |  |
| `detectors` | `list` | ['zscore', 'ewma', 'cusum', 'mahalanobis', 'iforest', 'pelt'] |  |
| `early_warning` | `EarlyWarningConfig` | EarlyWarningConfig(horizon_s=30.0, p_star=0.8) |  |

### StudyConfig.detect.early_warning

| Field | Type | Default | Description |
|---|---|---|---|
| `horizon_s` | `float` | 30.0 |  |
| `p_star` | `float` | 0.8 |  |

### StudyConfig.identify

| Field | Type | Default | Description |
|---|---|---|---|
| `params` | `list` | ['CL0', 'CLa', 'CD0', 'k', 'thrust_lapse_n'] |  |
| `bounds` | `dict` | {} |  |
| `sobol` | `SobolConfig` | SobolConfig(n=1024) |  |
| `regimes` | `RegimesConfig` | RegimesConfig(mach_bins=[0.0, 0.12, 0.18, 0.25]) |  |
| `revalidate` | `bool` | True |  |

### StudyConfig.identify.sobol

| Field | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | 1024 |  |

### StudyConfig.identify.regimes

| Field | Type | Default | Description |
|---|---|---|---|
| `mach_bins` | `list` | [0.0, 0.12, 0.18, 0.25] |  |

### StudyConfig.report

| Field | Type | Default | Description |
|---|---|---|---|
| `formats` | `list` | ['html'] |  |
| `pdf` | `bool` | False |  |
