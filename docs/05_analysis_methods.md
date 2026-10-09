# 05 — Analysis Methods & Experiment Design

Precise algorithmic definitions so implementation is unambiguous.

## 1. Time alignment

**Goal:** make simulated and observed signals comparable on a common clock.

1. Pick reference signals R (default `alt_m`, `vel_air_mps`; z-score each; use first differences to remove DC/trends).
2. Uniformly resample both series to a fine grid (e.g., 50 Hz) for correlation.
3. Compute normalised cross-correlation over lags ∈ [−L, L] (L = `max_lag_s`); sum across R; pick peak; refine with 3-point parabolic interpolation.
4. **Drift:** repeat in overlapping windows (60 s, 50% overlap); fit `lag(t) = a + b·t` by robust regression (Huber). Accept drift model only if the AIC improves and |b| > 2σ_b.
5. **CI:** moving-block bootstrap of windows → 95% CI for a, b.
6. Apply: `t_sim ← t_sim − (a + b·t)`; interpolate onto the telemetry clock.
7. Quality gate: peak NCC ≥ 0.6, else flag `sync_low_confidence`.
   *Note:* because variants in replay mode run from telemetry-timed controls, lag mostly reflects sensor/timestamp defects; sync stage is also applied between telemetry channels and truth in benchmark mode to verify recovery.

## 2. Cleaning & QC
- **Hampel filter:** window 2w+1; flag if |x − median| > n_σ · 1.4826 · MAD; replace with local median (flag retained in `mask_*`/`spike_*`).
- **Gap policy:** gap ≤ `max_gap_s` → linear interpolation (cubic for smooth signals optionally) with `interp_*` flag; > → NaN + mask.
- **Stuck sensor:** rolling variance ≈ 0 for > `stuck_s` while neighbouring signals vary → flag.
- **Range/physics checks:** registry ranges; consistency checks (e.g., |V_gnd − V_air| ≤ wind bound; dh/dt vs V sinγ within tolerance).
- QC score per flight = weighted fraction of valid data; flights below `qc_min` quarantined.

## 3. Feature engineering (per sample)
`mach, q_pa, rho, h_band, V_dot (SG), h_dot (SG), gamma_est, chi_dot (turn rate), bank_est = atan(V·χ̇/g), n_z_meas, specific_energy = h + V²/(2g), specific_power = d(Es)/dt, CL_eff = nz·m·g/(q S) (using telemetry mass or m0 − burn), CD_eff = (T_est cosα − nx·m·g)/(q S)` (T_est from throttle map of the variant under test — flag as model-dependent), `alpha_est`, wind_est = ground vel − air vel vector difference (needs heading), control rates.

**Phase segmentation rules (hysteresis):**
- `turn` if |bank_est| > 10° for > 3 s; `climb` if γ > 1.5° ; `descent` if γ < −1.5°; `accel` if |V̇| > 0.5 m/s²; `manoeuvre` if |q̇| or α̇ spike; else `cruise`.
- Minimum segment 3 s; merge short segments; priority: manoeuvre > turn > climb/descent > accel > cruise.

**Condition bins (default):** altitude [0–1.5, 1.5–3, 3–4.5, 4.5–6+] km; Mach [<0.10, 0.10–0.15, 0.15–0.20, 0.20–0.30, >0.30] (adjust to vehicle); |bank| [0–10, 10–30, 30–45, >45]°; n_z [<1.1, 1.1–1.5, 1.5–2.2, >2.2] g; wind [<3, 3–7, >7] m/s.

## 4. Validation metrics (definitions)

For residual e_i = obs_i − sim_i over valid samples n (angles wrapped):

- Bias `ME = mean(e)`; `MAE = mean|e|`; `RMSE = √mean(e²)`.
- `NRMSE_range = RMSE / (max(obs) − min(obs))`; `NRMSE_std = RMSE / std(obs)`.
- `MAPE = mean(|e| / max(|obs|, ε)) · 100` with ε = 1% of signal scale.
- `R² = 1 − Σe² / Σ(obs − mean(obs))²`; Pearson r.
- **Theil's U2** `= RMSE(sim vs obs) / RMSE(naive persistence vs obs)`; **TIC** `= RMSE / (√mean(sim²) + √mean(obs²))`.
- **Theil decomposition** of MSE: `U_M = (mean(sim) − mean(obs))² / MSE`, `U_S = (σ_sim − σ_obs)² / MSE`, `U_C = 2(1−r)σ_simσ_obs / MSE` (U_M+U_S+U_C = 1) — diagnoses bias vs scale vs phase error.
- Error growth: slope of `RMSE(h)` vs horizon h (windowed mode), fit log–log to detect drift (power-law exponent).
- Tolerance compliance: `P(|e| ≤ tol)`, time-to-first-exceedance (run length ≥ 1 s).
- Trajectory: `e_3D = ‖Δpos‖`, along/cross-track relative to the observed ground track direction; `CEP50/95` of horizontal error over flight.
- Circular statistics for χ/yaw (use wrapped differences; circular mean/std).

**Aggregation:** metrics per (flight, variant, signal, [phase|bin|horizon]); then flight-level aggregation (mean, median, IQR) — *flights are the unit of replication for inference*.

## 5. Variant comparison statistics

- **Composite score** per (flight, variant): `S = Σ_s w_s · (RMSE_{s,v} / RMSE_{s,ref})` with ref = V0 (or pooled median). Lower = better. Weights from config (default: alt 0.2, V 0.2, γ 0.1, χ 0.1, n_z 0.15, pitch 0.1, roll 0.05, horiz-pos 0.1).
- **Paired inference** over flights: Wilcoxon signed-rank on S differences, Holm–Bonferroni across pairs; Friedman test across all variants then Nemenyi post-hoc; report Cliff's δ / Hodges–Lehmann shift.
- **Within-flight block bootstrap** (block = `block_s`) of metric difference for time-level CI; **flight-level bootstrap** for ranking CIs (resample flights with replacement, recompute ranks, report rank distribution and P(variant is best)).
- **Rank agreement** with ground-truth fidelity order (Spearman ρ, Kendall τ) in benchmark mode.

## 6. Condition-dependence analysis (RQ2)
1. **Descriptive:** RMSE (or normalised |res|) heatmaps over 2-D bin grids with sample counts and CIs; hide cells with n < min.
2. **Model-based:** target `y = |res_s| / tol_s` (normalised); features = condition features + phase + variant-agnostic flight descriptors. Fit `HistGradientBoostingRegressor` with grouped CV by flight; report R² (CV), **permutation importance** (grouped), **partial dependence / ICE**, 2-D PDP for top pair; optionally SHAP.
3. **Hypothesis tests:** Kruskal–Wallis across bins per variable (flight-level medians), Dunn post-hoc with Holm; mixed-effects regression (`statsmodels.MixedLM`: `log|res| ~ cond + (1|flight)`) for adjusted effects.
4. **Regime-flaw recovery (benchmark):** compare ranked conditions to the known regime-flaw tag of each variant (S3).

## 7. Predictive modelling (RQ3, RQ4)

### 7.1 Tasks
- **A — Residual prediction:** `res_s(t+k) = f(features(t−L..t), controls, conditions, sim_s(t..t+k))`.
- **B — Hybrid:** `ŷ_s(t+k) = sim_s(t+k) + f̂_A(…)`; compare with sim-only (`sim`), persistence-of-residual (`sim + res(t)`), and pure ML forecast.
- **C — Forecast:** `y_s(t+k)` from history only (no sim) vs with sim as exogenous input.
Horizons k ∈ {1, 5, 10, 30} s.

### 7.2 Features
Lag window L = 10–30 s at 1–2 Hz sub-sampled; for each lag: obs state, controls, `sim` outputs, residual history, derived features (Mach, q, nz, bank, phase one-hot), summary stats (rolling mean/std/slope), condition bin codes. Standardise with train-fold statistics only. Sample weights optional to balance phases.

### 7.3 Models
Ridge/ElasticNet (baseline) → HistGradientBoosting / LightGBM → Random Forest → ARIMAX/VAR (per signal) → GRU/TCN (optional). Multi-output via per-target models or native multi-output.

### 7.4 Validation protocol
- **Splits:** leave-one-flight-out (LOFO) or grouped K-fold; plus **leave-condition-out** (hold out highest altitude band; hold out highest Mach band) for extrapolation.
- **Nested CV** for hyper-parameters (random search ≤ 30 trials).
- **Metrics:** RMSE/MAE, skill vs baselines, per-horizon curves, per-phase, per-condition; paired tests across flights.
- **Uncertainty:** (i) Quantile GBM at α/2, 1−α/2; (ii) **split-conformal** on calibration flights (held out from training) — normalised nonconformity |r|/σ̂(x); report coverage, mean width, interval score (Winkler), reliability plot.
- **Calibration of early-warning probabilities:** isotonic regression on exceedance probabilities; Brier score, reliability curve.

### 7.5 Anti-leakage checklist (tested)
Flight-level splits; scalers/imputers fit on train; no future samples in features (strict lag ≤ t); windowed sim init uses data ≤ t0 only; hyper-parameter search inside training folds; no truth-only columns (wind_truth, fault flags, truth params) in features.

## 8. Deviation detection

### 8.1 Residual normalisation
Per signal s (optionally per condition bin b): robust centre μ, scale σ via median/MAD on **nominal** (fault-free) training flights. `z = (res − μ)/σ`.

### 8.2 Detectors
| Detector | Statistic | Key params |
|---|---|---|
| z-score | `max_s |z_s|` > h for ≥ m consecutive samples | h≈4, m≈5 |
| EWMA | `E_t = λ z_t + (1−λ)E_{t−1}`; alarm if |E_t| > L σ_E | λ=0.1–0.3, L=3–4 |
| CUSUM | `S⁺ = max(0, S⁺ + z − k)`, `S⁻` mirrored; alarm S > H | k≈0.5, H calibrated |
| Page–Hinkley | cumulative deviation from running mean | δ, λ |
| Mahalanobis | `z' Σ⁻¹ z` vs χ²_d on whitened residual vector (Σ robust, e.g., MCD) | α for FAR |
| Isolation Forest / LOF / OC-SVM | score on windowed residual features (mean, std, slope, spectral energy, cross-signal correlations) | contamination via calibration |
| Autoencoder (opt.) | reconstruction error of residual windows | |
| Change-point (PELT/BinSeg) | segment mean/variance shifts, penalty by BIC | min_size |
| Predictive early-warning | Task-A/C predictor + conformal interval ⇒ P(|res(t+k)| > tol) ; alarm if ≥ p* | horizon 30 s |

**Threshold calibration:** choose threshold giving target FA/flight-hour on nominal calibration flights (empirical quantile of per-flight max statistic or Poisson-rate matching); report on separate test flights.

### 8.3 Evaluation
- Events: for each faulty flight: onset t_on (truth log), tolerance exceedance time t_exc (first time |res_s| > tol for ≥ 2 s, using *noise-free truth vs variant* in benchmark; using observed residual otherwise).
- **Detection delay** `t_alarm − t_on` (negative = early/predictive; false-early alarms before t_on counted as FA). **Lead time** `t_exc − t_alarm` (positive is good).
- **Recall** (detected within [t_on, t_on + T_max]); **precision**; **FA rate** per flight-hour; window-level ROC-AUC/PR-AUC; per-fault-type results; ablation by signal subset.
- Report distributions (median/IQR) across flights and seeds.

## 9. Experiments (map to research questions)

| ID | RQ | Experiment | Output |
|---|---|---|---|
| **E1** | RQ1 | Full-trajectory replay validation of V0–V6 on all flights; metrics table; composite ranking with CIs and significance | Ranking, forest plot |
| **E2** | RQ1 | Windowed multi-step validation (horizons 1–30 s); error growth exponent per variant | Error-vs-horizon curves |
| **E3** | RQ2 | Condition-dependence: heatmaps, GBM importance, mixed-effects; compare to regime-flaw tags | Heatmaps, importance, S3 score |
| **E4** | RQ3 | Residual prediction (Task A) across models/horizons/CV schemes | Skill tables |
| **E5** | RQ3/RQ4 | Hybrid correction (Task B) vs sim-only; extrapolation (leave-condition-out) | RMSE reduction (S4), degradation |
| **E6** | RQ3 | Deviation detection & early warning on injected faults | Delay, lead time, FA, ROC (S6) |
| **E7** | RQ4 | Sensitivity (Morris→Sobol) + local Jacobian + identifiability | Parameter ranking, identifiability table |
| **E8** | RQ4 | Regime-wise parameter fitting, refit variant V*, held-out re-validation, structural-term probe | Parameter CIs, before/after (S7, S8) |
| **E9** | all | Robustness: vary noise level, sync error, missing-data rate, number of flights (learning curves) | Sensitivity of conclusions |
| **E10** | all | Real-data dry-run via adapter on a public log (optional) | Pipeline sanity on non-synthetic |

Each experiment is a callable `experiments/eNN_*.py` that reads a config, writes artifacts, and contributes a report section; `fsa benchmark` runs E1–E8 over seeds {1…5}.

## 10. Parameter identification detail (E7/E8)

- **Parameter vector** θ ⊂ {CL0, CLα, CD0, k, T0, n, Mdd, mass offset}; transform to log-scale for positive params; bounds from config priors.
- **Output-error method:** minimise `J(θ) = Σ_s Σ_t ρ( w_s (obs_s(t) − sim_s(t; θ)) / σ_s )²` with soft-L1 loss, windowed re-initialisation (window 20–40 s) to avoid chaotic drift; weights `w_s` ∝ 1/σ_s² of residual noise.
- **Jacobian** by central finite differences with relative step 1e-4; covariance `Σ_θ ≈ s² (JᵀJ)⁻¹`; CIs via Wald; report condition number and correlation matrix; flag |corr| > 0.95 or cond > 1e6 as poorly identifiable; optional profile-likelihood.
- **Excitation check:** report which manoeuvres/regimes provide information (Fisher information per phase) — informs "what test points to fly next".
- **Regime-wise fits:** repeat per Mach/altitude/phase bin, include as a function-of-regime parameter estimate; test heterogeneity (Cochran's Q) → flags missing structure.
- **Global sensitivity:** Morris (r=20) to screen; Sobol (N=1024, Saltelli) on composite score; report S1, ST with bootstrap CIs.
- **Re-validation:** refit on training flights; apply to held-out flights; paired comparison (FR-VAL-005); also test on a *different scenario family* to detect overfitting.

## 11. Recommendation engine logic (FR-IDN-005)

Rule-based synthesis with numeric evidence:
1. For each variant and signal where tolerance compliance < threshold, locate top-ranked condition bins (E3).
2. Map signals/regimes to candidate parameters using the Jacobian (parameters whose sensitivity vector aligns, cosine > 0.6, with the residual signature in that regime).
3. Check fit results: parameter shift significant (CI excludes nominal) and refit improves held-out RMSE ≥ x%.
4. Rank by `impact = ΔRMSE_heldout_normalised × confidence`; emit recommendation with fields: `target_variant, parameter/structure, regime, estimated_change (CI), expected_improvement, evidence = [artifact paths], caveats`.
5. Never emit a recommendation without at least two independent evidence types (e.g., fit + sensitivity, or condition + detector attribution); otherwise label `hypothesis`.
