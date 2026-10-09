# 04 — Simulation & Synthetic Telemetry Specification

Reference vehicle: **generic fixed-wing UAV** (invented parameters, subsonic). Nothing here represents a real or restricted platform.

## 1. Purpose
Provide (a) a **truth model** that plays the role of "real flight", (b) several **variants** (the models under test) with known, deliberate deficiencies, (c) **sensor, timing and fault layers** that turn truth into realistic telemetry, and (d) a **scenario/manoeuvre generator** that creates varied flights across operating conditions.

## 2. Reference vehicle parameters (`configs/vehicles/generic_uav_v1.yaml`)

| Param | Value | Param | Value |
|---|---|---|---|
| Mass m0 | 25 kg (22–28 sampled) | Wing area S | 1.20 m² |
| Wingspan b | 3.2 m (AR ≈ 8.5) | Oswald e | 0.80 |
| CL0 | 0.25 | CLα | 5.0 /rad | 
| α_stall | 0.26 rad | CLmax | ≈ 1.45 |
| CD0 | 0.028 | k = 1/(π e AR) | ≈ 0.047 |
| Mdd (drag divergence) | 0.60 | wave-drag coeff | 20·(M−Mdd)⁴ for M>Mdd |
| Max thrust T0 (SL static) | 180 N | Thrust lapse exponent n | 0.85 (T ∝ (ρ/ρ0)ⁿ) |
| Thrust speed factor | T(M)=T·(1−0.25·M) | Fuel flow (SFC) | 1.5e-5 kg/(N·s) (≈ electric/prop-equivalent abstraction) |
| Max load factor | 3.5 g | Bank limit | 60° |

## 3. Dynamics (point-mass 3-D, wind-relative)

State `s = [x, y, h, V, γ, χ, m]` (V = true airspeed; γ, χ air-relative angles). Wind `w = (wN, wE, wD)`. Controls: throttle δt ∈ [0,1], angle of attack α, bank φ.

```
q   = ½ ρ V²              M = V / a
CL  = CL(α, M)            CD = CD(CL, M)
L   = q S CL              D = q S CD         T = T(δt, h, M)

m V̇        = T cosα − D − m g sinγ
m V γ̇      = (T sinα + L) cosφ − m g cosγ
m V cosγ χ̇ = (T sinα + L) sinφ

ẋ = V cosγ cosχ + wN      ẏ = V cosγ sinχ + wE      ḣ = V sinγ − wD
ṁ = −SFC · T
```
(Wind-rate terms in V̇/γ̇/χ̇ are included in the truth model when shear/turbulence is on: add `−(ẇ-projection)` terms; variants may omit them — a deliberate structural difference.)

Derived measurements:
```
nx = (T cosα − D) / (m g)      nz = (T sinα + L) / (m g)
ground speed = |(V cosγ cosχ + wN, V cosγ sinχ + wE)|
pitch θ ≈ γ + α (small sideslip, coordinated flight)   roll φ = bank   yaw ψ ≈ χ (− sideslip, ignored)
```

**Aerodynamics**
```
CL(α,M) = [CL0 + CLα·α]·PG(M)   for |α| < α_stall ; soft stall roll-off beyond (smooth tanh blend)
PG(M)   = 1/√(1−M²)  (Prandtl–Glauert, truth & V4 only; capped at M=0.85)
CD      = CD0 + k·CL² + CDwave(M) [+ CDfault]
CDwave  = 20·max(0, M − Mdd)⁴
```
**Propulsion:** `T = δt · T0 · (ρ/ρ0)^n · (1 − 0.25·M)`; actuator lag τ_T = 0.4 s in truth.

**Truth-only extras:** α first-order lag with τ_α = 0.15 s (short-period abstraction); elevator/aileron effectiveness scaling; Dryden turbulence; wind shear; ΔT atmosphere offset; mass depletion.

**Numerics:** RK4, dt = 0.01 s (truth), dt = 0.05 s default for variants (configurable; variant integration step intentionally different — numeric-error contribution is measured separately in the test plan).

## 4. Hidden truth perturbations

Truth parameters = vehicle nominal × (1 + δ_i), δ_i ~ N(0, σ_i) drawn **once per study** (fixed across flights) and saved in `truth_params.json`:

| Param | σ | Param | σ |
|---|---|---|---|
| CL0 | 4% | CLα | 5% |
| CD0 | 8% | k | 8% |
| Thrust T0 | 5% | Thrust lapse n | 4% |
| Mdd | 2% | Mass-offset | 1 kg |

Plus per-flight random: wind, turbulence seed, ΔT, mass (sampled in conditions).

## 5. Variants (models under test)

All variants share the same dynamics code with different parameters/switches (`StructuralSwitches`: `compressibility`, `wind_rate_terms`, `alpha_dynamics`, `thrust_lapse`, `isa_offset_known`, `wave_drag`).

| Variant | Description (known deficiency) | Expected fidelity rank (1 = best) |
|---|---|---|
| **V0** Nominal baseline | Vehicle nominal parameters; no compressibility; no wave drag; quasi-steady α; ISA, no wind | 4 |
| **V1** High-drag | V0 with CD0 +20% | 5 |
| **V2** Lift-curve error | V0 with CLα −12%, CL0 −10% | 4–5 (condition-dependent: bad in manoeuvres/high-load) |
| **V3** Atmosphere-naïve | V0 but constant ρ = ρ(h0); no lapse of thrust with altitude | 6 (bad at large altitude change) |
| **V4** Compressibility-aware | V0 + PG correction + wave drag; still no wind | 2 |
| **V5** Near-truth | Truth structure minus α dynamics; parameters = truth × (1 ± 2%); wind **estimated** from airspeed–groundspeed difference | 1 |
| **V6** (optional) Wind-blind but tuned | Parameters tuned to cruise-only data; fails in turns/transients | 3 |

The ground-truth fidelity ordering **V5 < V4 < V6 < V0 ≈ V2 < V1 < V3** (error ascending) is stored in the benchmark config and used to score S2; ties allowed where flagged "condition-dependent".

Each variant also carries a **regime-flaw tag** (e.g., V4: `none`; V0: `M>0.5 drag underestimate`; V2: `n_z>1.8`) used by S3.

## 6. Sensor models (`configs/sensors/default.yaml`)

| Signal | Noise σ | Bias | Other |
|---|---|---|---|
| alt_m (baro) | 1.5 m | slow random walk 0.02 m/√s | lag τ 0.2 s |
| x_m, y_m (GNSS) | 1.2 m | 0 | 1 Hz hold-and-interpolate option; occasional 2 s dropouts |
| vel_air_mps | 0.4 m/s | scale 1.005 | quantised 0.05 |
| vel_gnd_mps | 0.15 m/s | 0 | |
| roll/pitch/yaw | 0.4°/0.4°/1.0° | gyro-drift yaw 0.05°/√s | |
| nx_g, nz_g | 0.02 g | 0.005 g | vibration noise band |
| alpha_rad | 0.3° | 0.2° | optional vane |
| throttle, controls | recorded exactly (+ small quantisation) | | |

**Timing defects:** constant offset U(−0.8, 0.8) s; linear drift U(−50, 50) ppm; jitter σ = 5 ms; random dropouts (p = 0.5%); 0–2 burst gaps (0.5–3 s). Logged in meta JSON.

## 7. Fault library (`configs/faults/*.yaml`)

| Fault | Effect on **truth** (not variants) | Typical magnitude | Ramp |
|---|---|---|---|
| `cd0_step` | CD0 × (1 + m) | 0.10–0.30 | 0–5 s |
| `cla_degrade` | CLα × (1 − m) | 0.05–0.20 | 0–10 s |
| `thrust_loss` | T0 × (1 − m) | 0.10–0.40 | ramp 10–30 s |
| `icing` | CD0↑ + CLmax↓ + CLα↓ | 0.1–0.3 | ramp 30–90 s |
| `ctrl_eff_loss` | α response gain ×(1 − m) | 0.2–0.5 | step |
| `sensor_bias_step` | additive bias on one sensor | 3σ–10σ | step |
| `wind_shear_event` | wind step/gust | 5–15 m/s | 2–5 s |

Each fault logs `type, t_on_s, t_off_s, magnitude, affected_signals`. Probability per flight and type mix configurable; ≥ 1/3 of flights are fault-free (nominal pool for detector calibration).

## 8. Scenarios and manoeuvres

Manoeuvre primitives (parametric, composable):
`trim_cruise(T, V_cmd, h_cmd)` · `climb(rate or γ, h_target)` · `descend(...)` · `coordinated_turn(bank, heading_change)` · `s_turn(bank, period)` · `speed_step(ΔV)` · `pitch_doublet(α_amp, period)` · `loiter(radius)` · `throttle_chop(duration)`.

Controller stack (FR-SIM-011): speed-hold PI on throttle; altitude/γ-hold PID on α_cmd; bank-hold P with rate limit on φ; limiters for load factor.

Scenario templates (`configs/scenarios/*.yaml`):

| Scenario | Content | Duration |
|---|---|---|
| `mixed_mission` | climb → cruise → 2 turns → descent → speed step → doublet | ~15 min |
| `climb_cruise` | climb at 2 γ values then long cruise | ~8 min |
| `turn_series` | 4 turns at varying bank (20/35/45/55°) | ~8 min |
| `doublet_series` | 3 pitch doublets at different V/h | ~5 min |
| `high_speed_dash` | accelerate to M≈0.28–0.35 (stress compressibility/wave-drag region of variants) | ~6 min |

Condition sampling: Sobol sequence over `generate.conditions` ranges → one condition vector per flight; scenario assigned round-robin.

## 9. Generation pipeline (per flight)
1. Sample conditions + seeds.
2. Build control history by running controller stack against **truth** (closed loop) at 100 Hz → `truth` table with controls.
3. Optionally apply fault schedule inside the truth integration.
4. Apply sensor models + timing defects → `telemetry` (10 Hz nominal).
5. For each variant: **replay** recorded controls (from telemetry controls, i.e., after defects) with the initial state from first telemetry samples (smoothed) → full-trajectory outputs; and windowed outputs are computed later in `prep`/`validate` via `run_windowed`.
6. Save meta JSON, defect log, faults log.

## 10. Validation of the simulator itself (tests)
- ISA values vs table.
- Level-trim: with trim throttle/α computed analytically, 60 s stays within 0.1%.
- Ballistic (T=0, CL=0, CD=0): analytic parabola error < 1e-6 relative at dt=0.001.
- Energy: total specific energy changes only via work of T and D within 1e-3 relative.
- RK4 convergence slope ≈ 4 on log-log step refinement.
- Turn: steady coordinated turn radius R = V²/(g tanφ) within 0.5%; load factor 1/cosφ.
- Dryden turbulence: RMS and spectral shape tests.
- Variants V5 vs truth with no sensors: RMSE small (documented bound) → proves pipeline can see "good" models.
