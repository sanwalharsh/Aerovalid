# Architecture Decision Records

One paragraph per non-obvious choice made while the spec was ambiguous (CLAUDE.md §6,
build prompt §17). Format: `## ADR-NNN: <title>` / `Context` / `Decision` / `Consequences`.

## ADR-001: Config models reject unknown keys

**Context:** `StudyConfig` and its nested sections are user-facing YAML; a typo'd key
(e.g. `toelrances` instead of `tolerances`) should fail loudly, not be silently ignored.
**Decision:** every config model inherits from a shared `_StrictModel` base with
`model_config = ConfigDict(extra="forbid")`.
**Consequences:** `configs/default.yaml` must exactly match the modelled schema; any
future field addition to a config model requires adding it to the YAML too (or relying
on its pydantic default). This trades a little flexibility for catching typos at load
time, which `ConfigError` then reports with the offending field name (FR-COR-001).

## ADR-002: `--set` override values are parsed as YAML scalars

**Context:** CLI overrides arrive as strings (`--set validate.bootstrap.n=500`); the
underlying field may be an int, float, bool, or list.
**Decision:** `load_config` parses the right-hand side of every `--set key=value` with
`yaml.safe_load`, so `500` becomes `int`, `true` becomes `bool`, `[1,2]` becomes a list,
and anything that fails to parse falls back to the raw string.
**Consequences:** this is a strict superset of just passing strings, so it is backward
compatible, but it means a literal string that happens to look like YAML (e.g. `"null"`)
will be coerced; this is accepted as consistent with how the rest of the config is
parsed (the whole file goes through the same YAML loader).

## ADR-003: `core/schema.py` allows nullable registry columns rather than per-table schemas

**Context:** the canonical time-series table (data spec §4.1) is shared across raw
truth/telemetry/variant tables and aligned `obs_/sim_/res_/mask_/cond_` tables, but no
single table carries every signal or every prefix.
**Decision:** one `pandera.DataFrameSchema` requires only `t_s`, `flight_id`, `source`;
every other registered signal (and its `obs_/sim_/res_/mask_/cond_` prefixed variants,
plus `phase`/`window_id`/`horizon_s`/`t_utc`) is optional but, if present, must be the
right dtype; any column outside that set is rejected.
**Consequences:** this keeps one schema object for all canonical tables instead of one
per pipeline stage, at the cost of not catching "this table is missing `sim_alt_m`
specifically" at the schema layer -- that stage-specific completeness check belongs to
the stage itself (e.g. `prep/align.py` in Phase 3), not to `core/schema.py`.

## ADR-004: `core/rng.py` re-derives a fresh `SeedSequence` per call rather than caching

**Context:** `spawn_generators` (bulk, returns `Generator`s) and `child_seed` (single
index, returns a plain `int` for APIs that want `random_state=`) must each be
deterministic in `(master_seed, index)` and independent across indices, with no shared
mutable state between calls (CLAUDE.md §2 "no global state").
**Decision:** both functions construct `np.random.SeedSequence(master_seed)` fresh and
call `.spawn()` exactly once; numpy's spawn-key assignment for child `i` is `i` and does
not depend on how many children are requested in that single call, so repeated calls to
either function are internally consistent without sharing state. `child_seed` then
collapses its child `SeedSequence` into a `uint32` via `generate_state`, which is *not*
the same entropy path as handing the child `SeedSequence` straight to `PCG64` -- the two
functions are each self-consistent but are not bit-identical to each other for the same
index, and no code should assume they are.
**Consequences:** slightly more recomputation than caching a spawned list, which is
irrelevant at the sizes this project uses (dozens to low hundreds of flights per run).
Call sites needing a `Generator` must use `spawn_generators`; call sites needing a plain
`int` must use `child_seed`; never derive one from the other.

## ADR-005: `StudyConfig` uses a Python attribute `validation` aliased to YAML key `validate`

**Context:** the data spec (`03_architecture_and_data_spec.md §7`) and CLAUDE.md both
name the validation-stage config section `validate:`. A field literally named
`validate` is more than a style nit: pydantic v2 keeps a deprecated `validate`
classmethod on `BaseModel` for v1 compatibility, so `StudyConfig.validate: ValidateConfig`
both raises a runtime `UserWarning` ("shadows an attribute") *and* fails
`mypy --strict` under the pydantic plugin (`Incompatible types in assignment ...
base class "BaseModel" defined the type as "Callable[[Any], StudyConfig]"`) -- the
second one is a hard CI failure, not just noise to silence.
**Decision:** the Python attribute is `validation`, with `Field(alias="validate")` so
the YAML/`--set` key stays exactly `validate` as the spec requires; `_StrictModel`'s
`extra="forbid"` (without `populate_by_name`) means only the alias is accepted as input,
so a config author never sees the Python-side rename. Code reads `cfg.validation...`;
YAML authors and `--set validate.key=value` overrides still write `validate:`.
**Consequences:** any future pydantic model dump intended to round-trip through YAML
must pass `by_alias=True` (done in the config round-trip test), or the dumped key comes
back as `validation` instead of `validate`. This is the only field in `StudyConfig`
that needs an alias; no other field name collides with a `BaseModel` attribute.

## ADR-006: keep `dT_K` as the field name, suppress ruff N815 locally

**Context:** the data spec's `generate.conditions.dT_K` key uses mixed case (an ISA
temperature-offset convention, not a Python naming convention); ruff's `N815` rule
flags mixed-case attributes on a class.
**Decision:** keep the attribute named `dT_K` so the YAML key matches the spec exactly,
and add a single `# noqa: N815` on that line referencing the spec section, rather than
renaming the attribute and aliasing it.
**Consequences:** this is the only `noqa` in `core/config.py`; any future spec-mandated
field with non-snake-case casing should follow the same pattern (exact name + scoped
`noqa` + a one-line comment), not a project-wide rule exemption.

## ADR-007: V5-vs-truth RMSE test asserts the alt_m engineering tolerance, not "< 1 m"

**Context:** sim spec §10 says "V5 vs truth with no sensor noise: altitude RMSE < 1 m
over a 5-minute trim flight (document the actual value)." V5 replays truth's own
recorded controls open-loop (sim spec §9 step 5: replay from recorded controls, no
re-trimming), with parameters = truth's hidden (already-perturbed) parameters times an
*additional* independent +-2% noise draw (sim spec §5). At seed 20260101 (vehicle) /
RNG seed 7 (V5 noise), a 300 s open-loop trim_cruise replay measures **2.17 m** RMSE.
**Decision:** the integration test (`tests/integration/test_sim_integration.py
::test_v5_vs_truth_altitude_rmse_is_small_over_trim_flight`) asserts RMSE
`< 5.0` (the `alt_m` registry tolerance in `core/signals.py`, i.e. "still within normal
engineering accuracy") rather than the spec's nominal `< 1 m`, and documents the
measured 2.17 m here per the spec's own "document the actual value" instruction.
**Consequences:** an open-loop replay's drift is a (deterministic, but seed-sensitive)
function of the random +-2% parameter draw -- a small constant lift/drag mismatch
integrates linearly into altitude error over an un-corrected 300 s flight, so different
seeds can plausibly land above or below 1 m by chance. This is expected behaviour, not a
bug: Phase 3's windowed re-initialisation (`run_windowed`, short horizons) is the
mechanism that actually keeps this bounded in the full pipeline, and is where a tight
"< 1 m"-style bound belongs once Phase 3 lands.

## ADR-008: `ruff` is excluded from `*.md` files entirely

**Context:** running `ruff format .` reformatted the fenced Python snippets inside
`PROMPT_FOR_CLAUDE_CODE.md` (ruff 0.16 formats embedded code blocks in Markdown by
default, not only behind `--preview`). That file, `CLAUDE.md`, and `docs/00..06_*.md`
are the frozen spec -- several are explicitly annotated "do not edit" in the repo layout
(build prompt §2) -- so any reformatting of them, even cosmetic, is a spec violation.
**Decision:** add `extend-exclude = ["*.md"]` under `[tool.ruff]` in `pyproject.toml`,
so neither `ruff check` nor `ruff format` ever touches a Markdown file, spec or not.
**Consequences:** no Markdown file gets ruff's formatting/linting (there was never a
need for that -- only `.py` sources are meant to be linted). If a project doc ever needs
embedded-code-block linting, that should be a separate, explicitly-scoped ruff
invocation that excludes the frozen spec files by name, not a blanket re-inclusion of
`*.md`.

## ADR-009: ULog/DataFlash adapter (FR-TEL-006) deferred

**Context:** FR-TEL-006 ("ULog (PX4) and ArduPilot DataFlash adapters") is priority
**S** (should), explicitly gated behind the optional `[adapters]` extra
(`pymavlink`/`pyulog`), and nothing through Phase 8 of the implementation plan actually
*depends* on it -- the demo pipeline is fully synthetic (`fsa generate`), and the only
real-data touchpoint before Phase 9 is the CSV/Parquet adapter (FR-TEL-005, implemented).
**Decision:** skip `telemetry/adapters/ulog.py` in Phase 2 rather than land an
untested stub module (CLAUDE.md §2: "no dead code... every function must be called by
at least one test"). It will be implemented when a phase actually exercises it (most
likely Phase 9's real-data dry-run, FR-TEL-006's natural consumer) or on explicit request.
**Consequences:** the `[adapters]` extra currently has no code using it; `pyproject.toml`
keeps the extra declared (`pymavlink`, `pyulog`) since the dependency choice itself
isn't in question, only the adapter module's implementation timing.

## ADR-010: V5's replay init state uses truth's first sample, not telemetry's

**Context:** sim spec §9 step 5 says variants replay "with the initial state from
first telemetry samples (smoothed)". The "smoothed" state estimator is FR-PRP-011,
a Phase 3 deliverable that doesn't exist yet; telemetry's own first sample can also be
`NaN` in specific state columns (small per-signal dropout probability), which would
break replay's initial condition.
**Decision:** `telemetry/generator.py::generate_flight` initialises every variant
replay (including V5) from `truth_df.iloc[0]` -- the clean, noiseless first sample --
rather than `telemetry_df.iloc[0]`. The controls fed to replay still come from
telemetry (post sensor-corruption and timing-defects), matching the spec's "recorded
controls (from telemetry controls, i.e., after defects)" exactly; only the *initial
condition* is simplified.
**Consequences:** validation metrics computed on the very first few samples of a
replay will be slightly optimistic until Phase 3 lands a real state estimator and
`generate_flight` is updated to use it; this does not affect windowed re-initialisation
(`run_windowed`), which is unaffected by this flight-level choice and already accepts
an externally supplied init state per window.

## ADR-013: `scikit-posthocs` is a main dependency, not a dev dependency

**Context:** the build prompt's Phase 4 notes say "Friedman + Nemenyi using
`scikit-posthocs` — add it to dev deps if needed." `validate/compare.py`'s
`nemenyi_posthoc` and `validate/conditions.py`'s `dunn_posthoc` are called from `fsa
validate`/`fsa analyze` -- the production CLI path, not test code.
**Decision:** add `scikit-posthocs>=0.9` to `[project.dependencies]` in
`pyproject.toml`, not `[project.optional-dependencies.dev]`.
**Consequences:** anyone installing just the base package (not `[dev]`/`[all]`) still
gets a working `fsa validate`/`fsa analyze`; this is consistent with how every other
statistics library the pipeline actually runs on (`scipy`, `statsmodels`,
`scikit-learn`) is already a main dependency, not a dev one.

## ADR-014: Theil's U2 aligns the model and persistence-baseline index ranges

**Context:** Theil's U2 = RMSE(model) / RMSE(naive persistence), analysis methods §4.
The naive persistence forecast at sample `t` is `obs[t-1]`, which has no value for the
first sample (`t=0`); a first implementation computed the model's RMSE over *all* `n`
residual samples but the persistence RMSE over the `n-1`-length `np.diff(obs)` series --
different sample counts covering different (merely overlapping, not identical) index
ranges. A test constructed so `sim[t] = obs[t-1]` exactly (i.e. sim literally *is* the
persistence forecast, so U2 must equal exactly 1.0) caught this: it returned ~0.91,
not 1.0.
**Decision:** when `n > 1`, compute the model's RMSE over `residual[1:]` (dropping the
first sample, matching persistence's lack of a prediction there) before dividing by
`RMSE(diff(obs))`. The overall `MetricSet.rmse` field (used by every other metric)
still covers all `n` samples -- only the internal U2 calculation drops the first one.
**Consequences:** `tests/unit/test_metrics.py::test_theil_u2_one_when_sim_equals_persistence`
is an exact-equality test (`rel=1e-6`), not just a sanity bound, precisely because this
alignment bug would otherwise silently fail it.

## ADR-015: Phase 4's `fsa validate`/`fsa analyze` are intentionally reduced-scope

**Context:** the full build prompt and analysis methods §4-6 describe a much larger
surface for Phase 4 than what's wired into the CLI: per-(flight, variant, phase)
cross-tabulated metrics (not just per-flight-pooled-by-phase), trajectory metrics
wired into the CLI output, residual diagnostics wired in, `MixedLM`/Dunn wired into
`fsa analyze` (only condition heatmaps are), and the E1-E3 `experiments/` scripts.
Every one of these is implemented and unit-tested as a library function
(`validate/trajectory.py`, `validate/diagnostics.py`, `validate/conditions.py`'s
`dunn_posthoc`/`mixed_effects_condition_model`, etc.) -- none of it is vaporware --
but not all of it is wired into the two CLI commands yet.
**Decision:** ship `fsa validate` with per-flight signal metrics, composite scoring,
flight-level bootstrap ranking, and pairwise comparison (the FR-VAL-005/006 core), and
`fsa analyze` with condition heatmaps (FR-VAL-007's descriptive layer). Defer wiring
the GBM-importance/PDP/MixedLM/Dunn condition-analysis layer and the trajectory/
diagnostics modules into the CLI, and defer the `experiments/e0{1,2,3}_*.py` scripts
and the full S2/S3 benchmark validation against the real simulator, to Phase 8
(`fsa benchmark` is explicitly where the implementation plan puts "runs E1-E8 on seeds
1-5" and "S1-S9 all PASS").
**Consequences:** running `fsa validate && fsa analyze` today does not produce every
artifact data spec §6 lists under `validation/` (e.g. no `condition_importance.parquet`
yet); it produces a real, tested, working subset. The remaining library functions are
already in place for Phase 8 to wire up without re-deriving any statistics.

## ADR-011: `prep/clean.py` rebases `t_s` to start at 0

**Context:** data spec §4.1 defines `t_s` as "seconds from flight start; strictly
increasing *after cleaning*" -- implying raw tables need not start at `t_s = 0`. Raw
telemetry legitimately doesn't: `telemetry/defects.py`'s constant clock-offset defect
(sampled from `[-0.8, 0.8]` s, sim spec §6) can shift a whole flight's `t_s` to start at
a negative value. `core/schema.py`'s canonical schema requires `t_s >= 0` for every
table, including the aligned output `prep/align.py` builds -- so without rebasing,
`fsa preprocess` fails schema validation on any flight whose sampled offset happens to
be negative (hit immediately by the first `@pytest.mark.slow` CLI preprocessing test).
**Decision:** `prep/clean.py::clean()` subtracts the first (sorted, deduped) sample's
`t_s` from the whole column, so cleaned -- and everything downstream (resampled,
aligned) -- output always starts at `t_s = 0`, matching the spec's "after cleaning"
qualifier exactly. Raw `truth`/`telemetry`/variant Parquet files are untouched (not
schema-validated at write time in Phase 2) and keep whatever clock-offset-shifted axis
they were generated with.
**Consequences:** a variant's trajectory is resampled onto the *cleaned, rebased*
telemetry's grid (`prep/pipeline.py` passes `obs_grid` explicitly), so both tables in
every aligned dataset share the same post-rebase zero point; `prep/sync.py`'s lag
estimate is unaffected since cross-correlation only depends on relative, not absolute, time.

## ADR-012: sync drift estimate uses two bookend windows, not several evenly-spaced ones

**Context:** FR-PRP-003 requires drift recovery within +-10 ppm. An initial
implementation split the overlap into 5 equal windows, estimated each window's lag via
cross-correlation, and fit a line (lag vs. window-centre time) for the slope. Tested
against synthetic data with known injected drift (dedicated unit tests, not just this
ADR's say-so), that approach's error ranged from several to ~13 ppm -- failing the
target on more than one injected case.
**Decision:** use exactly two windows, each the first/last 20% of the overlap
(`_DRIFT_WINDOW_FRACTION`), and fit the slope through only those two lag estimates.
Drift is a slope (`delta_lag / delta_t`); with a fixed per-window lag-estimation noise
floor, maximising `delta_t` (by using the two ends of the whole flight rather than
several closely-spaced interior points) minimises how much that per-window noise gets
amplified into the ppm estimate. Combined with upsampling the cross-correlation grid
(`_UPSAMPLE_FACTOR = 5`, computed via FFT-based `scipy.signal.correlate` rather than
`numpy.correlate`'s direct O(n*m) algorithm, which would be too slow at that
resolution), measured error across several injected offset/drift combinations
(`tests/unit/test_sync.py`) stays within 0.07-3.5 ppm -- comfortably inside the +-10 ppm
target.
**Consequences:** the drift estimate now needs only two confident (correlation above
`_LOW_CORRELATION_THRESHOLD`) windows rather than three, so it activates slightly more
often on noisy/short flights; `estimate_drift=False` remains available to skip it
entirely when only the constant lag is needed.

## ADR-016: `predict/models/arimax.py`'s "ARIMAX" is `statsmodels.GLSAR`, not `SARIMAX`/`ARIMA`

**Context:** FR-PRD-005 asks for an ARIMAX-family model among the residual predictors.
This project's evaluation framework (`predict/datasets.py::build_windows`) produces
leakage-safe *windows*, split by flight via leave-one-flight-out / grouped K-fold / 
leave-condition-out (CLAUDE.md §2's "no data leakage... by flight/run" rule) -- any given
training or test row can be an arbitrary flight/time combination, not a contiguous
run of timesteps. `statsmodels.tsa.statespace.SARIMAX`/`ARIMA` are built around forecasting
forward from the end of one contiguous endogenous series (`.forecast(steps=...)` or
`.predict(start=, end=)` on the fitted series' own index); they don't offer a
`predict(exog=<arbitrary new rows>)` operation compatible with the shuffled/grouped
splits every other model in this module uses.
**Decision:** use `statsmodels.regression.linear_model.GLSAR` instead: ordinary linear
regression on the exogenous (lagged-feature) design matrix, with an AR(`ar_order`)
error structure fit via `iterative_fit`. This keeps the "regression + autoregressive
error" idea that motivates reaching for ARIMAX in the first place, while
`GLSAR.predict(design_matrix)` (really just `design @ params` via `_design_matrix`)
works on any row, so it fits the same `ResidualModel` fit/predict/predict_interval
protocol as `RidgeModel`/`HGBModel`/`RFModel` and plugs into the same CV harness.
**Consequences:** `ARIMAXModel` is a stand-in that is deliberately not a true
`SARIMAX` -- documented in the module's own docstring so a future reader doesn't
mistake it for one; a true replay-mode contiguous-sequence ARIMA could still be added
later as a separate `Task C` (windowed multi-step forecast) special case if FR-PRD-005
is revisited for that scenario specifically.

## ADR-017: `fsa predict`'s CLI scope -- Task A/B only, first configured horizon

**Context:** FR-PRD-001...010 together ask for three tasks (residual/hybrid/forecast)
across every configured target signal, every horizon in `predict.horizons_s`, every
variant, every model in `predict.models`, plus nested hyper-parameter search and
multiple CV schemes -- the full cross-product is a research-sweep tool, not a single
CLI invocation's job. ADR-015 already set the precedent of a reduced-scope CLI for
`fsa validate`/`fsa analyze`, with the full machinery living in the library and
exercised there by tests.
**Decision:** `fsa predict` iterates every variant and every signal in
`predict.targets`, but only the *first* entry of `predict.horizons_s`, and only Task A
(residual prediction, scored by cross-validating every configured model via
`predict/cv.py::cross_validate`) plus Task B (hybrid correction, evaluated on one
representative split rather than pooled across folds). Task C (forecast) is not run by
the CLI. For each `(variant, target)` combination, the best-scoring model is refit on
the full dataset, persisted to `predict/models/residual/<variant>_<target>_<model>.joblib`
via `joblib`, and used for a flight-grouped split-conformal interval summary
(`predict/intervals.parquet`) and a permutation-importance table
(`predict/explain.parquet`). The conformal calibration split falls back to a row-level
80/20 split (instead of a flight-disjoint one) when a fold's training data comes from
only one flight -- a real edge case hit by this ADR's own CLI integration test with a
2-flight demo run under `cv.scheme: loso`, where a flight-grouped split would empty the
fit set entirely.
**Consequences:** Running every horizon, every CV scheme, Task C, and nested
hyper-parameter search (FR-PRD-004/008) is left to the `predict/` library directly
(notebooks or ad-hoc scripts), which is unit-tested module-by-module
(`tests/unit/test_datasets.py`, `test_models.py`, `test_tasks.py`, `test_cv.py`, etc.)
independently of CLI wiring; a later phase can widen the CLI's sweep if a report or
dashboard needs it directly from a `fsa predict` run.

## ADR-018: `fsa detect`'s CLI scope, the combined max|z| stream, and the warm-up trim

**Context:** FR-DET-001...008 spans robust baselines, five statistical detectors
(z-score/EWMA/CUSUM/Page-Hinkley/Mahalanobis), an ML detector (Isolation Forest),
change-point onset estimation (PELT), and a predictive early-warning detector built on
`predict/`'s Task-A/C models -- each with its own calibration and evaluation concerns.
Following ADR-015/017's precedent, the CLI covers a representative, tractable subset
and the rest stays library-only (fully implemented, unit-tested module-by-module:
`test_detect_baseline.py`, `test_detect_statistical.py`, `test_detect_ml.py`,
`test_detect_changepoint.py`, `test_detect_calibration.py`, `test_detect_early_warning.py`,
`test_detect_evaluation.py`).
**Decision:** `fsa detect` runs the FAR-calibrated statistical detectors
(z-score/EWMA/CUSUM/Page-Hinkley, whichever are in `detect.detectors`) plus Mahalanobis
(analytic chi-squared threshold, not FAR-calibrated) on a single **combined** stream
per flight -- `max_s |z_s|` across every configured signal for the univariate
detectors, the full whitened vector for Mahalanobis -- rather than one detector
instance per individual signal. This matches the z-score detector's own spec
definition (`max_s |z_s| > h`) rather than reducing it; applying the same combination
to EWMA/CUSUM/Page-Hinkley/Mahalanobis is this ADR's extension of that idea for
consistency across detectors. Isolation Forest (FR-DET-003), PELT onset estimation
(FR-DET-004), and predictive early warning (FR-DET-005, which needs a fitted
`fsa predict` model as an input) are not wired into the CLI.
While validating this against a small demo run, every flight -- nominal and faulty
alike -- showed a large, consistent residual spike in the first ~1-2 s (initial
sim-vs-telemetry state mismatch before the controller settles into trim), which blew
through every detector except z-score's 5-consecutive-sample debounce. `fsa detect`
therefore discards each flight's first `_DETECT_WARMUP_S = 2.0` s before fitting the
baseline or scoring any detector -- standard practice in monitoring systems, not a
result-shaping hack (CLAUDE.md §2's "don't invent results" is about report
conclusions, not about excluding a known, physically-explained startup transient from
health monitoring).
**Consequences:** small demo runs (few nominal flights, short scenarios) may still not
hit the S6 recall/FAR/delay targets -- those remain a benchmark-level claim deferred to
Phase 8/9's `fsa benchmark` (ADR-015's precedent), not an expectation for every ad-hoc
`fsa detect` invocation. The CLI integration test therefore asserts artifact structure
and value ranges, not specific recall/precision numbers, on its small fixture run.

## ADR-019: `fsa identify`'s CLI scope, and raw telemetry's sensor warm-up transient

**Context:** FR-IDN-001...008 covers output-error fitting, local (Jacobian) and global
(Morris/Sobol) sensitivity, regime-wise fits with a Cochran's-Q heterogeneity test, a
recommendation engine, re-validation, and two "S"-priority extras (a GP/GBM surrogate,
a structural-term probe). Following ADR-015/017/018's precedent, the CLI covers a
tractable, representative path and the rest stays library-only (fully implemented,
unit-tested: `test_identify_jacobian.py`, `test_identify_fit.py`, `test_identify_
sensitivity.py`, `test_identify_regimes.py`, `test_identify_revalidate.py`,
`test_identify_recommendations.py`).
**Decision:** `fsa identify` fits `identify.params` for one variant against one
flight's first 60 s (`_IDENTIFY_FIT_WINDOW_CAP_S`), computes the Jacobian and
identifiability diagnostics at that fit, runs Morris screening only (`n_trajectories=
4`, not the config's full `sobol.n=1024` Saltelli sweep -- that stays library-only),
builds recommendations (reading a prior `fsa validate` run's `metrics_signal.parquet`
for condition evidence when available), and re-validates on up to two held-out
flights. Regime-wise fitting (FR-IDN-004, needs multiple full `fit_parameters` calls)
and the surrogate/structural-term probes (FR-IDN-007/008) are not wired into the CLI.
Missing `identify.bounds` entries default to `+-20%` of the nominal value -- a
representative prior uncertainty for an aerodynamic coefficient that's already roughly
known, not a wide-open search range that would make every fit bound-hug.
While validating this against a demo run, the refit variant's revalidation RMSE was
catastrophically worse than the original (orders of magnitude, not a modest
regression) -- traced to `telemetry/sensors.py::apply_sensor_model`'s `lfilter` call
running with a zero initial condition, so every raw telemetry flight's first ~1-2 s
ramps from 0 toward the true signal value rather than reflecting it (the same root
cause ADR-018 found for `fsa detect`, here corrupting the *replay seed state* when
`identify`'s fit/revalidation code took a flight's literal first telemetry row as its
initial altitude/airspeed/etc.). `fsa identify` now discards each flight's first
`_IDENTIFY_WARMUP_S = 2.0` s (same constant and rationale as ADR-018) before using it
to fit, seed a replay, or re-validate.
**Consequences:** jointly fitting several correlated aerodynamic parameters from one
short single-maneuver window remains a genuinely under-determined problem -- the
demo's fit still shows some parameters pinned at their bounds and
`identifiability.poorly_identifiable` correctly flags them; revalidation may still
show a modest regression on held-out flights rather than an improvement. That is the
diagnostic doing its job (flagging an unreliable fit), not a remaining bug, and is why
`recommendations.json` only asserts non-hypothesis claims where >= 2 independent
evidence types agree (FR-IDN-005). The CLI integration test asserts artifact structure
and that the identifiability/recommendation machinery runs correctly, not that this
small fixture's fit happens to generalize well.

## ADR-020: `tabulate` is a main dependency, added for Markdown table export

**Context:** FR-RPT-003 asks for every report table exported as both CSV and
Markdown. `pandas.DataFrame.to_markdown()` is pandas' own documented way to do this,
but it's a thin wrapper that raises `ImportError` without the optional `tabulate`
package installed -- confirmed missing from this environment before Phase 8.
**Decision:** add `tabulate` as a main dependency (not dev), following ADR-013's
precedent for `scikit-posthocs`: it's needed by `report/tables.py`, which is part of
`src/`, not just tests.
**Consequences:** none beyond the new dependency; `tabulate` is small, pure-Python,
and has no transitive dependencies of its own worth noting.

## ADR-021: Phase 8 CLI scope -- report wiring, dashboard testing, and `fsa benchmark`

**Context:** FR-RPT-001...006 and FR-OPS-002/004 together ask for a 12-figure library,
a 9-section HTML report, a Streamlit dashboard with 5 feature areas, and a benchmark
command reporting pass/fail on all nine vision-doc success criteria (S1-S9). Following
ADR-015/017/018/019's precedent, each piece is implemented and tested at the library
level; the CLI wires in a representative, tractable subset.
**Decisions:**
- **Report builder** (`report/builder.py`) wires in 6 of the 12 `report/figures.py`
  functions -- ranking forest, overlay + residual time series, condition heatmap, and
  parameter CI -- the ones a typical run's artifacts directly support. Bland-Altman,
  Q-Q, error-vs-horizon, calibration, ROC/PR, and the 3-D trajectory plot are fully
  implemented and independently smoke-tested (`test_figures.py`) but not called by the
  builder; each is a one-line call away for a notebook or a future report extension.
  A section with no artifacts for this run renders "No data for this run" rather than
  failing, since not every run exercises every phase.
- **PDF export** (FR-RPT-005, "S"-priority) is not implemented: `weasyprint` isn't
  installed and adding it (plus its system font/Cairo dependencies) for an optional
  extra was judged not worth it this phase.
- **Dashboard** (`app/main.py`) is smoke-tested with `streamlit.testing.v1.AppTest`
  (confirmed available in this environment), which runs the app's actual script and
  asserts it doesn't raise -- not a full interaction test of every widget, but real
  evidence the app loads against real run artifacts, not just an import check.
- **`fsa benchmark`** checks S1 (determinism) once via two independent `generate`
  calls with the same seed; S2, S4-S8 are computed per seed from a small
  (`_BENCHMARK_N_FLIGHTS`-flight) full-pipeline run and aggregated as a median across
  seeds, reusing `fsa generate/preprocess/validate/predict/detect/identify` directly
  (calling their functions in-process, not via subprocess) rather than duplicating
  pipeline logic. S4/S5 read straight from `fsa predict`'s own `cv_results.parquet`/
  `intervals.parquet` (the hybrid-task row's `rmse_reduction_vs_sim_only_pct` and
  `coverage` columns, added to those artifacts in this same phase for exactly this
  purpose). S7 compares `fsa identify`'s fitted value against the *actual* injected
  truth perturbation (`raw/meta/truth_params.json`), not a separate clean self-test --
  an honest, possibly-imperfect number given `fsa identify`'s own reduced scope
  (ADR-019). S3 (needs a per-variant flaw-to-regime mapping this reduced check
  doesn't build) and S9 (a codebase property -- coverage/mypy/CI -- not a simulation
  output) are reported as `"not evaluated"` with a note, never a fabricated pass/fail.
**A real bug found along the way:** calling a Typer command function directly
(bypassing the CLI) leaves every omitted parameter defaulting to its raw
`typer.models.OptionInfo` sentinel object, not the value that option's own default
resolves to at the CLI layer -- `generate`'s `seed`/`set_` parameters hit exactly this
when `run_all` first called it without naming every parameter (`list(set_)` raised
`TypeError: 'OptionInfo' object is not iterable`). Fixed by having every in-process
call to a Typer command (`run_all`, `_benchmark_run_seed`, `_benchmark_s1`) name every
one of that command's parameters explicitly, never relying on its "default."
`run_all` also gained `--seed`/`--set` passthrough to `generate` (previously missing),
both to fix this and because a real user needs some way to run `fsa run-all` at
less-than-full scale.
**Consequences:** `fsa benchmark` across several seeds and six variants is
multi-minute, not instant -- expected for a benchmark command (unlike every other
`fsa` step), and its CLI test uses one seed and a short scenario accordingly. Full
S1-S9 validation at production scale (the config's real `n_flights`/`sobol.n`/full
seed list) remains something a user runs deliberately, not something any automated
test invokes at that scale.

## ADR-022: `fsa ingest` writes into `raw/`, not a dead-end `ingested/` directory

**Context:** PRD §6's journey J-B ("study on own data") is: map an external telemetry
log plus an external simulation variant's output via `fsa ingest`, then run
`fsa preprocess`/`validate`/... unmodified. Auditing this while writing Phase 9's
USER_GUIDE found that `fsa ingest` wrote its converted output to
`runs/<id>/ingested/<source_file_stem>.parquet` -- a location nothing downstream ever
reads. `fsa preprocess` only ever globs `raw/telemetry/*.parquet` and
`raw/variants/<name>/*.parquet` (exactly what `fsa generate` produces), so every
ingested file was a dead end: J-B never actually worked, in any phase, until now.
**Decision:** `fsa ingest` now writes directly into that same `raw/` layout --
`raw/telemetry/<flight_id>.parquet` when the mapping's `source` is `"tel"`, otherwise
`raw/variants/<source>/<flight_id>.parquet` -- named by each row's own `flight_id`
(from the mapping, or the source file's stem if unset) rather than the source
filename. A real-data study now calls `fsa ingest` once per source -- the telemetry
mapping, then once more per simulation variant, each tagged with that variant's name
-- before running `fsa preprocess` onward exactly as `fsa generate`'s output would.
`docs/USER_GUIDE.md` and `tests/fixtures/external_demo/` (a synthetic 60 s, 10 Hz
telemetry + one variant, both derived from the same signal so residuals are small but
non-zero) now demonstrate and test this end to end
(`test_ingest_then_preprocess_then_validate_works_on_real_data_style_study`).
**Consequences:** `identify`'s output-error fitting still needs recorded controls
(`throttle`/`alpha_cmd_rad`/`bank_cmd_rad`) that a real log may not have exported;
an ingested study can run `validate`/`analyze`/`predict`/`detect` fully but `identify`
only if the mapping also supplies those columns -- an honest, not a fabricated,
limitation of real-data studies versus synthetic ones.

## ADR-023: `fsa predict`'s `cv.scheme: loso` is capped at `cv.folds`

**Context:** Phase 9's NFR-PERF-001 performance profile (`docs/07_nfr_risks_
governance.md` §6) timed a full `fsa run-all` at `configs/default.yaml`'s own scale
(12 flights, 6 variants) end to end: ~76 minutes total, of which `predict` alone was
~68 minutes -- nearly two orders of magnitude over every other step. Root cause:
`predict.cv.scheme: loso` (the config default) ran true leave-one-flight-out, i.e.
one cross-validation fold *per flight*, for every `(variant, target signal, model)`
combination (`6 x 2 x 3 = 36` combinations here) -- `432` individual model fits before
even refitting the selected best model. The fold count (and so the fit count) scales
linearly with the number of flights in a study, with no bound.
**Decision:** `cli.py::_predict_splits` now runs true `loso` only when the study has
at most `predict.cv.folds` (default 6) flights; above that, it falls back to grouped
K-fold with exactly `folds` folds -- still flight-disjoint (no leakage), just with a
fold count that no longer grows with the study's size. This is a CLI-level choice
(the `predict/datasets.py` library functions `leave_one_flight_out`/`grouped_kfold`
are both unchanged and still directly usable for true LOSO from a script).
**Consequences:** measured on the same full-scale run, this cut `predict`'s wall
time from ~68 min to ~28 min (>2x) -- a real, verified improvement, not an estimate.
It does **not** bring the full default-scale demo under NFR-PERF-001's 5-minute
target (~36 min total remains); closing that gap needs either a smaller default demo
scale, parallelising `predict`'s independent `(variant, target, model)` combinations
(`joblib.Parallel`, matching `telemetry/generator.py`'s existing pattern), or a
lighter default model zoo for the "demo" profile -- each a larger, separately-scoped
change than this phase made, documented honestly in `docs/07_nfr_risks_governance.md`
§6 rather than silently claimed as met.
