# 07 — Non-Functional Requirements, Risks & Governance

## 1. Non-functional requirements (NFRs)

Restated from `01_PRD.md §8` as testable targets.

| ID | NFR | Target | Verified by |
|---|---|---|---|
| NFR-PERF-001 | Demo pipeline latency | `make demo` completes in < 5 min on an 8-core laptop | `fsa benchmark` S1, CI e2e job |
| NFR-PERF-002 | Batch preprocessing throughput | 100 flights × 10 Hz × 1 h processed in < 30 min on 8-core laptop | Phase 9 performance profile (measured time documented) |
| NFR-REPRO-001 | Reproducibility | Same seed + config + platform → bit-identical Parquet outputs, excluding manifest timestamps | FR-COR-006 tests; `make demo` run twice |
| NFR-USE-001 | Usability | Every pipeline stage runs as a single `fsa <step>` command; config/validation errors name the offending field | `ConfigError` tests |
| NFR-PORT-001 | Portability | Runs on Linux, macOS, Windows under Python ≥ 3.11 | CI smoke matrix (`.github/workflows/ci.yml`) |
| NFR-QUAL-001 | Code quality | ≥ 85% test coverage; `mypy --strict` clean on `src/` | `make lint test`, CI |
| NFR-SEC-001 | Security / governance | No sensitive, proprietary, export-controlled, or classified data/parameters in the repo | This document; PR review checklist §4 |

## 2. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Synthetic data is too easy or unrepresentative, inflating validation/prediction scores | Medium | High | Variant fidelity spread V0–V5 with deliberately injected, documented flaws (§5 of the build prompt); `fsa benchmark` compares against known ground-truth ordering, not just internal consistency |
| R2 | ML models overfit on a small number of synthetic flights | Medium | Medium | Grouped CV by `flight_id` only (never by row); held-out calibration flights for conformal intervals; leakage tests enforced in CI (`tests/*/test_*leak*`) |
| R3 | Time-alignment (sync) fails silently and corrupts downstream metrics | Low | High | Sync quality gated (`peak < 0.6` → flagged, zero-lag fallback with warning, data spec §8); sync estimates persisted to `sync/sync_estimates.parquet` for audit |
| R4 | Residual/detector results are over-interpreted as certified V&V evidence | Medium | High | Report language and `docs/00_vision.md` explicitly scope this as a research/screening tool, not a certification artifact (non-goal NG at `01_PRD.md`) |
| R5 | A contributor accidentally adds real/classified vehicle data or weapon-guidance logic | Low | Critical | Hard rule in `CLAUDE.md §2`; reference vehicle is fixed as `generic_uav_v1` with invented parameters (build prompt §4); this document's governance checklist is part of PR review |
| R6 | Heavy optional dependencies (`torch`, `lightgbm`) break installs on contributors' machines | Medium | Low | Guarded imports (`try/except ImportError`) everywhere these are used; core pipeline never requires `[ml]` extra |
| R7 | Non-determinism creeps in via unseeded randomness (dict ordering across Python versions, unset RNG in a new module) | Medium | Medium | `core/rng.py` is the only sanctioned source of randomness; no module-level mutable RNG state (CLAUDE.md §2); reproducibility test in `tests/unit/test_rng.py` |

## 3. Data governance

- **Classification:** unclassified only, by construction. The reference vehicle (`configs/vehicles/generic_uav_v1.yaml`) is invented; see build prompt §4. No real platform's parameters are ever entered, even as a "close enough" stand-in.
- **External datasets** (D1–D8 in the build prompt §10) are referenced by URL only. None are bundled in the repository. Adapters validate and convert external data into the canonical schema (`core/schema.py`); they never assume column names match without an explicit mapping YAML.
- **Datasets requiring registration** (e.g. D1 NASA DASHlink) are documented here with their access terms; the data itself is never committed. Contributors fetch it locally for `experiments/` or `tests/fixtures/` (tiny derived fixtures only, never raw bulk data).
- **Licences:** D1, D3 — public domain / NASA public. D2 — community-uploaded, publicly accessible. D7 — CC BY 4.0 (OpenSky). D8 — published dataset with DOI (Agarwal et al. 2024, CC-licensed per source repository). Any dataset whose licence is unclear at time of use must be noted in `docs/DECISIONS.md` before use.

## 4. PR / review governance checklist

Every pull request into `main` must confirm:

- [ ] No real, proprietary, export-controlled, or classified vehicle parameters, weapon specs, or guidance logic were added.
- [ ] No bulk external dataset files were committed (`data/`, `runs/`, `reports/` remain git-ignored).
- [ ] Every new public function has a docstring with units/shapes and is covered by at least one test.
- [ ] Any new third-party dependency is declared in `pyproject.toml` and justified in `docs/DECISIONS.md`.
- [ ] `make lint test` is green locally and in CI.
- [ ] Any ambiguous requirement resolved during the change has a corresponding ADR in `docs/DECISIONS.md`.

## 5. Out of scope (non-goals, restated from `01_PRD.md`)

Real-time streaming; certified V&V claims; CFD; full nonlinear 6-DoF truth modelling beyond the documented truth model; multi-vehicle swarms; any classified/export-controlled data or weapon modelling (`01_PRD.md §`, `00_vision.md` NG3).

## 6. Phase 9 performance profile (measured, not estimated)

A full `fsa run-all` at `configs/default.yaml`'s own scale (12 flights, 4 scenario
types, 6 variants) was timed end to end on this development machine. **NFR-PERF-001
(< 5 min) is not met at this scale** -- the measured total was roughly **76 minutes**
before the fix described below, and roughly **36 minutes** after it. Per-step
breakdown (from `run_manifest.json`/log timestamps of the actual run):

| Step | Time (before) | Time (after `_predict_splits` fold cap) |
|---|---|---|
| `generate` | ~6 min 18 s | unchanged |
| `preprocess` | 80 s (12 flights) | unchanged |
| `validate` | 4 s | unchanged |
| `analyze` | 2 s | unchanged |
| `predict` | **68 min 4 s** | **28 min 11 s** |
| `detect` | 26 s | unchanged |
| `identify` | 29 s | unchanged |
| `report` | crashed (see below), ~2 s once fixed | ~2 s |
| **Total** | **~76 min** | **~36 min** |

`predict` dominates by nearly two orders of magnitude over every other step, for a
structural reason, not a one-off slowdown: with `predict.cv.scheme: loso` (the
config default), every `(variant, target signal, model)` combination -- `6 variants x
2 targets x 3 models = 36` combinations at this scale -- was cross-validated with
one fold *per flight* (12 folds), i.e. `432` individual model fits, before the model
selected as best was even refit and scored. The fold count scaled linearly with the
number of flights in the study, so it will only get worse as a study grows. Capping
`loso` at `predict.cv.folds` (default 6) folds -- falling back to grouped K-fold,
still flight-disjoint, once the flight count exceeds that -- cut the fold count (and
so the fit count) by 2x at this scale, for the measured >2x wall-clock improvement
above; see `docs/DECISIONS.md` ADR-023.

**This does not bring the full default-scale demo under the 5-minute target.**
Reaching that would need either (a) a smaller default "demo" scale than
`configs/default.yaml`'s current `generate.n_flights: 12` x `variants: 6`, (b)
parallelising `predict`'s per-`(variant, target, model)` loop (each combination is
independent -- the same `joblib.Parallel` pattern `telemetry/generator.py` already
uses across flights would apply directly), or (c) reducing the default model zoo
(e.g. dropping `rf`/`hgb` to just `ridge` for the "demo" profile specifically).
None of these were done in Phase 9: (a)/(c) would change what every other phase's
tests and this repo's own demo narrative have exercised throughout the build, and
(b) is a real, scoped follow-up rather than a quick fix. The CI `demo-smoke` job
(`.github/workflows/ci.yml`) and this session's own CLI integration tests
accordingly use a reduced flight/variant count (2-6, well under the fold cap), which
*does* complete in well under 5 minutes -- consistent with `fsa run-all` itself
being fast; it's specifically the full-scale default config that overruns the target.

**NFR-PERF-002** (100 flights x 10 Hz x 1 h in < 30 min) is extrapolated, not
measured at that literal scale (that would take many hours to actually run): the
measured `preprocess` throughput above is 80 s for 12 flights of ~300 s each, i.e.
~6.7 s per 5-minute flight. Scaling linearly to 1-hour flights (~80 s/flight) and to
100 such flights gives **~133 minutes**, also over target -- `preprocess()` runs its
per-flight loop sequentially (no `joblib.Parallel`, unlike `generate()`); the same
parallelisation note as above applies.

**Other findings from this full-scale run** (both already fixed, with regression
tests, before this document was written): `report/figures.py`'s `ranking_forest_plot`
raised `ValueError: 'xerr' must not contain negative values` because a bootstrap CI
bound can land on the "wrong" side of the point estimate with real data (now clipped
to non-negative -- `test_ranking_forest_plot_tolerates_a_ci_bound_crossing_the_point_estimate`);
`predict/models/ridge.py`'s `RidgeModel` logged (non-fatal) `LinAlgWarning:
ill-conditioned matrix` on several `(variant, target)` combinations, consistent with
some lag-feature columns having little variance within a short fitting window --
noted here as a known characteristic, not fixed, since Ridge's L2 term still
produces a usable (if less precise) fit and nothing downstream failed because of it.
