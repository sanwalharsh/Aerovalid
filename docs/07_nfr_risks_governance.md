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
