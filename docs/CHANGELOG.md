# Changelog

All notable changes to AeroValidate, grouped by build phase (see `06_implementation_plan.md`).

## Phase 0 — Scaffolding

- Added `pyproject.toml` (hatchling, src layout, extras `ml`/`adapters`/`dev`/`all`), `Makefile`
  (`setup`, `lint`, `test`, `test-fast`, `demo`, `dashboard`, `config-ref`), `.gitignore`,
  `.pre-commit-config.yaml` (ruff, mypy, nbstripout), `.github/workflows/ci.yml` (lint+test on
  Linux, smoke matrix on macOS/Windows).
- Added `src/aerovalid/core/`: `errors.py` (`AeroValidError` hierarchy), `logging.py`
  (structured, run_id-tagged), `rng.py` (`SeedSequence`-based deterministic generators),
  `angles.py` (wrap/circular stats), `signals.py` (canonical signal registry), `config.py`
  (pydantic `StudyConfig` tree + `load_config` with `include:` and `--set` overrides),
  `manifest.py` (`run_manifest.json` writer), `schema.py` (pandera canonical-table schema).
- Added `src/aerovalid/cli.py`: `fsa` typer app with all CLAUDE.md §5 subcommands stubbed
  (`NotImplementedError`, no silent `pass`), each naming the phase that implements it.
- Added `configs/default.yaml` matching `StudyConfig`'s defaults (data spec §7).
- Added `docs/DECISIONS.md` (ADR-001..004), `docs/07_nfr_risks_governance.md`,
  `scripts/gen_config_reference.py` (introspects the pydantic models to regenerate
  `docs/CONFIG_REFERENCE.md`).
- Added `tests/unit/` coverage for config (round-trip, overrides, `ConfigError` messages),
  manifest (write/read/update_step), RNG (determinism, independence), angles (wrap
  idempotence, circular mean), signals (lookup + unknown-signal error), schema (valid/invalid
  tables), and the CLI (`--help` lists every command; each stub raises `NotImplementedError`).

DoD: `pip install -e ".[dev]"` succeeds; `fsa --help` lists all commands; `make lint test`
green; config round-trip passes; manifest written correctly. See `docs/DECISIONS.md` for the
non-obvious calls made while resolving ambiguity.
