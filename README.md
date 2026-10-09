# AeroValidate

AI-assisted flight-simulation validation and predictive analysis for a generic,
unclassified fixed-wing UAV. Synthetic-first: ships its own truth-model simulator and
sensor/fault layer so it runs without any external data; real-data adapters are optional.

It does five things: generates/ingests multi-variant flight-simulation data, prepares
and aligns it against telemetry, validates each simulation variant against that
telemetry, predicts residuals and detects deviations, and explains the results
(sensitivity, parameter identification, ranked recommendations, an HTML report, and a
Streamlit dashboard). See `docs/00_vision.md` for the research questions and
`CLAUDE.md` for the full project rules.

## Setup

```bash
make setup
```

Creates `.venv/` and installs the package with every extra (`pip install -e ".[all]"`).
Requires Python >= 3.11.

## Quick start: a demo run

```bash
.venv/bin/fsa run-all --config configs/default.yaml --run-id demo --profile demo
```

This runs the full pipeline end to end — generate, preprocess, validate, analyze,
predict, detect, identify, report — and writes everything under `runs/demo/`,
including an HTML report at `runs/demo/reports/report.html`. On a laptop, a demo-scale
run (the config's default `generate.n_flights`) takes a few minutes.

To explore the result interactively:

```bash
.venv/bin/fsa dashboard --run-id demo
```

To run only part of the pipeline, invoke each step directly with the same
`--config`/`--run-id` (each step reads the previous step's artifacts from
`runs/<run-id>/`):

```bash
.venv/bin/fsa generate   --config configs/default.yaml --run-id demo
.venv/bin/fsa preprocess --config configs/default.yaml --run-id demo
.venv/bin/fsa validate   --config configs/default.yaml --run-id demo
.venv/bin/fsa analyze    --config configs/default.yaml --run-id demo
.venv/bin/fsa predict    --config configs/default.yaml --run-id demo
.venv/bin/fsa detect     --config configs/default.yaml --run-id demo
.venv/bin/fsa identify   --config configs/default.yaml --run-id demo
.venv/bin/fsa report     --config configs/default.yaml --run-id demo
```

Override any config value from the command line without editing the file, e.g. to run
a smaller/faster demo:

```bash
.venv/bin/fsa generate --config configs/default.yaml --run-id demo \
  --set generate.n_flights=4 --set generate.scenarios=[doublet_series] \
  --set variants=[V0,V1,V2]
```

`--seed` overrides `study.seed`; `--force` re-runs a step even if its artifacts and
config hash already match (steps are otherwise idempotent and skip re-running).

## CLI commands

| Command | What it does |
|---|---|
| `fsa generate` | Synthesize truth flight(s), sensor-corrupted telemetry, and every configured variant's trajectory. |
| `fsa ingest` | Load external telemetry/variant data via a column-mapping adapter instead of `generate`. |
| `fsa preprocess` | QC, clean, time-sync, resample, and align telemetry against each variant. |
| `fsa validate` | Per-signal/per-flight error metrics, composite scoring, bootstrap variant ranking, pairwise comparisons. |
| `fsa analyze` | Condition-dependent error analysis (heatmaps by altitude/Mach/etc. bin). |
| `fsa predict` | Residual prediction and hybrid (sim + predicted residual) correction models, with uncertainty. |
| `fsa detect` | Calibrated statistical + Mahalanobis deviation detectors on residual streams. |
| `fsa identify` | Output-error parameter fitting, sensitivity analysis, and evidence-linked recommendations. |
| `fsa report` | Build the HTML report, `summary.json`, and CSV/Markdown tables from whatever artifacts exist. |
| `fsa dashboard` | Launch the Streamlit dashboard for a run. |
| `fsa run-all` | Run every step above in order. |
| `fsa benchmark` | Run the pipeline across seeds and report the vision doc's S1-S9 success criteria. |

Run `fsa <command> --help` for a command's full option list; `fsa --help` lists all of
them. Several commands intentionally cover a reduced scope relative to their full
spec (the headline example per flight/signal/model rather than every combination) —
each says so in its own `--help` text and in `docs/DECISIONS.md`'s ADRs.

## Configuration

`configs/default.yaml` is the default study config; `configs/vehicles/`,
`configs/variants/`, `configs/scenarios/`, `configs/faults/`, and `configs/sensors/`
hold the generic vehicle's parameters, simulation-model variants (V0-V5, each with a
deliberately different known flaw), flight scenarios, fault types, and sensor models it
references. `docs/CONFIG_REFERENCE.md` (regenerate with `make config-ref`) documents
every config field and its default.

## Other Makefile targets

```bash
make lint        # ruff check, ruff format --check, mypy --strict
make test        # full pytest suite with coverage (85% gate)
make test-fast   # skip @pytest.mark.slow tests
make demo        # fsa run-all on the demo profile
make dashboard   # streamlit run src/aerovalid/app/main.py directly
```

## Docker

```bash
docker build -t aerovalid .
docker run -v "$PWD/runs:/app/runs" aerovalid run-all --config configs/default.yaml --run-id demo
```

Builds a lean image with the core pipeline (optional `torch`/`lightgbm`/`shap`/
`pymavlink`/`pyulog` extras aren't included; every module that needs them degrades
gracefully without them). Bind-mount `runs/` (and `data/`/`reports/` if you use them)
so output persists outside the container.

## More

- `docs/USER_GUIDE.md` — running the pipeline on your own (real) flight data.
- `docs/00_vision.md` — research questions and the S1-S9 success criteria.
- `docs/02_FRD.md` — functional requirements.
- `docs/03_architecture_and_data_spec.md` — module layout and on-disk artifact schema.
- `docs/05_analysis_methods.md` — the statistics/ML methodology behind each phase.
- `docs/06_implementation_plan.md` — phase-by-phase build plan and definition of done.
- `docs/DECISIONS.md` — architecture decision records for every non-obvious choice.
- `docs/CHANGELOG.md` — what each phase built, in order.
