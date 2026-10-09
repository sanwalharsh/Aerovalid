# User Guide: running AeroValidate on your own (real) data

This walks through PRD §6's journey J-B: validating your own simulation variant(s)
against your own flight telemetry, instead of the synthetic generator. It assumes
you've already done `make setup` (see `README.md`).

## 1. What AeroValidate needs from your data

Every table that flows through the pipeline is the canonical schema
(`src/aerovalid/core/schema.py`): a `t_s` (seconds from flight start), `flight_id`,
`source` column, plus whichever signal columns you have, each named and in the SI
units the signal registry expects (`src/aerovalid/core/signals.py` — e.g. `alt_m` in
metres, `vel_air_mps` in m/s, angles in radians). You don't need every signal or a
perfectly clean log: QC and cleaning (Phase 3) tolerate gaps and glitches, and every
downstream step only uses whichever signals are actually present.

You need, at minimum, **one telemetry file** (the real flight) and **one simulation
variant's output file** (what you want to validate), both covering the same flight
and containing at least one shared signal (e.g. `alt_m`). A timestamp column in
seconds, milliseconds, microseconds, or Unix epoch time is all `fsa ingest` needs to
find the time axis; everything else is handled by a small mapping file you write once
per data source.

## 2. Write a column-mapping file per data source

A mapping (`aerovalid.telemetry.adapters.base.ColumnMapping`) is a small YAML file
telling the CSV/Parquet adapter how to read your file. See
`tests/fixtures/external_demo/telemetry_mapping.yaml` for a complete worked example
(feet/knots/degrees converted to metres/m-per-s/radians):

```yaml
time_column: time_ms      # your file's timestamp column
time_unit: ms              # one of: s, ms, us, epoch_s, epoch_ms
flight_id: F_EXT_0001      # omit this to use each file's own filename stem instead
source: tel                 # "tel" for telemetry; a variant name (e.g. "V0") otherwise
columns:
  alt_m: altitude_ft        # canonical_name: your_column_name
  vel_air_mps: airspeed_kts
  pitch_rad: pitch_deg
  roll_rad: roll_deg
angle_columns_in_degrees: [pitch_rad, roll_rad]   # converted to radians first
unit_conversions:
  alt_m: 0.3048              # feet -> metres
  vel_air_mps: 0.5144444     # knots -> m/s
```

Write one mapping per **source**: one for your telemetry (`source: tel`), and one
more per simulation variant you want to compare against (`source: <variant-name>`,
e.g. `V0`) — see `tests/fixtures/external_demo/variant_v0_mapping.yaml`. If you're
validating against multiple variants, add a mapping (and an `fsa ingest` call) for
each; `flight_id` must match across a telemetry file and the variant file(s) it's
compared against, since that's how the pipeline pairs them up.

Put each source's file(s) in their own directory — `fsa ingest --source` reads every
`.csv`/`.parquet` file directly inside the directory you point it at (not recursive).

## 3. Ingest, then run the rest of the pipeline exactly as a synthetic study would

```bash
fsa ingest --run-id my_study \
  --source path/to/telemetry_logs/ \
  --mapping path/to/telemetry_mapping.yaml

fsa ingest --run-id my_study \
  --source path/to/v0_sim_outputs/ \
  --mapping path/to/v0_mapping.yaml

fsa preprocess --config configs/default.yaml --run-id my_study
fsa validate   --config configs/default.yaml --run-id my_study
fsa analyze    --config configs/default.yaml --run-id my_study
fsa report     --config configs/default.yaml --run-id my_study
```

`fsa ingest` writes straight into `runs/my_study/raw/telemetry/` and
`runs/my_study/raw/variants/<name>/` — the same layout `fsa generate` produces — so
every later step needs no awareness that the data came from a real log rather than
the synthetic simulator. Open `runs/my_study/reports/report.html` when it's done.

You can try this exact flow right now against the repo's own tiny synthetic
"external" fixture (a 60 s, 10 Hz telemetry trace and a slightly biased "V0" variant,
both under `tests/fixtures/external_demo/`):

```bash
fsa ingest --run-id demo_external \
  --source tests/fixtures/external_demo/telemetry \
  --mapping tests/fixtures/external_demo/telemetry_mapping.yaml
fsa ingest --run-id demo_external \
  --source tests/fixtures/external_demo/variant_v0 \
  --mapping tests/fixtures/external_demo/variant_v0_mapping.yaml
fsa preprocess --config configs/default.yaml --run-id demo_external
fsa validate   --config configs/default.yaml --run-id demo_external
```

## 4. What won't work on real data (and why)

- **`fsa identify`** (output-error parameter fitting) replays a candidate variant
  against your flight's *recorded control inputs* (`throttle`, `alpha_cmd_rad`,
  `bank_cmd_rad`). If your telemetry log doesn't export these, map them too (same
  `columns:` mechanism) or skip `fsa identify` for that study — `validate`/`analyze`/
  `predict`/`detect` don't need them.
- **`fsa detect`**'s false-alarm-rate calibration needs several *fault-free* flights
  to calibrate against (FR-DET-002); a single real flight isn't enough on its own —
  ingest several, and make sure at least a few are ones you believe are nominal.
- **ULog/DataFlash adapters** (PX4/ArduPilot native log formats) are deferred — see
  `docs/DECISIONS.md` ADR-009. Export to CSV/Parquet first (most ground-control
  software can do this), then use the mapping flow above.

## 5. Config reference

`docs/CONFIG_REFERENCE.md` (regenerate with `make config-ref`) documents every
`configs/default.yaml` field; you'll typically only need to adjust `validation.signals`
(to match whichever signals your mapping actually supplies) and `identify.params`/
`identify.bounds` if you do have control-input data and want to fit specific
aerodynamic parameters.
