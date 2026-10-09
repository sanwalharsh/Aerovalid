"""Canonical dataset schema validation (FR-COR-005), built on pandera. Every table that
crosses a module boundary (`sim` -> `telemetry` -> `prep` -> `validate` ...) is validated
on write and on read against the time-series schema from data spec §4.1.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from aerovalid.core.errors import SchemaError
from aerovalid.core.signals import known_signals

_REQUIRED_COLUMNS: dict[str, pa.Column] = {
    "t_s": pa.Column(float, checks=pa.Check.ge(0.0), nullable=False),
    "flight_id": pa.Column(str, nullable=False),
    "source": pa.Column(str, nullable=False),
}


def _signal_columns() -> dict[str, pa.Column]:
    """Optional float64 columns for every registered signal, each nullable (not every
    table carries every signal)."""
    return {name: pa.Column(float, nullable=True, required=False) for name in known_signals()}


def build_time_series_schema() -> pa.DataFrameSchema:
    """
    Build the canonical time-series table schema (data spec §4.1).

    Returns
    -------
    pandera.DataFrameSchema
        Requires `t_s`, `flight_id`, `source`; allows any registered signal column,
        plus `obs_<sig>` / `sim_<sig>` / `res_<sig>` / `mask_<sig>` / `cond_<name>` /
        `phase` / `window_id` / `horizon_s` prefixed columns used by aligned tables.
    """
    columns = {**_REQUIRED_COLUMNS, **_signal_columns()}
    return pa.DataFrameSchema(columns=columns, strict=False, coerce=False)


_TIME_SERIES_SCHEMA = build_time_series_schema()

_SIGNAL_PREFIXES = ("obs_", "sim_", "res_", "mask_")
_ALLOWED_EXTRA = frozenset({"phase", "window_id", "horizon_s", "t_utc"})


def _is_allowed_extra_column(column: str) -> bool:
    if column in _REQUIRED_COLUMNS or column in known_signals() or column in _ALLOWED_EXTRA:
        return True
    if column.startswith("cond_"):
        return True  # cond_<name> is a free-form condition descriptor, not a signal name
    return any(
        column.startswith(prefix) and column[len(prefix) :] in known_signals()
        for prefix in _SIGNAL_PREFIXES
    )


def validate_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate `df` against the canonical time-series schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Candidate canonical table.

    Returns
    -------
    pandas.DataFrame
        `df`, unchanged, if valid.

    Raises
    ------
    SchemaError
        Lists every offending column: missing required columns, wrong dtypes, or
        unrecognised column names outside the registry/prefix conventions.
    """
    unknown = [c for c in df.columns if not _is_allowed_extra_column(c)]
    if unknown:
        raise SchemaError(f"Unrecognised column(s) in canonical table: {unknown}")

    try:
        _TIME_SERIES_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        offending = sorted(set(exc.failure_cases["column"].dropna().tolist()))
        raise SchemaError(f"Schema validation failed for column(s): {offending}") from exc

    return df
