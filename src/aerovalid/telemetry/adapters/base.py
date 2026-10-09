"""Adapter protocol and column-mapping model (FR-TEL-005): every ingestion adapter
reads an external file plus a `ColumnMapping` and returns a canonical-schema DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class ColumnMapping(BaseModel):
    """Maps an external file's columns/units/time-base onto the canonical schema.

    Attributes
    ----------
    time_column : str
        Source column holding the timestamp.
    time_unit : str
        One of `"s"`, `"ms"`, `"us"`, `"epoch_s"`, `"epoch_ms"`. `epoch_*` variants
        are rebased to start at 0 (data spec §4.1: `t_s` is seconds from flight start).
    columns : dict[str, str]
        `canonical_name -> source_column_name`.
    unit_conversions : dict[str, float]
        `canonical_name -> multiplicative factor to SI`, applied after any degree
        conversion (e.g. feet to metres: `0.3048`).
    angle_columns_in_degrees : list[str]
        Canonical names whose source values are in degrees (converted to radians
        before `unit_conversions`, which therefore should not be set for these).
    flight_id : str, optional
        Constant flight ID to assign (the source file has none of its own).
    source : str
        Canonical `source` column value, e.g. `"tel"` or a variant name.
    """

    model_config = ConfigDict(extra="forbid")

    time_column: str
    time_unit: str = "s"
    columns: dict[str, str] = Field(default_factory=dict)
    unit_conversions: dict[str, float] = Field(default_factory=dict)
    angle_columns_in_degrees: list[str] = Field(default_factory=list)
    flight_id: str | None = None
    source: str = "ingested"


@runtime_checkable
class Adapter(Protocol):
    """Reads an external data file into the canonical schema given a `ColumnMapping`."""

    def read(self, path: Path, mapping: ColumnMapping) -> pd.DataFrame:
        """Return a canonical-schema DataFrame (`core/schema.py`) for `path`."""
        ...
