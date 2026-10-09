"""CSV/Parquet ingestion adapter (FR-TEL-005): column mapping + unit conversion +
time-base normalisation into the canonical schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aerovalid.core.errors import ConfigError
from aerovalid.core.schema import validate_canonical
from aerovalid.telemetry.adapters.base import ColumnMapping

_TIME_UNIT_TO_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


def _normalise_time(raw_time: pd.Series, time_unit: str) -> pd.Series:
    if time_unit in _TIME_UNIT_TO_SECONDS:
        return raw_time.astype(float) * _TIME_UNIT_TO_SECONDS[time_unit]
    if time_unit == "epoch_s":
        return raw_time.astype(float) - float(raw_time.iloc[0])
    if time_unit == "epoch_ms":
        return (raw_time.astype(float) - float(raw_time.iloc[0])) * 1e-3
    raise ConfigError(f"Unknown time_unit {time_unit!r}; expected s/ms/us/epoch_s/epoch_ms")


class CsvParquetAdapter:
    """Reads `.csv` or `.parquet` files via a `ColumnMapping` (FR-TEL-005)."""

    def read(self, path: Path, mapping: ColumnMapping) -> pd.DataFrame:
        """
        Read `path` and map it onto the canonical schema.

        Parameters
        ----------
        path : Path
            A `.csv` or `.parquet` file.
        mapping : ColumnMapping

        Returns
        -------
        pandas.DataFrame
            Canonical-schema table, validated via `core.schema.validate_canonical`.

        Raises
        ------
        ConfigError
            If `path`'s extension is unsupported, `mapping.time_column` or any
            `mapping.columns` source column is missing, or `time_unit` is unknown.
        """
        path = Path(path)
        if path.suffix == ".parquet":
            raw = pd.read_parquet(path)
        elif path.suffix == ".csv":
            raw = pd.read_csv(path)
        else:
            raise ConfigError(f"Unsupported file extension for adapter: {path.suffix}")

        if mapping.time_column not in raw.columns:
            raise ConfigError(f"Mapping time_column {mapping.time_column!r} not found in {path}")

        canonical = pd.DataFrame()
        canonical["t_s"] = _normalise_time(raw[mapping.time_column], mapping.time_unit)
        canonical["flight_id"] = mapping.flight_id or path.stem
        canonical["source"] = mapping.source

        for name, source_col in mapping.columns.items():
            if source_col not in raw.columns:
                raise ConfigError(
                    f"Mapping references missing source column {source_col!r} in {path}"
                )
            values: NDArray[np.float64] = raw[source_col].to_numpy(dtype=np.float64)
            if name in mapping.angle_columns_in_degrees:
                values = np.deg2rad(values)
            factor = mapping.unit_conversions.get(name, 1.0)
            canonical[name] = values * factor

        return validate_canonical(canonical)
