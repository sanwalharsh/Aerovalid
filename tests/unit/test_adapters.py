from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aerovalid.core.errors import ConfigError
from aerovalid.telemetry.adapters.base import ColumnMapping
from aerovalid.telemetry.adapters.csv_parquet import CsvParquetAdapter

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def mapping() -> ColumnMapping:
    raw = yaml.safe_load((_FIXTURES / "sample_external_flight_mapping.yaml").read_text())
    return ColumnMapping.model_validate(raw)


@pytest.mark.req("FR-TEL-005")
def test_csv_adapter_converts_non_si_units_to_canonical_schema(mapping: ColumnMapping) -> None:
    adapter = CsvParquetAdapter()
    df = adapter.read(_FIXTURES / "sample_external_flight.csv", mapping)

    assert list(df["t_s"]) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert df["alt_m"].iloc[0] == pytest.approx(4921.26 * 0.3048, rel=1e-6)
    assert df["vel_air_mps"].iloc[0] == pytest.approx(58.32 * 0.5144444, rel=1e-6)
    assert df["pitch_rad"].iloc[0] == pytest.approx(2.0 * 3.14159265 / 180, rel=1e-4)
    assert (df["flight_id"] == "F_EXTERNAL_0001").all()
    assert (df["source"] == "tel").all()


@pytest.mark.req("FR-TEL-005")
def test_csv_adapter_missing_time_column_raises(tmp_path: Path, mapping: ColumnMapping) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b\n1,2\n")
    with pytest.raises(ConfigError, match="time_column"):
        CsvParquetAdapter().read(bad_csv, mapping)


@pytest.mark.req("FR-TEL-005")
def test_csv_adapter_missing_mapped_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "partial.csv"
    csv_path.write_text("time_ms,altitude_ft\n0,100\n100,101\n")
    columns = {"alt_m": "altitude_ft", "vel_air_mps": "missing_col"}
    mapping = ColumnMapping(time_column="time_ms", time_unit="ms", columns=columns)
    with pytest.raises(ConfigError, match="missing_col"):
        CsvParquetAdapter().read(csv_path, mapping)


@pytest.mark.req("FR-TEL-005")
def test_unsupported_extension_raises(tmp_path: Path, mapping: ColumnMapping) -> None:
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("x")
    with pytest.raises(ConfigError, match="Unsupported"):
        CsvParquetAdapter().read(bad_file, mapping)
