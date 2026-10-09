from __future__ import annotations

import pandas as pd
import pytest

from aerovalid.core.errors import SchemaError
from aerovalid.core.schema import validate_canonical


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t_s": [0.0, 0.1, 0.2],
            "flight_id": ["F0001", "F0001", "F0001"],
            "source": ["truth", "truth", "truth"],
            "alt_m": [100.0, 100.1, 100.2],
            "obs_alt_m": [99.9, 100.0, 100.3],
            "sim_alt_m": [100.0, 100.1, 100.2],
            "res_alt_m": [-0.1, -0.1, 0.1],
            "mask_alt_m": [False, False, True],
            "cond_alt_band": ["low", "low", "low"],
            "phase": ["climb", "climb", "cruise"],
        }
    )


@pytest.mark.req("FR-COR-005")
def test_valid_canonical_table_passes() -> None:
    df = _valid_df()
    result = validate_canonical(df)
    assert result is df


@pytest.mark.req("FR-COR-005")
def test_missing_required_column_raises_schema_error() -> None:
    df = _valid_df().drop(columns=["flight_id"])
    with pytest.raises(SchemaError):
        validate_canonical(df)


@pytest.mark.req("FR-COR-005")
def test_unrecognised_column_raises_schema_error_naming_it() -> None:
    df = _valid_df()
    df["totally_made_up_column"] = 1.0
    with pytest.raises(SchemaError, match="totally_made_up_column"):
        validate_canonical(df)


@pytest.mark.req("FR-COR-005")
def test_wrong_dtype_raises_schema_error() -> None:
    df = _valid_df()
    df["t_s"] = df["t_s"].astype(str)
    with pytest.raises(SchemaError):
        validate_canonical(df)


@pytest.mark.req("FR-COR-005")
def test_minimal_required_only_table_passes() -> None:
    df = pd.DataFrame(
        {"t_s": [0.0, 1.0], "flight_id": ["F0001", "F0001"], "source": ["tel", "tel"]}
    )
    validate_canonical(df)
