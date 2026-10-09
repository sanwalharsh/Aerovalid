from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.validate.aggregate import aggregate_across_flights, compute_metrics_table


def _aligned_table(flight_id: str, phase_values: list[str]) -> pd.DataFrame:
    n = len(phase_values)
    return pd.DataFrame(
        {
            "flight_id": flight_id,
            "obs_alt_m": np.arange(n, dtype=float) + 100.0,
            "sim_alt_m": np.arange(n, dtype=float) + 100.0 - 1.0,
            "res_alt_m": np.full(n, 1.0),
            "mask_alt_m": np.zeros(n, dtype=bool),
            "phase": phase_values,
        }
    )


@pytest.mark.req("FR-VAL-002")
def test_compute_metrics_table_whole_flight() -> None:
    df = _aligned_table("F0001", ["cruise"] * 10)
    table = compute_metrics_table(df, signals=["alt_m"], tolerances={"alt_m": 0.5})
    assert len(table) == 1
    assert table.iloc[0]["signal"] == "alt_m"
    assert table.iloc[0]["n_samples"] == 10
    assert table.iloc[0]["me"] == pytest.approx(1.0)
    assert table.iloc[0]["tol_compliance"] == pytest.approx(0.0)


@pytest.mark.req("FR-VAL-002")
def test_compute_metrics_table_groups_by_phase() -> None:
    df = _aligned_table("F0001", ["climb"] * 5 + ["cruise"] * 5)
    table = compute_metrics_table(df, signals=["alt_m"], groupby="phase")
    assert set(table["group"]) == {"climb", "cruise"}
    assert (table["n_samples"] == 5).all()


@pytest.mark.req("FR-VAL-002")
def test_compute_metrics_table_excludes_masked_samples() -> None:
    df = _aligned_table("F0001", ["cruise"] * 10)
    df.loc[0:2, "mask_alt_m"] = True
    table = compute_metrics_table(df, signals=["alt_m"])
    assert table.iloc[0]["n_samples"] == 7


@pytest.mark.req("FR-VAL-002")
def test_compute_metrics_table_skips_signal_not_in_table() -> None:
    df = _aligned_table("F0001", ["cruise"] * 5)
    table = compute_metrics_table(df, signals=["alt_m", "vel_air_mps"])
    assert set(table["signal"]) == {"alt_m"}


@pytest.mark.req("FR-VAL-002")
def test_aggregate_across_flights_mean_median_iqr() -> None:
    metrics_table = pd.DataFrame(
        {
            "flight_id": ["F0001", "F0002", "F0003"],
            "signal": ["alt_m"] * 3,
            "rmse": [1.0, 2.0, 3.0],
        }
    )
    summary = aggregate_across_flights(metrics_table, value_columns=["rmse"])
    row = summary.iloc[0]
    assert row["n_flights"] == 3
    assert row["rmse_mean"] == pytest.approx(2.0)
    assert row["rmse_median"] == pytest.approx(2.0)
    assert row["rmse_iqr"] == pytest.approx(1.0)
