from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.prep.qc import qc_report


def _base_df(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t_s": np.arange(n) * 0.1,
            "flight_id": "F0001",
            "source": "tel",
            "alt_m": 1000.0 + np.sin(np.linspace(0, 10, n)),
            "vel_air_mps": np.full(n, 30.0),
        }
    )


@pytest.mark.req("FR-PRP-001")
def test_qc_report_basic_fields() -> None:
    df = _base_df()
    report = qc_report(df, "F0001")
    assert report.flight_id == "F0001"
    assert report.n_samples == 200
    assert report.sample_rate_hz == pytest.approx(10.0, rel=0.05)
    assert report.duplicate_timestamps == 0
    assert report.non_monotonic_count == 0
    assert report.severity == "ok"


@pytest.mark.req("FR-PRP-001")
def test_qc_report_detects_missing_fraction() -> None:
    df = _base_df()
    df.loc[5:14, "alt_m"] = np.nan
    report = qc_report(df, "F0001")
    assert report.missing_fraction["alt_m"] == pytest.approx(10 / 200)


@pytest.mark.req("FR-PRP-001")
def test_qc_report_quarantines_heavily_non_monotonic_flights() -> None:
    df = _base_df()
    t = df["t_s"].to_numpy().copy()
    t[10:200] = t[10] - np.arange(190) * 0.1  # strictly decreasing: forces dt < 0
    df["t_s"] = t
    report = qc_report(df, "F0001")
    assert report.severity == "quarantine"


@pytest.mark.req("FR-PRP-001")
def test_qc_report_detects_stuck_signal() -> None:
    df = _base_df()
    df.loc[:, "vel_air_mps"] = 30.0  # already constant for >= 20 samples
    report = qc_report(df, "F0001")
    assert "vel_air_mps" in report.stuck_signals


@pytest.mark.req("FR-PRP-001")
def test_qc_report_counts_spikes() -> None:
    df = _base_df()
    df.loc[50, "alt_m"] = 1e6  # injected spike
    report = qc_report(df, "F0001")
    assert report.spike_counts["alt_m"] >= 1


@pytest.mark.req("FR-PRP-001")
def test_qc_report_range_violation_uses_signal_registry() -> None:
    df = _base_df()
    df.loc[0, "alt_m"] = -10000.0  # outside alt_m's plausible range
    report = qc_report(df, "F0001")
    assert report.range_violations["alt_m"] >= 1
