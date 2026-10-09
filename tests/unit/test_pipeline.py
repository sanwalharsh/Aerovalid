from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.core.config import PrepConfig
from aerovalid.prep.pipeline import preprocess_flight


def _synthetic_telemetry(n: int = 600, dt: float = 0.1) -> pd.DataFrame:
    t = np.arange(n) * dt
    rng = np.random.default_rng(0)
    alt = 1000.0 + 5.0 * np.sin(2 * np.pi * 0.01 * t) + rng.normal(0, 0.5, n)
    alt[300] += 50.0  # spike
    alt[100:103] = np.nan  # short gap
    return pd.DataFrame(
        {
            "t_s": t,
            "flight_id": "F0001",
            "source": "tel",
            "alt_m": alt,
            "vel_air_mps": 30.0 + np.sin(2 * np.pi * 0.02 * t) + rng.normal(0, 0.1, n),
            "gamma_rad": np.zeros(n),
            "chi_rad": np.zeros(n),
            "mass_kg": np.full(n, 25.0),
            "roll_rad": np.zeros(n),
            "nz_g": np.ones(n),
            "throttle": np.full(n, 0.5),
            "alpha_cmd_rad": np.full(n, 0.05),
            "bank_cmd_rad": np.zeros(n),
        }
    )


def _synthetic_variant(telemetry: pd.DataFrame, offset_s: float = 0.2) -> pd.DataFrame:
    out = telemetry.copy()
    out["t_s"] = out["t_s"] + offset_s
    out["source"] = "V0"
    out["alt_m"] = out["alt_m"].interpolate().bfill() + 2.0
    return out


@pytest.mark.req("FR-PRP-009")
def test_preprocess_flight_produces_aligned_dataset_for_each_variant() -> None:
    telemetry = _synthetic_telemetry()
    variant = _synthetic_variant(telemetry)
    result = preprocess_flight("F0001", telemetry, {"V0": variant}, PrepConfig())

    assert result.qc.flight_id == "F0001"
    assert "V0" in result.aligned
    aligned_df = result.aligned["V0"].df
    assert "obs_alt_m" in aligned_df.columns
    assert "sim_alt_m" in aligned_df.columns
    assert "res_alt_m" in aligned_df.columns
    assert "phase" in aligned_df.columns


@pytest.mark.req("FR-PRP-002")
def test_preprocess_flight_cleans_injected_spike() -> None:
    telemetry = _synthetic_telemetry()
    variant = _synthetic_variant(telemetry)
    result = preprocess_flight("F0001", telemetry, {"V0": variant}, PrepConfig())
    assert result.clean_result.n_spikes_removed["alt_m"] >= 1


@pytest.mark.req("FR-PRP-003")
def test_preprocess_flight_estimates_sync_lag_close_to_injected_offset() -> None:
    telemetry = _synthetic_telemetry(n=3000)  # longer flight for a reliable lag estimate
    variant = _synthetic_variant(telemetry, offset_s=0.3)
    result = preprocess_flight("F0001", telemetry, {"V0": variant}, PrepConfig())
    estimate = result.sync_estimates["V0"]
    assert abs(estimate.lag_s + 0.3) < 1.0 or abs(estimate.lag_s - 0.3) < 1.0
