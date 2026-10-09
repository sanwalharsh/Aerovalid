from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerovalid.sim.protocols import VehicleParams
from aerovalid.sim.runners import controls_from_dataframe, replay_run, run_windowed, state_from_row
from aerovalid.sim.variants import load_variant_yaml, make_variant


@pytest.fixture
def flight_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    t_s = np.arange(0.0, 30.0, 0.1)
    tel_state = pd.DataFrame(
        {
            "t_s": t_s,
            "x_m": np.zeros_like(t_s),
            "y_m": np.zeros_like(t_s),
            "alt_m": np.full_like(t_s, 1500.0),
            "vel_air_mps": np.full_like(t_s, 30.0),
            "gamma_rad": np.zeros_like(t_s),
            "chi_rad": np.zeros_like(t_s),
            "mass_kg": np.full_like(t_s, 25.0),
            "alpha_rad": np.zeros_like(t_s),
        }
    )
    controls = pd.DataFrame(
        {
            "t_s": t_s,
            "throttle": np.full_like(t_s, 0.5),
            "alpha_cmd_rad": np.full_like(t_s, 0.04),
            "bank_cmd_rad": np.zeros_like(t_s),
        }
    )
    return tel_state, controls


@pytest.mark.req("FR-SIM-009")
def test_controls_from_dataframe_round_trips() -> None:
    df = pd.DataFrame(
        {
            "t_s": [0.0, 1.0],
            "throttle": [0.1, 0.2],
            "alpha_cmd_rad": [0.0, 0.01],
            "bank_cmd_rad": [0.0, 0.0],
        }
    )
    history = controls_from_dataframe(df)
    throttle, alpha, bank = history.at(0.5)
    assert throttle == pytest.approx(0.15)
    assert alpha == pytest.approx(0.005)
    assert bank == pytest.approx(0.0)


@pytest.mark.req("FR-SIM-009")
def test_state_from_row_maps_canonical_columns() -> None:
    row = pd.Series(
        {
            "t_s": 1.0,
            "x_m": 10.0,
            "y_m": 20.0,
            "alt_m": 1500.0,
            "vel_air_mps": 30.0,
            "gamma_rad": 0.01,
            "chi_rad": 0.2,
            "mass_kg": 25.0,
            "alpha_rad": 0.03,
        }
    )
    state = state_from_row(row)
    assert state.h_m == pytest.approx(1500.0)
    assert state.V_mps == pytest.approx(30.0)
    assert state.m_kg == pytest.approx(25.0)


@pytest.mark.req("FR-SIM-009")
def test_replay_run_follows_controls_without_nan(
    flight_frames: tuple[pd.DataFrame, pd.DataFrame],
    variants_dir: Path,
    test_vehicle: VehicleParams,
) -> None:
    tel_state, controls = flight_frames
    v0 = make_variant(load_variant_yaml("V0", variants_dir), test_vehicle)
    result = replay_run(v0, controls, tel_state.iloc[0], seed=0)
    assert result.status == "ok"
    assert len(result.trajectory) > 0


@pytest.mark.req("FR-SIM-010")
def test_run_windowed_tags_window_id_and_horizon(
    flight_frames: tuple[pd.DataFrame, pd.DataFrame],
    variants_dir: Path,
    test_vehicle: VehicleParams,
) -> None:
    tel_state, controls = flight_frames
    v0 = make_variant(load_variant_yaml("V0", variants_dir), test_vehicle)
    windowed = run_windowed(v0, tel_state, controls, horizon_s=5.0, stride_s=5.0, seed=0)

    assert "window_id" in windowed.columns
    assert "horizon_s" in windowed.columns
    assert windowed["window_id"].nunique() >= 2
    assert windowed["horizon_s"].min() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.req("FR-SIM-010")
def test_run_windowed_horizon_zero_matches_state_estimate(
    flight_frames: tuple[pd.DataFrame, pd.DataFrame],
    variants_dir: Path,
    test_vehicle: VehicleParams,
) -> None:
    tel_state, controls = flight_frames
    v0 = make_variant(load_variant_yaml("V0", variants_dir), test_vehicle)
    windowed = run_windowed(v0, tel_state, controls, horizon_s=5.0, stride_s=5.0, seed=0)

    first_window = windowed[windowed["window_id"] == 0]
    at_horizon_zero = first_window.loc[first_window["horizon_s"].idxmin()]
    assert at_horizon_zero["sim_alt_m"] == pytest.approx(tel_state.iloc[0]["alt_m"], abs=1e-6)
