"""Shared pytest fixtures. Phase 0 only needs tmp-path config fixtures; later phases add
tiny synthetic-flight fixtures (<=60 s, fixed seeds) here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aerovalid.sim.protocols import VehicleParams

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def default_config_path() -> Path:
    """Path to the repo's real `configs/default.yaml` (read-only)."""
    return _REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture
def minimal_config_path(tmp_path: Path) -> Path:
    """Write a minimal config (relies on defaults for everything else) and return its path."""
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.safe_dump({"study": {"name": "unit-test", "seed": 7}}))
    return path


@pytest.fixture
def vehicles_dir() -> Path:
    """Path to the repo's real `configs/vehicles/` (read-only)."""
    return _REPO_ROOT / "configs" / "vehicles"


@pytest.fixture
def variants_dir() -> Path:
    """Path to the repo's real `configs/variants/` (read-only)."""
    return _REPO_ROOT / "configs" / "variants"


@pytest.fixture
def scenarios_dir() -> Path:
    """Path to the repo's real `configs/scenarios/` (read-only)."""
    return _REPO_ROOT / "configs" / "scenarios"


@pytest.fixture
def test_vehicle() -> VehicleParams:
    """A `VehicleParams` matching `configs/vehicles/generic_uav_v1.yaml`, built in
    code so sim unit tests don't depend on YAML parsing."""
    return VehicleParams(
        name="generic_uav_v1",
        mass_nom_kg=25.0,
        mass_range_kg=(22.0, 28.0),
        wing_area_m2=1.2,
        wingspan_m=3.2,
        aspect_ratio=8.533,
        oswald_e=0.80,
        k=0.04696,
        CL0=0.25,
        CLa=5.0,
        alpha_stall_rad=0.2618,
        CLmax=1.45,
        CD0=0.028,
        Mdd=0.60,
        wave_drag_coeff=20.0,
        T0_N=180.0,
        thrust_lapse_n=0.85,
        thrust_mach_factor=0.25,
        SFC_kg_per_Ns=1.5e-5,
        nz_max_g=3.5,
        bank_limit_rad=1.0472,
        tau_thrust_s=0.4,
        tau_alpha_s=0.15,
    )


def _synthetic_flight(flight_id: str, n: int, dt: float, seed: int) -> pd.DataFrame:
    """One flight's worth of synthetic `prep/align.py`-shaped output: `alt_m`'s
    residual (obs - sim) is a smooth, flight-specific sinusoid -- learnable from
    recent lag features, unlike pure noise -- so `predict/` models have a real
    signal to beat the zero-residual/persistence baselines on.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    amplitude = rng.uniform(1.5, 3.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    period_s = 40.0
    res_alt_m = amplitude * np.sin(2.0 * np.pi * t / period_s + phase)
    obs_alt_m = 500.0 + 0.5 * t + rng.normal(0.0, 0.05, n)
    obs_vel_air_mps = 40.0 + 2.0 * np.sin(2.0 * np.pi * t / 60.0) + rng.normal(0.0, 0.1, n)

    return pd.DataFrame(
        {
            "flight_id": flight_id,
            "t_s": t,
            "obs_alt_m": obs_alt_m,
            "sim_alt_m": obs_alt_m - res_alt_m,
            "res_alt_m": res_alt_m,
            "mask_alt_m": np.zeros(n, dtype=bool),
            "obs_vel_air_mps": obs_vel_air_mps,
            "sim_vel_air_mps": obs_vel_air_mps,
            "res_vel_air_mps": np.zeros(n),
            "mask_vel_air_mps": np.zeros(n, dtype=bool),
            "phase": "cruise",
            "cond_alt_band": "mid",
        }
    )


@pytest.fixture
def synthetic_predict_aligned() -> pd.DataFrame:
    """Six 80-sample (dt=1.0 s) synthetic flights concatenated, shaped like
    `prep/align.py`'s aligned output, for `predict/` module tests."""
    return pd.concat(
        [_synthetic_flight(f"F{i:04d}", n=80, dt=1.0, seed=i) for i in range(6)],
        ignore_index=True,
    )
