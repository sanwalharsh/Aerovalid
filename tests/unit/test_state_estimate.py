from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.prep.state_estimate import estimate_state


@pytest.mark.req("FR-PRP-011")
def test_estimate_state_smooths_noisy_altitude_toward_truth() -> None:
    dt = 0.1
    t = np.arange(0.0, 60.0, dt)
    truth = 1000.0 + 5.0 * t
    rng = np.random.default_rng(0)
    noisy = truth + rng.normal(0, 1.5, size=len(t))
    telemetry = pd.DataFrame({"t_s": t, "alt_m": noisy, "vel_air_mps": np.full(len(t), 30.0)})

    estimate = estimate_state(telemetry, dt)

    noisy_error = np.mean(np.abs(noisy - truth))
    smoothed_error = np.mean(np.abs(estimate["alt_m"].to_numpy() - truth))
    assert smoothed_error < noisy_error


@pytest.mark.req("FR-PRP-011")
def test_estimate_state_wraps_angle_columns() -> None:
    dt = 0.1
    n = 100
    t = np.arange(n) * dt
    # chi_rad oscillating near the +-pi wraparound boundary
    chi = np.pi - 0.01 * np.sin(np.linspace(0, 10, n))
    telemetry = pd.DataFrame({"t_s": t, "chi_rad": chi})

    estimate = estimate_state(telemetry, dt)

    assert np.all(estimate["chi_rad"] <= np.pi + 1e-9)
    assert np.all(estimate["chi_rad"] > -np.pi - 1e-9)


@pytest.mark.req("FR-PRP-011")
def test_estimate_state_only_includes_present_columns() -> None:
    dt = 0.1
    telemetry = pd.DataFrame({"t_s": np.arange(10) * dt, "alt_m": np.full(10, 1000.0)})
    estimate = estimate_state(telemetry, dt)
    assert "alt_m" in estimate.columns
    assert "vel_air_mps" not in estimate.columns
