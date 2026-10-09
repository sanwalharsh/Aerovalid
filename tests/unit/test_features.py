from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.prep.features import add_derived_signals, savgol_derivative


@pytest.mark.req("FR-PRP-006")
def test_savgol_derivative_recovers_known_linear_slope() -> None:
    dt = 0.1
    t = np.arange(0.0, 20.0, dt)
    x = pd.Series(3.0 * t + 1.0)
    derivative = savgol_derivative(x, dt, window=11, order=2, deriv=1)
    np.testing.assert_allclose(derivative[5:-5], 3.0, atol=1e-6)


@pytest.mark.req("FR-PRP-006")
def test_savgol_derivative_smooths_noisy_sinusoid_below_threshold() -> None:
    dt = 0.05
    t = np.arange(0.0, 20.0, dt)
    rng = np.random.default_rng(0)
    clean = np.sin(t)
    noisy = pd.Series(clean + rng.normal(0, 0.02, size=len(t)))
    derivative = savgol_derivative(noisy, dt, window=11, order=3, deriv=1)
    analytic = np.cos(t)
    error = np.abs(derivative[20:-20] - analytic[20:-20])
    assert np.mean(error) < 0.1


@pytest.mark.req("FR-PRP-005")
def test_add_derived_signals_computes_mach_and_q() -> None:
    n = 50
    df = pd.DataFrame(
        {
            "t_s": np.arange(n) * 0.1,
            "alt_m": np.full(n, 1000.0),
            "vel_air_mps": np.full(n, 100.0),
        }
    )
    out = add_derived_signals(df, dt=0.1)
    assert "mach" in out.columns
    assert "q_pa" in out.columns
    assert out["mach"].iloc[0] > 0
    assert out["q_pa"].iloc[0] > 0


@pytest.mark.req("FR-PRP-005")
def test_add_derived_signals_climb_rate_matches_constant_slope() -> None:
    n = 100
    t = np.arange(n) * 0.1
    df = pd.DataFrame({"t_s": t, "alt_m": 1000.0 + 2.0 * t, "vel_air_mps": np.full(n, 30.0)})
    out = add_derived_signals(df, dt=0.1)
    assert out["climb_rate_mps"].iloc[20:-20].mean() == pytest.approx(2.0, abs=0.05)


@pytest.mark.req("FR-PRP-005")
def test_cl_eff_requires_wing_area() -> None:
    n = 10
    df = pd.DataFrame(
        {
            "t_s": np.arange(n) * 0.1,
            "alt_m": np.full(n, 1000.0),
            "vel_air_mps": np.full(n, 30.0),
            "nz_g": np.full(n, 1.2),
            "mass_kg": np.full(n, 25.0),
        }
    )
    without_area = add_derived_signals(df, dt=0.1)
    assert "cl_eff" not in without_area.columns

    with_area = add_derived_signals(df, dt=0.1, wing_area_m2=1.2)
    assert "cl_eff" in with_area.columns
    assert (with_area["cl_eff"] > 0).all()
