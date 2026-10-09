from __future__ import annotations

import numpy as np
import pytest

from aerovalid.sim.aero import (
    drag_coefficient,
    lift_coefficient,
    prandtl_glauert_factor,
    wave_drag_coefficient,
)
from aerovalid.sim.protocols import VehicleParams


@pytest.mark.req("FR-SIM-002")
def test_cl_linear_below_stall(test_vehicle: VehicleParams) -> None:
    alpha = np.deg2rad(5.0)
    cl = lift_coefficient(alpha, mach=0.1, params=test_vehicle, compressibility=False)
    expected = test_vehicle.CL0 + test_vehicle.CLa * alpha
    assert cl == pytest.approx(expected, rel=1e-3)


@pytest.mark.req("FR-SIM-002")
def test_cl_rolls_off_past_stall_instead_of_diverging(test_vehicle: VehicleParams) -> None:
    alpha = np.deg2rad(40.0)  # far past alpha_stall
    cl = lift_coefficient(alpha, mach=0.1, params=test_vehicle, compressibility=False)
    assert abs(float(cl)) <= test_vehicle.CLmax + 1e-6


@pytest.mark.req("FR-SIM-002")
def test_cl_continuous_across_stall_boundary(test_vehicle: VehicleParams) -> None:
    alphas = np.linspace(
        test_vehicle.alpha_stall_rad - 0.05, test_vehicle.alpha_stall_rad + 0.05, 200
    )
    cl = lift_coefficient(alphas, mach=0.1, params=test_vehicle, compressibility=False)
    assert np.max(np.abs(np.diff(cl))) < 0.05  # no jump/kink


@pytest.mark.req("FR-SIM-002")
def test_compressibility_increases_cl(test_vehicle: VehicleParams) -> None:
    alpha = np.deg2rad(5.0)
    cl_incompressible = lift_coefficient(
        alpha, mach=0.5, params=test_vehicle, compressibility=False
    )
    cl_compressible = lift_coefficient(alpha, mach=0.5, params=test_vehicle, compressibility=True)
    assert cl_compressible > cl_incompressible


@pytest.mark.req("FR-SIM-002")
def test_prandtl_glauert_capped_at_0_85() -> None:
    factor_at_cap = prandtl_glauert_factor(0.85)
    factor_above_cap = prandtl_glauert_factor(0.95)
    assert factor_at_cap == pytest.approx(factor_above_cap)


@pytest.mark.req("FR-SIM-002")
def test_wave_drag_zero_below_mdd(test_vehicle: VehicleParams) -> None:
    assert wave_drag_coefficient(0.3, test_vehicle) == pytest.approx(0.0)


@pytest.mark.req("FR-SIM-002")
def test_wave_drag_monotonic_rise_past_mdd(test_vehicle: VehicleParams) -> None:
    machs = np.linspace(test_vehicle.Mdd, test_vehicle.Mdd + 0.2, 50)
    cd_wave = wave_drag_coefficient(machs, test_vehicle)
    assert np.all(np.diff(cd_wave) >= 0)
    assert cd_wave[-1] > 0


@pytest.mark.req("FR-SIM-002")
def test_drag_coefficient_quadratic_in_cl(test_vehicle: VehicleParams) -> None:
    cd_zero_lift = drag_coefficient(0.0, mach=0.1, params=test_vehicle, wave_drag=False)
    assert cd_zero_lift == pytest.approx(test_vehicle.CD0)
    cd_at_cl = drag_coefficient(0.5, mach=0.1, params=test_vehicle, wave_drag=False)
    assert cd_at_cl == pytest.approx(test_vehicle.CD0 + test_vehicle.k * 0.25)


@pytest.mark.req("FR-SIM-002")
def test_drag_overrides_used_for_fault_injection(test_vehicle: VehicleParams) -> None:
    cd_default = drag_coefficient(0.3, mach=0.1, params=test_vehicle, wave_drag=False)
    cd_faulted = drag_coefficient(
        0.3, mach=0.1, params=test_vehicle, wave_drag=False, cd0_override=test_vehicle.CD0 * 1.3
    )
    assert cd_faulted > cd_default
