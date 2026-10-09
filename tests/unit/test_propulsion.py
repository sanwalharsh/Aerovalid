from __future__ import annotations

import pytest

from aerovalid.core.atmosphere import RHO0_KGM3, isa_atmosphere
from aerovalid.sim.propulsion import fuel_flow_kgps, thrust
from aerovalid.sim.protocols import VehicleParams


@pytest.mark.req("FR-SIM-003")
def test_thrust_at_sea_level_full_throttle_zero_mach(test_vehicle: VehicleParams) -> None:
    t = thrust(1.0, RHO0_KGM3, mach=0.0, params=test_vehicle, thrust_lapse=True)
    assert t == pytest.approx(test_vehicle.T0_N, rel=1e-6)


@pytest.mark.req("FR-SIM-003")
def test_thrust_scales_linearly_with_throttle(test_vehicle: VehicleParams) -> None:
    t_half = thrust(0.5, RHO0_KGM3, mach=0.0, params=test_vehicle, thrust_lapse=True)
    t_full = thrust(1.0, RHO0_KGM3, mach=0.0, params=test_vehicle, thrust_lapse=True)
    assert t_half == pytest.approx(t_full / 2.0, rel=1e-9)


@pytest.mark.req("FR-SIM-003")
def test_thrust_lapse_matches_formula(test_vehicle: VehicleParams) -> None:
    rho_5000 = float(isa_atmosphere(5000.0).density_kgm3)
    t = thrust(1.0, rho_5000, mach=0.0, params=test_vehicle, thrust_lapse=True)
    expected = test_vehicle.T0_N * (rho_5000 / RHO0_KGM3) ** test_vehicle.thrust_lapse_n
    assert t == pytest.approx(expected, rel=1e-9)


@pytest.mark.req("FR-SIM-003")
def test_thrust_lapse_disabled_ignores_density(test_vehicle: VehicleParams) -> None:
    rho_5000 = float(isa_atmosphere(5000.0).density_kgm3)
    t_sea_level = thrust(1.0, RHO0_KGM3, mach=0.0, params=test_vehicle, thrust_lapse=False)
    t_altitude = thrust(1.0, rho_5000, mach=0.0, params=test_vehicle, thrust_lapse=False)
    assert t_sea_level == pytest.approx(t_altitude, rel=1e-9)


@pytest.mark.req("FR-SIM-003")
def test_thrust_decreases_with_mach(test_vehicle: VehicleParams) -> None:
    t_static = thrust(1.0, RHO0_KGM3, mach=0.0, params=test_vehicle, thrust_lapse=True)
    t_moving = thrust(1.0, RHO0_KGM3, mach=0.2, params=test_vehicle, thrust_lapse=True)
    assert t_moving < t_static


@pytest.mark.req("FR-SIM-003")
def test_t0_override_used_for_thrust_loss_fault(test_vehicle: VehicleParams) -> None:
    t_nominal = thrust(1.0, RHO0_KGM3, mach=0.0, params=test_vehicle, thrust_lapse=True)
    t_faulted = thrust(
        1.0,
        RHO0_KGM3,
        mach=0.0,
        params=test_vehicle,
        thrust_lapse=True,
        t0_override=test_vehicle.T0_N * 0.7,
    )
    assert t_faulted == pytest.approx(t_nominal * 0.7, rel=1e-9)


@pytest.mark.req("FR-SIM-003")
def test_fuel_flow_nonnegative_and_proportional_to_thrust(test_vehicle: VehicleParams) -> None:
    flow_low = fuel_flow_kgps(50.0, test_vehicle)
    flow_high = fuel_flow_kgps(100.0, test_vehicle)
    assert flow_low >= 0.0
    assert flow_high == pytest.approx(2.0 * flow_low, rel=1e-9)
