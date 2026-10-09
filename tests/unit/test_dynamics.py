from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from aerovalid.sim.aero import drag_coefficient, lift_coefficient
from aerovalid.sim.dynamics import (
    FaultState,
    StepInputs,
    derived_measurements,
    rhs_quasi_steady,
)
from aerovalid.sim.integrators import integrate_fixed_step
from aerovalid.sim.protocols import StructuralSwitches, VehicleParams

_G0 = 9.80665
_SWITCHES = StructuralSwitches(
    compressibility=False,
    wind_rate_terms=False,
    alpha_dynamics=False,
    thrust_lapse=True,
    isa_offset_known=False,
    wave_drag=False,
)


def _trim_alpha_and_throttle(
    vehicle: VehicleParams, v_mps: float, h_m: float, bank_rad: float = 0.0
) -> tuple[float, float]:
    """Solve for (alpha, throttle) holding steady flight at (V, h, bank) -- test-only
    helper mirroring sim spec §10's "trim throttle/alpha computed analytically"."""
    from aerovalid.core.atmosphere import isa_atmosphere
    from aerovalid.sim.propulsion import thrust as compute_thrust

    atm = isa_atmosphere(h_m)
    mach = v_mps / float(atm.speed_of_sound_mps)
    q_pa = 0.5 * float(atm.density_kgm3) * v_mps**2
    target_vertical_n = vehicle.mass_nom_kg * _G0 / np.cos(bank_rad)

    def residual(alpha: float) -> float:
        cl = float(lift_coefficient(alpha, mach, vehicle, compressibility=False))
        drag_n = (
            q_pa
            * vehicle.wing_area_m2
            * float(drag_coefficient(cl, mach, vehicle, wave_drag=False))
        )
        lift_n = q_pa * vehicle.wing_area_m2 * cl
        # From T*cos(a)=D and T*sin(a) + L = target: tan(a) = (target - L) / D
        return np.tan(alpha) * drag_n - (target_vertical_n - lift_n)

    alpha = brentq(residual, -0.2, 0.3, xtol=1e-10)
    cl = float(lift_coefficient(alpha, mach, vehicle, compressibility=False))
    drag_n = (
        q_pa * vehicle.wing_area_m2 * float(drag_coefficient(cl, mach, vehicle, wave_drag=False))
    )
    thrust_required_n = drag_n / np.cos(alpha)
    throttle = thrust_required_n / float(
        compute_thrust(1.0, float(atm.density_kgm3), mach, vehicle, thrust_lapse=True)
    )
    return alpha, throttle


@pytest.mark.req("FR-SIM-001")
def test_ballistic_matches_analytic_parabola(test_vehicle: VehicleParams) -> None:
    zero_lift_vehicle = test_vehicle.model_copy(
        update={"CL0": 0.0, "CLa": 0.0, "CD0": 0.0, "k": 0.0, "T0_N": 0.0}
    )
    v0, gamma0, h0 = 50.0, 0.1, 1000.0
    y0 = np.array([0.0, 0.0, h0, v0, gamma0, 0.0, zero_lift_vehicle.mass_nom_kg])
    inputs = StepInputs(throttle=0.0, alpha_cmd_rad=0.0, bank_cmd_rad=0.0)

    duration_s, dt = 5.0, 0.001
    _, states = integrate_fixed_step(
        rhs_quasi_steady,
        y0,
        0.0,
        duration_s,
        dt,
        inputs,
        zero_lift_vehicle,
        _SWITCHES,
        FaultState(),
    )
    x_final, h_final = states[-1, 0], states[-1, 2]

    vx0, vz0 = v0 * np.cos(gamma0), v0 * np.sin(gamma0)
    x_exact = vx0 * duration_s
    h_exact = h0 + vz0 * duration_s - 0.5 * _G0 * duration_s**2

    assert x_final == pytest.approx(x_exact, rel=1e-6)
    assert h_final == pytest.approx(h_exact, rel=1e-6)


@pytest.mark.req("FR-SIM-001")
def test_level_trim_holds_altitude_and_speed_for_60s(test_vehicle: VehicleParams) -> None:
    v_trim, h_trim = 32.0, 2000.0
    alpha, throttle = _trim_alpha_and_throttle(test_vehicle, v_trim, h_trim)
    assert 0.0 < throttle <= 1.0

    y0 = np.array([0.0, 0.0, h_trim, v_trim, 0.0, 0.0, test_vehicle.mass_nom_kg])
    inputs = StepInputs(throttle=throttle, alpha_cmd_rad=alpha, bank_cmd_rad=0.0)
    _, states = integrate_fixed_step(
        rhs_quasi_steady, y0, 0.0, 60.0, 0.01, inputs, test_vehicle, _SWITCHES, FaultState()
    )

    h_final, v_final = states[-1, 2], states[-1, 3]
    assert h_final == pytest.approx(h_trim, rel=1e-3)
    assert v_final == pytest.approx(v_trim, rel=1e-3)


@pytest.mark.req("FR-SIM-001")
def test_steady_turn_radius_matches_formula(test_vehicle: VehicleParams) -> None:
    v_trim, h_trim, bank = 32.0, 2000.0, np.deg2rad(30.0)
    alpha, throttle = _trim_alpha_and_throttle(test_vehicle, v_trim, h_trim, bank_rad=bank)

    y0 = np.array([0.0, 0.0, h_trim, v_trim, 0.0, 0.0, test_vehicle.mass_nom_kg])
    inputs = StepInputs(throttle=throttle, alpha_cmd_rad=alpha, bank_cmd_rad=bank)
    times, states = integrate_fixed_step(
        rhs_quasi_steady, y0, 0.0, 10.0, 0.01, inputs, test_vehicle, _SWITCHES, FaultState()
    )

    chi = states[:, 5]
    chi_dot_numeric = np.gradient(chi, times)
    chi_dot_steady = float(np.median(chi_dot_numeric[len(chi_dot_numeric) // 2 :]))

    expected_chi_dot = _G0 * np.tan(bank) / v_trim
    expected_radius = v_trim**2 / (_G0 * np.tan(bank))
    numeric_radius = v_trim / chi_dot_steady

    assert chi_dot_steady == pytest.approx(expected_chi_dot, rel=5e-3)
    assert numeric_radius == pytest.approx(expected_radius, rel=5e-3)

    measurements = derived_measurements(
        h_trim,
        v_trim,
        0.0,
        0.0,
        test_vehicle.mass_nom_kg,
        alpha,
        throttle * test_vehicle.T0_N,
        inputs,
        test_vehicle,
        _SWITCHES,
    )
    assert measurements.nz_g == pytest.approx(1.0 / np.cos(bank), rel=5e-3)


@pytest.mark.req("FR-SIM-007")
def test_atmosphere_h_override_freezes_density_regardless_of_true_altitude(
    test_vehicle: VehicleParams,
) -> None:
    """V3 "atmosphere-naive" mechanism: with `atmosphere_h_override_m` set, the
    derivative at two different true altitudes must be identical."""
    y_low = np.array([0.0, 0.0, 500.0, 32.0, 0.0, 0.0, test_vehicle.mass_nom_kg])
    y_high = np.array([0.0, 0.0, 4000.0, 32.0, 0.0, 0.0, test_vehicle.mass_nom_kg])
    inputs = StepInputs(
        throttle=0.6, alpha_cmd_rad=0.05, bank_cmd_rad=0.0, atmosphere_h_override_m=500.0
    )

    dydt_low = rhs_quasi_steady(0.0, y_low, inputs, test_vehicle, _SWITCHES, FaultState())
    dydt_high = rhs_quasi_steady(0.0, y_high, inputs, test_vehicle, _SWITCHES, FaultState())

    # V_dot depends only on the (frozen) atmosphere, not on the true h in the state.
    assert dydt_low[3] == pytest.approx(dydt_high[3], rel=1e-9)

    # Without the override, the two altitudes give genuinely different V_dot.
    inputs_no_override = StepInputs(throttle=0.6, alpha_cmd_rad=0.05, bank_cmd_rad=0.0)
    dydt_low_real = rhs_quasi_steady(
        0.0, y_low, inputs_no_override, test_vehicle, _SWITCHES, FaultState()
    )
    dydt_high_real = rhs_quasi_steady(
        0.0, y_high, inputs_no_override, test_vehicle, _SWITCHES, FaultState()
    )
    assert dydt_low_real[3] != pytest.approx(dydt_high_real[3], rel=1e-9)


@pytest.mark.req("FR-SIM-001")
def test_energy_balance_within_tolerance(test_vehicle: VehicleParams) -> None:
    """Specific energy change should equal the work of thrust minus drag (no climb)."""
    v_trim, h_trim = 32.0, 2000.0
    alpha, throttle = _trim_alpha_and_throttle(test_vehicle, v_trim, h_trim)
    y0 = np.array([0.0, 0.0, h_trim, v_trim, 0.0, 0.0, test_vehicle.mass_nom_kg])
    inputs = StepInputs(throttle=throttle, alpha_cmd_rad=alpha, bank_cmd_rad=0.0)
    _, states = integrate_fixed_step(
        rhs_quasi_steady, y0, 0.0, 20.0, 0.01, inputs, test_vehicle, _SWITCHES, FaultState()
    )

    h0, v0 = states[0, 2], states[0, 3]
    h1, v1 = states[-1, 2], states[-1, 3]
    specific_energy_0 = _G0 * h0 + 0.5 * v0**2
    specific_energy_1 = _G0 * h1 + 0.5 * v1**2
    # At trim, net work is ~0 by construction; energy should stay essentially flat.
    assert specific_energy_1 == pytest.approx(specific_energy_0, rel=1e-3)
