"""Point-mass 3-D flight dynamics (FR-SIM-001, sim spec §3): the state derivative and
derived measurements (nx, nz, ground speed, attitude, Mach, q). Two state vectors are
supported: a 7-element quasi-steady one (alpha and thrust track their commands
algebraically -- used by all variants) and a 9-element one with first-order alpha and
thrust-actuator lag (truth only, `StructuralSwitches.alpha_dynamics=True`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aerovalid.core.atmosphere import isa_atmosphere
from aerovalid.sim.aero import drag_coefficient, lift_coefficient
from aerovalid.sim.propulsion import thrust as compute_thrust
from aerovalid.sim.protocols import StructuralSwitches, VehicleParams

_G0 = 9.80665  # m/s^2

# State layouts.
QUASI_STEADY_DIM = 7  # x, y, h, V, gamma, chi, m
ACTUATOR_LAG_DIM = 9  # + alpha, T_actual


@dataclass(frozen=True)
class StepInputs:
    """Per-step control and environment inputs held fixed across one RK4 step."""

    throttle: float
    alpha_cmd_rad: float
    bank_cmd_rad: float
    wind_n_mps: float = 0.0
    wind_e_mps: float = 0.0
    wind_d_mps: float = 0.0
    delta_t_k: float = 0.0
    atmosphere_h_override_m: float | None = None
    """Fixed altitude to evaluate density/sound-speed at instead of the true `h_m`
    (V3 "atmosphere-naive", sim spec §5: constant rho = rho(h0))."""


@dataclass(frozen=True)
class FaultState:
    """Time-varying parameter multipliers the fault schedule applies to truth only."""

    cd0_mult: float = 1.0
    cla_mult: float = 1.0
    clmax_mult: float = 1.0
    t0_mult: float = 1.0
    alpha_gain: float = 1.0


_NO_FAULT = FaultState()


def _core_derivatives(
    h_m: float,
    v_mps: float,
    gamma_rad: float,
    chi_rad: float,
    m_kg: float,
    alpha_rad: float,
    thrust_n: float,
    inputs: StepInputs,
    vehicle: VehicleParams,
    switches: StructuralSwitches,
    faults: FaultState,
    density_kgm3: float,
    speed_of_sound_mps: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Shared translational equations of motion (sim spec §3). `density_kgm3` /
    `speed_of_sound_mps` are evaluated by the caller (at `h_m`, or at V3's frozen
    `atmosphere_h_override_m`). Returns
    `(xdot, ydot, hdot, Vdot, gammadot, chidot, mdot, mach, q_pa)`."""
    mach = v_mps / speed_of_sound_mps
    q_pa = 0.5 * density_kgm3 * v_mps**2

    cl = lift_coefficient(alpha_rad, mach, vehicle, compressibility=switches.compressibility)
    cl = cl * faults.cla_mult if faults.cla_mult != 1.0 else cl
    cl = np.clip(cl, -vehicle.CLmax * faults.clmax_mult, vehicle.CLmax * faults.clmax_mult)
    cd = drag_coefficient(
        cl, mach, vehicle, wave_drag=switches.wave_drag, cd0_override=vehicle.CD0 * faults.cd0_mult
    )
    lift_n = q_pa * vehicle.wing_area_m2 * float(cl)
    drag_n = q_pa * vehicle.wing_area_m2 * float(cd)

    phi = inputs.bank_cmd_rad
    weight_n = m_kg * _G0

    v_dot = (thrust_n * np.cos(alpha_rad) - drag_n - weight_n * np.sin(gamma_rad)) / m_kg
    gamma_dot = (
        (thrust_n * np.sin(alpha_rad) + lift_n) * np.cos(phi) - weight_n * np.cos(gamma_rad)
    ) / (m_kg * max(v_mps, 1e-6))
    chi_dot = ((thrust_n * np.sin(alpha_rad) + lift_n) * np.sin(phi)) / (
        m_kg * max(v_mps, 1e-6) * max(np.cos(gamma_rad), 1e-6)
    )

    x_dot = v_mps * np.cos(gamma_rad) * np.cos(chi_rad) + inputs.wind_n_mps
    y_dot = v_mps * np.cos(gamma_rad) * np.sin(chi_rad) + inputs.wind_e_mps
    h_dot = v_mps * np.sin(gamma_rad) - inputs.wind_d_mps

    m_dot = -vehicle.SFC_kg_per_Ns * thrust_n

    return (
        float(x_dot),
        float(y_dot),
        float(h_dot),
        float(v_dot),
        float(gamma_dot),
        float(chi_dot),
        float(m_dot),
        float(mach),
        float(q_pa),
    )


def rhs_quasi_steady(
    t: float,
    y: NDArray[np.float64],
    inputs: StepInputs,
    vehicle: VehicleParams,
    switches: StructuralSwitches,
    faults: FaultState = _NO_FAULT,
) -> NDArray[np.float64]:
    """
    State derivative for the 7-element quasi-steady state (all variants).

    `y = [x_m, y_m, h_m, V_mps, gamma_rad, chi_rad, m_kg]`; alpha tracks
    `inputs.alpha_cmd_rad` algebraically and thrust tracks its commanded value
    algebraically (no actuator lag).
    """
    x, y_pos, h, v, gamma, chi, m = y
    alpha = inputs.alpha_cmd_rad
    atm_h = inputs.atmosphere_h_override_m if inputs.atmosphere_h_override_m is not None else h
    atm = isa_atmosphere(atm_h, delta_t_k=inputs.delta_t_k if switches.isa_offset_known else 0.0)
    density = float(atm.density_kgm3)
    sound_speed = float(atm.speed_of_sound_mps)
    mach = v / sound_speed
    thrust_n = float(
        compute_thrust(
            inputs.throttle,
            density,
            mach,
            vehicle,
            thrust_lapse=switches.thrust_lapse,
            t0_override=vehicle.T0_N * faults.t0_mult if faults.t0_mult != 1.0 else None,
        )
    )
    x_dot, y_dot, h_dot, v_dot, gamma_dot, chi_dot, m_dot, _, _ = _core_derivatives(
        h,
        v,
        gamma,
        chi,
        m,
        alpha,
        thrust_n,
        inputs,
        vehicle,
        switches,
        faults,
        density,
        sound_speed,
    )
    return np.array([x_dot, y_dot, h_dot, v_dot, gamma_dot, chi_dot, m_dot], dtype=np.float64)


def rhs_actuator_lag(
    t: float,
    y: NDArray[np.float64],
    inputs: StepInputs,
    vehicle: VehicleParams,
    switches: StructuralSwitches,
    faults: FaultState = _NO_FAULT,
) -> NDArray[np.float64]:
    """
    State derivative for the 9-element truth state (first-order alpha + thrust lag).

    `y = [x_m, y_m, h_m, V_mps, gamma_rad, chi_rad, m_kg, alpha_rad, thrust_n]`.
    """
    x, y_pos, h, v, gamma, chi, m, alpha, thrust_n = y
    atm_h = inputs.atmosphere_h_override_m if inputs.atmosphere_h_override_m is not None else h
    atm = isa_atmosphere(atm_h, delta_t_k=inputs.delta_t_k if switches.isa_offset_known else 0.0)
    density = float(atm.density_kgm3)
    sound_speed = float(atm.speed_of_sound_mps)
    mach = v / sound_speed
    thrust_cmd = float(
        compute_thrust(
            inputs.throttle,
            density,
            mach,
            vehicle,
            thrust_lapse=switches.thrust_lapse,
            t0_override=vehicle.T0_N * faults.t0_mult if faults.t0_mult != 1.0 else None,
        )
    )
    alpha_dot = faults.alpha_gain * (inputs.alpha_cmd_rad - alpha) / vehicle.tau_alpha_s
    thrust_dot = (thrust_cmd - thrust_n) / vehicle.tau_thrust_s

    x_dot, y_dot, h_dot, v_dot, gamma_dot, chi_dot, m_dot, _, _ = _core_derivatives(
        h,
        v,
        gamma,
        chi,
        m,
        alpha,
        thrust_n,
        inputs,
        vehicle,
        switches,
        faults,
        density,
        sound_speed,
    )
    return np.array(
        [x_dot, y_dot, h_dot, v_dot, gamma_dot, chi_dot, m_dot, alpha_dot, thrust_dot],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class DerivedMeasurements:
    """Instantaneous measurements derived from a state + inputs (sim spec §3)."""

    nx_g: float
    nz_g: float
    vel_gnd_mps: float
    pitch_rad: float
    roll_rad: float
    yaw_rad: float
    mach: float
    q_pa: float


def derived_measurements(
    h_m: float,
    v_mps: float,
    gamma_rad: float,
    chi_rad: float,
    m_kg: float,
    alpha_rad: float,
    thrust_n: float,
    inputs: StepInputs,
    vehicle: VehicleParams,
    switches: StructuralSwitches,
    faults: FaultState = _NO_FAULT,
) -> DerivedMeasurements:
    """
    Compute nx, nz, ground speed, attitude, Mach, q at one instant (sim spec §3).

    Parameters mirror `_core_derivatives`; `thrust_n` is the actual thrust (lagged
    for truth, commanded for variants).

    Returns
    -------
    DerivedMeasurements
    """
    atm_h = inputs.atmosphere_h_override_m if inputs.atmosphere_h_override_m is not None else h_m
    atm = isa_atmosphere(atm_h, delta_t_k=inputs.delta_t_k if switches.isa_offset_known else 0.0)
    mach = v_mps / float(atm.speed_of_sound_mps)
    q_pa = 0.5 * float(atm.density_kgm3) * v_mps**2

    cl = lift_coefficient(alpha_rad, mach, vehicle, compressibility=switches.compressibility)
    cl = np.clip(cl, -vehicle.CLmax, vehicle.CLmax)
    lift_n = q_pa * vehicle.wing_area_m2 * float(cl)
    drag_n = (
        q_pa
        * vehicle.wing_area_m2
        * float(drag_coefficient(cl, mach, vehicle, wave_drag=switches.wave_drag))
    )

    weight_n = m_kg * _G0
    nx_g = (thrust_n * np.cos(alpha_rad) - drag_n) / weight_n
    nz_g = (thrust_n * np.sin(alpha_rad) + lift_n) / weight_n

    v_n = v_mps * np.cos(gamma_rad) * np.cos(chi_rad) + inputs.wind_n_mps
    v_e = v_mps * np.cos(gamma_rad) * np.sin(chi_rad) + inputs.wind_e_mps
    ground_speed = float(np.hypot(v_n, v_e))

    pitch_rad = gamma_rad + alpha_rad
    roll_rad = inputs.bank_cmd_rad
    yaw_rad = chi_rad

    return DerivedMeasurements(
        nx_g=float(nx_g),
        nz_g=float(nz_g),
        vel_gnd_mps=ground_speed,
        pitch_rad=float(pitch_rad),
        roll_rad=float(roll_rad),
        yaw_rad=float(yaw_rad),
        mach=float(mach),
        q_pa=float(q_pa),
    )
