"""Controller stack (FR-SIM-011, sim spec §3, §8): speed-hold PI on throttle,
altitude-hold PID on angle-of-attack command, and a rate-limited bank tracker. Used by
the truth model's closed-loop run to turn a scenario's manoeuvre setpoints into
control histories (throttle, alpha_cmd, bank_cmd).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerovalid.sim.protocols import State, VehicleParams


@dataclass
class PIController:
    """Anti-windup PI controller."""

    kp: float
    ki: float
    output_min: float
    output_max: float
    integral: float = 0.0

    def step(self, error: float, dt_s: float) -> float:
        """Advance by `dt_s` given the current `error`; returns the clipped output."""
        candidate_integral = self.integral + error * dt_s
        raw_output = self.kp * error + self.ki * candidate_integral
        output = float(np.clip(raw_output, self.output_min, self.output_max))
        if output == raw_output:
            self.integral = candidate_integral  # only integrate while not saturated
        return output


@dataclass
class PIDController:
    """Anti-windup PID controller."""

    kp: float
    ki: float
    kd: float
    output_min: float
    output_max: float
    integral: float = 0.0
    prev_error: float | None = None

    def step(self, error: float, dt_s: float) -> float:
        """Advance by `dt_s` given the current `error`; returns the clipped output."""
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt_s
        candidate_integral = self.integral + error * dt_s
        raw_output = self.kp * error + self.ki * candidate_integral + self.kd * derivative
        output = float(np.clip(raw_output, self.output_min, self.output_max))
        if output == raw_output:
            self.integral = candidate_integral
        self.prev_error = error
        return output


def rate_limit(target: float, current: float, max_rate: float, dt_s: float) -> float:
    """
    Move `current` toward `target` at most `max_rate` per second over `dt_s`.

    Parameters
    ----------
    target : float
        Desired value.
    current : float
        Current value.
    max_rate : float
        Maximum rate of change, units/s.
    dt_s : float
        Time step, s.

    Returns
    -------
    float
        `current`, moved toward `target` by at most `max_rate * dt_s`.
    """
    max_delta = max_rate * dt_s
    delta = float(np.clip(target - current, -max_delta, max_delta))
    return current + delta


@dataclass
class Setpoint:
    """Commanded targets for one controller step (`sim/maneuvers.py` produces these)."""

    v_cmd_mps: float
    h_cmd_m: float
    bank_cmd_rad: float
    alpha_cmd_override_rad: float | None = None
    throttle_override: float | None = None


@dataclass
class ControllerStack:
    """Speed/altitude/bank controller stack used to fly a scenario closed-loop."""

    vehicle: VehicleParams
    speed_pi: PIController = field(
        default_factory=lambda: PIController(kp=0.08, ki=0.02, output_min=0.0, output_max=1.0)
    )
    altitude_pid: PIDController = field(
        default_factory=lambda: PIDController(
            kp=0.015, ki=0.0015, kd=0.01, output_min=-0.20, output_max=0.20
        )
    )
    bank_rate_limit_radps: float = np.deg2rad(20.0)
    _current_bank_rad: float = 0.0

    def step(self, state: State, setpoint: Setpoint, dt_s: float) -> tuple[float, float, float]:
        """
        Advance the controller stack by `dt_s`; returns `(throttle, alpha_cmd_rad, bank_cmd_rad)`.

        Parameters
        ----------
        state : State
            Current vehicle state.
        setpoint : Setpoint
            This instant's commanded targets (from the active manoeuvre segment).
        dt_s : float
            Time step, s.
        """
        if setpoint.throttle_override is not None:
            throttle = setpoint.throttle_override
        else:
            throttle = self.speed_pi.step(setpoint.v_cmd_mps - state.V_mps, dt_s)

        if setpoint.alpha_cmd_override_rad is not None:
            alpha_cmd = setpoint.alpha_cmd_override_rad
        else:
            alpha_cmd = self.altitude_pid.step(setpoint.h_cmd_m - state.h_m, dt_s)
            alpha_cmd = float(
                np.clip(alpha_cmd, -self.vehicle.alpha_stall_rad, self.vehicle.alpha_stall_rad)
            )

        bank_target = float(
            np.clip(
                setpoint.bank_cmd_rad, -self.vehicle.bank_limit_rad, self.vehicle.bank_limit_rad
            )
        )
        self._current_bank_rad = rate_limit(
            bank_target, self._current_bank_rad, self.bank_rate_limit_radps, dt_s
        )
        return throttle, alpha_cmd, self._current_bank_rad
