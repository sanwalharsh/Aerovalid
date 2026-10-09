from __future__ import annotations

import pytest

from aerovalid.sim.controllers import (
    ControllerStack,
    PIController,
    PIDController,
    Setpoint,
    rate_limit,
)
from aerovalid.sim.protocols import State, VehicleParams


@pytest.mark.req("FR-SIM-011")
def test_pi_controller_converges_toward_zero_error() -> None:
    pi = PIController(kp=0.5, ki=0.1, output_min=-10.0, output_max=10.0)
    error = 5.0
    outputs = [pi.step(error, 0.1) for _ in range(20)]
    assert outputs[-1] > outputs[0]  # integral action increases the output over time


@pytest.mark.req("FR-SIM-011")
def test_pi_controller_respects_output_bounds() -> None:
    pi = PIController(kp=10.0, ki=10.0, output_min=0.0, output_max=1.0)
    for _ in range(50):
        output = pi.step(100.0, 0.1)
    assert 0.0 <= output <= 1.0


@pytest.mark.req("FR-SIM-011")
def test_pid_controller_derivative_reacts_to_changing_error() -> None:
    pid = PIDController(kp=1.0, ki=0.0, kd=1.0, output_min=-100.0, output_max=100.0)
    first = pid.step(1.0, 0.1)
    second = pid.step(1.0, 0.1)  # error unchanged -> derivative term should vanish
    assert second == pytest.approx(first, abs=1e-9)


@pytest.mark.req("FR-SIM-011")
def test_rate_limit_caps_change_per_step() -> None:
    result = rate_limit(target=10.0, current=0.0, max_rate=2.0, dt_s=1.0)
    assert result == pytest.approx(2.0)


@pytest.mark.req("FR-SIM-011")
def test_rate_limit_reaches_target_when_within_budget() -> None:
    result = rate_limit(target=1.0, current=0.9, max_rate=5.0, dt_s=1.0)
    assert result == pytest.approx(1.0)


@pytest.mark.req("FR-SIM-011")
def test_controller_stack_tracks_speed_and_altitude_setpoint(test_vehicle: VehicleParams) -> None:
    stack = ControllerStack(vehicle=test_vehicle)
    state = State(
        t_s=0.0, x_m=0.0, y_m=0.0, h_m=1990.0, V_mps=31.0, gamma_rad=0.0, chi_rad=0.0, m_kg=25.0
    )
    setpoint = Setpoint(v_cmd_mps=32.0, h_cmd_m=2000.0, bank_cmd_rad=0.0)
    throttle, alpha_cmd, bank_cmd = stack.step(state, setpoint, dt_s=0.1)
    assert 0.0 <= throttle <= 1.0
    assert alpha_cmd > 0.0  # climbing toward a higher altitude setpoint needs positive alpha
    assert bank_cmd == pytest.approx(0.0)


@pytest.mark.req("FR-SIM-011")
def test_controller_stack_bank_command_is_rate_limited(test_vehicle: VehicleParams) -> None:
    stack = ControllerStack(vehicle=test_vehicle, bank_rate_limit_radps=0.1)
    state = State(
        t_s=0.0, x_m=0.0, y_m=0.0, h_m=2000.0, V_mps=32.0, gamma_rad=0.0, chi_rad=0.0, m_kg=25.0
    )
    setpoint = Setpoint(v_cmd_mps=32.0, h_cmd_m=2000.0, bank_cmd_rad=0.5)
    _, _, bank_cmd = stack.step(state, setpoint, dt_s=0.1)
    assert bank_cmd == pytest.approx(0.01, abs=1e-9)  # 0.1 rad/s * 0.1 s
