from __future__ import annotations

import numpy as np
import pytest

from aerovalid.sim.integrators import integrate_adaptive, integrate_fixed_step, rk4_step


def _exponential_decay_rhs(t: float, y: np.ndarray, k: float) -> np.ndarray:
    return -k * y


@pytest.mark.req("FR-SIM-005")
def test_rk4_single_step_matches_analytic_for_linear_ode() -> None:
    k = 1.5
    y0 = np.array([2.0])
    y1 = rk4_step(_exponential_decay_rhs, 0.0, y0, 0.01, k)
    assert y1[0] == pytest.approx(2.0 * np.exp(-k * 0.01), rel=1e-6)


@pytest.mark.req("FR-SIM-005")
def test_rk4_convergence_order_is_approximately_4() -> None:
    k = 2.0
    y0 = np.array([1.0])
    t0, t1 = 0.0, 1.0
    dts = [0.08, 0.04, 0.02, 0.01]
    errors = []
    for dt in dts:
        _, states = integrate_fixed_step(_exponential_decay_rhs, y0, t0, t1, dt, k)
        errors.append(abs(states[-1, 0] - np.exp(-k * t1)))

    slope, _ = np.polyfit(np.log(dts), np.log(errors), 1)
    assert slope == pytest.approx(4.0, abs=0.3)


@pytest.mark.req("FR-SIM-005")
def test_integrate_fixed_step_includes_both_endpoints() -> None:
    times, states = integrate_fixed_step(
        _exponential_decay_rhs, np.array([1.0]), 0.0, 1.0, 0.1, 1.0
    )
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)
    assert states.shape == (len(times), 1)


@pytest.mark.req("FR-SIM-005")
def test_on_step_hook_can_patch_state() -> None:
    def clamp_to_zero(t: float, y: np.ndarray) -> np.ndarray:
        return np.zeros_like(y) if t > 0.5 else None

    _, states = integrate_fixed_step(
        _exponential_decay_rhs, np.array([1.0]), 0.0, 1.0, 0.1, 1.0, on_step=clamp_to_zero
    )
    assert states[-1, 0] == pytest.approx(0.0)


@pytest.mark.req("FR-SIM-005")
def test_adaptive_integrator_runs_and_matches_analytic() -> None:
    k = 1.0
    times, states = integrate_adaptive(
        _exponential_decay_rhs, np.array([1.0]), 0.0, 2.0, k, max_step=0.1
    )
    assert states[-1, 0] == pytest.approx(np.exp(-k * times[-1]), rel=1e-3)
