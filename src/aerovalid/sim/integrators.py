"""Fixed-step RK4 (default) and an adaptive wrapper around `scipy.integrate.solve_ivp`
(FR-SIM-005). `rhs(t, y, *args) -> dydt` is any right-hand-side callable returning an
array the same shape as `y`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

RhsFn = Callable[..., NDArray[np.float64]]


def rk4_step(
    rhs: RhsFn, t: float, y: NDArray[np.float64], dt: float, *args: object
) -> NDArray[np.float64]:
    """
    One classical 4th-order Runge-Kutta step.

    Parameters
    ----------
    rhs : callable
        `rhs(t, y, *args) -> dydt`, same shape as `y`.
    t : float
        Current time, s.
    y : ndarray, shape (n,)
        Current state.
    dt : float
        Step size, s.
    *args
        Extra positional arguments forwarded to `rhs`.

    Returns
    -------
    ndarray, shape (n,)
        State at `t + dt`.
    """
    k1 = rhs(t, y, *args)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4 = rhs(t + dt, y + dt * k3, *args)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_fixed_step(
    rhs: RhsFn,
    y0: NDArray[np.float64],
    t0: float,
    t1: float,
    dt: float,
    *args: object,
    on_step: Callable[[float, NDArray[np.float64]], NDArray[np.float64] | None] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Integrate `rhs` from `t0` to `t1` with fixed-step RK4.

    Parameters
    ----------
    rhs : callable
        `rhs(t, y, *args) -> dydt`.
    y0 : ndarray, shape (n,)
        Initial state at `t0`.
    t0, t1 : float
        Start/end time, s.
    dt : float
        Step size, s (the final step is shortened to land exactly on `t1`).
    *args
        Extra positional arguments forwarded to `rhs`.
    on_step : callable, optional
        `on_step(t, y) -> y_or_None`, called after every accepted step; if it
        returns an array, that value replaces `y` (used to clamp/patch state, e.g.
        algebraic angle-of-attack tracking in quasi-steady variants).

    Returns
    -------
    tuple[ndarray, ndarray]
        `times, shape (m,)` and `states, shape (m, n)`, including both endpoints.
    """
    n_steps = max(1, int(round((t1 - t0) / dt)))
    times = np.linspace(t0, t1, n_steps + 1)
    states = np.empty((n_steps + 1, y0.shape[0]), dtype=np.float64)
    states[0] = y0
    y = y0
    for i in range(n_steps):
        step_dt = times[i + 1] - times[i]
        y = rk4_step(rhs, times[i], y, step_dt, *args)
        if on_step is not None:
            patched = on_step(times[i + 1], y)
            if patched is not None:
                y = patched
        states[i + 1] = y
    return times, states


def integrate_adaptive(
    rhs: RhsFn, y0: NDArray[np.float64], t0: float, t1: float, *args: object, max_step: float = 1.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Integrate `rhs` from `t0` to `t1` with `scipy.integrate.solve_ivp` (RK45).

    Parameters
    ----------
    rhs : callable
        `rhs(t, y, *args) -> dydt`.
    y0 : ndarray, shape (n,)
        Initial state.
    t0, t1 : float
        Start/end time, s.
    *args
        Extra positional arguments forwarded to `rhs`.
    max_step : float, default 1.0
        Maximum adaptive step size, s.

    Returns
    -------
    tuple[ndarray, ndarray]
        `times, shape (m,)` and `states, shape (m, n)`.
    """
    solution = solve_ivp(
        rhs, (t0, t1), y0, args=args, method="RK45", max_step=max_step, dense_output=False
    )
    return solution.t, solution.y.T
