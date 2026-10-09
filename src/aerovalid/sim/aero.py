"""Aerodynamics (FR-SIM-002, sim spec §3): lift and drag coefficients with a smooth
stall roll-off, optional Prandtl-Glauert compressibility correction, and wave drag.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aerovalid.sim.protocols import VehicleParams

_PG_MACH_CAP = 0.85
_STALL_BLEND_WIDTH_RAD = 0.035  # ~2 deg, smooths the tanh roll-off around alpha_stall


def prandtl_glauert_factor(mach: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """
    Prandtl-Glauert compressibility factor `1/sqrt(1 - M^2)`, capped at `M = 0.85`.

    Parameters
    ----------
    mach : ndarray or float
        Mach number, dimensionless.

    Returns
    -------
    ndarray or float
        Compressibility factor, >= 1.
    """
    m_capped = np.minimum(np.asarray(mach, dtype=np.float64), _PG_MACH_CAP)
    return 1.0 / np.sqrt(1.0 - m_capped**2)


def lift_coefficient(
    alpha_rad: NDArray[np.float64] | float,
    mach: NDArray[np.float64] | float,
    params: VehicleParams,
    *,
    compressibility: bool,
) -> NDArray[np.float64]:
    """
    Lift coefficient CL(alpha, M) with a smooth stall roll-off (sim spec §3).

    Parameters
    ----------
    alpha_rad : ndarray or float
        Angle of attack, rad.
    mach : ndarray or float
        Mach number.
    params : VehicleParams
        Vehicle aerodynamic parameters.
    compressibility : bool
        If True, apply the Prandtl-Glauert factor (truth and V4 only, sim spec §3).

    Returns
    -------
    ndarray
        Lift coefficient, same shape as `alpha_rad`.
    """
    alpha = np.asarray(alpha_rad, dtype=np.float64)
    linear_cl = params.CL0 + params.CLa * alpha

    # Smooth blend to a post-stall plateau near +-CLmax using a tanh roll-off, so CL
    # is continuous (and differentiable) across alpha_stall rather than kinked.
    over_stall = np.abs(alpha) - params.alpha_stall_rad
    blend = 0.5 * (1.0 - np.tanh(over_stall / _STALL_BLEND_WIDTH_RAD))
    sign = np.sign(np.where(alpha == 0, 1.0, alpha))
    plateau_cl = sign * params.CLmax
    cl = blend * linear_cl + (1.0 - blend) * plateau_cl

    if compressibility:
        cl = cl * prandtl_glauert_factor(mach)
    return np.asarray(cl, dtype=np.float64)


def wave_drag_coefficient(
    mach: NDArray[np.float64] | float, params: VehicleParams
) -> NDArray[np.float64]:
    """
    Wave drag rise past the drag-divergence Mach number: `coeff * max(0, M - Mdd)^4`.

    Parameters
    ----------
    mach : ndarray or float
        Mach number.
    params : VehicleParams
        Vehicle parameters (`Mdd`, `wave_drag_coeff`).

    Returns
    -------
    ndarray
        Non-negative, monotonically non-decreasing in `mach` past `Mdd`.
    """
    excess = np.maximum(0.0, np.asarray(mach, dtype=np.float64) - params.Mdd)
    return np.asarray(params.wave_drag_coeff * excess**4, dtype=np.float64)


def drag_coefficient(
    cl: NDArray[np.float64] | float,
    mach: NDArray[np.float64] | float,
    params: VehicleParams,
    *,
    wave_drag: bool,
    cd0_override: float | None = None,
    k_override: float | None = None,
) -> NDArray[np.float64]:
    """
    Drag coefficient `CD0 + k*CL^2 [+ wave drag]` (sim spec §3).

    Parameters
    ----------
    cl : ndarray or float
        Lift coefficient.
    mach : ndarray or float
        Mach number.
    params : VehicleParams
        Vehicle parameters.
    wave_drag : bool
        If True, add `wave_drag_coefficient(mach, params)` (truth and V4 only).
    cd0_override, k_override : float, optional
        Fault-perturbed values to use instead of `params.CD0` / `params.k`
        (`telemetry/faults.py` injects these as time-varying parameters).

    Returns
    -------
    ndarray
        Drag coefficient, same shape as `cl`.
    """
    cd0 = params.CD0 if cd0_override is None else cd0_override
    k = params.k if k_override is None else k_override
    cl_arr = np.asarray(cl, dtype=np.float64)
    cd = cd0 + k * cl_arr**2
    if wave_drag:
        cd = cd + wave_drag_coefficient(mach, params)
    return np.asarray(cd, dtype=np.float64)
