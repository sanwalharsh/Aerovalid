"""Propulsion (FR-SIM-003, sim spec §3): thrust vs throttle/altitude/Mach, and the
abstracted "fuel" (mass) flow used for mass depletion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aerovalid.core.atmosphere import RHO0_KGM3
from aerovalid.sim.protocols import VehicleParams


def thrust(
    throttle: NDArray[np.float64] | float,
    rho_kgm3: NDArray[np.float64] | float,
    mach: NDArray[np.float64] | float,
    params: VehicleParams,
    *,
    thrust_lapse: bool,
    t0_override: float | None = None,
) -> NDArray[np.float64]:
    """
    Thrust: `delta_t * T0 * (rho/rho0)^n * (1 - thrust_mach_factor * M)` (sim spec §3).

    Parameters
    ----------
    throttle : ndarray or float
        Throttle command, 0-1.
    rho_kgm3 : ndarray or float
        Local air density, kg/m^3.
    mach : ndarray or float
        Mach number.
    params : VehicleParams
        Vehicle propulsion parameters.
    thrust_lapse : bool
        If False, altitude lapse is disabled (V3 "atmosphere-naive", sim spec §5):
        thrust uses sea-level density regardless of `rho_kgm3`.
    t0_override : float, optional
        Fault-perturbed `T0` to use instead of `params.T0_N` (`thrust_loss` fault).

    Returns
    -------
    ndarray
        Thrust, N.
    """
    t0 = params.T0_N if t0_override is None else t0_override
    density_ratio = np.asarray(rho_kgm3, dtype=np.float64) / RHO0_KGM3 if thrust_lapse else 1.0
    lapse = density_ratio**params.thrust_lapse_n if thrust_lapse else 1.0
    mach_factor = 1.0 - params.thrust_mach_factor * np.asarray(mach, dtype=np.float64)
    return np.asarray(throttle, dtype=np.float64) * t0 * lapse * mach_factor


def fuel_flow_kgps(
    thrust_n: NDArray[np.float64] | float, params: VehicleParams
) -> NDArray[np.float64]:
    """
    Mass flow (abstracted "fuel") for mass depletion: `SFC * T` (sim spec §3).

    Parameters
    ----------
    thrust_n : ndarray or float
        Thrust, N.
    params : VehicleParams
        Vehicle parameters (`SFC_kg_per_Ns`).

    Returns
    -------
    ndarray
        Mass flow rate, kg/s (always >= 0).
    """
    return np.asarray(
        params.SFC_kg_per_Ns * np.asarray(thrust_n, dtype=np.float64), dtype=np.float64
    )
