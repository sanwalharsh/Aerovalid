"""Derived signals and smoothed derivatives (FR-PRP-005, FR-PRP-006): Mach, dynamic
pressure, climb/turn rate, specific energy/power, effective CL/CD, and Savitzky-Golay
derivatives for noisy rate signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from aerovalid.core.atmosphere import isa_atmosphere

_G0 = 9.80665


def savgol_derivative(
    x: pd.Series, dt: float, window: int = 11, order: int = 3, deriv: int = 1
) -> np.ndarray:
    """
    Savitzky-Golay derivative of a (possibly noisy) signal on a regular grid.

    Parameters
    ----------
    x : pandas.Series
    dt : float
        Grid spacing, s.
    window : int, default 11
        Filter window length, samples (odd; reduced automatically if longer than `x`).
    order : int, default 3
        Polynomial order (must be < window).
    deriv : int, default 1
        Derivative order (0 = pure smoothing).

    Returns
    -------
    ndarray
        Same length as `x`.
    """
    values = x.to_numpy(dtype=np.float64)
    win = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    win = max(win, order + 2 if (order + 2) % 2 == 1 else order + 3)
    if win > len(values):
        return np.gradient(values, dt) if deriv == 1 else np.zeros_like(values)
    polyorder = min(order, win - 1)
    filtered = savgol_filter(values, window_length=win, polyorder=polyorder, deriv=deriv, delta=dt)
    return np.asarray(filtered)


def add_derived_signals(
    df: pd.DataFrame,
    dt: float,
    sg_window: int = 11,
    sg_order: int = 3,
    wing_area_m2: float | None = None,
) -> pd.DataFrame:
    """
    Add Mach/q (if missing), climb/turn rate, specific energy/power, and effective
    CL to a resampled table (FR-PRP-005, FR-PRP-006).

    Parameters
    ----------
    df : pandas.DataFrame
        Must have `alt_m`, `vel_air_mps`; `gamma_rad`/`chi_rad`/`mass_kg`/`nz_g` are
        used when present.
    dt : float
        Grid spacing, s (regular grid assumed, e.g. `prep/resample.py`'s output).
    sg_window, sg_order : Savitzky-Golay parameters (`preprocess.sg` config).
    wing_area_m2 : float, optional
        Vehicle wing area; `cl_eff = (nz_g * mass_kg * g) / (q_pa * wing_area_m2)` is
        only added when this is given (otherwise there is no principled way to turn
        a load factor into a lift *coefficient*, only a lift force).

    Returns
    -------
    pandas.DataFrame
        `df` plus `mach`, `q_pa`, `climb_rate_mps`, `turn_rate_radps`,
        `specific_energy_j_per_kg`, `specific_power_w_per_kg`, and `cl_eff` (whichever
        inputs/`wing_area_m2` are available).
    """
    out = df.copy()
    atm = isa_atmosphere(out["alt_m"].to_numpy(dtype=np.float64))

    if "mach" not in out.columns:
        out["mach"] = out["vel_air_mps"].to_numpy() / atm.speed_of_sound_mps
    if "q_pa" not in out.columns:
        out["q_pa"] = 0.5 * atm.density_kgm3 * out["vel_air_mps"].to_numpy() ** 2

    out["climb_rate_mps"] = savgol_derivative(out["alt_m"], dt, sg_window, sg_order, deriv=1)

    if "chi_rad" in out.columns:
        chi_unwrapped = np.unwrap(out["chi_rad"].to_numpy(dtype=np.float64))
        out["turn_rate_radps"] = savgol_derivative(
            pd.Series(chi_unwrapped), dt, sg_window, sg_order, deriv=1
        )

    specific_energy = _G0 * out["alt_m"].to_numpy() + 0.5 * out["vel_air_mps"].to_numpy() ** 2
    out["specific_energy_j_per_kg"] = specific_energy
    out["specific_power_w_per_kg"] = savgol_derivative(
        pd.Series(specific_energy), dt, sg_window, sg_order, deriv=1
    )

    if wing_area_m2 is not None and "nz_g" in out.columns and "mass_kg" in out.columns:
        weight_n = out["mass_kg"].to_numpy() * _G0
        lift_n = out["nz_g"].to_numpy() * weight_n
        out["cl_eff"] = lift_n / np.maximum(out["q_pa"].to_numpy() * wing_area_m2, 1e-6)

    return out
