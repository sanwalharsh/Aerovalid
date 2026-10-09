"""ISA atmosphere model (FR-COR-003): temperature, pressure, density, speed of sound,
and dynamic viscosity vs geometric altitude, 0-20 km, with an optional constant
temperature offset. Vectorised; humidity is ignored per the data spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_G0 = 9.80665  # m/s^2
_R_SPECIFIC = 287.052874  # J/(kg K), dry air
_GAMMA_AIR = 1.4
_SUTHERLAND_MU0 = 1.716e-5  # Pa s, reference viscosity at T0
_SUTHERLAND_T0 = 273.15  # K
_SUTHERLAND_S = 110.4  # K

# ISA layer base altitude (m), base temperature (K), lapse rate (K/m), base pressure (Pa)
# for the troposphere (0-11 km) and the first isothermal stratosphere layer (11-20 km).
_T0 = 288.15
_P0 = 101325.0
_LAPSE_TROPO = -0.0065
_H_TROPOPAUSE = 11000.0
_T_TROPOPAUSE = _T0 + _LAPSE_TROPO * _H_TROPOPAUSE
_P_TROPOPAUSE = _P0 * (_T_TROPOPAUSE / _T0) ** (-_G0 / (_LAPSE_TROPO * _R_SPECIFIC))
_H_STRATO_TOP = 20000.0


@dataclass(frozen=True)
class AtmosphereState:
    """ISA atmosphere properties at one or more altitudes.

    Attributes
    ----------
    temperature_k : ndarray, shape (N,)
        Static temperature, K.
    pressure_pa : ndarray, shape (N,)
        Static pressure, Pa.
    density_kgm3 : ndarray, shape (N,)
        Air density, kg/m^3.
    speed_of_sound_mps : ndarray, shape (N,)
        Speed of sound, m/s.
    dynamic_viscosity_pas : ndarray, shape (N,)
        Dynamic viscosity, Pa s (Sutherland's law).
    """

    temperature_k: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    density_kgm3: NDArray[np.float64]
    speed_of_sound_mps: NDArray[np.float64]
    dynamic_viscosity_pas: NDArray[np.float64]


def isa_atmosphere(alt_m: NDArray[np.float64] | float, delta_t_k: float = 0.0) -> AtmosphereState:
    """
    ISA atmosphere (0-20 km geometric altitude) with an optional temperature offset.

    Parameters
    ----------
    alt_m : ndarray or float, shape (N,)
        Geometric altitude, m. Values are clipped to [0, 20000] m (ISA is only
        defined for this project's two lowest layers).
    delta_t_k : float, default 0.0
        Constant temperature offset added uniformly at all altitudes, K.

    Returns
    -------
    AtmosphereState
        Temperature, pressure, density, speed of sound, viscosity at `alt_m`.

    References
    ----------
    FR-COR-003: matches the standard ISA table within 0.1% at 0, 5, 11, 15 km.
    """
    h = np.clip(np.asarray(alt_m, dtype=np.float64), 0.0, _H_STRATO_TOP)
    is_tropo = h <= _H_TROPOPAUSE

    temperature_isa = np.where(h <= _H_TROPOPAUSE, _T0 + _LAPSE_TROPO * h, _T_TROPOPAUSE)
    pressure = np.where(
        is_tropo,
        _P0 * (temperature_isa / _T0) ** (-_G0 / (_LAPSE_TROPO * _R_SPECIFIC)),
        _P_TROPOPAUSE * np.exp(-_G0 * (h - _H_TROPOPAUSE) / (_R_SPECIFIC * _T_TROPOPAUSE)),
    )

    temperature = temperature_isa + delta_t_k
    density = pressure / (_R_SPECIFIC * temperature)
    speed_of_sound = np.sqrt(_GAMMA_AIR * _R_SPECIFIC * temperature)
    viscosity = (
        _SUTHERLAND_MU0
        * (temperature / _SUTHERLAND_T0) ** 1.5
        * (_SUTHERLAND_T0 + _SUTHERLAND_S)
        / (temperature + _SUTHERLAND_S)
    )

    return AtmosphereState(
        temperature_k=temperature,
        pressure_pa=pressure,
        density_kgm3=density,
        speed_of_sound_mps=speed_of_sound,
        dynamic_viscosity_pas=viscosity,
    )


SEA_LEVEL = isa_atmosphere(0.0)
RHO0_KGM3 = float(SEA_LEVEL.density_kgm3)
