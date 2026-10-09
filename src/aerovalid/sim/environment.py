"""Environment (FR-SIM-004, sim spec §3, §4): steady wind + linear shear, seeded
Dryden-style turbulence, and the ISA temperature offset passed straight through to
`core/atmosphere.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class WindParams(BaseModel):
    """Steady wind + shear + turbulence configuration for one flight."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_n_mps: float = 0.0
    mean_e_mps: float = 0.0
    shear_per_m: float = 0.0  # linear increase in horizontal wind speed with altitude
    turbulence_rms_mps: float = 0.0
    turbulence_scale_length_m: float = 200.0


@dataclass
class DrydenTurbulence:
    """Seeded, first-order-filtered turbulence (a simplified Dryden/von-Karman model):
    white noise shaped by a single-pole low-pass with the von-Karman-style time
    constant `L / V`, giving the right RMS and a roughly correct low-frequency rise.

    Attributes
    ----------
    rms_mps : float
        Target turbulence RMS, m/s.
    scale_length_m : float
        Turbulence scale length L, m.
    rng : numpy.random.Generator
        Seeded generator (FR-COR-006); never shared across flights.
    """

    rms_mps: float
    scale_length_m: float
    rng: np.random.Generator
    _state_n: float = 0.0
    _state_e: float = 0.0
    _state_d: float = 0.0

    def step(self, dt_s: float, airspeed_mps: float) -> tuple[float, float, float]:
        """
        Advance the turbulence filter by `dt_s` and return `(gust_n, gust_e, gust_d)`.

        Parameters
        ----------
        dt_s : float
            Integration step, s.
        airspeed_mps : float
            Current true airspeed, used for the von-Karman time constant `L/V`, m/s.

        Returns
        -------
        tuple[float, float, float]
            Gust velocity components, m/s.
        """
        if self.rms_mps <= 0.0:
            return 0.0, 0.0, 0.0
        tau = self.scale_length_m / max(airspeed_mps, 1.0)
        alpha = dt_s / tau
        alpha = min(alpha, 1.0)
        # Discrete-time shaping filter driven by unit-variance white noise, scaled so
        # the stationary RMS of the filtered signal equals `rms_mps`.
        drive_std = self.rms_mps * np.sqrt(2.0 * alpha - alpha**2) if alpha < 2.0 else self.rms_mps
        noise = self.rng.normal(0.0, drive_std, size=3)
        self._state_n = (1.0 - alpha) * self._state_n + noise[0]
        self._state_e = (1.0 - alpha) * self._state_e + noise[1]
        self._state_d = (1.0 - alpha) * self._state_d + noise[2]
        return self._state_n, self._state_e, self._state_d


def steady_wind_with_shear(
    alt_m: NDArray[np.float64] | float, params: WindParams
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Steady wind with a linear altitude shear on the horizontal components.

    Parameters
    ----------
    alt_m : ndarray or float
        Altitude MSL, m.
    params : WindParams
        Wind configuration.

    Returns
    -------
    tuple[ndarray, ndarray]
        `(wind_n_mps, wind_e_mps)`.
    """
    alt = np.asarray(alt_m, dtype=np.float64)
    shear_factor = 1.0 + params.shear_per_m * alt
    return params.mean_n_mps * shear_factor, params.mean_e_mps * shear_factor
