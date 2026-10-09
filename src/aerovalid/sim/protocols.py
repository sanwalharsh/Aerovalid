"""Shared simulator types (data spec §3): `State`, `ControlHistory`, `SimResult`, the
`SimulatorProtocol` every simulator (truth, variant, or a future external plug-in)
implements, and `VehicleParams`/`StructuralSwitches` loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from aerovalid.core.errors import ConfigError, SimulationError


class VehicleParams(BaseModel):
    """Invented generic fixed-wing UAV parameters (sim spec §2, build prompt §4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    mass_nom_kg: float
    mass_range_kg: tuple[float, float]
    wing_area_m2: float
    wingspan_m: float
    aspect_ratio: float
    oswald_e: float
    k: float
    CL0: float
    CLa: float
    alpha_stall_rad: float
    CLmax: float
    CD0: float
    Mdd: float
    wave_drag_coeff: float
    T0_N: float
    thrust_lapse_n: float
    thrust_mach_factor: float
    SFC_kg_per_Ns: float
    nz_max_g: float
    bank_limit_rad: float
    tau_thrust_s: float
    tau_alpha_s: float


def load_vehicle_params(ref: str, vehicles_dir: Path) -> VehicleParams:
    """
    Load a vehicle's parameters from `<vehicles_dir>/<ref>.yaml`.

    Parameters
    ----------
    ref : str
        Vehicle name, e.g. `"generic_uav_v1"` (`StudyConfig.vehicle.ref`).
    vehicles_dir : Path
        Directory containing vehicle YAML files (`configs/vehicles/`).

    Returns
    -------
    VehicleParams

    Raises
    ------
    ConfigError
        If the file is missing or fails validation.
    """
    path = Path(vehicles_dir) / f"{ref}.yaml"
    if not path.exists():
        raise ConfigError(f"Vehicle config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    try:
        return VehicleParams.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, actionable ConfigError
        raise ConfigError(f"Invalid vehicle config {path}: {exc}") from exc


class StructuralSwitches(BaseModel):
    """Structural model differences between variants (sim spec §5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compressibility: bool = False
    wind_rate_terms: bool = False
    alpha_dynamics: bool = False
    thrust_lapse: bool = True
    isa_offset_known: bool = False
    wave_drag: bool = False


@dataclass(frozen=True)
class State:
    """A single flight state vector (data spec §3).

    Attributes
    ----------
    t_s : float
        Time since flight start, s.
    x_m, y_m : float
        Local NE position, m.
    h_m : float
        Altitude MSL, m.
    V_mps : float
        True airspeed, m/s.
    gamma_rad : float
        Flight-path angle, rad.
    chi_rad : float
        Track angle, rad (0 = North, clockwise).
    m_kg : float
        Mass, kg.
    alpha_rad : float
        Angle of attack, rad (dynamic only when `StructuralSwitches.alpha_dynamics`).
    """

    t_s: float
    x_m: float
    y_m: float
    h_m: float
    V_mps: float
    gamma_rad: float
    chi_rad: float
    m_kg: float
    alpha_rad: float = 0.0


@dataclass(frozen=True)
class ControlHistory:
    """A recorded or commanded control time series, interpolated at arbitrary `t_s`.

    Attributes
    ----------
    t_s : ndarray, shape (N,)
        Monotonically increasing sample times, s.
    throttle : ndarray, shape (N,)
        Throttle command, 0-1.
    alpha_cmd_rad : ndarray, shape (N,)
        Commanded angle of attack, rad.
    bank_cmd_rad : ndarray, shape (N,)
        Commanded bank angle, rad.
    """

    t_s: NDArray[np.float64]
    throttle: NDArray[np.float64]
    alpha_cmd_rad: NDArray[np.float64]
    bank_cmd_rad: NDArray[np.float64]

    def at(
        self, t_s: NDArray[np.float64] | float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate (throttle, alpha_cmd_rad, bank_cmd_rad) at `t_s` (clamped to ends)."""
        t_query = np.asarray(t_s, dtype=np.float64)
        throttle = np.interp(t_query, self.t_s, self.throttle)
        alpha_cmd = np.interp(t_query, self.t_s, self.alpha_cmd_rad)
        bank_cmd = np.interp(t_query, self.t_s, self.bank_cmd_rad)
        return throttle, alpha_cmd, bank_cmd


@dataclass
class SimResult:
    """Output of one simulator run.

    Attributes
    ----------
    trajectory : pandas.DataFrame
        Canonical-schema time series (`core/schema.py`), `source` set to the
        simulator's name (`"truth"` or a variant name).
    status : str
        `"ok"` or `"diverged"`.
    diverged_at_s : float, optional
        Simulation time of the first non-finite state, if `status == "diverged"`.
    """

    trajectory: pd.DataFrame
    status: str = "ok"
    diverged_at_s: float | None = None


@dataclass(frozen=True)
class Scenario:
    """A parsed scenario: initial conditions plus an ordered list of manoeuvre segments.

    Attributes
    ----------
    name : str
        Scenario name.
    init : State
        Initial state at `t_s = 0`.
    segments : list[dict]
        Each a `{"type": ..., **params, "duration_s": ...}` manoeuvre primitive spec
        (`sim/maneuvers.py` interprets these).
    """

    name: str
    init: State
    segments: list[dict[str, Any]]

    @property
    def duration_s(self) -> float:
        return float(sum(float(segment["duration_s"]) for segment in self.segments))


@runtime_checkable
class SimulatorProtocol(Protocol):
    """Common interface for truth, variants, and future external plug-ins."""

    def run(
        self,
        scenario: Scenario | None,
        controls: ControlHistory | None,
        init: State | None,
        *,
        seed: int,
    ) -> SimResult:
        """
        Run the simulator either closed-loop against `scenario` (`controls is None`,
        the controller stack drives the vehicle and the recorded controls are
        included in the output trajectory) or in replay mode against fixed
        `controls` and `init` (`scenario` ignored).
        """
        ...


def assert_finite_or_truncate(y: NDArray[np.float64], t_s: float, context: str) -> None:
    """Raise `SimulationError` eagerly if `y` is non-finite (used internally by sim/)."""
    if not np.all(np.isfinite(y)):
        raise SimulationError(f"{context}: non-finite state at t={t_s:.3f}s: {y}")
