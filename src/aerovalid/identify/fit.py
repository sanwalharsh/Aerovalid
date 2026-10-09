"""Output-error parameter identification (FR-IDN-002, analysis methods §10): fit
selected `VehicleParams` fields to telemetry via `scipy.optimize.least_squares` with a
robust (soft-L1) loss, using windowed re-initialisation (`sim/runners.py::run_windowed`)
to avoid chaotic long-horizon drift, exactly as `predict/`'s Task-C windowing does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import least_squares

from aerovalid.sim.protocols import StructuralSwitches, VehicleParams
from aerovalid.sim.runners import run_windowed
from aerovalid.sim.variants import VariantModel

_REQUIRED_STATE_COLUMNS = [
    "t_s",
    "x_m",
    "y_m",
    "alt_m",
    "vel_air_mps",
    "gamma_rad",
    "chi_rad",
    "mass_kg",
]


def _apply_params(
    nominal: VehicleParams, param_names: list[str], values: NDArray[np.float64]
) -> VehicleParams:
    return nominal.model_copy(update=dict(zip(param_names, values, strict=True)))


def residual_vector(
    values: NDArray[np.float64],
    param_names: list[str],
    nominal: VehicleParams,
    switches: StructuralSwitches,
    dt_s: float,
    telemetry: pd.DataFrame,
    signals: list[str],
    weights: dict[str, float],
    horizon_s: float,
    stride_s: float,
) -> NDArray[np.float64]:
    """
    Weighted `obs - sim` residual vector for one candidate parameter point, pooled
    across every `(signal, window)` pair (analysis methods §10's `J(theta)` summand).

    Parameters
    ----------
    values : ndarray, shape (n,)
        Candidate values for `param_names`.
    param_names : list[str]
        `VehicleParams` field names being fit.
    nominal : VehicleParams
        Base vehicle; only `param_names` fields are overridden.
    switches : StructuralSwitches
        The model structure being fit against (e.g. a variant's own switches).
    dt_s : float
        Integration step for the candidate variant.
    telemetry : pandas.DataFrame
        Raw telemetry (`t_s, throttle, alpha_cmd_rad, bank_cmd_rad`, plus each
        `signals` entry by its plain name) -- same source `controls_from_dataframe`
        and `state_from_row` already consume elsewhere (sim/runners.py).
    signals : list[str]
        Signals to include in the residual.
    weights : dict[str, float]
        Per-signal weight (typically `1/sigma_s`); signals not present default to 1.0.
    horizon_s, stride_s : float
        `run_windowed`'s re-initialisation window and stride, s.

    Returns
    -------
    ndarray
        Concatenated weighted residuals across every signal and window.
    """
    vehicle = _apply_params(nominal, param_names, values)
    variant = VariantModel(name="_fit_candidate", vehicle=vehicle, switches=switches, dt_s=dt_s)
    tel_state = telemetry[[c for c in _REQUIRED_STATE_COLUMNS if c in telemetry.columns]]
    windowed = run_windowed(variant, tel_state, telemetry, horizon_s, stride_s, seed=0)

    chunks = []
    for signal in signals:
        sim_col = f"sim_{signal}"
        if sim_col not in windowed.columns or signal not in telemetry.columns:
            continue
        merged = pd.merge_asof(
            windowed[["t_s", sim_col]].sort_values("t_s"),
            telemetry[["t_s", signal]].sort_values("t_s"),
            on="t_s",
            direction="nearest",
        )
        weight = weights.get(signal, 1.0)
        chunks.append(weight * (merged[signal] - merged[sim_col]).to_numpy(dtype=np.float64))
    return np.concatenate(chunks) if chunks else np.zeros(0)


@dataclass(frozen=True)
class FitResult:
    """One output-error fit's outcome."""

    param_names: list[str]
    fitted_values: NDArray[np.float64]
    residuals: NDArray[np.float64]
    cost: float
    success: bool

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.param_names, self.fitted_values, strict=True))


def fit_parameters(
    param_names: list[str],
    bounds: dict[str, tuple[float, float]],
    nominal: VehicleParams,
    switches: StructuralSwitches,
    dt_s: float,
    telemetry: pd.DataFrame,
    signals: list[str],
    weights: dict[str, float],
    horizon_s: float = 30.0,
    stride_s: float = 15.0,
    x0: NDArray[np.float64] | None = None,
) -> FitResult:
    """
    Output-error fit of `param_names` to `telemetry` (FR-IDN-002).

    Parameters
    ----------
    param_names : list[str]
    bounds : dict[str, tuple[float, float]]
        Per-parameter `(lower, upper)`; every name in `param_names` must have an entry.
    nominal, switches, dt_s, telemetry, signals, weights, horizon_s, stride_s
        Forwarded to `residual_vector`.
    x0 : ndarray, optional
        Starting point; defaults to `nominal`'s own values for `param_names`.

    Returns
    -------
    FitResult
    """
    x0 = x0 if x0 is not None else np.array([getattr(nominal, name) for name in param_names])
    lower = np.array([bounds[name][0] for name in param_names])
    upper = np.array([bounds[name][1] for name in param_names])

    result = least_squares(
        residual_vector,
        x0,
        args=(
            param_names,
            nominal,
            switches,
            dt_s,
            telemetry,
            signals,
            weights,
            horizon_s,
            stride_s,
        ),
        bounds=(lower, upper),
        loss="soft_l1",
    )
    return FitResult(
        param_names=list(param_names),
        fitted_values=result.x,
        residuals=result.fun,
        cost=float(result.cost),
        success=bool(result.success),
    )
