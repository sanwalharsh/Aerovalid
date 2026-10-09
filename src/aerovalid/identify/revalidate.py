"""Re-validation (FR-IDN-006, analysis methods §10): apply a fitted parameter vector to
build a candidate variant V*, replay it on held-out flights, and compare its metrics
against the original (flawed) variant's -- paired, per flight, reusing `validate/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aerovalid.sim.protocols import State, StructuralSwitches, VehicleParams
from aerovalid.sim.runners import controls_from_dataframe
from aerovalid.sim.variants import VariantModel
from aerovalid.validate.metrics import MetricSet, compute_metrics


def build_variant_star(
    nominal: VehicleParams,
    fitted_params: dict[str, float],
    switches: StructuralSwitches,
    dt_s: float,
    name: str = "V_star",
) -> VariantModel:
    """Build the refit candidate variant (FR-IDN-006): `nominal` with `fitted_params`
    substituted in, same structural switches as the variant being improved on."""
    vehicle = nominal.model_copy(update=fitted_params)
    return VariantModel(name=name, vehicle=vehicle, switches=switches, dt_s=dt_s)


@dataclass(frozen=True)
class RevalidationRow:
    flight_id: str
    signal: str
    rmse_original: float
    rmse_refit: float
    rmse_reduction_pct: float


def revalidate_on_held_out_flights(
    variant_star: VariantModel,
    original_trajectories: dict[str, pd.DataFrame],
    held_out_telemetry: dict[str, pd.DataFrame],
    signals: list[str],
) -> pd.DataFrame:
    """
    Replay `variant_star` on each held-out flight and compare its per-signal RMSE
    against the original variant's own trajectory on the same flight (FR-IDN-006).

    Parameters
    ----------
    variant_star : VariantModel
    original_trajectories : dict[str, pandas.DataFrame]
        The original (pre-refit) variant's trajectory per held-out flight (already
        computed, e.g. from `raw/variants/<name>/<flight>.parquet`).
    held_out_telemetry : dict[str, pandas.DataFrame]
        Each held-out flight's raw telemetry (`t_s, throttle, alpha_cmd_rad,
        bank_cmd_rad`, plus each `signals` entry).
    signals : list[str]

    Returns
    -------
    pandas.DataFrame
        One row per `(flight_id, signal)`: `RevalidationRow` fields.
    """
    rows: list[dict[str, object]] = []
    for flight_id, telemetry in held_out_telemetry.items():
        controls = controls_from_dataframe(telemetry)
        init_row = telemetry.iloc[0]
        init = State(
            t_s=float(init_row["t_s"]),
            x_m=float(init_row.get("x_m", 0.0)),
            y_m=float(init_row.get("y_m", 0.0)),
            h_m=float(init_row["alt_m"]),
            V_mps=float(init_row["vel_air_mps"]),
            gamma_rad=float(init_row.get("gamma_rad", 0.0)),
            chi_rad=float(init_row.get("chi_rad", 0.0)),
            m_kg=float(init_row.get("mass_kg", 0.0) or 0.0),
        )
        refit_result = variant_star.run(None, controls, init, seed=0)
        refit_traj = refit_result.trajectory
        original_traj = original_trajectories.get(flight_id)
        if original_traj is None or original_traj.empty or refit_traj.empty:
            continue

        for signal in signals:
            if signal not in telemetry.columns or signal not in refit_traj.columns:
                continue
            obs = telemetry[["t_s", signal]].sort_values("t_s")
            refit_merged = pd.merge_asof(
                obs,
                refit_traj[["t_s", signal]].sort_values("t_s"),
                on="t_s",
                direction="nearest",
                suffixes=("", "_refit"),
            )
            original_merged = pd.merge_asof(
                obs,
                original_traj[["t_s", signal]].sort_values("t_s"),
                on="t_s",
                direction="nearest",
                suffixes=("", "_orig"),
            )
            rmse_refit = compute_metrics(
                refit_merged[signal].to_numpy(dtype=np.float64),
                refit_merged[f"{signal}_refit"].to_numpy(dtype=np.float64),
            ).rmse
            rmse_original = compute_metrics(
                original_merged[signal].to_numpy(dtype=np.float64),
                original_merged[f"{signal}_orig"].to_numpy(dtype=np.float64),
            ).rmse
            reduction = (
                100.0 * (rmse_original - rmse_refit) / rmse_original
                if rmse_original > 0
                else float("nan")
            )
            rows.append(
                {
                    "flight_id": flight_id,
                    "signal": signal,
                    "rmse_original": rmse_original,
                    "rmse_refit": rmse_refit,
                    "rmse_reduction_pct": reduction,
                }
            )
    return pd.DataFrame(rows)


__all__ = ["MetricSet", "build_variant_star", "revalidate_on_held_out_flights", "RevalidationRow"]
