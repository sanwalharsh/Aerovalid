"""Replay and windowed runners (FR-SIM-009, FR-SIM-010, data spec §4.2): drive a
variant from recorded controls, either once end-to-end (replay) or re-initialised from
a (smoothed) telemetry state estimate every `stride_s` for `horizon_s` (windowed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aerovalid.core.signals import known_signals
from aerovalid.sim.protocols import ControlHistory, SimResult, State
from aerovalid.sim.variants import VariantModel


def controls_from_dataframe(df: pd.DataFrame) -> ControlHistory:
    """
    Build a `ControlHistory` from a DataFrame with `t_s, throttle, alpha_cmd_rad,
    bank_cmd_rad` columns (e.g. a slice of truth's or telemetry's recorded controls).
    """
    return ControlHistory(
        t_s=df["t_s"].to_numpy(dtype=np.float64),
        throttle=df["throttle"].to_numpy(dtype=np.float64),
        alpha_cmd_rad=df["alpha_cmd_rad"].to_numpy(dtype=np.float64),
        bank_cmd_rad=df["bank_cmd_rad"].to_numpy(dtype=np.float64),
    )


def state_from_row(row: pd.Series) -> State:
    """Build a `State` from one row of a telemetry/truth DataFrame (data spec §4.1 names)."""
    return State(
        t_s=float(row["t_s"]),
        x_m=float(row.get("x_m", 0.0)),
        y_m=float(row.get("y_m", 0.0)),
        h_m=float(row["alt_m"]),
        V_mps=float(row["vel_air_mps"]),
        gamma_rad=float(row.get("gamma_rad", 0.0)),
        chi_rad=float(row.get("chi_rad", 0.0)),
        m_kg=float(row.get("mass_kg", 0.0) or 0.0),
        alpha_rad=float(row.get("alpha_rad", 0.0) or 0.0),
    )


def replay_run(
    variant: VariantModel,
    controls: pd.DataFrame,
    init_row: pd.Series,
    *,
    seed: int,
    wind_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SimResult:
    """
    Run `variant` once, end to end, following `controls` from `init_row` (FR-SIM-009).

    Parameters
    ----------
    variant : VariantModel
    controls : pandas.DataFrame
        Full-flight recorded controls (`t_s, throttle, alpha_cmd_rad, bank_cmd_rad`).
    init_row : pandas.Series
        A telemetry/truth row giving the initial state (data spec §4.1 column names).
    seed : int
        Forwarded to `variant.run` (unused by variants, kept for protocol symmetry).
    wind_mps : tuple[float, float, float], default (0, 0, 0)
        Constant wind estimate for the whole run (V5 only; sim spec §5).

    Returns
    -------
    SimResult
    """
    return variant.run(
        None,
        controls_from_dataframe(controls),
        state_from_row(init_row),
        seed=seed,
        wind_mps=wind_mps,
    )


def run_windowed(
    variant: VariantModel,
    tel_state: pd.DataFrame,
    controls: pd.DataFrame,
    horizon_s: float,
    stride_s: float,
    *,
    seed: int = 0,
    wind_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> pd.DataFrame:
    """
    Re-initialise `variant` from `tel_state` every `stride_s`, run forward `horizon_s`
    (FR-SIM-010, data spec §4.2).

    Parameters
    ----------
    variant : VariantModel
    tel_state : pandas.DataFrame
        Smoothed telemetry state estimate (`t_s, x_m, y_m, alt_m, vel_air_mps,
        gamma_rad, chi_rad, mass_kg`), one row per sample.
    controls : pandas.DataFrame
        Full-flight recorded controls (`t_s, throttle, alpha_cmd_rad, bank_cmd_rad`).
    horizon_s : float
        How far forward to run from each window's start, s.
    stride_s : float
        Spacing between successive window starts, s.
    seed, wind_mps
        Forwarded to each window's `variant.run` call.

    Returns
    -------
    pandas.DataFrame
        Concatenated windows with `window_id`, `horizon_s` (elapsed time since the
        window's own start, not `t_s`), and signal columns prefixed `sim_<name>`
        (data spec §4.2). At `horizon_s == 0` the output equals the state-estimate
        input (FR-SIM-010's acceptance criterion).
    """
    t_min, t_max = float(tel_state["t_s"].min()), float(tel_state["t_s"].max())
    window_starts = np.arange(t_min, max(t_max - horizon_s, t_min) + 1e-9, stride_s)
    signal_columns = set(known_signals())
    frames: list[pd.DataFrame] = []

    for window_id, t0 in enumerate(window_starts):
        init_pos = int(np.argmin(np.abs(tel_state["t_s"].to_numpy() - t0)))
        init_row = tel_state.iloc[init_pos]
        window_controls = controls[(controls["t_s"] >= t0) & (controls["t_s"] <= t0 + horizon_s)]
        if len(window_controls) < 2:
            continue
        result = variant.run(
            None,
            controls_from_dataframe(window_controls),
            state_from_row(init_row),
            seed=seed,
            wind_mps=wind_mps,
        )
        trajectory = result.trajectory.rename(
            columns={
                col: f"sim_{col}" for col in result.trajectory.columns if col in signal_columns
            }
        )
        trajectory["window_id"] = window_id
        trajectory["horizon_s"] = trajectory["t_s"] - t0
        frames.append(trajectory)

    if not frames:
        return pd.DataFrame(columns=["t_s", "source", "window_id", "horizon_s"])
    return pd.concat(frames, ignore_index=True)
