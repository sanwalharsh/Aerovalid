"""Per-flight preprocessing orchestration (FR-PRP-001...011): QC -> clean -> sync ->
resample -> features -> phases/conditions -> align, for one flight's telemetry against
every variant. Pure in-memory (CLAUDE.md §7: I/O stays in `cli.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aerovalid.core.config import PrepConfig
from aerovalid.prep.align import AlignedDataset, build_aligned
from aerovalid.prep.clean import CleanResult, clean
from aerovalid.prep.features import add_derived_signals
from aerovalid.prep.phases import segment_phases, tag_conditions
from aerovalid.prep.qc import QCReport, qc_report
from aerovalid.prep.resample import resample
from aerovalid.prep.state_estimate import estimate_state
from aerovalid.prep.sync import SyncEstimate, apply_sync, estimate_sync


@dataclass
class FlightPreprocessResult:
    """Everything produced by preprocessing one flight."""

    flight_id: str
    qc: QCReport
    clean_result: CleanResult
    state_estimate: pd.DataFrame
    sync_estimates: dict[str, SyncEstimate]
    aligned: dict[str, AlignedDataset]


def preprocess_flight(
    flight_id: str,
    telemetry: pd.DataFrame,
    variant_trajectories: dict[str, pd.DataFrame],
    cfg: PrepConfig,
    wing_area_m2: float | None = None,
) -> FlightPreprocessResult:
    """
    Run the full preprocessing pipeline for one flight.

    Parameters
    ----------
    flight_id : str
    telemetry : pandas.DataFrame
        Raw (pre-clean) telemetry for this flight.
    variant_trajectories : dict[str, pandas.DataFrame]
        Each variant's raw replayed trajectory.
    cfg : PrepConfig
        `StudyConfig.preprocess`.
    wing_area_m2 : float, optional
        Forwarded to `prep/features.py::add_derived_signals` for `cl_eff`.

    Returns
    -------
    FlightPreprocessResult
    """
    qc = qc_report(telemetry, flight_id)
    clean_result = clean(
        telemetry,
        max_gap_s=cfg.max_gap_s,
        hampel_window=cfg.hampel.window,
        hampel_n_sigma=cfg.hampel.n_sigma,
    )

    dt = 1.0 / cfg.grid_hz
    obs_resampled = resample(clean_result.df, dt)
    obs_resampled = add_derived_signals(
        obs_resampled, dt, cfg.sg.window, cfg.sg.order, wing_area_m2
    )
    obs_resampled["phase"] = segment_phases(obs_resampled, dt).to_numpy()
    obs_resampled = tag_conditions(obs_resampled)

    state_estimate = estimate_state(obs_resampled, dt, cfg.sg.window, cfg.sg.order)

    obs_grid = obs_resampled["t_s"].to_numpy()
    resampled_mask = _resample_mask(clean_result, obs_grid)
    sync_estimates: dict[str, SyncEstimate] = {}
    aligned: dict[str, AlignedDataset] = {}
    for variant_name, trajectory in variant_trajectories.items():
        estimate = estimate_sync(
            obs_resampled, trajectory, cfg.sync.ref_signals, cfg.sync.max_lag_s, dt
        )
        sync_estimates[variant_name] = estimate
        synced = apply_sync(trajectory, estimate)
        sim_on_obs_grid = resample(synced, dt, t_grid=obs_grid)
        aligned[variant_name] = build_aligned(
            obs_resampled, sim_on_obs_grid, flight_id, variant_name, resampled_mask
        )

    return FlightPreprocessResult(
        flight_id=flight_id,
        qc=qc,
        clean_result=clean_result,
        state_estimate=state_estimate,
        sync_estimates=sync_estimates,
        aligned=aligned,
    )


def _resample_mask(clean_result: CleanResult, t_grid: np.ndarray) -> pd.DataFrame:
    """Nearest-neighbour resample of a boolean exclusion mask onto `t_grid` (linear
    interpolation would turn `True`/`False` into meaningless fractions)."""
    t_source = clean_result.df["t_s"].to_numpy(dtype=np.float64)
    t_target = np.asarray(t_grid, dtype=np.float64)
    nearest_idx = np.clip(np.searchsorted(t_source, t_target), 0, len(t_source) - 1)
    return pd.DataFrame(
        {
            column: clean_result.mask[column].to_numpy()[nearest_idx]
            for column in clean_result.mask.columns
        }
    )
