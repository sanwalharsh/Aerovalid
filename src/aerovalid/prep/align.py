"""Aligned dataset builder (FR-PRP-009): one table per (flight, variant) with
`obs_<sig>`, `sim_<sig>`, `res_<sig>` (angle-wrapped), `mask_<sig>`, `cond_*`, `phase`
columns, validated against the canonical schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aerovalid.core.angles import angle_diff
from aerovalid.core.schema import validate_canonical
from aerovalid.core.signals import known_signals

_ANGLE_SIGNALS = frozenset(
    {"gamma_rad", "chi_rad", "roll_rad", "pitch_rad", "yaw_rad", "alpha_rad"}
)


@dataclass
class AlignedDataset:
    """A validated aligned table for one (flight, variant) pair."""

    flight_id: str
    variant: str
    df: pd.DataFrame


def build_aligned(
    obs: pd.DataFrame,
    sim: pd.DataFrame,
    flight_id: str,
    variant: str,
    mask: pd.DataFrame | None = None,
) -> AlignedDataset:
    """
    Build one aligned (flight, variant) table (FR-PRP-009).

    Parameters
    ----------
    obs : pandas.DataFrame
        Cleaned, resampled, synced telemetry (the "observation").
    sim : pandas.DataFrame
        The variant's trajectory, resampled/synced onto `obs`'s exact time grid
        (same length and `t_s` values; `prep/resample.py` + `prep/sync.py`'s job).
    flight_id, variant : str
    mask : pandas.DataFrame, optional
        Per-signal exclusion mask from `prep/clean.py` (`CleanResult.mask`); any
        signal not in `mask` is assumed never excluded.

    Returns
    -------
    AlignedDataset
        Schema-validated; row count equals `len(obs)` (FR-PRP-009's acceptance
        criterion: "row counts equal across variants" holds when every variant is
        aligned against the same `obs`).

    Raises
    ------
    SchemaError
        If the assembled table fails canonical-schema validation.
    """
    if len(obs) != len(sim):
        raise ValueError(
            f"obs and sim must share the same time grid: {len(obs)} != {len(sim)} rows"
        )

    aligned = pd.DataFrame({"t_s": obs["t_s"].to_numpy()})
    aligned["flight_id"] = flight_id
    aligned["source"] = variant

    for name in known_signals():
        has_obs, has_sim = name in obs.columns, name in sim.columns
        if not has_obs and not has_sim:
            continue
        if has_obs:
            aligned[f"obs_{name}"] = obs[name].to_numpy()
        if has_sim:
            aligned[f"sim_{name}"] = sim[name].to_numpy()
        if has_obs and has_sim:
            obs_vals = obs[name].to_numpy(dtype=np.float64)
            sim_vals = sim[name].to_numpy(dtype=np.float64)
            aligned[f"res_{name}"] = (
                angle_diff(obs_vals, sim_vals) if name in _ANGLE_SIGNALS else obs_vals - sim_vals
            )
        if has_obs:
            excluded = (
                mask[name].to_numpy()
                if mask is not None and name in mask.columns
                else np.zeros(len(obs), dtype=bool)
            )
            aligned[f"mask_{name}"] = excluded | np.isnan(
                aligned[f"obs_{name}"].to_numpy(dtype=np.float64)
            )

    for column in obs.columns:
        if column.startswith("cond_") or column == "phase":
            aligned[column] = obs[column].to_numpy()

    validate_canonical(aligned)
    return AlignedDataset(flight_id=flight_id, variant=variant, df=aligned)
