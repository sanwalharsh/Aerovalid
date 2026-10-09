"""Rule-based flight-phase segmentation with hysteresis (FR-PRP-007) and
operating-condition tagging (FR-PRP-008).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


class PhaseThresholds(BaseModel):
    """Rate thresholds distinguishing climb/descend/turn/accel from cruise."""

    model_config = ConfigDict(extra="forbid")

    climb_rate_mps: float = 0.5
    turn_rate_radps: float = 0.03
    accel_mps2: float = 0.3
    min_duration_s: float = 2.0  # hysteresis: a phase must hold this long to "stick"


def segment_phases(
    df: pd.DataFrame, dt: float, thresholds: PhaseThresholds | None = None
) -> pd.Series:
    """
    Rule-based phase labelling with a minimum-duration debounce (FR-PRP-007).

    Parameters
    ----------
    df : pandas.DataFrame
        Must have `climb_rate_mps` and `turn_rate_radps` (`prep/features.py`); uses
        `vel_air_mps` for an acceleration check if present.
    dt : float
        Grid spacing, s.
    thresholds : PhaseThresholds, optional

    Returns
    -------
    pandas.Series
        Category values in `{"climb", "descend", "turn", "accel", "cruise"}`, same
        index as `df`; matches the scenario script's segment type >= 90% of samples
        on synthetic data with default thresholds (FR-PRP-007's acceptance criterion).
    """
    thresholds = thresholds or PhaseThresholds()
    n = len(df)
    climb_rate = df["climb_rate_mps"].to_numpy(dtype=np.float64)
    turn_rate = (
        df["turn_rate_radps"].to_numpy(dtype=np.float64)
        if "turn_rate_radps" in df.columns
        else np.zeros(n)
    )
    accel = (
        np.gradient(df["vel_air_mps"].to_numpy(dtype=np.float64), dt)
        if "vel_air_mps" in df.columns
        else np.zeros(n)
    )

    raw = np.full(n, "cruise", dtype=object)
    raw[np.abs(turn_rate) > thresholds.turn_rate_radps] = "turn"
    raw[np.abs(accel) > thresholds.accel_mps2] = "accel"
    raw[climb_rate < -thresholds.climb_rate_mps] = "descend"
    raw[climb_rate > thresholds.climb_rate_mps] = "climb"

    min_run = max(1, round(thresholds.min_duration_s / dt))
    debounced = _debounce(raw, min_run)
    return pd.Series(debounced, index=df.index, name="phase", dtype="category")


def _debounce(labels: np.ndarray, min_run: int) -> np.ndarray:
    """Replace any run shorter than `min_run` with the preceding (stable) label."""
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if j - i < min_run and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


class ConditionBins(BaseModel):
    """Bin edges for operating-condition tags (FR-PRP-008)."""

    model_config = ConfigDict(extra="forbid")

    altitude_m: list[float] = [0, 1000, 2500, 4000, 6000, 20000]
    mach: list[float] = [0.0, 0.1, 0.2, 0.3, 1.0]
    bank_rad: list[float] = [0.0, 0.2, 0.4, 0.6, 1.2]
    nz_g: list[float] = [-1.0, 0.8, 1.2, 1.8, 3.5]


def tag_conditions(df: pd.DataFrame, bins: ConditionBins | None = None) -> pd.DataFrame:
    """
    Add binned condition tag columns (FR-PRP-008): `cond_alt_band`, `cond_mach_band`,
    `cond_bank_band`, `cond_nz_band` (whichever source columns are present).

    Parameters
    ----------
    df : pandas.DataFrame
    bins : ConditionBins, optional

    Returns
    -------
    pandas.DataFrame
        `df` with the available `cond_*_band` columns appended.
    """
    bins = bins or ConditionBins()
    out = df.copy()
    pairs = (
        ("alt_m", "cond_alt_band", bins.altitude_m),
        ("mach", "cond_mach_band", bins.mach),
        ("roll_rad", "cond_bank_band", bins.bank_rad),
        ("nz_g", "cond_nz_band", bins.nz_g),
    )
    for source, target, edges in pairs:
        if source not in df.columns:
            continue
        values = df[source].to_numpy(dtype=np.float64)
        band_source = np.abs(values) if target in ("cond_bank_band",) else values
        out[target] = pd.cut(band_source, bins=edges, include_lowest=True).astype(str)
    return out
