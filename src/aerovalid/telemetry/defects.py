"""Timing defect injectors (FR-TEL-002, sim spec §6): constant clock offset, linear
drift, per-sample jitter, dropped samples, duplicate timestamps, and burst gaps.
Applied to the *timestamp* array, independently of `telemetry/sensors.py`'s per-signal
value corruption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from aerovalid.core.errors import ConfigError


class _Range(BaseModel):
    model_config = ConfigDict(extra="forbid")
    low: float
    high: float


class TimingDefectParams(BaseModel):
    """Timing-defect distribution parameters (sim spec §6)."""

    model_config = ConfigDict(extra="forbid")

    offset_s: _Range = Field(default_factory=lambda: _Range(low=-0.8, high=0.8))
    drift_ppm: _Range = Field(default_factory=lambda: _Range(low=-50.0, high=50.0))
    jitter_sigma_ms: float = 5.0
    dropout_p_per_sample: float = 0.005
    duplicate_p_per_sample: float = 0.002
    burst_gaps_max_n: int = 2
    burst_gaps_duration_s: tuple[float, float] = (0.5, 3.0)


@dataclass
class TimingDefectLog:
    """What `apply_timing_defects` actually drew/injected for one flight."""

    offset_s: float
    drift_ppm: float
    gaps_s: list[tuple[float, float]] = field(default_factory=list)
    n_duplicated: int = 0
    n_dropped: int = 0


def load_timing_defect_params(path: Path) -> TimingDefectParams:
    """Load the `timing_defects:` section of a sensors YAML (`configs/sensors/default.yaml`)."""
    if not path.exists():
        raise ConfigError(f"Sensor config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    timing_raw = raw.get("timing_defects", {})
    try:
        return TimingDefectParams.model_validate(timing_raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, actionable ConfigError
        raise ConfigError(f"Invalid timing_defects config in {path}: {exc}") from exc


def apply_timing_defects(
    t_s: NDArray[np.float64], params: TimingDefectParams, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.bool_], TimingDefectLog]:
    """
    Corrupt a timestamp array with constant offset, linear drift, jitter, duplicate
    timestamps, random dropouts, and burst gaps.

    Parameters
    ----------
    t_s : ndarray, shape (N,)
        Clean, strictly increasing sample times, s.
    params : TimingDefectParams
    rng : numpy.random.Generator
        Seeded per-flight generator (FR-COR-006).

    Returns
    -------
    tuple[ndarray, ndarray, TimingDefectLog]
        Corrupted timestamps (same length as `t_s`; duplicated entries collapse onto
        their predecessor's corrupted time rather than inserting rows), a boolean
        `keep_mask` (False where the sample was dropped by dropout or a burst gap),
        and the realised defect parameters for `meta/<flight>.json`.
    """
    n = t_s.shape[0]
    offset_s = float(rng.uniform(params.offset_s.low, params.offset_s.high))
    drift_ppm = float(rng.uniform(params.drift_ppm.low, params.drift_ppm.high))
    jitter = rng.normal(0.0, params.jitter_sigma_ms / 1000.0, size=n)
    t_corrupted = t_s * (1.0 + drift_ppm * 1e-6) + offset_s + jitter

    dup_draw: NDArray[np.float64] = rng.random(size=(n,))
    dup_mask = dup_draw < params.duplicate_p_per_sample
    dup_mask[0] = False
    # Propagate each duplicate's nearest preceding non-duplicated sample's corrupted
    # time forward -- correctly collapses a *run* of consecutive duplicates onto one
    # shared anchor value, unlike a naive `t[dup] = t[dup - 1]` shift (numpy evaluates
    # that assignment's RHS once, so it just shifts the whole run instead of merging it).
    anchor_idx = np.maximum.accumulate(np.where(~dup_mask, np.arange(n), 0))
    t_corrupted = np.where(dup_mask, t_corrupted[anchor_idx], t_corrupted)

    dropout_draw: NDArray[np.float64] = rng.random(size=(n,))
    dropout_mask = dropout_draw < params.dropout_p_per_sample

    gaps_s: list[tuple[float, float]] = []
    gap_mask: NDArray[np.bool_] = np.zeros(n, dtype=bool)
    t_min, t_max = float(t_s[0]), float(t_s[-1])
    n_gaps = int(rng.integers(0, params.burst_gaps_max_n + 1))
    for _ in range(n_gaps):
        duration = float(rng.uniform(*params.burst_gaps_duration_s))
        start = float(rng.uniform(t_min, max(t_min, t_max - duration)))
        end = start + duration
        gaps_s.append((start, end))
        gap_mask |= (t_s >= start) & (t_s <= end)

    keep_mask: NDArray[np.bool_] = ~dropout_mask & ~gap_mask
    log = TimingDefectLog(
        offset_s=offset_s,
        drift_ppm=drift_ppm,
        gaps_s=gaps_s,
        n_duplicated=int(dup_mask.sum()),
        n_dropped=int((~keep_mask).sum()),
    )
    return t_corrupted, keep_mask, log
