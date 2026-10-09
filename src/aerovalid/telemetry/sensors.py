"""Sensor noise/bias/lag/dropout models (FR-TEL-001, sim spec §6). Each signal gets
an independent `SensorModel`; `apply_sensors` corrupts a whole truth DataFrame into a
telemetry DataFrame in one call, returning a per-signal defect log for QC/audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.signal import lfilter

from aerovalid.core.errors import ConfigError


class SensorModel(BaseModel):
    """One signal's sensor corruption model (sim spec §6). All fields are additive/
    multiplicative knobs that default to "no effect" so a signal can opt into only
    the defects it needs."""

    model_config = ConfigDict(extra="forbid")

    sigma: float = 0.0
    sigma_deg: float | None = None
    bias: float = 0.0
    bias_deg: float | None = None
    bias_rw: float = 0.0
    drift_deg_per_sqrts: float | None = None
    scale: float = 1.0
    lag_s: float = 0.0
    quantise: float = 0.0
    dropout_p: float = 0.0
    enabled: bool = True

    def resolved_sigma(self) -> float:
        """White-noise std in the signal's native unit (deg inputs converted to rad)."""
        return float(np.deg2rad(self.sigma_deg)) if self.sigma_deg is not None else self.sigma

    def resolved_bias(self) -> float:
        """Constant bias in the signal's native unit (deg inputs converted to rad)."""
        return float(np.deg2rad(self.bias_deg)) if self.bias_deg is not None else self.bias


@dataclass
class SensorDefectLog:
    """Per-signal record of what `apply_sensors` actually injected."""

    signal: str
    dropout_indices: list[int] = field(default_factory=list)
    final_bias: float = 0.0


def load_sensor_suite(path: Path) -> dict[str, SensorModel]:
    """
    Load per-signal `SensorModel`s from a sensors YAML (e.g. `configs/sensors/default.yaml`).

    The `timing_defects` key (if present) is ignored here; it belongs to
    `telemetry/defects.py`'s `TimingDefectParams`.

    Parameters
    ----------
    path : Path
        Path to the sensors YAML file.

    Returns
    -------
    dict[str, SensorModel]
        Keyed by canonical signal name.
    """
    if not path.exists():
        raise ConfigError(f"Sensor config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    raw.pop("timing_defects", None)
    try:
        return {name: SensorModel.model_validate(spec) for name, spec in raw.items()}
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, actionable ConfigError
        raise ConfigError(f"Invalid sensor config {path}: {exc}") from exc


def apply_sensor_model(
    signal: NDArray[np.float64],
    t_s: NDArray[np.float64],
    model: SensorModel,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], SensorDefectLog]:
    """
    Corrupt one signal's samples on a (regular) time grid with `model`.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Clean (truth) signal values.
    t_s : ndarray, shape (N,)
        Sample times, s (assumed on a regular grid, as telemetry is before timing
        defects are applied -- `telemetry/defects.py` runs separately, afterward).
    model : SensorModel
    rng : numpy.random.Generator
        Seeded per-flight generator (FR-COR-006).

    Returns
    -------
    tuple[ndarray, SensorDefectLog]
        Corrupted signal (dropped samples are `NaN`) and what was injected.
    """
    n = signal.shape[0]
    if not model.enabled or n == 0:
        return signal.copy(), SensorDefectLog(signal="")

    dt = float(np.median(np.diff(t_s))) if n > 1 else 0.0
    y = signal * model.scale

    bias_path = np.full(n, model.resolved_bias())
    if model.bias_rw > 0 and dt > 0:
        steps = rng.normal(0.0, model.bias_rw * np.sqrt(dt), size=n)
        bias_path = bias_path + np.cumsum(steps)
    if model.drift_deg_per_sqrts and dt > 0:
        drift_rate = np.deg2rad(model.drift_deg_per_sqrts)
        steps = rng.normal(0.0, drift_rate * np.sqrt(dt), size=n)
        bias_path = bias_path + np.cumsum(steps)
    y = y + bias_path

    sigma = model.resolved_sigma()
    if sigma > 0:
        y = y + rng.normal(0.0, sigma, size=n)

    if model.lag_s > 0 and dt > 0:
        alpha = dt / (model.lag_s + dt)
        y = lfilter([alpha], [1.0, -(1.0 - alpha)], y)

    if model.quantise > 0:
        y = np.round(y / model.quantise) * model.quantise

    dropout_indices: list[int] = []
    if model.dropout_p > 0:
        dropout_mask = rng.random(n) < model.dropout_p
        dropout_indices = np.flatnonzero(dropout_mask).tolist()
        y = y.copy()
        y[dropout_mask] = np.nan

    return y, SensorDefectLog(
        signal="", dropout_indices=dropout_indices, final_bias=float(bias_path[-1])
    )


def apply_sensors(
    truth: pd.DataFrame, sensor_suite: dict[str, SensorModel], rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, SensorDefectLog]]:
    """
    Corrupt every modelled signal column of a truth DataFrame into telemetry
    (architecture spec §3: `apply_sensors(truth, cfg, rng) -> (DataFrame, DefectLog)`).

    Parameters
    ----------
    truth : pandas.DataFrame
        Canonical-schema truth table (must have `t_s` and the signal columns the
        suite models).
    sensor_suite : dict[str, SensorModel]
        From `load_sensor_suite`.
    rng : numpy.random.Generator
        Seeded per-flight generator; every signal draws from the same generator in a
        fixed (sorted) order so the result is deterministic (FR-COR-006).

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, SensorDefectLog]]
        Telemetry DataFrame (same shape as `truth`, modelled columns corrupted,
        `source="tel"`) and each corrupted signal's defect log.
    """
    telemetry = truth.copy()
    t_s = truth["t_s"].to_numpy(dtype=np.float64)
    logs: dict[str, SensorDefectLog] = {}

    for signal_name in sorted(sensor_suite):
        if signal_name not in truth.columns:
            continue
        model = sensor_suite[signal_name]
        signal_values = truth[signal_name].to_numpy(dtype=np.float64)
        corrupted, log = apply_sensor_model(signal_values, t_s, model, rng)
        log.signal = signal_name
        telemetry[signal_name] = corrupted
        logs[signal_name] = log

    telemetry["source"] = "tel"
    return telemetry, logs
