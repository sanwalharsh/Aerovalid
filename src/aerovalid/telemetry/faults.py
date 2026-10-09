"""Fault injection library (FR-TEL-003, sim spec §7): drag-rise step, lift-curve-slope
degradation, thrust-loss ramp, icing, control-effectiveness loss, sensor-bias step, and
wind-shear event. Faults affect truth only, never variants. At most one fault is
sampled per flight, and `probability_per_flight` (default 0.5) keeps >= 1/3 of flights
fault-free for detector calibration (sim spec §7).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from aerovalid.core.errors import ConfigError
from aerovalid.sim.dynamics import FaultState
from aerovalid.telemetry.sensors import SensorModel

_PARAM_FAULT_TYPES = frozenset({"cd0_step", "cla_degrade", "thrust_loss", "icing", "ctrl_eff_loss"})
_EVENT_FAULT_DURATION_S = 30.0  # sensor_bias_step / wind_shear_event last this long after onset


class FaultSpec(BaseModel):
    """One fault type's library entry (`configs/faults/*.yaml`)."""

    model_config = ConfigDict(extra="forbid")

    type: str
    magnitude_range: tuple[float, float]
    ramp_s_range: tuple[float, float]
    affected_signals: list[str] = []


class FaultEvent(BaseModel):
    """One sampled, realised fault instance, saved verbatim to `faults.json`."""

    model_config = ConfigDict(extra="forbid")

    type: str
    t_on_s: float
    t_off_s: float
    magnitude: float
    ramp_s: float
    affected_signals: list[str] = []


def load_fault_library(faults_dir: Path) -> dict[str, FaultSpec]:
    """
    Load every `*.yaml` fault definition in `faults_dir` (e.g. `configs/faults/`).

    Returns
    -------
    dict[str, FaultSpec]
        Keyed by fault `type`.

    Raises
    ------
    ConfigError
        If the directory has no fault YAML files.
    """
    library: dict[str, FaultSpec] = {}
    for path in sorted(Path(faults_dir).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        try:
            spec = FaultSpec.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed, actionable ConfigError
            raise ConfigError(f"Invalid fault config {path}: {exc}") from exc
        library[spec.type] = spec
    if not library:
        raise ConfigError(f"No fault definitions found in {faults_dir}")
    return library


def sample_fault_schedule(
    library: dict[str, FaultSpec],
    flight_duration_s: float,
    rng: np.random.Generator,
    probability_per_flight: float,
) -> list[FaultEvent]:
    """
    Sample at most one fault for a flight (sim spec §7).

    Parameters
    ----------
    library : dict[str, FaultSpec]
        From `load_fault_library`.
    flight_duration_s : float
        This flight's total duration, s.
    rng : numpy.random.Generator
        Seeded per-flight generator (FR-COR-006).
    probability_per_flight : float
        Probability this flight has a fault at all (`generate.faults.probability_per_flight`).

    Returns
    -------
    list[FaultEvent]
        Empty if this flight is fault-free (the common case at the default 0.5).
    """
    if rng.random() >= probability_per_flight:
        return []

    fault_type = str(rng.choice(sorted(library)))
    spec = library[fault_type]
    magnitude = float(rng.uniform(*spec.magnitude_range))
    ramp_s = float(rng.uniform(*spec.ramp_s_range))
    t_on_s = float(rng.uniform(0.1 * flight_duration_s, 0.7 * flight_duration_s))
    if fault_type in _PARAM_FAULT_TYPES:
        t_off_s = flight_duration_s
    else:
        t_off_s = min(flight_duration_s, t_on_s + max(ramp_s, 1.0) + _EVENT_FAULT_DURATION_S)

    return [
        FaultEvent(
            type=fault_type,
            t_on_s=t_on_s,
            t_off_s=t_off_s,
            magnitude=magnitude,
            ramp_s=ramp_s,
            affected_signals=spec.affected_signals,
        )
    ]


def _ramped_magnitude(t_s: float, event: FaultEvent) -> float:
    if t_s < event.t_on_s or t_s > event.t_off_s:
        return 0.0
    if event.ramp_s <= 0:
        return event.magnitude
    return event.magnitude * min(1.0, (t_s - event.t_on_s) / event.ramp_s)


def build_param_fault_fn(events: list[FaultEvent]) -> Callable[[float], FaultState]:
    """
    Build the `t_s -> FaultState` closure `TruthModel.run(param_fault_fn=...)` wants.

    Parameters
    ----------
    events : list[FaultEvent]
        From `sample_fault_schedule`; non-parameter events (`sensor_bias_step`,
        `wind_shear_event`) are ignored here.

    Returns
    -------
    callable
        `t_s -> FaultState`, ramping linearly from `t_on_s` to `t_on_s + ramp_s`.
    """
    param_events = [e for e in events if e.type in _PARAM_FAULT_TYPES]
    if not param_events:
        return lambda t_s: FaultState()

    def fault_state_at(t_s: float) -> FaultState:
        state = FaultState()
        for event in param_events:
            m = _ramped_magnitude(t_s, event)
            if m <= 0.0:
                continue
            if event.type == "cd0_step":
                state = replace(state, cd0_mult=state.cd0_mult * (1.0 + m))
            elif event.type == "cla_degrade":
                state = replace(state, cla_mult=state.cla_mult * (1.0 - m))
            elif event.type == "thrust_loss":
                state = replace(state, t0_mult=state.t0_mult * (1.0 - m))
            elif event.type == "icing":
                state = replace(
                    state,
                    cd0_mult=state.cd0_mult * (1.0 + m),
                    clmax_mult=state.clmax_mult * (1.0 - m),
                    cla_mult=state.cla_mult * (1.0 - 0.5 * m),
                )
            elif event.type == "ctrl_eff_loss":
                state = replace(state, alpha_gain=state.alpha_gain * (1.0 - m))
        return state

    return fault_state_at


def build_wind_shear_fn(events: list[FaultEvent]) -> Callable[[float], tuple[float, float]]:
    """Build the `t_s -> (delta_wind_n, delta_wind_e)` closure for `wind_shear_event`."""
    shear_events = [e for e in events if e.type == "wind_shear_event"]
    if not shear_events:
        return lambda t_s: (0.0, 0.0)

    def shear_at(t_s: float) -> tuple[float, float]:
        delta_n = sum(_ramped_magnitude(t_s, event) for event in shear_events)
        return delta_n, 0.0

    return shear_at


def apply_sensor_bias_faults(
    telemetry: pd.DataFrame, events: list[FaultEvent], sensor_suite: dict[str, SensorModel]
) -> pd.DataFrame:
    """
    Apply `sensor_bias_step` events: an additive step of `magnitude * sigma` on each
    affected signal from `t_on_s` onward.

    Parameters
    ----------
    telemetry : pandas.DataFrame
        Already sensor-corrupted telemetry (`telemetry/sensors.py`'s `apply_sensors`).
    events : list[FaultEvent]
    sensor_suite : dict[str, SensorModel]
        Used to resolve each signal's sigma (the step is specified in sigma units).

    Returns
    -------
    pandas.DataFrame
        `telemetry`, with the bias step added where applicable.
    """
    out = telemetry.copy()
    for event in events:
        if event.type != "sensor_bias_step":
            continue
        for signal in event.affected_signals:
            if signal not in out.columns or signal not in sensor_suite:
                continue
            step = event.magnitude * sensor_suite[signal].resolved_sigma()
            active = out["t_s"] >= event.t_on_s
            out.loc[active, signal] = out.loc[active, signal] + step
    return out
