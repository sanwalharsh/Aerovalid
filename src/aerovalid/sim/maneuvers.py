"""Manoeuvre primitives and the scenario YAML loader (FR-SIM-008, sim spec §8).

A scenario is an ordered list of segments, each a small dict with a `type` and that
primitive's parameters. `build_scenario` fills in anything that depends on the running
flight-plan state (segment duration for a `coordinated_turn`, the altitude/speed
baseline a `climb`/`speed_step` ramps from/to), producing a `Scenario` whose segments
are ready for `setpoint_at` to turn into per-instant `Setpoint`s during closed-loop flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aerovalid.core.errors import ConfigError
from aerovalid.sim.controllers import Setpoint
from aerovalid.sim.protocols import Scenario, State

_G0 = 9.80665


@dataclass(frozen=True)
class _SegmentContext:
    """Bundles a resolved segment with its elapsed time and flight-plan baselines."""

    segment: dict[str, Any]
    t_s: float
    duration_s: float
    v_start_mps: float
    h_start_m: float


def _turn_duration_s(bank_rad: float, heading_change_rad: float, v_mps: float) -> float:
    turn_rate_radps = float(_G0 * np.tan(abs(bank_rad)) / max(v_mps, 1e-6))
    return abs(heading_change_rad) / max(turn_rate_radps, 1e-9)


def build_scenario(name: str, raw_segments: list[dict[str, Any]], init: State) -> Scenario:
    """
    Resolve a raw segment list into a `Scenario` with every segment's `duration_s`,
    `v_cmd_start_mps`, and `h_cmd_start_m` filled in from the running flight plan.

    Parameters
    ----------
    name : str
        Scenario name.
    raw_segments : list[dict]
        Each a manoeuvre primitive spec (`type` + its own params); `coordinated_turn`
        may omit `duration_s` (computed from `bank_rad`/`heading_change_rad`/`v_mps`).
    init : State
        Initial state; the flight plan's starting speed/altitude baseline.

    Returns
    -------
    Scenario
    """
    v_baseline = init.V_mps
    h_baseline = init.h_m
    segments: list[dict[str, Any]] = []

    for raw in raw_segments:
        segment = dict(raw)
        segment_type = segment.get("type")
        segment["v_cmd_start_mps"] = v_baseline
        segment["h_cmd_start_m"] = h_baseline

        if segment_type == "coordinated_turn" and "duration_s" not in segment:
            segment["duration_s"] = _turn_duration_s(
                segment["bank_rad"], segment["heading_change_rad"], segment.get("v_mps", v_baseline)
            )

        if "duration_s" not in segment:
            raise ConfigError(f"Scenario {name!r} segment {segment_type!r} is missing duration_s")

        if segment_type in ("climb", "descend"):
            v_baseline = segment.get("v_mps", v_baseline)
            h_baseline = segment["h_target_m"]
        elif segment_type == "speed_step":
            v_baseline = v_baseline + segment["delta_v_mps"]
        elif "v_mps" in segment:
            v_baseline = segment["v_mps"]

        segments.append(segment)

    return Scenario(name=name, init=init, segments=segments)


def _setpoint_trim_cruise(ctx: _SegmentContext) -> Setpoint:
    return Setpoint(
        v_cmd_mps=ctx.segment.get("v_mps", ctx.v_start_mps),
        h_cmd_m=ctx.segment.get("h_m", ctx.h_start_m),
        bank_cmd_rad=0.0,
    )


def _setpoint_climb_descend(ctx: _SegmentContext) -> Setpoint:
    h_target = float(ctx.segment["h_target_m"])
    frac = min(ctx.t_s / ctx.duration_s, 1.0) if ctx.duration_s > 0 else 1.0
    h_cmd = ctx.h_start_m + frac * (h_target - ctx.h_start_m)
    return Setpoint(
        v_cmd_mps=ctx.segment.get("v_mps", ctx.v_start_mps), h_cmd_m=h_cmd, bank_cmd_rad=0.0
    )


def _setpoint_coordinated_turn(ctx: _SegmentContext) -> Setpoint:
    return Setpoint(
        v_cmd_mps=ctx.segment.get("v_mps", ctx.v_start_mps),
        h_cmd_m=ctx.h_start_m,
        bank_cmd_rad=ctx.segment["bank_rad"],
    )


def _setpoint_s_turn(ctx: _SegmentContext) -> Setpoint:
    period_s = float(ctx.segment["period_s"])
    bank_rad = (
        ctx.segment["bank_rad"]
        if (ctx.t_s % period_s) < period_s / 2.0
        else -ctx.segment["bank_rad"]
    )
    return Setpoint(
        v_cmd_mps=ctx.segment.get("v_mps", ctx.v_start_mps),
        h_cmd_m=ctx.h_start_m,
        bank_cmd_rad=bank_rad,
    )


def _setpoint_speed_step(ctx: _SegmentContext) -> Setpoint:
    return Setpoint(
        v_cmd_mps=ctx.v_start_mps + ctx.segment["delta_v_mps"],
        h_cmd_m=ctx.h_start_m,
        bank_cmd_rad=0.0,
    )


def _setpoint_pitch_doublet(ctx: _SegmentContext) -> Setpoint:
    period_s = float(ctx.segment["period_s"])
    amp = float(ctx.segment["alpha_amp_rad"])
    if ctx.t_s < period_s / 2.0:
        alpha_override = amp
    elif ctx.t_s < period_s:
        alpha_override = -amp
    else:
        alpha_override = 0.0
    return Setpoint(
        v_cmd_mps=ctx.v_start_mps,
        h_cmd_m=ctx.h_start_m,
        bank_cmd_rad=0.0,
        alpha_cmd_override_rad=alpha_override,
    )


def _setpoint_loiter(ctx: _SegmentContext) -> Setpoint:
    v_cmd = ctx.segment.get("v_mps", ctx.v_start_mps)
    bank_rad = float(np.arctan(v_cmd**2 / (_G0 * float(ctx.segment["radius_m"]))))
    return Setpoint(v_cmd_mps=v_cmd, h_cmd_m=ctx.h_start_m, bank_cmd_rad=bank_rad)


def _setpoint_throttle_chop(ctx: _SegmentContext) -> Setpoint:
    return Setpoint(
        v_cmd_mps=ctx.v_start_mps,
        h_cmd_m=ctx.h_start_m,
        bank_cmd_rad=0.0,
        throttle_override=float(ctx.segment.get("throttle", 0.05)),
    )


_SETPOINT_HANDLERS = {
    "trim_cruise": _setpoint_trim_cruise,
    "climb": _setpoint_climb_descend,
    "descend": _setpoint_climb_descend,
    "coordinated_turn": _setpoint_coordinated_turn,
    "s_turn": _setpoint_s_turn,
    "speed_step": _setpoint_speed_step,
    "pitch_doublet": _setpoint_pitch_doublet,
    "loiter": _setpoint_loiter,
    "throttle_chop": _setpoint_throttle_chop,
}


def setpoint_at(segment: dict[str, Any], t_in_segment_s: float) -> Setpoint:
    """
    Evaluate one manoeuvre segment's commanded `Setpoint` at `t_in_segment_s`.

    Parameters
    ----------
    segment : dict
        A resolved segment from `build_scenario` (carries `v_cmd_start_mps` /
        `h_cmd_start_m` baselines and a concrete `duration_s`).
    t_in_segment_s : float
        Elapsed time since this segment started, s (0 <= t <= `segment["duration_s"]`).

    Returns
    -------
    Setpoint
    """
    handler = _SETPOINT_HANDLERS.get(segment["type"])
    if handler is None:
        raise ConfigError(f"Unknown manoeuvre segment type: {segment['type']!r}")
    ctx = _SegmentContext(
        segment=segment,
        t_s=t_in_segment_s,
        duration_s=float(segment["duration_s"]),
        v_start_mps=float(segment["v_cmd_start_mps"]),
        h_start_m=float(segment["h_cmd_start_m"]),
    )
    return handler(ctx)


def load_scenario_yaml(name: str, scenarios_dir: Path, init: State) -> Scenario:
    """
    Load `<scenarios_dir>/<name>.yaml` and resolve it into a `Scenario`.

    Parameters
    ----------
    name : str
        Scenario name (file stem).
    scenarios_dir : Path
        Directory containing scenario YAML files (`configs/scenarios/`).
    init : State
        Initial state to resolve segment baselines against.

    Returns
    -------
    Scenario
    """
    path = Path(scenarios_dir) / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"Scenario config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    segments = raw.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ConfigError(f"Scenario {path} must have a non-empty `segments` list")
    return build_scenario(name, segments, init)
