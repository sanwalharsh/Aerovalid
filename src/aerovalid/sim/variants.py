"""Variant models (FR-SIM-007, sim spec §5): YAML-driven parameter overrides and
structural switches on top of the same dynamics code truth uses, run in replay mode
against fixed controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from aerovalid.core.atmosphere import isa_atmosphere
from aerovalid.core.errors import ConfigError
from aerovalid.sim.dynamics import FaultState, StepInputs, derived_measurements, rhs_quasi_steady
from aerovalid.sim.integrators import rk4_step
from aerovalid.sim.propulsion import thrust as compute_thrust
from aerovalid.sim.protocols import (
    ControlHistory,
    Scenario,
    SimResult,
    State,
    StructuralSwitches,
    VehicleParams,
)

_NO_FAULT = FaultState()


class ParamOverrides(BaseModel):
    """Multiplicative overrides on the nominal vehicle's aerodynamic parameters."""

    model_config = ConfigDict(extra="forbid")

    CD0_mult: float = 1.0
    CLa_mult: float = 1.0
    CL0_mult: float = 1.0


class VariantSpec(BaseModel):
    """A variant's YAML definition (sim spec §5)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    known_flaw: str = ""
    overrides: ParamOverrides = Field(default_factory=ParamOverrides)
    switches: StructuralSwitches = Field(default_factory=StructuralSwitches)
    atmosphere_naive: bool = False
    dt_s: float = 0.05


def load_variant_yaml(name: str, variants_dir: Path) -> VariantSpec:
    """Load `<variants_dir>/<name>.yaml` into a `VariantSpec`."""
    path = Path(variants_dir) / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"Variant config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    try:
        return VariantSpec.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, actionable ConfigError
        raise ConfigError(f"Invalid variant config {path}: {exc}") from exc


@dataclass
class VariantModel:
    """A variant simulator: fixed parameters/switches, run in replay mode.

    Attributes
    ----------
    name : str
        Variant name (becomes the `source` column in its output trajectory).
    vehicle : VehicleParams
        This variant's (possibly overridden) parameters.
    switches : StructuralSwitches
        Structural differences from truth.
    atmosphere_naive : bool
        If True (V3), density/sound-speed are frozen at the initial altitude for the
        whole run rather than tracking the true altitude (sim spec §5).
    dt_s : float
        Integration step, s (intentionally coarser than truth's by default).
    """

    name: str
    vehicle: VehicleParams
    switches: StructuralSwitches
    atmosphere_naive: bool = False
    dt_s: float = 0.05

    def run(
        self,
        scenario: Scenario | None,
        controls: ControlHistory | None,
        init: State | None,
        *,
        seed: int,
        wind_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
        delta_t_k: float = 0.0,
    ) -> SimResult:
        """
        Replay fixed `controls` from `init` (`scenario` ignored: variants never fly
        closed-loop).

        Parameters
        ----------
        scenario : None
            Ignored.
        controls : ControlHistory
            Required: recorded (throttle, alpha_cmd_rad, bank_cmd_rad) to follow.
        init : State
            Required: initial state to replay from.
        seed : int
            Unused (variants are deterministic given controls/init); kept for
            `SimulatorProtocol` compatibility.
        wind_mps : tuple[float, float, float], default (0, 0, 0)
            Constant wind estimate for the whole run (V5 estimates this from
            airspeed - ground speed; V0-V4 assume zero, sim spec §5).
        delta_t_k : float, default 0.0
            ISA temperature offset, used only if `switches.isa_offset_known`.

        Returns
        -------
        SimResult
        """
        if controls is None or init is None:
            raise ValueError("VariantModel.run requires both controls and init (replay mode)")

        y = np.array(
            [init.x_m, init.y_m, init.h_m, init.V_mps, init.gamma_rad, init.chi_rad, init.m_kg]
        )
        atmosphere_override = init.h_m if self.atmosphere_naive else None

        t0, t1 = float(controls.t_s[0]), float(controls.t_s[-1])
        n_steps = max(1, int(round((t1 - t0) / self.dt_s)))
        records: list[dict[str, float | str]] = []
        status, diverged_at = "ok", None

        for i in range(n_steps + 1):
            t = min(t0 + i * self.dt_s, t1)
            throttle, alpha_cmd, bank_cmd = (float(v) for v in controls.at(t))
            inputs = StepInputs(
                throttle=throttle,
                alpha_cmd_rad=alpha_cmd,
                bank_cmd_rad=bank_cmd,
                wind_n_mps=wind_mps[0],
                wind_e_mps=wind_mps[1],
                wind_d_mps=wind_mps[2],
                delta_t_k=delta_t_k,
                atmosphere_h_override_m=atmosphere_override,
            )
            thrust_n = _commanded_thrust(
                y[2], y[3], inputs, self.vehicle, self.switches, atmosphere_override
            )
            measurements = derived_measurements(
                y[2],
                y[3],
                y[4],
                y[5],
                y[6],
                alpha_cmd,
                thrust_n,
                inputs,
                self.vehicle,
                self.switches,
            )
            records.append(
                {
                    "t_s": t,
                    "x_m": y[0],
                    "y_m": y[1],
                    "alt_m": y[2],
                    "vel_air_mps": y[3],
                    "gamma_rad": y[4],
                    "chi_rad": y[5],
                    "mass_kg": y[6],
                    "alpha_rad": alpha_cmd,
                    "vel_gnd_mps": measurements.vel_gnd_mps,
                    "pitch_rad": measurements.pitch_rad,
                    "roll_rad": measurements.roll_rad,
                    "yaw_rad": measurements.yaw_rad,
                    "nx_g": measurements.nx_g,
                    "nz_g": measurements.nz_g,
                    "mach": measurements.mach,
                    "q_pa": measurements.q_pa,
                    "throttle": throttle,
                    "alpha_cmd_rad": alpha_cmd,
                    "bank_cmd_rad": bank_cmd,
                }
            )

            if i == n_steps or not np.all(np.isfinite(y)):
                break
            step_dt = min(self.dt_s, t1 - t)
            if step_dt <= 0:
                continue
            y = rk4_step(
                rhs_quasi_steady, t, y, step_dt, inputs, self.vehicle, self.switches, _NO_FAULT
            )
            if not np.all(np.isfinite(y)):
                status, diverged_at = "diverged", t + step_dt

        trajectory = pd.DataFrame.from_records(records)
        trajectory["source"] = self.name
        return SimResult(trajectory=trajectory, status=status, diverged_at_s=diverged_at)


def _commanded_thrust(
    h_m: float,
    v_mps: float,
    inputs: StepInputs,
    vehicle: VehicleParams,
    switches: StructuralSwitches,
    atmosphere_override: float | None,
) -> float:
    atm_h = atmosphere_override if atmosphere_override is not None else h_m
    atm = isa_atmosphere(atm_h, delta_t_k=inputs.delta_t_k if switches.isa_offset_known else 0.0)
    mach = v_mps / float(atm.speed_of_sound_mps)
    return float(
        compute_thrust(
            inputs.throttle,
            float(atm.density_kgm3),
            mach,
            vehicle,
            thrust_lapse=switches.thrust_lapse,
        )
    )


def make_variant(spec: VariantSpec, nominal: VehicleParams) -> VariantModel:
    """
    Build a `VariantModel` from a `VariantSpec` applied to the nominal vehicle (V0-V4).

    Parameters
    ----------
    spec : VariantSpec
        Loaded variant definition.
    nominal : VehicleParams
        The study's nominal vehicle parameters (never truth's perturbed ones).

    Returns
    -------
    VariantModel
    """
    updated = nominal.model_copy(
        update={
            "CD0": nominal.CD0 * spec.overrides.CD0_mult,
            "CLa": nominal.CLa * spec.overrides.CLa_mult,
            "CL0": nominal.CL0 * spec.overrides.CL0_mult,
        }
    )
    return VariantModel(
        name=spec.name,
        vehicle=updated,
        switches=spec.switches,
        atmosphere_naive=spec.atmosphere_naive,
        dt_s=spec.dt_s,
    )


V5_NOISE_SIGMA = 0.02
V5_PARAMS = ("CL0", "CLa", "CD0", "k", "T0_N", "thrust_lapse_n", "Mdd")


def make_variant_v5(
    truth_vehicle: VehicleParams, rng: np.random.Generator, dt_s: float = 0.05
) -> VariantModel:
    """
    Build V5 "near-truth": truth's structure minus alpha dynamics, with truth's
    (hidden) parameters each perturbed by an additional +-2% noise (sim spec §5).

    Parameters
    ----------
    truth_vehicle : VehicleParams
        `TruthModel.vehicle` (already perturbed from nominal; hidden from every
        other variant).
    rng : numpy.random.Generator
        Seeded generator for the additional noise draw.
    dt_s : float, default 0.05
        Integration step, s.

    Returns
    -------
    VariantModel
        Named `"V5"`, wind-estimating (`switches.wind_rate_terms=True`), without
        alpha dynamics.
    """
    updates = {
        name: getattr(truth_vehicle, name) * (1.0 + float(rng.normal(0.0, V5_NOISE_SIGMA)))
        for name in V5_PARAMS
    }
    vehicle = truth_vehicle.model_copy(update=updates)
    switches = StructuralSwitches(
        compressibility=True,
        wind_rate_terms=True,
        alpha_dynamics=False,
        thrust_lapse=True,
        isa_offset_known=True,
        wave_drag=True,
    )
    return VariantModel(name="V5", vehicle=vehicle, switches=switches, dt_s=dt_s)


def load_all_variants(
    names: list[str], variants_dir: Path, nominal: VehicleParams
) -> dict[str, VariantModel]:
    """Load every YAML-defined variant in `names` (does not build "V5", see `make_variant_v5`)."""
    return {name: make_variant(load_variant_yaml(name, variants_dir), nominal) for name in names}
