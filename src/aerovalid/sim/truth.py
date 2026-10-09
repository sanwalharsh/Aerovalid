"""Truth model (FR-SIM-006, sim spec §3-4): the high-fidelity closed-loop simulator
that plays the role of "real flight". Hidden parameter perturbations are drawn once
per study (fixed across every flight) and never exposed to variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aerovalid.sim.controllers import ControllerStack
from aerovalid.sim.dynamics import (
    ACTUATOR_LAG_DIM,
    FaultState,
    StepInputs,
    derived_measurements,
    rhs_actuator_lag,
)
from aerovalid.sim.environment import DrydenTurbulence, WindParams, steady_wind_with_shear
from aerovalid.sim.integrators import rk4_step
from aerovalid.sim.maneuvers import setpoint_at
from aerovalid.sim.protocols import (
    ControlHistory,
    Scenario,
    SimResult,
    State,
    StructuralSwitches,
    VehicleParams,
)

# Relative std. dev. of each truth hidden-parameter perturbation, drawn once per study
# (sim spec §4). Mass offset is an absolute kg perturbation, not relative.
TRUTH_PARAM_SIGMAS: dict[str, float] = {
    "CL0": 0.04,
    "CLa": 0.05,
    "CD0": 0.08,
    "k": 0.08,
    "T0_N": 0.05,
    "thrust_lapse_n": 0.04,
    "Mdd": 0.02,
}
MASS_OFFSET_SIGMA_KG = 1.0

TRUTH_SWITCHES = StructuralSwitches(
    compressibility=True,
    wind_rate_terms=True,
    alpha_dynamics=True,
    thrust_lapse=True,
    isa_offset_known=True,
    wave_drag=True,
)

_DT_S = 0.01


def _segment_at(scenario: Scenario, t_s: float) -> tuple[dict[str, Any], float]:
    elapsed = 0.0
    for segment in scenario.segments:
        duration = float(segment["duration_s"])
        if t_s <= elapsed + duration or segment is scenario.segments[-1]:
            return segment, max(0.0, t_s - elapsed)
        elapsed += duration
    return scenario.segments[-1], 0.0


@dataclass
class TruthModel:
    """High-fidelity truth simulator with hidden, study-fixed parameter perturbations.

    Attributes
    ----------
    vehicle : VehicleParams
        The *perturbed* vehicle parameters (never exposed to variants).
    perturbations : dict[str, float]
        The relative (or, for mass, absolute) perturbation drawn for each hidden
        parameter; saved verbatim to `truth_params.json` by the caller.
    switches : StructuralSwitches
        Always `TRUTH_SWITCHES` (full fidelity).
    """

    vehicle: VehicleParams
    perturbations: dict[str, float]
    switches: StructuralSwitches = field(default_factory=lambda: TRUTH_SWITCHES)
    dt_s: float = _DT_S

    @classmethod
    def from_nominal(cls, nominal: VehicleParams, master_seed: int) -> TruthModel:
        """
        Draw the study-fixed hidden perturbations from `master_seed` and build a
        `TruthModel`. Calling this twice with the same `master_seed` reproduces the
        same perturbed vehicle (FR-COR-006).
        """
        rng = np.random.Generator(np.random.PCG64(master_seed))
        perturbations: dict[str, float] = {}
        updates: dict[str, float] = {}
        for name, sigma in TRUTH_PARAM_SIGMAS.items():
            delta = float(rng.normal(0.0, sigma))
            perturbations[name] = delta
            updates[name] = getattr(nominal, name) * (1.0 + delta)
        mass_delta = float(rng.normal(0.0, MASS_OFFSET_SIGMA_KG))
        perturbations["mass_offset_kg"] = mass_delta
        updates["mass_nom_kg"] = nominal.mass_nom_kg + mass_delta
        perturbed_vehicle = nominal.model_copy(update=updates)
        return cls(vehicle=perturbed_vehicle, perturbations=perturbations)

    def run(
        self,
        scenario: Scenario | None,
        controls: ControlHistory | None = None,
        init: State | None = None,
        *,
        seed: int,
        wind: WindParams | None = None,
        fault_schedule: list[tuple[float, float, FaultState]] | None = None,
    ) -> SimResult:
        """
        Fly `scenario` closed-loop (the controller stack drives the vehicle).

        Parameters
        ----------
        scenario : Scenario
            Required; truth always runs closed-loop against a scenario.
        controls : None
            Ignored (truth never replays fixed controls).
        init : State, optional
            Overrides `scenario.init` if given.
        seed : int
            Per-flight seed (turbulence RNG); independent of the study's master seed
            used for the hidden-parameter draw.
        wind : WindParams, optional
            This flight's wind/turbulence condition; `None` disables wind entirely.
        fault_schedule : list[(t_on_s, t_off_s, FaultState)], optional
            Time-varying parameter multipliers (`telemetry/faults.py` builds this);
            `None` means no faults this flight.

        Returns
        -------
        SimResult
        """
        if scenario is None:
            raise ValueError("TruthModel.run requires a scenario (closed-loop only)")
        start_state = init if init is not None else scenario.init
        rng = np.random.Generator(np.random.PCG64(seed))
        turbulence = DrydenTurbulence(
            rms_mps=wind.turbulence_rms_mps if wind else 0.0,
            scale_length_m=wind.turbulence_scale_length_m if wind else 200.0,
            rng=rng,
        )
        controller = ControllerStack(vehicle=self.vehicle)

        y = np.zeros(ACTUATOR_LAG_DIM, dtype=np.float64)
        y[:7] = [
            start_state.x_m,
            start_state.y_m,
            start_state.h_m,
            start_state.V_mps,
            start_state.gamma_rad,
            start_state.chi_rad,
            start_state.m_kg,
        ]
        y[7] = start_state.alpha_rad

        duration_s = scenario.duration_s
        n_steps = max(1, int(round(duration_s / self.dt_s)))
        records: list[dict[str, float | str]] = []
        status, diverged_at = "ok", None

        for i in range(n_steps + 1):
            t = min(i * self.dt_s, duration_s)
            segment, t_in_segment = _segment_at(scenario, t)
            setpoint = setpoint_at(segment, t_in_segment)
            state = State(t, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7])
            throttle, alpha_cmd, bank_cmd = controller.step(state, setpoint, self.dt_s)

            if wind:
                wind_n_raw, wind_e_raw = steady_wind_with_shear(y[2], wind)
                wind_n, wind_e = float(wind_n_raw), float(wind_e_raw)
            else:
                wind_n, wind_e = 0.0, 0.0
            gust_n, gust_e, gust_d = turbulence.step(self.dt_s, max(y[3], 1.0))
            fault_state = _fault_state_at(t, fault_schedule)

            inputs = StepInputs(
                throttle=throttle,
                alpha_cmd_rad=alpha_cmd,
                bank_cmd_rad=bank_cmd,
                wind_n_mps=wind_n + gust_n,
                wind_e_mps=wind_e + gust_e,
                wind_d_mps=gust_d,
            )
            thrust_n = float(y[8])
            measurements = derived_measurements(
                y[2],
                y[3],
                y[4],
                y[5],
                y[6],
                y[7],
                thrust_n,
                inputs,
                self.vehicle,
                self.switches,
                fault_state,
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
                    "alpha_rad": y[7],
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
            step_dt = min(self.dt_s, duration_s - t)
            if step_dt <= 0:
                continue
            y = rk4_step(
                rhs_actuator_lag, t, y, step_dt, inputs, self.vehicle, self.switches, fault_state
            )
            if not np.all(np.isfinite(y)):
                status, diverged_at = "diverged", t + step_dt

        trajectory = pd.DataFrame.from_records(records)
        trajectory["source"] = "truth"
        return SimResult(trajectory=trajectory, status=status, diverged_at_s=diverged_at)


def _fault_state_at(
    t_s: float, schedule: list[tuple[float, float, FaultState]] | None
) -> FaultState:
    if not schedule:
        return FaultState()
    for t_on, t_off, state in schedule:
        if t_on <= t_s <= t_off:
            return state
    return FaultState()
