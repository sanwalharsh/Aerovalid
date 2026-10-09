"""Synthetic dataset generator (FR-TEL-004, sim spec §9): Sobol-sampled conditions,
one closed-loop truth flight each, sensor + timing defects into telemetry, and every
variant replayed from (post-defect) recorded controls. Pure in-memory orchestration --
no disk I/O here; `cli.py`'s `generate` command serialises `FlightArtifacts` to
`runs/<run_id>/raw/...` (CLAUDE.md §7: side effects only in `cli.py`/adapters/`report/`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

from aerovalid.core.config import GenerateConditions, StudyConfig
from aerovalid.core.rng import child_seed
from aerovalid.sim.environment import WindParams
from aerovalid.sim.maneuvers import load_scenario_yaml
from aerovalid.sim.protocols import State, VehicleParams
from aerovalid.sim.runners import controls_from_dataframe, state_from_row
from aerovalid.sim.truth import TruthModel
from aerovalid.sim.variants import VariantModel
from aerovalid.telemetry.defects import TimingDefectLog, TimingDefectParams, apply_timing_defects
from aerovalid.telemetry.faults import (
    FaultEvent,
    FaultSpec,
    apply_sensor_bias_faults,
    build_param_fault_fn,
    build_wind_shear_fn,
    sample_fault_schedule,
)
from aerovalid.telemetry.sensors import SensorModel, apply_sensors

_TURB_RMS_MPS = {"none": 0.0, "light": 0.5, "moderate": 1.5}
_CONDITION_DIMS = ("alt0_m", "v0_mps", "mass_kg", "wind_mean_mps", "dT_K")


@dataclass
class FlightArtifacts:
    """Everything generated for one flight, ready for `cli.py` to serialise.

    Attributes
    ----------
    flight_id : str
    scenario_name : str
    seed : int
        This flight's derived per-flight seed (`core/rng.child_seed`).
    conditions : dict
        Sampled condition vector (data spec §4.3 `meta.conditions`).
    truth_df, telemetry_df : pandas.DataFrame
    variant_dfs : dict[str, pandas.DataFrame]
    fault_events : list[FaultEvent]
    timing_log : TimingDefectLog
    status : str
        `"ok"` or `"diverged"` (truth's status; data spec §8).
    """

    flight_id: str
    scenario_name: str
    seed: int
    conditions: dict[str, float | str]
    truth_df: pd.DataFrame
    telemetry_df: pd.DataFrame
    variant_dfs: dict[str, pd.DataFrame]
    fault_events: list[FaultEvent]
    timing_log: TimingDefectLog
    status: str


def sample_conditions(
    n_flights: int, cfg: GenerateConditions, master_seed: int
) -> list[dict[str, float | str]]:
    """
    Sobol-sample `n_flights` condition vectors over `cfg`'s ranges (sim spec §8).

    Parameters
    ----------
    n_flights : int
    cfg : GenerateConditions
        Ranges for `alt0_m`, `v0_mps`, `mass_kg`, `wind_mean_mps`, `dT_K`, and the
        discrete `turb` options.
    master_seed : int
        Study seed; the Sobol sampler and `turb` assignment are both deterministic
        given this (FR-COR-006).

    Returns
    -------
    list[dict]
        One dict per flight with keys matching `_CONDITION_DIMS` plus `"turb"`.
    """
    # Sobol's balance properties require a power-of-2 sample count; draw the next
    # power of 2 and truncate rather than calling .random(n) with an arbitrary n
    # (which still works, but silently degrades the sequence's balance).
    sampler = qmc.Sobol(d=len(_CONDITION_DIMS), scramble=True, seed=master_seed)
    exponent = int(np.ceil(np.log2(max(n_flights, 1))))
    unit_points = sampler.random_base2(m=exponent)[:n_flights]
    ranges = {
        "alt0_m": cfg.alt0_m,
        "v0_mps": cfg.v0_mps,
        "mass_kg": cfg.mass_kg,
        "wind_mean_mps": cfg.wind_mean_mps,
        "dT_K": cfg.dT_K,
    }

    conditions: list[dict[str, float | str]] = []
    for i in range(n_flights):
        point: dict[str, float | str] = {
            dim: float(ranges[dim][0] + unit_points[i, d] * (ranges[dim][1] - ranges[dim][0]))
            for d, dim in enumerate(_CONDITION_DIMS)
        }
        point["turb"] = cfg.turb[i % len(cfg.turb)]
        conditions.append(point)
    return conditions


def estimate_wind_from_telemetry(telemetry: pd.DataFrame) -> tuple[float, float, float]:
    """
    Estimate a constant wind vector from ground speed minus airspeed (sim spec §5,
    V5's wind-estimating structural switch), assuming ground and air velocity share
    `chi_rad`'s direction (valid when wind is small relative to airspeed).

    Parameters
    ----------
    telemetry : pandas.DataFrame
        Must have `vel_gnd_mps`, `vel_air_mps`, `chi_rad` columns.

    Returns
    -------
    tuple[float, float, float]
        `(wind_n_mps, wind_e_mps, 0.0)`.
    """
    diff = (telemetry["vel_gnd_mps"] - telemetry["vel_air_mps"]).to_numpy()
    chi = telemetry["chi_rad"].to_numpy()
    wind_n = float(np.nanmean(diff * np.cos(chi))) if len(diff) else 0.0
    wind_e = float(np.nanmean(diff * np.sin(chi))) if len(diff) else 0.0
    return wind_n, wind_e, 0.0


def generate_flight(
    index: int,
    scenario_name: str,
    conditions: dict[str, float | str],
    config: StudyConfig,
    truth: TruthModel,
    variants: dict[str, VariantModel],
    sensor_suite: dict[str, SensorModel],
    timing_params: TimingDefectParams,
    fault_library: dict[str, FaultSpec],
    scenarios_dir: Path,
) -> FlightArtifacts:
    """
    Generate one complete flight: truth, telemetry, and every variant's replay.

    Parameters
    ----------
    index : int
        Zero-based flight index (determines `flight_id` and the derived seed).
    scenario_name : str
        Scenario YAML name (`configs/scenarios/<name>.yaml`).
    conditions : dict
        One entry from `sample_conditions`.
    config : StudyConfig
    truth : TruthModel
    variants : dict[str, VariantModel]
        Every variant to replay (V5's wind estimate is computed from this flight's
        telemetry automatically if `variants` includes `"V5"`).
    sensor_suite : dict[str, SensorModel]
    timing_params : TimingDefectParams
    fault_library : dict[str, FaultSpec]
    scenarios_dir : Path

    Returns
    -------
    FlightArtifacts
    """
    flight_id = f"F{index + 1:04d}"
    seed = child_seed(config.study.seed, index)
    rng = np.random.Generator(np.random.PCG64(seed))

    init = State(
        t_s=0.0,
        x_m=0.0,
        y_m=0.0,
        h_m=float(conditions["alt0_m"]),
        V_mps=float(conditions["v0_mps"]),
        gamma_rad=0.0,
        chi_rad=0.0,
        m_kg=float(conditions["mass_kg"]),
    )
    scenario = load_scenario_yaml(scenario_name, scenarios_dir, init)

    wind_dir_rad = float(rng.uniform(0.0, 2.0 * np.pi))
    wind_mean = float(conditions["wind_mean_mps"])
    wind = WindParams(
        mean_n_mps=wind_mean * np.cos(wind_dir_rad),
        mean_e_mps=wind_mean * np.sin(wind_dir_rad),
        turbulence_rms_mps=_TURB_RMS_MPS[str(conditions["turb"])],
        turbulence_scale_length_m=200.0,
    )

    fault_events = sample_fault_schedule(
        fault_library, scenario.duration_s, rng, config.generate.faults.probability_per_flight
    )
    truth_result = truth.run(
        scenario,
        seed=seed,
        wind=wind,
        param_fault_fn=build_param_fault_fn(fault_events),
        wind_shear_fn=build_wind_shear_fn(fault_events),
        delta_t_k=float(conditions["dT_K"]),
    )
    truth_df = truth_result.trajectory
    truth_df["flight_id"] = flight_id

    decimation = max(1, round(config.generate.truth_rate_hz / config.generate.telemetry_rate_hz))
    telemetry_base = truth_df.iloc[::decimation].reset_index(drop=True)
    telemetry_df, _sensor_logs = apply_sensors(telemetry_base, sensor_suite, rng)
    telemetry_df = apply_sensor_bias_faults(telemetry_df, fault_events, sensor_suite)

    t_corrupted, keep_mask, timing_log = apply_timing_defects(
        telemetry_df["t_s"].to_numpy(dtype=np.float64), timing_params, rng
    )
    telemetry_df = telemetry_df.loc[keep_mask].reset_index(drop=True)
    telemetry_df["t_s"] = t_corrupted[keep_mask]
    telemetry_df["flight_id"] = flight_id

    control_cols = ["t_s", "throttle", "alpha_cmd_rad", "bank_cmd_rad"]
    controls_df = telemetry_df[control_cols].dropna().sort_values("t_s").drop_duplicates("t_s")
    controls = controls_from_dataframe(controls_df)
    # Phase 3 will replace this with a real smoothed state estimate (FR-PRP-011);
    # Phase 2 replays from truth's own first sample (data spec §9 step 5).
    init_row = truth_df.iloc[0]

    variant_dfs: dict[str, pd.DataFrame] = {}
    for name, variant in variants.items():
        wind_mps = estimate_wind_from_telemetry(telemetry_df) if name == "V5" else (0.0, 0.0, 0.0)
        result = variant.run(
            None,
            controls,
            state_from_row(init_row),
            seed=seed,
            wind_mps=wind_mps,
            delta_t_k=float(conditions["dT_K"]),
        )
        df = result.trajectory
        df["flight_id"] = flight_id
        variant_dfs[name] = df

    return FlightArtifacts(
        flight_id=flight_id,
        scenario_name=scenario_name,
        seed=seed,
        conditions=conditions,
        truth_df=truth_df,
        telemetry_df=telemetry_df,
        variant_dfs=variant_dfs,
        fault_events=fault_events,
        timing_log=timing_log,
        status=truth_result.status,
    )


def generate_all_flights(
    config: StudyConfig,
    nominal_vehicle: VehicleParams,
    truth: TruthModel,
    variants: dict[str, VariantModel],
    sensor_suite: dict[str, SensorModel],
    timing_params: TimingDefectParams,
    fault_library: dict[str, FaultSpec],
    scenarios_dir: Path,
    *,
    n_jobs: int = 1,
) -> list[FlightArtifacts]:
    """
    Generate every flight for a study (FR-TEL-004): `fsa generate --profile demo`
    produces exactly `config.generate.n_flights` flights, deterministically.

    Parameters
    ----------
    config : StudyConfig
    nominal_vehicle : VehicleParams
        Unused directly (kept for signature symmetry / future per-flight vehicle
        variation); `truth` and `variants` already carry their own parameters.
    truth : TruthModel
    variants : dict[str, VariantModel]
    sensor_suite : dict[str, SensorModel]
    timing_params : TimingDefectParams
    fault_library : dict[str, FaultSpec]
    scenarios_dir : Path
    n_jobs : int, default 1
        `joblib.Parallel` worker count; each flight's RNG is seeded independently of
        execution order (`core/rng.child_seed`), so results are identical regardless
        of `n_jobs` (FR-COR-006).

    Returns
    -------
    list[FlightArtifacts]
        Length `config.generate.n_flights`.
    """
    n_flights = config.generate.n_flights
    conditions = sample_conditions(n_flights, config.generate.conditions, config.study.seed)
    scenario_names = config.generate.scenarios

    def _one(index: int) -> FlightArtifacts:
        scenario_name = scenario_names[index % len(scenario_names)]
        return generate_flight(
            index,
            scenario_name,
            conditions[index],
            config,
            truth,
            variants,
            sensor_suite,
            timing_params,
            fault_library,
            scenarios_dir,
        )

    results: list[FlightArtifacts] = Parallel(n_jobs=n_jobs)(
        delayed(_one)(i) for i in range(n_flights)
    )
    return results
