from __future__ import annotations

import numpy as np
import pytest

from aerovalid.sim.maneuvers import build_scenario
from aerovalid.sim.protocols import Scenario, State, VehicleParams
from aerovalid.sim.truth import TRUTH_PARAM_SIGMAS, TruthModel

_INIT = State(
    t_s=0.0, x_m=0.0, y_m=0.0, h_m=1500.0, V_mps=30.0, gamma_rad=0.0, chi_rad=0.0, m_kg=25.0
)


def _short_cruise_scenario(duration_s: float = 3.0) -> Scenario:
    return build_scenario(
        "short_cruise",
        [{"type": "trim_cruise", "v_mps": 30.0, "h_m": 1500.0, "duration_s": duration_s}],
        _INIT,
    )


@pytest.mark.req("FR-SIM-006")
def test_hidden_perturbations_are_deterministic_given_master_seed(
    test_vehicle: VehicleParams,
) -> None:
    truth_a = TruthModel.from_nominal(test_vehicle, master_seed=12345)
    truth_b = TruthModel.from_nominal(test_vehicle, master_seed=12345)
    assert truth_a.perturbations == truth_b.perturbations
    assert pytest.approx(truth_b.vehicle.CD0) == truth_a.vehicle.CD0


@pytest.mark.req("FR-SIM-006")
def test_different_master_seed_gives_different_perturbations(test_vehicle: VehicleParams) -> None:
    truth_a = TruthModel.from_nominal(test_vehicle, master_seed=1)
    truth_b = TruthModel.from_nominal(test_vehicle, master_seed=2)
    assert truth_a.perturbations != truth_b.perturbations


@pytest.mark.req("FR-SIM-006")
def test_perturbations_cover_every_hidden_parameter_plus_mass(test_vehicle: VehicleParams) -> None:
    truth = TruthModel.from_nominal(test_vehicle, master_seed=7)
    assert set(truth.perturbations) == set(TRUTH_PARAM_SIGMAS) | {"mass_offset_kg"}
    assert truth.vehicle.CL0 != test_vehicle.CL0  # perturbation actually applied


@pytest.mark.req("FR-SIM-006")
def test_truth_run_produces_finite_trajectory_without_wind(test_vehicle: VehicleParams) -> None:
    truth = TruthModel.from_nominal(test_vehicle, master_seed=42)
    scenario = _short_cruise_scenario()
    result = truth.run(scenario, seed=1)

    assert result.status == "ok"
    trajectory = result.trajectory
    assert len(trajectory) > 0
    assert trajectory["source"].iloc[0] == "truth"
    numeric_cols = trajectory.select_dtypes(include="number").columns
    assert np.all(np.isfinite(trajectory[numeric_cols].to_numpy()))


@pytest.mark.req("FR-SIM-006")
def test_truth_run_requires_scenario(test_vehicle: VehicleParams) -> None:
    truth = TruthModel.from_nominal(test_vehicle, master_seed=42)
    with pytest.raises(ValueError, match="requires a scenario"):
        truth.run(None, seed=1)


@pytest.mark.slow
@pytest.mark.req("FR-SIM-006")
def test_truth_run_with_wind_and_turbulence_stays_finite(test_vehicle: VehicleParams) -> None:
    from aerovalid.sim.environment import WindParams

    truth = TruthModel.from_nominal(test_vehicle, master_seed=5)
    scenario = _short_cruise_scenario(duration_s=20.0)
    wind = WindParams(
        mean_n_mps=4.0, mean_e_mps=2.0, turbulence_rms_mps=1.5, turbulence_scale_length_m=200.0
    )
    result = truth.run(scenario, seed=99, wind=wind)

    assert result.status == "ok"
    numeric_cols = result.trajectory.select_dtypes(include="number").columns
    assert np.all(np.isfinite(result.trajectory[numeric_cols].to_numpy()))
