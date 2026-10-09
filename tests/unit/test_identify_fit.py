from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.identify.fit import fit_parameters
from aerovalid.sim.protocols import StructuralSwitches, VehicleParams
from aerovalid.sim.runners import controls_from_dataframe, state_from_row
from aerovalid.sim.variants import VariantModel


def _synthetic_telemetry(vehicle: VehicleParams, switches: StructuralSwitches) -> pd.DataFrame:
    """Replay a known vehicle to produce synthetic "telemetry" with a known ground
    truth -- the same model structure generates and fits the data, isolating pure
    parameter identifiability from any structural model error (S7's intent)."""
    t_s = np.arange(0.0, 20.0, 0.05)
    control_df = pd.DataFrame(
        {
            "t_s": t_s,
            "throttle": np.full_like(t_s, 0.55),
            "alpha_cmd_rad": np.full_like(t_s, 0.05),
            "bank_cmd_rad": np.zeros_like(t_s),
        }
    )
    init_row = pd.Series(
        {
            "t_s": 0.0,
            "alt_m": 1500.0,
            "vel_air_mps": 30.0,
            "gamma_rad": 0.0,
            "chi_rad": 0.0,
            "mass_kg": vehicle.mass_nom_kg,
        }
    )
    variant = VariantModel(name="_truth", vehicle=vehicle, switches=switches, dt_s=0.05)
    result = variant.run(
        None, controls_from_dataframe(control_df), state_from_row(init_row), seed=0
    )
    trajectory = result.trajectory[
        ["t_s", "alt_m", "vel_air_mps", "gamma_rad", "chi_rad", "mass_kg"]
    ]
    return trajectory.merge(control_df, on="t_s")


@pytest.mark.slow
@pytest.mark.req("FR-IDN-002")
def test_fit_parameters_recovers_an_injected_cd0_error(test_vehicle: VehicleParams) -> None:
    switches = StructuralSwitches(thrust_lapse=True)  # same structure for data + fit (S7)
    true_cd0 = test_vehicle.CD0 * 1.2  # a +20% "unmodelled drag" error, like V0's known_flaw
    true_vehicle = test_vehicle.model_copy(update={"CD0": true_cd0})

    telemetry = _synthetic_telemetry(true_vehicle, switches)

    result = fit_parameters(
        param_names=["CD0"],
        bounds={"CD0": (0.01, 0.1)},
        nominal=test_vehicle,
        switches=switches,
        dt_s=0.05,
        telemetry=telemetry,
        signals=["alt_m", "vel_air_mps"],
        weights={"alt_m": 1.0, "vel_air_mps": 1.0},
        horizon_s=8.0,
        stride_s=4.0,
    )

    fitted_cd0 = result.params["CD0"]
    assert abs(fitted_cd0 - true_cd0) < abs(test_vehicle.CD0 - true_cd0) * 0.3
    assert result.success


@pytest.mark.slow
@pytest.mark.req("FR-IDN-002")
def test_fit_parameters_at_the_true_value_has_near_zero_residual(
    test_vehicle: VehicleParams,
) -> None:
    switches = StructuralSwitches(thrust_lapse=True)
    telemetry = _synthetic_telemetry(test_vehicle, switches)

    result = fit_parameters(
        param_names=["CD0"],
        bounds={"CD0": (0.01, 0.1)},
        nominal=test_vehicle,
        switches=switches,
        dt_s=0.05,
        telemetry=telemetry,
        signals=["alt_m"],
        weights={"alt_m": 1.0},
        horizon_s=8.0,
        stride_s=4.0,
        x0=np.array([test_vehicle.CD0]),
    )
    assert np.max(np.abs(result.residuals)) < 1e-6
