from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.identify.revalidate import build_variant_star, revalidate_on_held_out_flights
from aerovalid.sim.protocols import StructuralSwitches, VehicleParams
from aerovalid.sim.runners import controls_from_dataframe, state_from_row
from aerovalid.sim.variants import VariantModel


def _control_df() -> pd.DataFrame:
    t_s = np.arange(0.0, 15.0, 0.05)
    return pd.DataFrame(
        {
            "t_s": t_s,
            "throttle": np.full_like(t_s, 0.55),
            "alpha_cmd_rad": np.full_like(t_s, 0.05),
            "bank_cmd_rad": np.zeros_like(t_s),
        }
    )


def _run_trajectory(
    vehicle: VehicleParams, switches: StructuralSwitches, control_df: pd.DataFrame
) -> pd.DataFrame:
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
    variant = VariantModel(name="_gen", vehicle=vehicle, switches=switches, dt_s=0.05)
    result = variant.run(
        None, controls_from_dataframe(control_df), state_from_row(init_row), seed=0
    )
    return result.trajectory


@pytest.mark.slow
@pytest.mark.req("FR-IDN-006")
def test_revalidation_shows_refit_improves_rmse_vs_original(test_vehicle: VehicleParams) -> None:
    switches = StructuralSwitches(thrust_lapse=True)
    control_df = _control_df()

    true_vehicle = test_vehicle.model_copy(update={"CD0": test_vehicle.CD0 * 1.2})
    truth_trajectory = _run_trajectory(true_vehicle, switches, control_df)

    # "telemetry" = truth's own trajectory (noise-free, for a clean comparison) plus
    # the recorded controls.
    telemetry = truth_trajectory[
        ["t_s", "alt_m", "vel_air_mps", "gamma_rad", "chi_rad", "mass_kg"]
    ].merge(control_df, on="t_s")

    # The "original" (flawed) variant uses the nominal CD0 as-is.
    original_variant = VariantModel(
        name="original", vehicle=test_vehicle, switches=switches, dt_s=0.05
    )
    original_trajectory = original_variant.run(
        None,
        controls_from_dataframe(control_df),
        state_from_row(telemetry.iloc[0]),
        seed=0,
    ).trajectory

    # The refit variant recovers (most of) the true CD0.
    variant_star = build_variant_star(
        test_vehicle, {"CD0": true_vehicle.CD0 * 0.95}, switches, dt_s=0.05, name="V_star"
    )

    table = revalidate_on_held_out_flights(
        variant_star,
        original_trajectories={"F0001": original_trajectory},
        held_out_telemetry={"F0001": telemetry},
        signals=["alt_m", "vel_air_mps"],
    )

    assert set(table["signal"]) == {"alt_m", "vel_air_mps"}
    for _, row in table.iterrows():
        assert row["rmse_refit"] < row["rmse_original"]
        assert row["rmse_reduction_pct"] > 0.0


@pytest.mark.req("FR-IDN-006")
def test_build_variant_star_overrides_only_the_fitted_fields(test_vehicle: VehicleParams) -> None:
    switches = StructuralSwitches()
    variant_star = build_variant_star(
        test_vehicle, {"CD0": 0.05}, switches, dt_s=0.05, name="V_star"
    )
    assert pytest.approx(0.05) == variant_star.vehicle.CD0
    assert pytest.approx(test_vehicle.CL0) == variant_star.vehicle.CL0
    assert variant_star.name == "V_star"
