from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerovalid.core.errors import ConfigError
from aerovalid.sim.protocols import ControlHistory, State, VehicleParams
from aerovalid.sim.truth import TruthModel
from aerovalid.sim.variants import (
    load_all_variants,
    load_variant_yaml,
    make_variant,
    make_variant_v5,
)

_INIT = State(
    t_s=0.0, x_m=0.0, y_m=0.0, h_m=1500.0, V_mps=30.0, gamma_rad=0.0, chi_rad=0.0, m_kg=25.0
)


def _constant_controls(
    duration_s: float, throttle: float = 0.5, alpha: float = 0.05, bank: float = 0.0
) -> ControlHistory:
    t_s = np.array([0.0, duration_s])
    return ControlHistory(
        t_s=t_s,
        throttle=np.full(2, throttle),
        alpha_cmd_rad=np.full(2, alpha),
        bank_cmd_rad=np.full(2, bank),
    )


@pytest.mark.req("FR-SIM-007")
def test_make_variant_applies_multiplicative_overrides(
    variants_dir: Path, test_vehicle: VehicleParams
) -> None:
    v1_spec = load_variant_yaml("V1", variants_dir)
    v1 = make_variant(v1_spec, test_vehicle)
    assert pytest.approx(test_vehicle.CD0 * 1.20) == v1.vehicle.CD0
    assert v1.vehicle.CLa == pytest.approx(test_vehicle.CLa)  # untouched

    v2_spec = load_variant_yaml("V2", variants_dir)
    v2 = make_variant(v2_spec, test_vehicle)
    assert v2.vehicle.CLa == pytest.approx(test_vehicle.CLa * 0.88)
    assert pytest.approx(test_vehicle.CL0 * 0.90) == v2.vehicle.CL0


@pytest.mark.req("FR-SIM-007")
def test_v3_is_marked_atmosphere_naive(variants_dir: Path, test_vehicle: VehicleParams) -> None:
    v3 = make_variant(load_variant_yaml("V3", variants_dir), test_vehicle)
    assert v3.atmosphere_naive is True
    assert v3.switches.thrust_lapse is False


@pytest.mark.req("FR-SIM-007")
def test_v4_enables_compressibility_and_wave_drag(
    variants_dir: Path, test_vehicle: VehicleParams
) -> None:
    v4 = make_variant(load_variant_yaml("V4", variants_dir), test_vehicle)
    assert v4.switches.compressibility is True
    assert v4.switches.wave_drag is True


@pytest.mark.req("FR-SIM-007")
def test_load_all_variants_v0_through_v4(variants_dir: Path, test_vehicle: VehicleParams) -> None:
    variants = load_all_variants(["V0", "V1", "V2", "V3", "V4"], variants_dir, test_vehicle)
    assert set(variants) == {"V0", "V1", "V2", "V3", "V4"}


@pytest.mark.req("FR-SIM-007")
def test_missing_variant_raises_config_error(variants_dir: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_variant_yaml("V99", variants_dir)


@pytest.mark.req("FR-SIM-007")
def test_variant_replay_runs_without_nan(variants_dir: Path, test_vehicle: VehicleParams) -> None:
    v0 = make_variant(load_variant_yaml("V0", variants_dir), test_vehicle)
    controls = _constant_controls(duration_s=5.0)
    result = v0.run(None, controls, _INIT, seed=0)

    assert result.status == "ok"
    assert result.trajectory["source"].iloc[0] == "V0"
    numeric_cols = result.trajectory.select_dtypes(include="number").columns
    assert np.all(np.isfinite(result.trajectory[numeric_cols].to_numpy()))


@pytest.mark.req("FR-SIM-007")
def test_variant_replay_requires_controls_and_init(
    variants_dir: Path, test_vehicle: VehicleParams
) -> None:
    v0 = make_variant(load_variant_yaml("V0", variants_dir), test_vehicle)
    with pytest.raises(ValueError, match="requires both controls and init"):
        v0.run(None, None, None, seed=0)


@pytest.mark.req("FR-SIM-007")
def test_make_variant_v5_applies_small_noise_to_truth_params(test_vehicle: VehicleParams) -> None:
    truth = TruthModel.from_nominal(test_vehicle, master_seed=11)
    rng = np.random.Generator(np.random.PCG64(99))
    v5 = make_variant_v5(truth.vehicle, rng)

    assert v5.name == "V5"
    assert v5.switches.alpha_dynamics is False
    # within a generous multiple of the 2% noise sigma
    assert pytest.approx(truth.vehicle.CD0, rel=0.1) == v5.vehicle.CD0
    assert v5.vehicle.CD0 != truth.vehicle.CD0


@pytest.mark.req("FR-SIM-007")
def test_make_variant_v5_is_deterministic_given_rng_seed(test_vehicle: VehicleParams) -> None:
    truth = TruthModel.from_nominal(test_vehicle, master_seed=11)
    v5_a = make_variant_v5(truth.vehicle, np.random.Generator(np.random.PCG64(5)))
    v5_b = make_variant_v5(truth.vehicle, np.random.Generator(np.random.PCG64(5)))
    assert pytest.approx(v5_b.vehicle.CD0) == v5_a.vehicle.CD0
