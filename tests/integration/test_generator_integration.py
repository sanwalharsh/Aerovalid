"""Phase 2 integration checks (FR-TEL-004): `generate_all_flights` produces exactly
`n_flights` flights deterministically (hash-identical re-run). Uses a short custom
scenario (not the real multi-minute ones) to keep this fast enough to not be marked slow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerovalid.core.config import StudyConfig
from aerovalid.sim.protocols import VehicleParams
from aerovalid.sim.truth import TruthModel
from aerovalid.sim.variants import load_all_variants, make_variant_v5
from aerovalid.telemetry.defects import load_timing_defect_params
from aerovalid.telemetry.faults import load_fault_library
from aerovalid.telemetry.generator import generate_all_flights, sample_conditions
from aerovalid.telemetry.sensors import load_sensor_suite

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tiny_scenarios_dir(tmp_path: Path) -> Path:
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    (scenarios_dir / "tiny.yaml").write_text(
        "segments:\n  - {type: trim_cruise, v_mps: 30.0, h_m: 1500.0, duration_s: 4.0}\n"
    )
    return scenarios_dir


@pytest.fixture
def tiny_config() -> StudyConfig:
    config = StudyConfig()
    config.study.seed = 4242
    config.generate.n_flights = 2
    config.generate.scenarios = ["tiny"]
    config.variants = ["V0", "V1", "V5"]
    return config


def _build_and_run(config: StudyConfig, test_vehicle: VehicleParams, scenarios_dir: Path) -> list:
    truth = TruthModel.from_nominal(test_vehicle, master_seed=config.study.seed)
    variants = load_all_variants(["V0", "V1"], _REPO_ROOT / "configs" / "variants", test_vehicle)
    v5_rng = np.random.Generator(np.random.PCG64(config.study.seed + 1))
    variants["V5"] = make_variant_v5(truth.vehicle, v5_rng)
    sensors_path = _REPO_ROOT / "configs" / "sensors" / "default.yaml"
    sensor_suite = load_sensor_suite(sensors_path)
    timing_params = load_timing_defect_params(sensors_path)
    fault_library = load_fault_library(_REPO_ROOT / "configs" / "faults")

    return generate_all_flights(
        config,
        test_vehicle,
        truth,
        variants,
        sensor_suite,
        timing_params,
        fault_library,
        scenarios_dir,
        n_jobs=1,
    )


@pytest.mark.req("FR-TEL-004")
def test_generate_all_flights_produces_exactly_n_flights(
    tiny_config: StudyConfig, test_vehicle: VehicleParams, tiny_scenarios_dir: Path
) -> None:
    flights = _build_and_run(tiny_config, test_vehicle, tiny_scenarios_dir)
    assert len(flights) == 2
    assert [f.flight_id for f in flights] == ["F0001", "F0002"]
    for flight in flights:
        assert flight.status == "ok"
        assert set(flight.variant_dfs) == {"V0", "V1", "V5"}
        assert len(flight.telemetry_df) <= len(flight.truth_df)


@pytest.mark.req("FR-TEL-004")
def test_generate_all_flights_is_deterministic(
    tiny_config: StudyConfig, test_vehicle: VehicleParams, tiny_scenarios_dir: Path
) -> None:
    flights_a = _build_and_run(tiny_config, test_vehicle, tiny_scenarios_dir)
    flights_b = _build_and_run(tiny_config, test_vehicle, tiny_scenarios_dir)

    for flight_a, flight_b in zip(flights_a, flights_b, strict=True):
        pd.testing.assert_frame_equal(flight_a.truth_df, flight_b.truth_df)
        pd.testing.assert_frame_equal(flight_a.telemetry_df, flight_b.telemetry_df)
        for name in flight_a.variant_dfs:
            pd.testing.assert_frame_equal(flight_a.variant_dfs[name], flight_b.variant_dfs[name])


@pytest.mark.req("FR-TEL-004")
def test_sample_conditions_within_configured_ranges(tiny_config: StudyConfig) -> None:
    conditions = sample_conditions(5, tiny_config.generate.conditions, tiny_config.study.seed)
    assert len(conditions) == 5
    alt_low, alt_high = tiny_config.generate.conditions.alt0_m
    for point in conditions:
        assert alt_low <= point["alt0_m"] <= alt_high
        assert point["turb"] in tiny_config.generate.conditions.turb
