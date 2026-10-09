from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerovalid.telemetry.sensors import (
    SensorModel,
    apply_sensor_model,
    apply_sensors,
    load_sensor_suite,
)


@pytest.fixture
def default_sensors_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "sensors" / "default.yaml"


@pytest.mark.req("FR-TEL-001")
def test_resolved_sigma_prefers_degrees() -> None:
    model = SensorModel(sigma=0.0, sigma_deg=1.0)
    assert model.resolved_sigma() == pytest.approx(np.deg2rad(1.0))


@pytest.mark.req("FR-TEL-001")
def test_resolved_bias_prefers_degrees() -> None:
    model = SensorModel(bias=0.0, bias_deg=2.0)
    assert model.resolved_bias() == pytest.approx(np.deg2rad(2.0))


@pytest.mark.req("FR-TEL-001")
def test_white_noise_sigma_recovered_within_5_percent() -> None:
    rng = np.random.Generator(np.random.PCG64(1))
    n = 20_000
    t_s = np.arange(n) * 0.1
    signal = np.zeros(n)
    model = SensorModel(sigma=1.5)

    corrupted, _log = apply_sensor_model(signal, t_s, model, rng)

    assert np.std(corrupted) == pytest.approx(1.5, rel=0.05)


@pytest.mark.req("FR-TEL-001")
def test_bias_recovered_within_5_percent() -> None:
    rng = np.random.Generator(np.random.PCG64(2))
    n = 20_000
    t_s = np.arange(n) * 0.1
    signal = np.zeros(n)
    model = SensorModel(sigma=0.1, bias=2.0)

    corrupted, _log = apply_sensor_model(signal, t_s, model, rng)

    assert np.mean(corrupted) == pytest.approx(2.0, rel=0.05)


@pytest.mark.req("FR-TEL-001")
def test_scale_factor_applied() -> None:
    rng = np.random.Generator(np.random.PCG64(3))
    t_s = np.arange(100) * 0.1
    signal = np.full(100, 10.0)
    model = SensorModel(scale=1.005)

    corrupted, _log = apply_sensor_model(signal, t_s, model, rng)

    assert np.mean(corrupted) == pytest.approx(10.05, rel=1e-3)


@pytest.mark.req("FR-TEL-001")
def test_quantisation_rounds_to_step() -> None:
    rng = np.random.Generator(np.random.PCG64(4))
    t_s = np.arange(10) * 0.1
    signal = np.array([0.0, 0.021, 0.049, 0.074, 0.1, 0.12, 0.17, 0.19, 0.2, 0.3])
    model = SensorModel(quantise=0.05)

    corrupted, _log = apply_sensor_model(signal, t_s, model, rng)

    remainder = np.abs(np.round(corrupted / 0.05) * 0.05 - corrupted)
    assert np.all(remainder < 1e-9)


@pytest.mark.req("FR-TEL-001")
def test_dropout_produces_expected_fraction_of_nan() -> None:
    rng = np.random.Generator(np.random.PCG64(5))
    n = 20_000
    t_s = np.arange(n) * 0.1
    signal = np.ones(n)
    model = SensorModel(dropout_p=0.05)

    corrupted, log = apply_sensor_model(signal, t_s, model, rng)

    nan_fraction = np.mean(np.isnan(corrupted))
    assert nan_fraction == pytest.approx(0.05, abs=0.01)
    assert len(log.dropout_indices) == int(np.isnan(corrupted).sum())


@pytest.mark.req("FR-TEL-001")
def test_lag_smooths_a_step_change() -> None:
    rng = np.random.Generator(np.random.PCG64(6))
    t_s = np.arange(200) * 0.1
    signal = np.concatenate([np.zeros(100), np.ones(100)])
    model = SensorModel(lag_s=1.0)

    corrupted, _log = apply_sensor_model(signal, t_s, model, rng)

    # immediately after the step, the lagged signal should not have fully caught up
    assert corrupted[100] < 0.5
    # but should converge back toward 1.0 well after the step
    assert corrupted[-1] == pytest.approx(1.0, abs=0.05)


@pytest.mark.req("FR-TEL-001")
def test_disabled_sensor_passes_through_unchanged() -> None:
    rng = np.random.Generator(np.random.PCG64(7))
    t_s = np.arange(10) * 0.1
    signal = np.arange(10, dtype=float)
    model = SensorModel(sigma=5.0, enabled=False)

    corrupted, log = apply_sensor_model(signal, t_s, model, rng)

    assert np.array_equal(corrupted, signal)
    assert log.dropout_indices == []


@pytest.mark.req("FR-TEL-001")
def test_load_sensor_suite_from_real_config(default_sensors_path: Path) -> None:
    suite = load_sensor_suite(default_sensors_path)
    assert "alt_m" in suite
    assert "timing_defects" not in suite
    assert suite["alt_m"].sigma == pytest.approx(1.5)
    assert suite["yaw_rad"].drift_deg_per_sqrts == pytest.approx(0.05)


@pytest.mark.req("FR-TEL-001")
def test_apply_sensors_corrupts_a_truth_dataframe(default_sensors_path: Path) -> None:
    suite = load_sensor_suite(default_sensors_path)
    rng = np.random.Generator(np.random.PCG64(8))
    n = 100
    truth = pd.DataFrame(
        {
            "t_s": np.arange(n) * 0.1,
            "flight_id": "F0001",
            "source": "truth",
            "alt_m": np.full(n, 1000.0),
            "vel_air_mps": np.full(n, 30.0),
        }
    )

    telemetry, logs = apply_sensors(truth, suite, rng)

    assert telemetry["source"].iloc[0] == "tel"
    assert "alt_m" in logs
    assert len(telemetry) == len(truth)
    assert telemetry["alt_m"].to_numpy() is not truth["alt_m"].to_numpy()
