from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerovalid.sim.dynamics import FaultState
from aerovalid.telemetry.faults import (
    FaultEvent,
    apply_sensor_bias_faults,
    build_param_fault_fn,
    build_wind_shear_fn,
    load_fault_library,
    sample_fault_schedule,
)
from aerovalid.telemetry.sensors import SensorModel


@pytest.fixture
def faults_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "faults"


@pytest.mark.req("FR-TEL-003")
def test_load_fault_library_has_all_seven_types(faults_dir: Path) -> None:
    library = load_fault_library(faults_dir)
    expected = {
        "cd0_step",
        "cla_degrade",
        "thrust_loss",
        "icing",
        "ctrl_eff_loss",
        "sensor_bias_step",
        "wind_shear_event",
    }
    assert set(library) == expected


@pytest.mark.req("FR-TEL-003")
def test_zero_probability_gives_fault_free_flight(faults_dir: Path) -> None:
    library = load_fault_library(faults_dir)
    rng = np.random.Generator(np.random.PCG64(1))
    events = sample_fault_schedule(library, 300.0, rng, probability_per_flight=0.0)
    assert events == []


@pytest.mark.req("FR-TEL-003")
def test_certain_probability_samples_exactly_one_event(faults_dir: Path) -> None:
    library = load_fault_library(faults_dir)
    rng = np.random.Generator(np.random.PCG64(2))
    events = sample_fault_schedule(library, 300.0, rng, probability_per_flight=1.0)
    assert len(events) == 1
    event = events[0]
    assert event.type in library
    spec = library[event.type]
    assert spec.magnitude_range[0] <= event.magnitude <= spec.magnitude_range[1]
    assert 0.0 <= event.t_on_s < event.t_off_s <= 300.0


@pytest.mark.req("FR-TEL-003")
def test_cd0_step_ramps_then_holds() -> None:
    event = FaultEvent(type="cd0_step", t_on_s=10.0, t_off_s=100.0, magnitude=0.2, ramp_s=5.0)
    fault_fn = build_param_fault_fn([event])

    assert fault_fn(5.0) == FaultState()  # before onset
    mid_ramp = fault_fn(12.5)
    assert 1.0 < mid_ramp.cd0_mult < 1.2
    full = fault_fn(50.0)
    assert full.cd0_mult == pytest.approx(1.2)
    assert fault_fn(200.0) == FaultState()  # after t_off


@pytest.mark.req("FR-TEL-003")
def test_icing_affects_three_parameters() -> None:
    event = FaultEvent(type="icing", t_on_s=0.0, t_off_s=100.0, magnitude=0.2, ramp_s=0.0)
    fault_fn = build_param_fault_fn([event])
    state = fault_fn(50.0)
    assert state.cd0_mult == pytest.approx(1.2)
    assert state.clmax_mult == pytest.approx(0.8)
    assert state.cla_mult == pytest.approx(0.9)


@pytest.mark.req("FR-TEL-003")
def test_non_param_events_ignored_by_param_fault_fn() -> None:
    event = FaultEvent(
        type="wind_shear_event", t_on_s=0.0, t_off_s=30.0, magnitude=10.0, ramp_s=2.0
    )
    fault_fn = build_param_fault_fn([event])
    assert fault_fn(10.0) == FaultState()


@pytest.mark.req("FR-TEL-003")
def test_wind_shear_fn_ramps() -> None:
    event = FaultEvent(
        type="wind_shear_event", t_on_s=10.0, t_off_s=40.0, magnitude=10.0, ramp_s=2.0
    )
    shear_fn = build_wind_shear_fn([event])
    assert shear_fn(5.0) == (0.0, 0.0)
    delta_n, delta_e = shear_fn(12.0)
    assert delta_n == pytest.approx(10.0)
    assert delta_e == pytest.approx(0.0)


@pytest.mark.req("FR-TEL-003")
def test_apply_sensor_bias_faults_adds_step() -> None:
    telemetry = pd.DataFrame({"t_s": [0.0, 1.0, 2.0, 3.0], "alt_m": [100.0, 100.0, 100.0, 100.0]})
    event = FaultEvent(
        type="sensor_bias_step",
        t_on_s=2.0,
        t_off_s=10.0,
        magnitude=5.0,
        ramp_s=0.0,
        affected_signals=["alt_m"],
    )
    sensor_suite = {"alt_m": SensorModel(sigma=1.5)}

    out = apply_sensor_bias_faults(telemetry, [event], sensor_suite)

    assert out["alt_m"].tolist() == [100.0, 100.0, 107.5, 107.5]
