from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerovalid.core.errors import ConfigError
from aerovalid.sim.maneuvers import build_scenario, load_scenario_yaml, setpoint_at
from aerovalid.sim.protocols import State

_INIT = State(
    t_s=0.0, x_m=0.0, y_m=0.0, h_m=1000.0, V_mps=30.0, gamma_rad=0.0, chi_rad=0.0, m_kg=25.0
)


@pytest.mark.req("FR-SIM-008")
def test_trim_cruise_setpoint_is_constant() -> None:
    scenario = build_scenario(
        "t", [{"type": "trim_cruise", "v_mps": 35.0, "h_m": 1500.0, "duration_s": 10.0}], _INIT
    )
    setpoint = setpoint_at(scenario.segments[0], 5.0)
    assert setpoint.v_cmd_mps == pytest.approx(35.0)
    assert setpoint.h_cmd_m == pytest.approx(1500.0)
    assert setpoint.bank_cmd_rad == pytest.approx(0.0)


@pytest.mark.req("FR-SIM-008")
def test_climb_ramps_altitude_linearly() -> None:
    scenario = build_scenario(
        "t", [{"type": "climb", "h_target_m": 2000.0, "v_mps": 30.0, "duration_s": 100.0}], _INIT
    )
    segment = scenario.segments[0]
    assert setpoint_at(segment, 0.0).h_cmd_m == pytest.approx(1000.0)
    assert setpoint_at(segment, 50.0).h_cmd_m == pytest.approx(1500.0)
    assert setpoint_at(segment, 100.0).h_cmd_m == pytest.approx(2000.0)


@pytest.mark.req("FR-SIM-008")
def test_coordinated_turn_duration_computed_from_heading_change() -> None:
    bank, heading_change, v = 0.5236, np.pi, 30.0  # 30 deg bank, 180 deg turn
    scenario = build_scenario(
        "t",
        [
            {
                "type": "coordinated_turn",
                "bank_rad": bank,
                "heading_change_rad": heading_change,
                "v_mps": v,
            }
        ],
        _INIT,
    )
    expected_duration = heading_change / (9.80665 * np.tan(bank) / v)
    assert scenario.segments[0]["duration_s"] == pytest.approx(expected_duration, rel=1e-6)
    assert setpoint_at(scenario.segments[0], 1.0).bank_cmd_rad == pytest.approx(bank)


@pytest.mark.req("FR-SIM-008")
def test_s_turn_alternates_bank_sign() -> None:
    scenario = build_scenario(
        "t",
        [{"type": "s_turn", "bank_rad": 0.4, "period_s": 10.0, "duration_s": 20.0, "v_mps": 30.0}],
        _INIT,
    )
    segment = scenario.segments[0]
    assert setpoint_at(segment, 1.0).bank_cmd_rad == pytest.approx(0.4)
    assert setpoint_at(segment, 6.0).bank_cmd_rad == pytest.approx(-0.4)


@pytest.mark.req("FR-SIM-008")
def test_speed_step_offsets_baseline() -> None:
    scenario = build_scenario(
        "t", [{"type": "speed_step", "delta_v_mps": 5.0, "duration_s": 10.0}], _INIT
    )
    assert setpoint_at(scenario.segments[0], 0.0).v_cmd_mps == pytest.approx(35.0)


@pytest.mark.req("FR-SIM-008")
def test_pitch_doublet_overrides_alpha_in_two_halves() -> None:
    scenario = build_scenario(
        "t",
        [{"type": "pitch_doublet", "alpha_amp_rad": 0.05, "period_s": 4.0, "duration_s": 4.0}],
        _INIT,
    )
    segment = scenario.segments[0]
    assert setpoint_at(segment, 1.0).alpha_cmd_override_rad == pytest.approx(0.05)
    assert setpoint_at(segment, 3.0).alpha_cmd_override_rad == pytest.approx(-0.05)


@pytest.mark.req("FR-SIM-008")
def test_loiter_bank_matches_turn_radius_formula() -> None:
    radius, v = 150.0, 28.0
    scenario = build_scenario(
        "t", [{"type": "loiter", "radius_m": radius, "v_mps": v, "duration_s": 60.0}], _INIT
    )
    expected_bank = np.arctan(v**2 / (9.80665 * radius))
    assert setpoint_at(scenario.segments[0], 10.0).bank_cmd_rad == pytest.approx(expected_bank)


@pytest.mark.req("FR-SIM-008")
def test_throttle_chop_overrides_throttle() -> None:
    scenario = build_scenario(
        "t", [{"type": "throttle_chop", "throttle": 0.02, "duration_s": 5.0}], _INIT
    )
    setpoint = setpoint_at(scenario.segments[0], 1.0)
    assert setpoint.throttle_override == pytest.approx(0.02)


@pytest.mark.req("FR-SIM-008")
def test_unknown_segment_type_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="Unknown manoeuvre"):
        setpoint_at(
            {
                "type": "warp_drive",
                "duration_s": 1.0,
                "v_cmd_start_mps": 30.0,
                "h_cmd_start_m": 1000.0,
            },
            0.0,
        )


@pytest.mark.req("FR-SIM-008")
def test_segment_missing_duration_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="missing duration_s"):
        build_scenario("t", [{"type": "trim_cruise", "v_mps": 30.0}], _INIT)


@pytest.mark.req("FR-SIM-008")
def test_load_scenario_yaml_resolves_real_mixed_mission(scenarios_dir: Path) -> None:
    scenario = load_scenario_yaml("mixed_mission", scenarios_dir, _INIT)
    assert len(scenario.segments) > 0
    assert scenario.duration_s > 0
    for segment in scenario.segments:
        setpoint = setpoint_at(segment, 0.0)
        assert setpoint.v_cmd_mps > 0


@pytest.mark.req("FR-SIM-008")
def test_load_missing_scenario_raises_config_error(scenarios_dir: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_scenario_yaml("does_not_exist", scenarios_dir, _INIT)
