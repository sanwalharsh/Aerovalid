from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.prep.phases import PhaseThresholds, segment_phases, tag_conditions


def _scripted_flight(dt: float = 0.1) -> tuple[pd.DataFrame, list[str]]:
    """60 s: 20 s climb, 20 s cruise, 20 s turn (known ground truth)."""
    n_per_segment = int(20.0 / dt)
    climb_rate = np.concatenate(
        [np.full(n_per_segment, 2.0), np.zeros(n_per_segment), np.zeros(n_per_segment)]
    )
    turn_rate = np.concatenate(
        [np.zeros(n_per_segment), np.zeros(n_per_segment), np.full(n_per_segment, 0.2)]
    )
    n = 3 * n_per_segment
    return pd.DataFrame(
        {
            "t_s": np.arange(n) * dt,
            "climb_rate_mps": climb_rate,
            "turn_rate_radps": turn_rate,
            "vel_air_mps": np.full(n, 30.0),
        }
    ), ["climb"] * n_per_segment + ["cruise"] * n_per_segment + ["turn"] * n_per_segment


@pytest.mark.req("FR-PRP-007")
def test_segment_phases_matches_scripted_flight_at_least_90_percent() -> None:
    df, truth = _scripted_flight()
    predicted = segment_phases(df, dt=0.1)
    accuracy = float(np.mean(predicted.to_numpy() == np.array(truth)))
    assert accuracy >= 0.90


@pytest.mark.req("FR-PRP-007")
def test_segment_phases_debounces_short_blips() -> None:
    n = 200
    climb_rate = np.zeros(n)
    climb_rate[100:102] = 10.0  # 0.2 s blip, well under default min_duration_s
    df = pd.DataFrame(
        {"t_s": np.arange(n) * 0.1, "climb_rate_mps": climb_rate, "vel_air_mps": np.full(n, 30.0)}
    )
    predicted = segment_phases(df, dt=0.1)
    assert (predicted == "cruise").all()


@pytest.mark.req("FR-PRP-007")
def test_custom_thresholds_change_sensitivity() -> None:
    n = 100
    df = pd.DataFrame(
        {
            "t_s": np.arange(n) * 0.1,
            "climb_rate_mps": np.full(n, 0.3),
            "vel_air_mps": np.full(n, 30.0),
        }
    )
    loose = segment_phases(df, dt=0.1, thresholds=PhaseThresholds(climb_rate_mps=0.5))
    tight_thresholds = PhaseThresholds(climb_rate_mps=0.1, min_duration_s=0.5)
    tight = segment_phases(df, dt=0.1, thresholds=tight_thresholds)
    assert (loose == "cruise").all()
    assert (tight == "climb").all()


@pytest.mark.req("FR-PRP-008")
def test_tag_conditions_adds_band_columns() -> None:
    n = 50
    df = pd.DataFrame(
        {
            "alt_m": np.linspace(0, 5000, n),
            "mach": np.linspace(0.0, 0.3, n),
            "roll_rad": np.linspace(-0.5, 0.5, n),
            "nz_g": np.linspace(0.5, 2.0, n),
        }
    )
    tagged = tag_conditions(df)
    assert "cond_alt_band" in tagged.columns
    assert "cond_mach_band" in tagged.columns
    assert "cond_bank_band" in tagged.columns
    assert "cond_nz_band" in tagged.columns
    assert tagged["cond_alt_band"].notna().all()
