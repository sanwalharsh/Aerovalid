from __future__ import annotations

import numpy as np
import pytest

from aerovalid.detect.calibration import calibrate_threshold, count_alarm_episodes
from aerovalid.detect.statistical import ZScoreDetector, detect_batch


@pytest.mark.req("FR-DET-002")
def test_count_alarm_episodes_counts_rising_edges_not_samples() -> None:
    alarms = np.array([False, True, True, True, False, True, False])
    assert count_alarm_episodes(alarms) == 2


@pytest.mark.req("FR-DET-002")
def test_count_alarm_episodes_empty_array() -> None:
    assert count_alarm_episodes(np.array([], dtype=bool)) == 0


@pytest.mark.req("FR-DET-002")
def test_calibrate_threshold_hits_target_far_within_25_percent() -> None:
    rng = np.random.default_rng(0)
    n_flights = 20
    samples_per_flight = 3000  # 300 s at 10 Hz
    dt = 0.1
    nominal_z = [rng.normal(0.0, 1.0, samples_per_flight) for _ in range(n_flights)]
    durations = [samples_per_flight * dt] * n_flights
    target_far_per_hour = 2.0

    def build_alarms(h: float) -> list[np.ndarray]:
        return [detect_batch(ZScoreDetector(h=h, m=3), z) for z in nominal_z]

    threshold = calibrate_threshold(
        build_alarms, durations, target_far_per_hour, search_range=(1.0, 6.0)
    )
    achieved_far = sum(count_alarm_episodes(a) for a in build_alarms(threshold)) / (
        sum(durations) / 3600.0
    )
    assert achieved_far == pytest.approx(target_far_per_hour, rel=0.25)
