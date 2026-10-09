"""Threshold calibration to a target false-alarm rate (FR-DET-002): bisection search
over a detector's threshold parameter so its false-alarm rate on nominal (fault-free)
calibration flights matches `target_far_per_hour`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def count_alarm_episodes(alarms: NDArray[np.bool_]) -> int:
    """Count contiguous `True` runs (rising edges), not raw alarming samples -- a
    multi-sample sustained alarm is one false-alarm event, not many."""
    if alarms.size == 0:
        return 0
    edges = np.diff(alarms.astype(np.int8), prepend=0)
    return int(np.sum(edges == 1))


def calibrate_threshold(
    build_alarms: Callable[[float], list[NDArray[np.bool_]]],
    flight_durations_s: list[float],
    target_far_per_hour: float,
    search_range: tuple[float, float],
    max_iter: int = 40,
) -> float:
    """
    Bisect for the threshold giving `target_far_per_hour` false alarms on nominal
    calibration flights (FR-DET-002).

    Parameters
    ----------
    build_alarms : callable
        `threshold -> [per-flight alarm bool array, ...]`, evaluated on nominal
        (fault-free) flights only. Must be non-increasing in `threshold` (a higher
        threshold alarms no more often than a lower one).
    flight_durations_s : list[float]
        Each nominal flight's duration, s (same order as `build_alarms`'s output).
    target_far_per_hour : float
    search_range : tuple[float, float]
        `(low, high)` bounds the search must bracket: `build_alarms(low)` should alarm
        at least as often as target, `build_alarms(high)` no more often.
    max_iter : int, default 40

    Returns
    -------
    float
        The calibrated threshold.
    """
    total_hours = sum(flight_durations_s) / 3600.0

    def far_at(threshold: float) -> float:
        alarms = build_alarms(threshold)
        n_events = sum(count_alarm_episodes(a) for a in alarms)
        return n_events / total_hours if total_hours > 0 else float("inf")

    low, high = search_range
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if far_at(mid) > target_far_per_hour:
            low = mid
        else:
            high = mid
    return high
