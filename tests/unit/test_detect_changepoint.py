from __future__ import annotations

import numpy as np
import pytest

from aerovalid.detect.changepoint import nearest_onset_error, pelt_onsets


@pytest.mark.req("FR-DET-004")
def test_pelt_onsets_detects_a_mean_shift_near_the_true_time() -> None:
    rng = np.random.default_rng(0)
    dt = 0.1
    n = 600  # 60 s
    t_s = np.arange(n) * dt
    onset_idx = 300  # t = 30.0 s
    signal = rng.normal(0.0, 0.3, n)
    signal[onset_idx:] += 5.0

    onsets = pelt_onsets(t_s, signal, penalty=15.0, min_size=5)
    error = nearest_onset_error(onsets, true_onset_s=30.0)
    assert error is not None
    assert error <= 3.0


@pytest.mark.req("FR-DET-004")
def test_pelt_onsets_median_error_across_seeds_is_within_3s() -> None:
    dt = 0.1
    n = 600
    t_s = np.arange(n) * dt
    true_onset_s = 30.0
    errors = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        signal = rng.normal(0.0, 0.3, n)
        signal[300:] += 5.0
        onsets = pelt_onsets(t_s, signal, penalty=15.0, min_size=5)
        error = nearest_onset_error(onsets, true_onset_s)
        errors.append(error if error is not None else 100.0)
    assert float(np.median(errors)) <= 3.0


@pytest.mark.req("FR-DET-004")
def test_nearest_onset_error_returns_none_without_changepoints() -> None:
    assert nearest_onset_error([], true_onset_s=10.0) is None


@pytest.mark.req("FR-DET-004")
def test_nearest_onset_error_picks_closest_candidate() -> None:
    error = nearest_onset_error([5.0, 29.0, 60.0], true_onset_s=30.0)
    assert error == pytest.approx(1.0)
