from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.prep.sync import SyncEstimate, apply_sync, estimate_sync

_DT = 0.1
_DURATION_S = 900.0


def _richly_correlated_signal(t: np.ndarray) -> np.ndarray:
    return (
        np.sin(2 * np.pi * 0.05 * t)
        + 0.5 * np.sin(2 * np.pi * 0.13 * t)
        + 0.3 * np.sin(2 * np.pi * 0.02 * t)
    )


def _synthetic_pair(offset_s: float, drift_ppm: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    t_ref = np.arange(0.0, _DURATION_S, _DT)
    values = _richly_correlated_signal(t_ref)
    ref = pd.DataFrame({"t_s": t_ref, "alt_m": values})
    other_t = t_ref * (1.0 + drift_ppm * 1e-6) + offset_s
    other = pd.DataFrame({"t_s": other_t, "alt_m": values})
    return ref, other


@pytest.mark.req("FR-PRP-003")
def test_offset_recovered_within_half_sample() -> None:
    ref, other = _synthetic_pair(offset_s=0.37, drift_ppm=0.0)
    rng = np.random.default_rng(1)
    estimate = estimate_sync(ref, other, ["alt_m"], max_lag_s=3.0, dt=_DT, n_bootstrap=50, rng=rng)

    assert not estimate.low_confidence
    assert abs(estimate.lag_s - 0.37) <= 0.5 * _DT


@pytest.mark.req("FR-PRP-003")
def test_negative_offset_recovered_within_half_sample() -> None:
    ref, other = _synthetic_pair(offset_s=-0.6, drift_ppm=0.0)
    estimate = estimate_sync(ref, other, ["alt_m"], max_lag_s=3.0, dt=_DT, n_bootstrap=0)
    assert abs(estimate.lag_s - (-0.6)) <= 0.5 * _DT


@pytest.mark.req("FR-PRP-003")
@pytest.mark.parametrize(("offset_s", "drift_ppm"), [(0.37, 30.0), (-0.9, -45.0), (0.05, 5.0)])
def test_drift_recovered_within_10ppm(offset_s: float, drift_ppm: float) -> None:
    ref, other = _synthetic_pair(offset_s=offset_s, drift_ppm=drift_ppm)
    estimate = estimate_sync(ref, other, ["alt_m"], max_lag_s=3.0, dt=_DT, n_bootstrap=0)

    assert abs(estimate.lag_s - offset_s) <= 0.5 * _DT
    assert abs(estimate.drift_ppm - drift_ppm) <= 10.0


@pytest.mark.req("FR-PRP-003")
def test_no_drift_estimated_when_disabled() -> None:
    ref, other = _synthetic_pair(offset_s=0.1, drift_ppm=40.0)
    estimate = estimate_sync(
        ref, other, ["alt_m"], max_lag_s=3.0, dt=_DT, estimate_drift=False, n_bootstrap=0
    )
    assert estimate.drift_ppm == 0.0


@pytest.mark.req("FR-PRP-003")
def test_uncorrelated_signals_fall_back_to_zero_lag_low_confidence() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 60.0, _DT)
    ref = pd.DataFrame({"t_s": t, "alt_m": rng.normal(size=len(t))})
    other = pd.DataFrame({"t_s": t, "alt_m": rng.normal(size=len(t))})
    estimate = estimate_sync(ref, other, ["alt_m"], max_lag_s=3.0, dt=_DT, n_bootstrap=0)
    assert estimate.low_confidence
    assert estimate.lag_s == 0.0


@pytest.mark.req("FR-PRP-003")
def test_too_short_overlap_is_low_confidence() -> None:
    t = np.arange(0.0, 1.0, _DT)
    ref = pd.DataFrame({"t_s": t, "alt_m": np.zeros(len(t))})
    other = pd.DataFrame({"t_s": t, "alt_m": np.zeros(len(t))})
    estimate = estimate_sync(ref, other, ["alt_m"], max_lag_s=3.0, dt=_DT, n_bootstrap=0)
    assert estimate.low_confidence


@pytest.mark.req("FR-PRP-003")
def test_apply_sync_shifts_and_corrects_timestamps() -> None:
    df = pd.DataFrame({"t_s": [0.0, 1.0, 2.0], "alt_m": [1.0, 2.0, 3.0]})
    estimate = SyncEstimate(
        lag_s=0.5, drift_ppm=0.0, peak_correlation=1.0, lag_ci_s=(0.5, 0.5), low_confidence=False
    )
    corrected = apply_sync(df, estimate)
    assert list(corrected["t_s"]) == pytest.approx([-0.5, 0.5, 1.5])
