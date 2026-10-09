from __future__ import annotations

import numpy as np
import pytest

from aerovalid.detect.statistical import (
    CUSUMDetector,
    EWMADetector,
    MahalanobisDetector,
    PageHinkleyDetector,
    ZScoreDetector,
    detect_batch,
    fit_mahalanobis,
)

_STREAMING_DETECTORS = {
    "zscore": lambda: ZScoreDetector(h=4.0, m=5),
    "ewma": lambda: EWMADetector(lam=0.2, upper_limit=3.5),
    "cusum": lambda: CUSUMDetector(k=0.5, h=5.0),
    "page_hinkley": lambda: PageHinkleyDetector(delta=0.5, lam=10.0),
}


def _nominal_then_shifted_stream(
    seed: int = 0, n_nominal: int = 200, n_shifted: int = 100
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nominal = rng.normal(0.0, 1.0, n_nominal)
    shifted = rng.normal(6.0, 1.0, n_shifted)
    return np.concatenate([nominal, shifted])


@pytest.mark.req("FR-DET-002")
@pytest.mark.parametrize("name", list(_STREAMING_DETECTORS))
def test_detector_stays_quiet_on_nominal_noise(name: str) -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 1.0, 300)
    detector = _STREAMING_DETECTORS[name]()
    alarms = detect_batch(detector, z)
    assert alarms.sum() <= 3  # a handful of spurious alarms is acceptable, not a flood


@pytest.mark.req("FR-DET-002")
@pytest.mark.parametrize("name", list(_STREAMING_DETECTORS))
def test_detector_eventually_alarms_on_a_sustained_shift(name: str) -> None:
    z = _nominal_then_shifted_stream()
    detector = _STREAMING_DETECTORS[name]()
    alarms = detect_batch(detector, z)
    assert alarms[200:].any()


@pytest.mark.req("FR-DET-008")
@pytest.mark.parametrize("name", list(_STREAMING_DETECTORS))
def test_batch_matches_streaming_replay(name: str) -> None:
    z = _nominal_then_shifted_stream()
    batch_detector = _STREAMING_DETECTORS[name]()
    batch_alarms = detect_batch(batch_detector, z)

    streaming_detector = _STREAMING_DETECTORS[name]()
    streaming_detector.reset()
    streaming_alarms = np.array([streaming_detector.update(float(value)) for value in z])

    np.testing.assert_array_equal(batch_alarms, streaming_alarms)


@pytest.mark.req("FR-DET-002")
def test_mahalanobis_detector_flags_a_multivariate_outlier() -> None:
    rng = np.random.default_rng(2)
    nominal_z = rng.normal(0.0, 1.0, (500, 3))
    detector = fit_mahalanobis(nominal_z, alpha=0.01)

    in_distribution = np.array([0.1, -0.2, 0.3])
    assert not detector.update(in_distribution)

    outlier = np.array([10.0, 10.0, 10.0])
    assert detector.update(outlier)


@pytest.mark.req("FR-DET-002")
def test_mahalanobis_detector_reset_is_a_noop() -> None:
    detector = MahalanobisDetector(inv_cov=np.eye(2), threshold=5.0)
    detector.reset()  # memoryless; must not raise
    assert detector.statistic(np.array([1.0, 1.0])) == pytest.approx(2.0)
