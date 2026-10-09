from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.detect.ml import IsolationForestDetector, extract_window_features


@pytest.mark.req("FR-DET-003")
def test_extract_window_features_shape() -> None:
    z_df = pd.DataFrame({"z_alt_m": np.arange(100.0), "z_vel_air_mps": np.arange(100.0) * 2.0})
    features, window_end_idx = extract_window_features(z_df, window_samples=10, stride_samples=5)
    assert features.shape[1] == 3 * 2  # mean/std/slope per signal
    assert len(window_end_idx) == len(features)
    assert window_end_idx[0] == 9


@pytest.mark.req("FR-DET-003")
def test_extract_window_features_empty_when_too_short() -> None:
    z_df = pd.DataFrame({"z_alt_m": np.arange(5.0)})
    features, window_end_idx = extract_window_features(z_df, window_samples=10, stride_samples=5)
    assert len(features) == 0
    assert len(window_end_idx) == 0


@pytest.mark.req("FR-DET-003")
def test_isolation_forest_scores_anomalous_window_higher() -> None:
    rng = np.random.default_rng(0)
    nominal = rng.normal(0.0, 1.0, (300, 6))
    detector = IsolationForestDetector(n_estimators=100, contamination=0.02, seed=0)
    detector.fit(nominal)

    nominal_window = np.zeros(6)
    anomalous_window = np.full(6, 20.0)
    assert detector.score(anomalous_window) > detector.score(nominal_window)


@pytest.mark.req("FR-DET-003")
def test_isolation_forest_score_batch_matches_individual_scores() -> None:
    rng = np.random.default_rng(0)
    nominal = rng.normal(0.0, 1.0, (200, 4))
    detector = IsolationForestDetector(seed=0)
    detector.fit(nominal)

    test_windows = rng.normal(0.0, 1.0, (10, 4))
    batch_scores = detector.score_batch(test_windows)
    individual_scores = np.array([detector.score(w) for w in test_windows])
    np.testing.assert_allclose(batch_scores, individual_scores)
