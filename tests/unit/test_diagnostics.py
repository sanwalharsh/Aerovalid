from __future__ import annotations

import numpy as np
import pytest

from aerovalid.validate.diagnostics import power_spectrum, residual_diagnostics


@pytest.mark.req("FR-VAL-008")
def test_white_noise_flagged_as_white_and_stationary() -> None:
    rng = np.random.default_rng(0)
    residual = rng.normal(0, 1.0, 500)
    result = residual_diagnostics(residual)
    assert result.is_white_noise
    assert result.is_stationary


@pytest.mark.req("FR-VAL-008")
def test_strongly_autocorrelated_residual_flagged_non_white() -> None:
    rng = np.random.default_rng(1)
    n = 500
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.95 * ar[i - 1] + rng.normal(0, 0.1)
    result = residual_diagnostics(ar)
    assert not result.is_white_noise
    assert result.acf_values[1] > 0.5  # strong lag-1 autocorrelation


@pytest.mark.req("FR-VAL-008")
def test_normal_residual_flagged_normal() -> None:
    rng = np.random.default_rng(2)
    residual = rng.normal(0, 1.0, 1000)
    result = residual_diagnostics(residual)
    assert result.is_normal


@pytest.mark.req("FR-VAL-008")
def test_non_stationary_trend_flagged_non_stationary() -> None:
    n = 500
    trend = np.linspace(0, 50, n) + np.random.default_rng(3).normal(0, 0.5, n)
    result = residual_diagnostics(trend)
    assert not result.is_stationary


@pytest.mark.req("FR-VAL-008")
def test_too_few_samples_returns_nan_flags() -> None:
    result = residual_diagnostics(np.array([1.0, 2.0, 3.0]))
    assert result.n_samples == 3
    assert not result.is_white_noise
    assert np.isnan(result.ljung_box_p)


@pytest.mark.req("FR-VAL-008")
def test_power_spectrum_peak_at_known_frequency() -> None:
    fs = 100.0
    t = np.arange(0, 10, 1.0 / fs)
    signal = np.sin(2 * np.pi * 5.0 * t)
    freqs, psd = power_spectrum(signal, fs)
    peak_freq = freqs[np.argmax(psd)]
    assert peak_freq == pytest.approx(5.0, abs=0.5)
