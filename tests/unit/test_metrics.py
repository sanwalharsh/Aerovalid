from __future__ import annotations

import numpy as np
import pytest

from aerovalid.validate.metrics import (
    compute_metrics,
    compute_metrics_from_residual,
    error_growth_exponent,
)


@pytest.mark.req("FR-VAL-001")
def test_theil_decomposition_sums_to_one() -> None:
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    sim = obs + 0.1
    metrics = compute_metrics(obs, sim)
    assert metrics.theil_u_m + metrics.theil_u_s + metrics.theil_u_c == pytest.approx(1.0, abs=1e-9)


@pytest.mark.req("FR-VAL-001")
def test_rmse_ge_abs_me() -> None:
    rng = np.random.default_rng(0)
    obs = rng.normal(size=200)
    sim = obs + rng.normal(0, 0.5, size=200)
    metrics = compute_metrics(obs, sim)
    assert metrics.rmse >= abs(metrics.me)


@pytest.mark.req("FR-VAL-001")
def test_hand_computed_bias_mae_rmse() -> None:
    obs = np.array([10.0, 12.0, 14.0, 16.0])
    sim = np.array([9.0, 12.0, 15.0, 15.0])
    # residual = obs - sim = [1, 0, -1, 1]
    metrics = compute_metrics(obs, sim)
    assert metrics.me == pytest.approx(0.25)
    assert metrics.mae == pytest.approx(0.75)
    assert metrics.rmse == pytest.approx(np.sqrt((1 + 0 + 1 + 1) / 4))


@pytest.mark.req("FR-VAL-001")
def test_perfect_fit_gives_zero_error_and_r2_one() -> None:
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_metrics(obs, obs.copy())
    assert metrics.rmse == pytest.approx(0.0, abs=1e-12)
    assert metrics.mae == pytest.approx(0.0, abs=1e-12)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.pearson_r == pytest.approx(1.0)


@pytest.mark.req("FR-VAL-001")
def test_theil_u2_one_when_sim_equals_persistence() -> None:
    # sim(t) = obs(t-1): this is exactly the naive persistence forecast, so U2 should be ~1
    obs = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    sim = np.concatenate([[obs[0]], obs[:-1]])
    metrics = compute_metrics(obs, sim)
    assert metrics.theil_u2 == pytest.approx(1.0, rel=1e-6)


@pytest.mark.req("FR-VAL-001")
def test_nrmse_std_scales_with_signal_variance() -> None:
    rng = np.random.default_rng(1)
    obs_narrow = rng.normal(0, 1.0, size=500)
    obs_wide = obs_narrow * 10.0
    sim_narrow = obs_narrow + rng.normal(0, 0.2, size=500)
    sim_wide = obs_wide + rng.normal(0, 0.2, size=500)
    m_narrow = compute_metrics(obs_narrow, sim_narrow)
    m_wide = compute_metrics(obs_wide, sim_wide)
    # same absolute noise applied to a much larger-scale signal -> much smaller NRMSE_std
    assert m_wide.nrmse_std < m_narrow.nrmse_std


@pytest.mark.req("FR-VAL-001")
def test_masked_nonfinite_samples_excluded() -> None:
    obs = np.array([1.0, 2.0, np.nan, 4.0])
    sim = np.array([1.0, 2.0, 100.0, 4.0])
    metrics = compute_metrics(obs, sim)
    assert metrics.n_samples == 3
    assert metrics.rmse == pytest.approx(0.0, abs=1e-12)


@pytest.mark.req("FR-VAL-001")
def test_compute_metrics_from_residual_matches_direct_difference() -> None:
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([0.5, 2.5, 2.0])
    direct = compute_metrics(obs, sim)
    via_residual = compute_metrics_from_residual(obs, sim, obs - sim)
    assert direct == via_residual


@pytest.mark.req("FR-VAL-001")
def test_error_growth_exponent_recovers_known_power_law() -> None:
    horizons = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
    true_exponent = 0.5
    rmse = 2.0 * horizons**true_exponent
    exponent = error_growth_exponent(horizons, rmse)
    assert exponent == pytest.approx(true_exponent, abs=1e-6)


@pytest.mark.req("FR-VAL-001")
def test_empty_input_returns_nan_metrics() -> None:
    metrics = compute_metrics(np.array([]), np.array([]))
    assert metrics.n_samples == 0
    assert np.isnan(metrics.rmse)
