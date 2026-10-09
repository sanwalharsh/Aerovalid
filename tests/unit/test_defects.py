from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerovalid.telemetry.defects import (
    TimingDefectParams,
    apply_timing_defects,
    load_timing_defect_params,
)


@pytest.fixture
def default_sensors_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "sensors" / "default.yaml"


@pytest.mark.req("FR-TEL-002")
def test_load_timing_defect_params_from_real_config(default_sensors_path: Path) -> None:
    params = load_timing_defect_params(default_sensors_path)
    assert params.offset_s.low == pytest.approx(-0.8)
    assert params.offset_s.high == pytest.approx(0.8)
    assert params.dropout_p_per_sample == pytest.approx(0.005)


@pytest.mark.req("FR-TEL-002")
def test_offset_and_drift_within_configured_range() -> None:
    params = TimingDefectParams()
    t_s = np.arange(1000) * 0.1
    rng = np.random.Generator(np.random.PCG64(1))

    _t_corrupted, _keep, log = apply_timing_defects(t_s, params, rng)

    assert params.offset_s.low <= log.offset_s <= params.offset_s.high
    assert params.drift_ppm.low <= log.drift_ppm <= params.drift_ppm.high


@pytest.mark.req("FR-TEL-002")
def test_deterministic_given_seeded_rng() -> None:
    params = TimingDefectParams()
    t_s = np.arange(1000) * 0.1

    t_a, keep_a, log_a = apply_timing_defects(t_s, params, np.random.Generator(np.random.PCG64(42)))
    t_b, keep_b, log_b = apply_timing_defects(t_s, params, np.random.Generator(np.random.PCG64(42)))

    assert np.array_equal(t_a, t_b)
    assert np.array_equal(keep_a, keep_b)
    assert log_a == log_b


@pytest.mark.req("FR-TEL-002")
def test_dropout_reduces_kept_sample_count() -> None:
    params = TimingDefectParams(
        dropout_p_per_sample=0.5, burst_gaps_max_n=0, duplicate_p_per_sample=0.0
    )
    t_s = np.arange(2000) * 0.1
    rng = np.random.Generator(np.random.PCG64(2))

    _t_corrupted, keep_mask, log = apply_timing_defects(t_s, params, rng)

    assert np.mean(keep_mask) == pytest.approx(0.5, abs=0.05)
    assert log.n_dropped == int((~keep_mask).sum())


@pytest.mark.req("FR-TEL-002")
def test_burst_gaps_are_logged_and_removed() -> None:
    params = TimingDefectParams(
        dropout_p_per_sample=0.0,
        duplicate_p_per_sample=0.0,
        burst_gaps_max_n=2,
        burst_gaps_duration_s=(1.0, 1.0),
    )
    t_s = np.arange(1000) * 0.1  # 100 s flight
    rng = np.random.Generator(np.random.PCG64(3))

    _t_corrupted, keep_mask, log = apply_timing_defects(t_s, params, rng)

    assert 0 <= len(log.gaps_s) <= 2
    if log.gaps_s:
        assert not np.all(keep_mask)


@pytest.mark.req("FR-TEL-002")
def test_duplicate_timestamps_are_injected() -> None:
    params = TimingDefectParams(
        duplicate_p_per_sample=1.0, dropout_p_per_sample=0.0, burst_gaps_max_n=0
    )
    t_s = np.arange(100) * 0.1
    rng = np.random.Generator(np.random.PCG64(4))

    t_corrupted, _keep, log = apply_timing_defects(t_s, params, rng)

    assert log.n_duplicated > 0
    # every sample after the first collided with its predecessor's corrupted time
    assert np.sum(np.diff(t_corrupted) == 0.0) == log.n_duplicated
