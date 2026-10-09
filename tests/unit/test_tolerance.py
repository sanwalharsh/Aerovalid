from __future__ import annotations

import numpy as np
import pytest

from aerovalid.validate.tolerance import time_to_first_exceedance, tolerance_compliance


@pytest.mark.req("FR-VAL-004")
def test_tolerance_compliance_hand_computed() -> None:
    residual = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert tolerance_compliance(residual, tol=0.3) == pytest.approx(3 / 5)


@pytest.mark.req("FR-VAL-004")
def test_tolerance_compliance_ignores_nan() -> None:
    residual = np.array([0.1, np.nan, 0.3, np.nan])
    assert tolerance_compliance(residual, tol=0.2) == pytest.approx(0.5)


@pytest.mark.req("FR-VAL-004")
def test_tolerance_compliance_empty_is_nan() -> None:
    assert np.isnan(tolerance_compliance(np.array([]), tol=0.1))


@pytest.mark.req("FR-VAL-004")
def test_time_to_first_exceedance_hand_computed() -> None:
    t = np.arange(0.0, 10.0, 1.0)
    residual = np.array([0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # sustained exceedance starts at t=2, lasts 3 s >= min_run_s
    assert time_to_first_exceedance(t, residual, tol=1.0, min_run_s=2.0) == pytest.approx(2.0)


@pytest.mark.req("FR-VAL-004")
def test_time_to_first_exceedance_ignores_single_sample_blip() -> None:
    t = np.arange(0.0, 10.0, 1.0)
    residual = np.zeros(10)
    residual[5] = 5.0  # single-sample blip, 1s wide, under a 2s min run
    assert time_to_first_exceedance(t, residual, tol=1.0, min_run_s=2.0) is None


@pytest.mark.req("FR-VAL-004")
def test_time_to_first_exceedance_never_exceeds_returns_none() -> None:
    t = np.arange(0.0, 10.0, 1.0)
    residual = np.zeros(10)
    assert time_to_first_exceedance(t, residual, tol=1.0) is None
