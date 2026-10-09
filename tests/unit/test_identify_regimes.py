from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.identify.regimes import bin_telemetry_by_regime, cochrans_q_heterogeneity


@pytest.mark.req("FR-IDN-004")
def test_cochrans_q_low_for_homogeneous_estimates() -> None:
    estimates = np.array([1.01, 0.99, 1.00, 1.02])
    variances = np.full(4, 0.01**2)
    q_statistic, p_value = cochrans_q_heterogeneity(estimates, variances)
    assert p_value > 0.05
    assert q_statistic < 10.0


@pytest.mark.req("FR-IDN-004")
def test_cochrans_q_high_for_heterogeneous_estimates() -> None:
    # Same tiny variances, but the estimates themselves clearly disagree across
    # regimes -- the signature of a parameter that actually drifts with regime.
    estimates = np.array([1.0, 1.5, 2.0, 2.5])
    variances = np.full(4, 0.001**2)
    q_statistic, p_value = cochrans_q_heterogeneity(estimates, variances)
    assert p_value < 0.01
    assert q_statistic > 20.0


@pytest.mark.req("FR-IDN-004")
def test_cochrans_q_single_regime_returns_nan_p_value() -> None:
    _q, p_value = cochrans_q_heterogeneity(np.array([1.0]), np.array([0.01]))
    assert np.isnan(p_value)


@pytest.mark.req("FR-IDN-004")
def test_bin_telemetry_by_regime_splits_on_derived_column() -> None:
    telemetry = pd.DataFrame({"vel_air_mps": [10.0, 20.0, 30.0, 40.0]})
    bins = bin_telemetry_by_regime(
        telemetry, regime_column=lambda df: df["vel_air_mps"], bin_edges=[0.0, 25.0, 50.0]
    )
    assert sum(len(df) for df in bins.values()) == 4
    assert len(bins) == 2
