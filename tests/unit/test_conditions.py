from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.validate.conditions import (
    condition_heatmap_data,
    dunn_posthoc,
    fit_condition_importance,
    kruskal_wallis_across_bins,
    mixed_effects_condition_model,
    partial_dependence_curve,
)


@pytest.mark.req("FR-VAL-007")
def test_condition_heatmap_hides_sparse_cells() -> None:
    df = pd.DataFrame({"cond_alt_band": ["low"] * 10 + ["high"] * 2, "value": [1.0] * 12})
    heatmap = condition_heatmap_data(df, ["cond_alt_band"], "value", min_n=5)
    low_row = heatmap[heatmap["cond_alt_band"] == "low"].iloc[0]
    high_row = heatmap[heatmap["cond_alt_band"] == "high"].iloc[0]
    assert not np.isnan(low_row["mean"])
    assert np.isnan(high_row["mean"])


@pytest.mark.req("FR-VAL-007")
def test_fit_condition_importance_ranks_informative_feature_first() -> None:
    rng = np.random.default_rng(0)
    n_flights = 20
    rows_per_flight = 30
    rows = []
    for flight_idx in range(n_flights):
        mach = rng.uniform(0, 0.3, rows_per_flight)
        noise_feature = rng.uniform(0, 1, rows_per_flight)
        target = mach * 2.0 + rng.normal(0, 0.01, rows_per_flight)  # strongly driven by mach only
        for m, noise, t in zip(mach, noise_feature, target, strict=True):
            rows.append(
                {"flight_id": f"F{flight_idx:04d}", "mach": m, "noise_feature": noise, "target": t}
            )
    df = pd.DataFrame(rows)

    result = fit_condition_importance(df, ["mach", "noise_feature"], "target")
    top_feature = result.importance.iloc[0]["feature"]
    assert top_feature == "mach"
    assert result.cv_r2 > 0.8


@pytest.mark.req("FR-VAL-007")
def test_partial_dependence_curve_increasing_for_informative_feature() -> None:
    rng = np.random.default_rng(1)
    n = 500
    x = rng.uniform(0, 10, n)
    y = x + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"flight_id": [f"F{i % 20:04d}" for i in range(n)], "x": x, "target": y})
    result = fit_condition_importance(df, ["x"], "target")
    pdp = partial_dependence_curve(result.model, df[["x"]], "x")
    assert pdp["partial_dependence"].iloc[-1] > pdp["partial_dependence"].iloc[0]


@pytest.mark.req("FR-VAL-007")
def test_kruskal_wallis_detects_different_bin_distributions() -> None:
    rng = np.random.default_rng(2)
    values_by_bin = {
        "low": rng.normal(1.0, 0.1, 30),
        "mid": rng.normal(2.0, 0.1, 30),
        "high": rng.normal(3.0, 0.1, 30),
    }
    _stat, p_value = kruskal_wallis_across_bins(values_by_bin)
    assert p_value < 0.01


@pytest.mark.req("FR-VAL-007")
def test_dunn_posthoc_returns_square_matrix() -> None:
    rng = np.random.default_rng(3)
    values_by_bin = {
        "low": rng.normal(size=20),
        "mid": rng.normal(size=20),
        "high": rng.normal(size=20),
    }
    result = dunn_posthoc(values_by_bin)
    assert list(result.index) == ["low", "mid", "high"]
    assert result.shape == (3, 3)


@pytest.mark.req("FR-VAL-007")
def test_mixed_effects_condition_model_fits() -> None:
    rng = np.random.default_rng(4)
    n_flights = 15
    rows = []
    for flight_idx in range(n_flights):
        flight_offset = rng.normal(0, 0.5)
        mach = rng.uniform(0, 0.3, 10)
        log_abs_res = 0.5 * mach + flight_offset + rng.normal(0, 0.05, 10)
        for m, r in zip(mach, log_abs_res, strict=True):
            rows.append({"flight_id": f"F{flight_idx:04d}", "mach": m, "log_abs_res": r})
    df = pd.DataFrame(rows)

    result = mixed_effects_condition_model(df, "log_abs_res ~ mach")
    assert result.params["mach"] == pytest.approx(0.5, abs=0.2)
