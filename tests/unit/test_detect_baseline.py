from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerovalid.detect.baseline import fit_baseline


def _nominal_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "res_alt_m": rng.normal(2.0, 0.5, n),
            "res_vel_air_mps": rng.normal(-1.0, 0.2, n),
            "cond_alt_band": rng.choice(["low", "high"], n),
        }
    )


@pytest.mark.req("FR-DET-001")
def test_fit_baseline_recovers_median_and_mad_scale() -> None:
    df = _nominal_df()
    baseline = fit_baseline(df, signals=["alt_m", "vel_air_mps"])
    assert baseline.center["_global_"]["alt_m"] == pytest.approx(2.0, abs=0.15)
    assert baseline.scale["_global_"]["alt_m"] == pytest.approx(0.5, rel=0.3)


@pytest.mark.req("FR-DET-001")
def test_z_scores_whiten_nominal_residuals_to_near_unit_scale() -> None:
    df = _nominal_df()
    baseline = fit_baseline(df, signals=["alt_m", "vel_air_mps"])
    z = baseline.z_scores(df)
    assert set(z.columns) == {"z_alt_m", "z_vel_air_mps"}
    assert z["z_alt_m"].std() == pytest.approx(1.0, rel=0.3)
    assert z["z_alt_m"].mean() == pytest.approx(0.0, abs=0.3)


@pytest.mark.req("FR-DET-001")
def test_per_bin_baseline_falls_back_to_global_for_small_bins() -> None:
    df = _nominal_df()
    df.loc[df.index[:5], "cond_alt_band"] = "rare_bin"  # fewer than min_bin_samples
    baseline = fit_baseline(df, signals=["alt_m"], bin_column="cond_alt_band", min_bin_samples=30)
    assert "rare_bin" not in baseline.center
    assert "low" in baseline.center or "high" in baseline.center

    z = baseline.z_scores(df)
    assert np.all(np.isfinite(z["z_alt_m"]))


@pytest.mark.req("FR-DET-001")
def test_baseline_json_roundtrip(tmp_path: Path) -> None:
    from aerovalid.detect.baseline import ResidualBaseline

    df = _nominal_df()
    baseline = fit_baseline(df, signals=["alt_m", "vel_air_mps"])
    path = tmp_path / "baseline.json"
    baseline.save(path)
    reloaded = ResidualBaseline.load(path)
    assert reloaded.center == baseline.center
    assert reloaded.scale == baseline.scale
    assert reloaded.signals == baseline.signals
