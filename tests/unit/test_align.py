from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.prep.align import build_aligned


def _tables(n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = np.arange(n) * 0.1
    obs = pd.DataFrame(
        {
            "t_s": t,
            "flight_id": "F0001",
            "source": "tel",
            "alt_m": 1000.0 + t,
            "gamma_rad": np.full(n, 3.1),
        }
    )
    sim = pd.DataFrame(
        {
            "t_s": t,
            "flight_id": "F0001",
            "source": "V0",
            "alt_m": 1000.0 + t + 0.5,
            "gamma_rad": np.full(n, -3.1),  # near +-pi wraparound from obs's 3.1
        }
    )
    return obs, sim


@pytest.mark.req("FR-PRP-009")
def test_build_aligned_has_obs_sim_res_columns() -> None:
    obs, sim = _tables()
    aligned = build_aligned(obs, sim, "F0001", "V0")
    df = aligned.df
    assert "obs_alt_m" in df.columns
    assert "sim_alt_m" in df.columns
    assert "res_alt_m" in df.columns
    np.testing.assert_allclose(df["res_alt_m"].to_numpy(), -0.5, atol=1e-9)


@pytest.mark.req("FR-PRP-009")
def test_build_aligned_wraps_angle_residuals() -> None:
    obs, sim = _tables()
    aligned = build_aligned(obs, sim, "F0001", "V0")
    residual = aligned.df["res_gamma_rad"].to_numpy()
    # obs=3.1, sim=-3.1: the wrapped difference should be small, not ~6.2
    assert np.all(np.abs(residual) < 0.5)


@pytest.mark.req("FR-PRP-009")
def test_build_aligned_row_counts_equal_across_variants() -> None:
    obs, sim_a = _tables()
    _, sim_b = _tables()
    sim_b["source"] = "V1"
    aligned_a = build_aligned(obs, sim_a, "F0001", "V0")
    aligned_b = build_aligned(obs, sim_b, "F0001", "V1")
    assert len(aligned_a.df) == len(aligned_b.df) == len(obs)


@pytest.mark.req("FR-PRP-009")
def test_build_aligned_mismatched_lengths_raises() -> None:
    obs, sim = _tables()
    with pytest.raises(ValueError, match="same time grid"):
        build_aligned(obs, sim.iloc[:-1], "F0001", "V0")


@pytest.mark.req("FR-PRP-009")
def test_build_aligned_mask_reflects_missing_obs() -> None:
    obs, sim = _tables()
    obs.loc[5, "alt_m"] = np.nan
    aligned = build_aligned(obs, sim, "F0001", "V0")
    assert aligned.df["mask_alt_m"].iloc[5]
    assert not aligned.df["mask_alt_m"].iloc[0]


@pytest.mark.req("FR-PRP-009")
def test_build_aligned_validates_canonical_schema() -> None:
    obs, sim = _tables()
    aligned = build_aligned(obs, sim, "F0001", "V0")
    assert set(aligned.df["flight_id"]) == {"F0001"}
    assert set(aligned.df["source"]) == {"V0"}
