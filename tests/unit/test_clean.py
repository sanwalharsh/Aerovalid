from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.prep.clean import clean, hampel_filter


@pytest.mark.req("FR-PRP-002")
def test_hampel_filter_removes_injected_spikes() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    clean_signal = np.sin(np.linspace(0, 20, n))
    noisy = clean_signal + rng.normal(0, 0.01, n)

    spike_idx = rng.choice(n, size=50, replace=False)
    spiked = noisy.copy()
    spiked[spike_idx] += rng.choice([-1, 1], size=50) * 5.0

    cleaned, outlier_mask = hampel_filter(pd.Series(spiked), window=7, n_sigma=4.0)

    detected = set(np.flatnonzero(outlier_mask.to_numpy()))
    recovered_fraction = len(detected & set(spike_idx)) / len(spike_idx)
    assert recovered_fraction >= 0.95


@pytest.mark.req("FR-PRP-002")
def test_hampel_filter_preserves_nan() -> None:
    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
    cleaned, _mask = hampel_filter(series, window=3, n_sigma=3.0)
    assert pd.isna(cleaned.iloc[2])


@pytest.mark.req("FR-PRP-002")
def test_clean_sorts_and_dedupes_timestamps() -> None:
    df = pd.DataFrame(
        {
            "t_s": [0.0, 0.2, 0.1, 0.2, 0.3],
            "flight_id": "F0001",
            "source": "tel",
            "alt_m": [100.0, 102.0, 101.0, 999.0, 103.0],
        }
    )
    result = clean(df)
    assert list(result.df["t_s"]) == [0.0, 0.1, 0.2, 0.3]


@pytest.mark.req("FR-PRP-002")
def test_clean_interpolates_short_gaps_and_masks_long_ones() -> None:
    n = 100
    t_s = np.arange(n) * 0.1
    alt = 1000.0 + np.arange(n) * 0.5
    alt_with_gaps = alt.copy()
    alt_with_gaps[10:12] = np.nan  # short gap: 0.2s < max_gap_s
    alt_with_gaps[50:70] = np.nan  # long gap: 2.0s > max_gap_s

    df = pd.DataFrame({"t_s": t_s, "flight_id": "F0001", "source": "tel", "alt_m": alt_with_gaps})
    result = clean(df, max_gap_s=0.5)

    assert not result.df["alt_m"].iloc[10:12].isna().any()
    assert result.mask["alt_m"].iloc[50:70].all()
    assert result.n_gaps_filled["alt_m"] == 2
    assert result.n_gaps_masked["alt_m"] == 20
