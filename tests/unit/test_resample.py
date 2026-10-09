from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.prep.resample import resample


@pytest.mark.req("FR-PRP-004")
def test_resample_preserves_values_at_on_grid_points() -> None:
    t_s = np.arange(0.0, 10.0, 0.1)
    df = pd.DataFrame({"t_s": t_s, "flight_id": "F0001", "source": "tel", "alt_m": t_s * 10.0})
    out = resample(df, dt=0.1)
    np.testing.assert_allclose(out["alt_m"].to_numpy(), t_s * 10.0, atol=1e-6)


@pytest.mark.req("FR-PRP-004")
def test_resample_flags_original_samples() -> None:
    t_s = np.arange(0.0, 10.0, 0.1)
    df = pd.DataFrame({"t_s": t_s, "flight_id": "F0001", "source": "tel", "alt_m": t_s})
    out = resample(df, dt=0.1)
    assert out["is_original_sample"].all()


@pytest.mark.req("FR-PRP-004")
def test_resample_onto_explicit_grid() -> None:
    t_s = np.arange(0.0, 10.0, 0.05)
    df = pd.DataFrame({"t_s": t_s, "flight_id": "F0001", "source": "tel", "alt_m": np.sin(t_s)})
    explicit_grid = np.arange(0.0, 9.9, 0.1)
    out = resample(df, dt=0.1, t_grid=explicit_grid)
    assert len(out) == len(explicit_grid)
    np.testing.assert_allclose(out["t_s"].to_numpy(), explicit_grid)


@pytest.mark.req("FR-PRP-004")
def test_anti_alias_reduces_high_frequency_energy() -> None:
    from aerovalid.prep.resample import anti_alias_filter

    t_s = np.arange(0.0, 20.0, 0.01)  # 100 Hz source
    noisy = np.sin(2 * np.pi * 0.5 * t_s) + 0.5 * np.sin(2 * np.pi * 40.0 * t_s)
    filtered = anti_alias_filter(pd.Series(noisy), source_rate_hz=100.0, target_rate_hz=10.0)
    # low-frequency content preserved, high-frequency largely removed
    low_freq = np.sin(2 * np.pi * 0.5 * t_s)
    assert np.std(filtered - low_freq) < np.std(noisy - low_freq)
