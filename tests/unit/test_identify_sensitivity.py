from __future__ import annotations

import numpy as np
import pytest

from aerovalid.identify.sensitivity import morris_screen, sobol_indices

# A simple additive linear model over [0, 1]^3 with no interactions: "a" dominates,
# "b" is weak, "c" has no effect at all. Sobol S1/ST should reflect exactly that, and
# should each sum to ~1 since the model is purely additive (no interaction terms).
_PARAM_NAMES = ["a", "b", "c"]
_BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]


def _linear_model(x: np.ndarray) -> float:
    return float(10.0 * x[0] + 0.5 * x[1] + 0.0 * x[2])


@pytest.mark.req("FR-IDN-001")
def test_sobol_indices_sum_to_approximately_one_for_additive_model() -> None:
    table = sobol_indices(_linear_model, _PARAM_NAMES, _BOUNDS, n=512, seed=0)
    assert table["S1"].sum() == pytest.approx(1.0, abs=0.1)
    assert table["ST"].sum() == pytest.approx(1.0, abs=0.1)


@pytest.mark.req("FR-IDN-001")
def test_sobol_indices_rank_dominant_parameter_first() -> None:
    table = sobol_indices(_linear_model, _PARAM_NAMES, _BOUNDS, n=512, seed=0)
    ranked = table.sort_values("ST", ascending=False)
    assert ranked.iloc[0]["parameter"] == "a"


@pytest.mark.req("FR-IDN-001")
def test_sobol_indices_assigns_near_zero_sensitivity_to_inert_parameter() -> None:
    table = sobol_indices(_linear_model, _PARAM_NAMES, _BOUNDS, n=512, seed=0)
    inert = table[table["parameter"] == "c"].iloc[0]
    assert inert["ST"] < 0.05


@pytest.mark.req("FR-IDN-001")
def test_morris_screen_ranks_dominant_parameter_first() -> None:
    table = morris_screen(_linear_model, _PARAM_NAMES, _BOUNDS, n_trajectories=20, seed=0)
    assert table.iloc[0]["parameter"] == "a"
    assert set(table["parameter"]) == set(_PARAM_NAMES)
