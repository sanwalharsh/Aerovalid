from __future__ import annotations

import numpy as np
import pytest

from aerovalid.validate.compare import (
    cliffs_delta,
    compare_variant_pairs,
    friedman_test,
    hodges_lehmann_shift,
    nemenyi_posthoc,
)


@pytest.mark.req("FR-VAL-005")
def test_cliffs_delta_complete_separation() -> None:
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(a, b) == pytest.approx(1.0)
    assert cliffs_delta(b, a) == pytest.approx(-1.0)


@pytest.mark.req("FR-VAL-005")
def test_cliffs_delta_identical_distributions_near_zero() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert cliffs_delta(a, a) == pytest.approx(0.0)


@pytest.mark.req("FR-VAL-005")
def test_hodges_lehmann_shift_hand_computed() -> None:
    # differences = [1, 2, 3]; Walsh averages (i<=j): 1, 1.5, 2, 2, 2.5, 3 -> median 2.0
    differences = np.array([1.0, 2.0, 3.0])
    assert hodges_lehmann_shift(differences) == pytest.approx(2.0)


@pytest.mark.req("FR-VAL-005")
def test_hodges_lehmann_shift_symmetric_around_zero() -> None:
    differences = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert hodges_lehmann_shift(differences) == pytest.approx(0.0)


@pytest.mark.req("FR-VAL-005")
def test_compare_variant_pairs_detects_known_better_variant() -> None:
    rng = np.random.default_rng(0)
    n_flights = 30
    # V_good's score is consistently lower (better) than V_bad's
    v_good = rng.normal(1.0, 0.1, n_flights)
    v_bad = rng.normal(2.0, 0.1, n_flights)
    results = compare_variant_pairs({"V_good": v_good, "V_bad": v_bad})
    assert len(results) == 1
    result = results[0]
    assert result.p_value_holm < 0.01
    # sign depends on alphabetical (variant_a, variant_b) ordering, not which one we
    # called "good"/"bad"; either way the separation should be (near-)complete.
    expected_sign = 1.0 if result.variant_a == "V_bad" else -1.0
    assert result.cliffs_delta == pytest.approx(expected_sign, abs=0.1)


@pytest.mark.req("FR-VAL-005")
def test_compare_variant_pairs_null_pair_not_significant() -> None:
    rng = np.random.default_rng(1)
    n_flights = 30
    a = rng.normal(1.0, 0.2, n_flights)
    b = a + rng.normal(0, 0.01, n_flights)  # same distribution, tiny noise
    results = compare_variant_pairs({"A": a, "B": b})
    assert results[0].p_value_holm > 0.05


@pytest.mark.req("FR-VAL-005")
def test_compare_variant_pairs_holm_correction_applied_across_all_pairs() -> None:
    rng = np.random.default_rng(2)
    scores = {name: rng.normal(1.0, 0.1, 20) for name in ("V0", "V1", "V2")}
    results = compare_variant_pairs(scores)
    assert len(results) == 3  # 3 choose 2
    for r in results:
        assert r.p_value_holm >= r.p_value  # Holm correction never decreases p


@pytest.mark.req("FR-VAL-005")
def test_friedman_test_detects_systematic_variant_differences() -> None:
    rng = np.random.default_rng(3)
    n_flights = 20
    scores = {
        "V0": rng.normal(2.0, 0.1, n_flights),
        "V1": rng.normal(1.0, 0.1, n_flights),
        "V2": rng.normal(3.0, 0.1, n_flights),
    }
    _stat, p_value = friedman_test(scores)
    assert p_value < 0.01


@pytest.mark.req("FR-VAL-005")
def test_nemenyi_posthoc_returns_square_matrix() -> None:
    rng = np.random.default_rng(4)
    scores = {name: rng.normal(size=15) for name in ("V0", "V1", "V2")}
    result = nemenyi_posthoc(scores)
    assert list(result.index) == ["V0", "V1", "V2"]
    assert list(result.columns) == ["V0", "V1", "V2"]
    assert result.shape == (3, 3)
