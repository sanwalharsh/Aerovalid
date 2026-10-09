from __future__ import annotations

import numpy as np
import pytest

from aerovalid.core.rng import child_seed, spawn_generators


@pytest.mark.req("FR-COR-006")
def test_same_seed_same_labels_identical_streams() -> None:
    gens_a = spawn_generators(42, ["F0001", "F0002", "F0003"])
    gens_b = spawn_generators(42, ["F0001", "F0002", "F0003"])
    for label in gens_a:
        draw_a = gens_a[label].random(10)
        draw_b = gens_b[label].random(10)
        assert np.array_equal(draw_a, draw_b)


@pytest.mark.req("FR-COR-006")
def test_different_labels_get_independent_streams() -> None:
    gens = spawn_generators(42, ["F0001", "F0002"])
    draw_1 = gens["F0001"].random(100)
    draw_2 = gens["F0002"].random(100)
    assert not np.array_equal(draw_1, draw_2)


@pytest.mark.req("FR-COR-006")
def test_different_master_seed_changes_stream() -> None:
    gens_a = spawn_generators(1, ["F0001"])
    gens_b = spawn_generators(2, ["F0001"])
    assert not np.array_equal(gens_a["F0001"].random(10), gens_b["F0001"].random(10))


@pytest.mark.req("FR-COR-006")
def test_child_seed_deterministic_and_index_sensitive() -> None:
    assert child_seed(42, 0) == child_seed(42, 0)
    assert child_seed(42, 0) != child_seed(42, 1)
    assert child_seed(42, 5) != child_seed(7, 5)


@pytest.mark.req("FR-COR-006")
def test_child_seed_values_are_pairwise_independent() -> None:
    seeds = [child_seed(99, i) for i in range(5)]
    assert len(set(seeds)) == len(seeds)
    # calling twice for the same index is bit-identical
    assert [child_seed(99, i) for i in range(5)] == seeds
