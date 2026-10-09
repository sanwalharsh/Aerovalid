from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aerovalid.core.angles import angle_diff, circular_mean, wrap_to_pi


@pytest.mark.req("FR-COR-004")
@pytest.mark.parametrize(
    ("angle_in", "expected"),
    [
        (0.0, 0.0),
        (np.pi, np.pi),
        (-np.pi, np.pi),
        (3 * np.pi, np.pi),
        (-3 * np.pi, np.pi),
        (2 * np.pi, 0.0),
    ],
)
def test_wrap_to_pi_scalar(angle_in: float, expected: float) -> None:
    assert wrap_to_pi(angle_in) == pytest.approx(expected, abs=1e-12)


@pytest.mark.req("FR-COR-004")
def test_wrap_to_pi_array_is_in_range() -> None:
    angles = np.linspace(-20, 20, 1000)
    wrapped = wrap_to_pi(angles)
    assert np.all(wrapped > -np.pi - 1e-9)
    assert np.all(wrapped <= np.pi + 1e-9)


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_wrap_to_pi_idempotent(angle: float) -> None:
    once = wrap_to_pi(angle)
    twice = wrap_to_pi(once)
    assert once == pytest.approx(twice, abs=1e-9)


@pytest.mark.req("FR-COR-004")
def test_angle_diff_shortest_path() -> None:
    # 350 deg vs 10 deg should differ by -20 deg (wrapping the long way around)
    a = np.deg2rad(350.0)
    b = np.deg2rad(10.0)
    diff = angle_diff(a, b)
    assert diff == pytest.approx(np.deg2rad(-20.0), abs=1e-9)


@pytest.mark.req("FR-COR-004")
def test_circular_mean_of_symmetric_angles_near_zero() -> None:
    angles = np.array([-0.1, 0.1, -0.05, 0.05])
    assert circular_mean(angles) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.req("FR-COR-004")
def test_circular_mean_wraparound() -> None:
    # angles clustered around +-pi should average near pi, not 0
    angles = np.array([np.pi - 0.1, -np.pi + 0.1])
    mean = circular_mean(angles)
    assert abs(abs(mean) - np.pi) < 0.2
