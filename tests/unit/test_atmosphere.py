from __future__ import annotations

import numpy as np
import pytest

from aerovalid.core.atmosphere import RHO0_KGM3, isa_atmosphere

# Standard ISA table reference values (temperature K, pressure Pa, density kg/m^3).
_ISA_TABLE = {
    0: (288.15, 101325.0, 1.2250),
    5000: (255.65, 54048.0, 0.73612),
    11000: (216.65, 22632.0, 0.36392),
    15000: (216.65, 12044.0, 0.19367),
}


@pytest.mark.req("FR-COR-003")
@pytest.mark.parametrize(("alt_m", "expected"), list(_ISA_TABLE.items()))
def test_isa_matches_table_within_0_1_percent(
    alt_m: float, expected: tuple[float, float, float]
) -> None:
    atm = isa_atmosphere(alt_m)
    expected_t, expected_p, expected_rho = expected
    assert atm.temperature_k == pytest.approx(expected_t, rel=1e-3)
    assert atm.pressure_pa == pytest.approx(expected_p, rel=1e-3)
    assert atm.density_kgm3 == pytest.approx(expected_rho, rel=1e-3)


@pytest.mark.req("FR-COR-003")
def test_sea_level_density_constant() -> None:
    assert pytest.approx(1.225, rel=1e-3) == RHO0_KGM3


@pytest.mark.req("FR-COR-003")
def test_delta_t_offset_increases_temperature_and_reduces_density() -> None:
    baseline = isa_atmosphere(2000.0)
    offset = isa_atmosphere(2000.0, delta_t_k=10.0)
    assert offset.temperature_k == pytest.approx(baseline.temperature_k + 10.0)
    assert offset.density_kgm3 < baseline.density_kgm3


@pytest.mark.req("FR-COR-003")
def test_vectorised_input() -> None:
    altitudes = np.array([0.0, 1000.0, 5000.0, 11000.0])
    atm = isa_atmosphere(altitudes)
    assert atm.density_kgm3.shape == altitudes.shape
    assert np.all(np.diff(atm.density_kgm3) < 0)  # density strictly decreases with altitude


@pytest.mark.req("FR-COR-003")
def test_speed_of_sound_decreases_with_altitude_in_troposphere() -> None:
    low = isa_atmosphere(0.0)
    high = isa_atmosphere(10000.0)
    assert high.speed_of_sound_mps < low.speed_of_sound_mps
