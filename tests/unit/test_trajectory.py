from __future__ import annotations

import numpy as np
import pytest

from aerovalid.validate.trajectory import trajectory_error


@pytest.mark.req("FR-VAL-003")
def test_trajectory_error_constant_offset_hand_computed() -> None:
    n = 10
    obs_x = np.zeros(n)
    obs_y = np.zeros(n)
    obs_alt = np.full(n, 100.0)
    sim_x = np.full(n, 3.0)
    sim_y = np.full(n, 4.0)
    sim_alt = np.full(n, 95.0)
    chi = np.zeros(n)  # heading = North

    result = trajectory_error(obs_x, obs_y, obs_alt, sim_x, sim_y, sim_alt, chi)

    # dx=-3, dy=-4 => horizontal error = 5 exactly; dz=5
    assert result.rmse_horizontal_m == pytest.approx(5.0)
    assert result.rmse_vertical_m == pytest.approx(5.0)
    assert result.rmse_3d_m == pytest.approx(np.sqrt(3**2 + 4**2 + 5**2))
    assert result.cep50_m == pytest.approx(5.0)
    assert result.cep95_m == pytest.approx(5.0)


@pytest.mark.req("FR-VAL-003")
def test_along_track_error_when_heading_north() -> None:
    # obs is ahead of sim by 10m to the North (along-track) with chi=0 (heading North)
    obs_x = np.array([10.0])
    obs_y = np.array([0.0])
    obs_alt = np.array([0.0])
    sim_x = np.array([0.0])
    sim_y = np.array([0.0])
    sim_alt = np.array([0.0])
    chi = np.array([0.0])

    result = trajectory_error(obs_x, obs_y, obs_alt, sim_x, sim_y, sim_alt, chi)
    assert result.along_track_rmse_m == pytest.approx(10.0)
    assert result.cross_track_rmse_m == pytest.approx(0.0, abs=1e-9)


@pytest.mark.req("FR-VAL-003")
def test_cross_track_error_when_heading_north() -> None:
    # obs is offset to the East (cross-track) relative to a North-heading track
    obs_x = np.array([0.0])
    obs_y = np.array([10.0])
    obs_alt = np.array([0.0])
    sim_x = np.array([0.0])
    sim_y = np.array([0.0])
    sim_alt = np.array([0.0])
    chi = np.array([0.0])

    result = trajectory_error(obs_x, obs_y, obs_alt, sim_x, sim_y, sim_alt, chi)
    assert result.cross_track_rmse_m == pytest.approx(10.0)
    assert result.along_track_rmse_m == pytest.approx(0.0, abs=1e-9)


@pytest.mark.req("FR-VAL-003")
def test_empty_input_returns_nan() -> None:
    empty = np.array([])
    result = trajectory_error(empty, empty, empty, empty, empty, empty, empty)
    assert result.n_samples == 0
    assert np.isnan(result.rmse_3d_m)
