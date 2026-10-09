"""Trajectory metrics (FR-VAL-003, analysis methods §4): 3-D position error,
along/cross-track decomposition relative to the observed ground track, and CEP50/95.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Position-error summary over one flight's aligned samples."""

    n_samples: int
    rmse_3d_m: float
    rmse_horizontal_m: float
    rmse_vertical_m: float
    along_track_rmse_m: float
    cross_track_rmse_m: float
    cep50_m: float
    cep95_m: float


def trajectory_error(
    obs_x_m: NDArray[np.float64],
    obs_y_m: NDArray[np.float64],
    obs_alt_m: NDArray[np.float64],
    sim_x_m: NDArray[np.float64],
    sim_y_m: NDArray[np.float64],
    sim_alt_m: NDArray[np.float64],
    obs_chi_rad: NDArray[np.float64],
) -> TrajectoryMetrics:
    """
    Compute 3-D/horizontal/vertical position error, along/cross-track error (relative
    to the observed ground-track bearing `obs_chi_rad`, 0 = North, clockwise), and
    the horizontal-error CEP50/95.

    Parameters
    ----------
    obs_x_m, obs_y_m, obs_alt_m : ndarray, shape (N,)
        Observed local-NE position and altitude.
    sim_x_m, sim_y_m, sim_alt_m : ndarray, shape (N,)
        Simulated counterparts.
    obs_chi_rad : ndarray, shape (N,)
        Observed track angle at each sample, used to rotate the horizontal error
        into along-track / cross-track components.

    Returns
    -------
    TrajectoryMetrics
    """
    dx = np.asarray(obs_x_m, dtype=np.float64) - np.asarray(sim_x_m, dtype=np.float64)
    dy = np.asarray(obs_y_m, dtype=np.float64) - np.asarray(sim_y_m, dtype=np.float64)
    dz = np.asarray(obs_alt_m, dtype=np.float64) - np.asarray(sim_alt_m, dtype=np.float64)
    chi = np.asarray(obs_chi_rad, dtype=np.float64)

    valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz) & np.isfinite(chi)
    dx, dy, dz, chi = dx[valid], dy[valid], dz[valid], chi[valid]
    n = dx.shape[0]
    if n == 0:
        return TrajectoryMetrics(0, *([float("nan")] * 7))

    horizontal_err = np.hypot(dx, dy)
    err_3d = np.sqrt(dx**2 + dy**2 + dz**2)

    along_track = dx * np.cos(chi) + dy * np.sin(chi)
    cross_track = -dx * np.sin(chi) + dy * np.cos(chi)

    return TrajectoryMetrics(
        n_samples=n,
        rmse_3d_m=float(np.sqrt(np.mean(err_3d**2))),
        rmse_horizontal_m=float(np.sqrt(np.mean(horizontal_err**2))),
        rmse_vertical_m=float(np.sqrt(np.mean(dz**2))),
        along_track_rmse_m=float(np.sqrt(np.mean(along_track**2))),
        cross_track_rmse_m=float(np.sqrt(np.mean(cross_track**2))),
        cep50_m=float(np.percentile(horizontal_err, 50)),
        cep95_m=float(np.percentile(horizontal_err, 95)),
    )
