"""Angle wrapping and circular statistics. All attitude/track residuals must be wrapped
with `wrap_to_pi` before use in metrics (CLAUDE.md §4, data spec §5).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wrap_to_pi(angle_rad: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """
    Wrap angle(s) into the half-open interval (-pi, pi].

    Parameters
    ----------
    angle_rad : ndarray or float, shape (N,) or scalar
        Angle(s) in radians, any magnitude.

    Returns
    -------
    ndarray or float
        Wrapped angle(s) in radians, same shape as input, in (-pi, pi].
    """
    wrapped = np.asarray(angle_rad, dtype=np.float64)
    wrapped = ((wrapped + np.pi) % (2.0 * np.pi)) - np.pi
    wrapped = np.where(wrapped <= -np.pi, wrapped + 2.0 * np.pi, wrapped)
    if np.isscalar(angle_rad) or (isinstance(angle_rad, np.ndarray) and angle_rad.ndim == 0):
        return float(wrapped)
    return wrapped


def angle_diff(a_rad: NDArray[np.float64], b_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Shortest signed angular difference `a - b`, wrapped to (-pi, pi].

    Parameters
    ----------
    a_rad, b_rad : ndarray, shape (N,)
        Angles in radians.

    Returns
    -------
    ndarray, shape (N,)
        `wrap_to_pi(a - b)`.
    """
    result = wrap_to_pi(np.asarray(a_rad, dtype=np.float64) - np.asarray(b_rad, dtype=np.float64))
    return np.asarray(result, dtype=np.float64)


def circular_mean(
    angles_rad: NDArray[np.float64], axis: int | None = None
) -> NDArray[np.float64] | float:
    """
    Mean of angles on the unit circle.

    Parameters
    ----------
    angles_rad : ndarray, shape (N,)
        Angles in radians.
    axis : int, optional
        Axis to reduce over; default reduces the whole array.

    Returns
    -------
    ndarray or float
        Circular mean in (-pi, pi].
    """
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    mean_sin = np.mean(np.sin(angles_rad), axis=axis)
    mean_cos = np.mean(np.cos(angles_rad), axis=axis)
    result = np.arctan2(mean_sin, mean_cos)
    return wrap_to_pi(result)
