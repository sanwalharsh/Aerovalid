"""Local sensitivity (Jacobian) and identifiability diagnostics (FR-IDN-003, analysis
methods §10): central finite-difference Jacobian of a residual vector wrt parameters,
Wald confidence intervals from `(J^T J)^-1`, and condition-number/correlation flags for
poorly identifiable parameter combinations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_CORRELATION_FLAG_THRESHOLD = 0.95
_CONDITION_NUMBER_FLAG_THRESHOLD = 1e6
_Z_95 = 1.959963985


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Covariance-based identifiability diagnostics at one fitted parameter point."""

    param_names: list[str]
    jacobian: NDArray[np.float64]
    covariance: NDArray[np.float64]
    std_errors: NDArray[np.float64]
    ci_95: dict[str, tuple[float, float]]
    correlation: NDArray[np.float64]
    condition_number: float
    poorly_identifiable: list[str]


def central_difference_jacobian(
    residual_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    rel_step: float = 1e-4,
) -> NDArray[np.float64]:
    """
    Central finite-difference Jacobian of `residual_fn` at `x0` (analysis methods §10).

    Parameters
    ----------
    residual_fn : callable
        `params -> residual vector`, shape `(m,)`.
    x0 : ndarray, shape (n,)
        The point to differentiate at (typically the fitted parameters).
    rel_step : float, default 1e-4
        Relative finite-difference step per parameter (absolute step floored at
        `1e-8` so a zero-valued parameter still gets perturbed).

    Returns
    -------
    ndarray, shape (m, n)
    """
    x0 = np.asarray(x0, dtype=np.float64)
    f0 = residual_fn(x0)
    jacobian = np.zeros((len(f0), len(x0)))
    for i in range(len(x0)):
        step = rel_step * max(abs(x0[i]), 1e-8)
        x_plus, x_minus = x0.copy(), x0.copy()
        x_plus[i] += step
        x_minus[i] -= step
        jacobian[:, i] = (residual_fn(x_plus) - residual_fn(x_minus)) / (2.0 * step)
    return jacobian


def identifiability(
    jacobian: NDArray[np.float64],
    residuals: NDArray[np.float64],
    fitted_values: NDArray[np.float64],
    param_names: list[str],
) -> IdentifiabilityResult:
    """
    Parameter covariance, Wald CIs, and identifiability flags from a fit's Jacobian
    (analysis methods §10): `Sigma_theta ~= s^2 (J^T J)^-1`.

    Parameters
    ----------
    jacobian : ndarray, shape (m, n)
    residuals : ndarray, shape (m,)
        The fit's residual vector at `fitted_values` (used for the `s^2` scale).
    fitted_values : ndarray, shape (n,)
    param_names : list[str]

    Returns
    -------
    IdentifiabilityResult
    """
    m, n = jacobian.shape
    dof = max(m - n, 1)
    s_squared = float(np.sum(residuals**2) / dof)
    jtj = jacobian.T @ jacobian
    condition_number = (
        float(np.linalg.cond(jtj)) if n > 1 else float(np.linalg.cond(jtj.reshape(1, 1)))
    )
    covariance: NDArray[np.float64] = (s_squared * np.linalg.pinv(jtj)).astype(np.float64)
    std_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    ci_95 = {
        name: (float(value - _Z_95 * se), float(value + _Z_95 * se))
        for name, value, se in zip(param_names, fitted_values, std_errors, strict=True)
    }

    denom = np.outer(std_errors, std_errors)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(denom > 0, covariance / denom, 0.0)
    np.fill_diagonal(correlation, 1.0)

    flagged: set[str] = set()
    if condition_number > _CONDITION_NUMBER_FLAG_THRESHOLD:
        flagged.update(param_names)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(correlation[i, j]) > _CORRELATION_FLAG_THRESHOLD:
                flagged.add(param_names[i])
                flagged.add(param_names[j])

    return IdentifiabilityResult(
        param_names=list(param_names),
        jacobian=jacobian,
        covariance=covariance,
        std_errors=std_errors,
        ci_95=ci_95,
        correlation=correlation,
        condition_number=condition_number,
        poorly_identifiable=sorted(flagged),
    )
