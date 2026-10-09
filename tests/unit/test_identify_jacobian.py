from __future__ import annotations

import numpy as np
import pytest

from aerovalid.identify.jacobian import central_difference_jacobian, identifiability


@pytest.mark.req("FR-IDN-003")
def test_central_difference_jacobian_matches_known_linear_coefficients() -> None:
    # residual(x) = A @ x, so the Jacobian is exactly A.
    a = np.array([[2.0, -1.0, 0.0], [0.0, 3.0, 5.0]])

    def residual_fn(x: np.ndarray) -> np.ndarray:
        return a @ x

    jacobian = central_difference_jacobian(residual_fn, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(jacobian, a, atol=1e-6)


@pytest.mark.req("FR-IDN-003")
def test_identifiability_recovers_small_std_errors_for_well_conditioned_fit() -> None:
    rng = np.random.default_rng(0)
    n_samples = 500
    x = rng.normal(0.0, 1.0, (n_samples, 2))
    jacobian = x  # residual = x @ theta_true + noise; Jacobian wrt theta is just x
    true_theta = np.array([2.0, -1.0])
    noise = rng.normal(0.0, 0.01, n_samples)  # the fit's residual at the optimum is pure noise

    result = identifiability(jacobian, noise, true_theta, ["a", "b"])
    assert result.std_errors.shape == (2,)
    assert np.all(result.std_errors < 0.05)
    assert result.condition_number < 100
    assert result.poorly_identifiable == []
    for name in ("a", "b"):
        lower, upper = result.ci_95[name]
        assert lower < true_theta[0 if name == "a" else 1] < upper


@pytest.mark.req("FR-IDN-003")
def test_identifiability_flags_collinear_parameters() -> None:
    rng = np.random.default_rng(1)
    n_samples = 200
    base = rng.normal(0.0, 1.0, n_samples)
    # param "b" is (almost) a copy of "a"'s column: the fit can't tell them apart.
    jacobian = np.column_stack([base, base + rng.normal(0.0, 1e-6, n_samples)])
    residuals = rng.normal(0.0, 0.1, n_samples)

    result = identifiability(jacobian, residuals, np.array([1.0, 1.0]), ["a", "b"])
    assert set(result.poorly_identifiable) == {"a", "b"}
    assert abs(result.correlation[0, 1]) > 0.95


@pytest.mark.req("FR-IDN-003")
def test_identifiability_correlation_diagonal_is_one() -> None:
    rng = np.random.default_rng(2)
    jacobian = rng.normal(0.0, 1.0, (100, 3))
    residuals = rng.normal(0.0, 1.0, 100)
    result = identifiability(jacobian, residuals, np.array([1.0, 2.0, 3.0]), ["a", "b", "c"])
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
