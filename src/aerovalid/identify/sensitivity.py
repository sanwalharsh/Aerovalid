"""Global sensitivity analysis (FR-IDN-001, analysis methods §10): Morris screening to
rank parameters cheaply, then Sobol (Saltelli sampling) for variance-based indices with
bootstrap confidence intervals, of any scalar model/metric wrt its parameters.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _problem(param_names: list[str], bounds: list[tuple[float, float]]) -> dict[str, object]:
    return {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": [list(b) for b in bounds],
    }


def morris_screen(
    model_fn: Callable[[NDArray[np.float64]], float],
    param_names: list[str],
    bounds: list[tuple[float, float]],
    n_trajectories: int = 20,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Morris elementary-effects screening (FR-IDN-001, analysis methods §10).

    Parameters
    ----------
    model_fn : callable
        `params (shape (n,)) -> scalar metric`.
    param_names : list[str]
    bounds : list[tuple[float, float]]
        Per-parameter `(lower, upper)`, same order as `param_names`.
    n_trajectories : int, default 20
    seed : int, optional

    Returns
    -------
    pandas.DataFrame
        `parameter, mu_star, sigma, mu_star_conf`, sorted most-influential first.
    """
    from SALib.analyze import morris as morris_analyze
    from SALib.sample import morris as morris_sample

    problem = _problem(param_names, bounds)
    samples = morris_sample.sample(problem, N=n_trajectories, seed=seed)
    outputs = np.array([model_fn(row) for row in samples])
    result = morris_analyze.analyze(problem, samples, outputs, seed=seed)
    return pd.DataFrame(
        {
            "parameter": param_names,
            "mu_star": result["mu_star"],
            "sigma": result["sigma"],
            "mu_star_conf": result["mu_star_conf"],
        }
    ).sort_values("mu_star", ascending=False, ignore_index=True)


def sobol_indices(
    model_fn: Callable[[NDArray[np.float64]], float],
    param_names: list[str],
    bounds: list[tuple[float, float]],
    n: int = 1024,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Sobol variance-based sensitivity indices via Saltelli sampling (FR-IDN-001).

    Parameters
    ----------
    model_fn : callable
        `params (shape (n,)) -> scalar metric`.
    param_names : list[str]
    bounds : list[tuple[float, float]]
    n : int, default 1024
        Saltelli base sample size (total evaluations: `n * (2*d + 2)`).
    seed : int, optional

    Returns
    -------
    pandas.DataFrame
        `parameter, S1, S1_conf, ST, ST_conf`. `S1.sum()` and `ST.sum()` are each
        approximately 1 for an additive model with no interactions; see
        `test_sensitivity.py` for a worked analytic check.
    """
    from SALib.analyze.sobol import analyze as sobol_analyze
    from SALib.sample.sobol import sample as sobol_sample

    problem = _problem(param_names, bounds)
    samples = sobol_sample(problem, N=n, seed=seed)
    outputs = np.array([model_fn(row) for row in samples])
    result = sobol_analyze(problem, outputs, seed=seed)
    return pd.DataFrame(
        {
            "parameter": param_names,
            "S1": result["S1"],
            "S1_conf": result["S1_conf"],
            "ST": result["ST"],
            "ST_conf": result["ST_conf"],
        }
    )
