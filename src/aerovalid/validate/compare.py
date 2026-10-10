"""Variant comparison statistics (FR-VAL-005, analysis methods §5): block bootstrap,
paired Wilcoxon + Holm, Friedman + Nemenyi, and effect sizes (Cliff's delta,
Hodges-Lehmann shift). Flights are the unit of replication for inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from numpy.typing import NDArray
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def block_bootstrap_ci(
    values: NDArray[np.float64],
    block_size: int,
    n_bootstrap: int = 2000,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Moving-block bootstrap CI for the mean of a (possibly autocorrelated) time series.

    Parameters
    ----------
    values : ndarray, shape (N,)
        E.g. a per-sample metric-difference time series.
    block_size : int
        Block length, samples (`block_s / dt`).
    n_bootstrap : int, default 2000
    rng : numpy.random.Generator, optional
    alpha : float, default 0.05
        CI level: returns the `(alpha/2, 1 - alpha/2)` percentiles.

    Returns
    -------
    tuple[float, float]
        `(lower, upper)` bootstrap CI for `mean(values)`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan")
    block_size = max(1, min(block_size, n))
    n_blocks = max(1, n // block_size)

    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([v[s : s + block_size] for s in starts])
        means[i] = np.mean(sample)
    return float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2))
    )


def cliffs_delta(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """
    Cliff's delta effect size: `(#[a_i > b_j] - #[a_i < b_j]) / (len(a) * len(b))`.

    Returns
    -------
    float
        In `[-1, 1]`; 0 = no stochastic dominance, +-1 = complete separation.
    """
    a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = a_arr[:, None] - b_arr[None, :]
    n_pos, n_neg = np.count_nonzero(diff > 0), np.count_nonzero(diff < 0)
    return float((n_pos - n_neg) / (len(a_arr) * len(b_arr)))


def hodges_lehmann_shift(differences: NDArray[np.float64]) -> float:
    """
    Hodges-Lehmann point estimate for paired data: the median of all pairwise Walsh
    averages `(d_i + d_j) / 2` (`i <= j`) of the per-pair differences -- the estimator
    naturally paired with the Wilcoxon signed-rank test.

    Parameters
    ----------
    differences : ndarray, shape (N,)
        Per-flight paired differences (e.g. variant A's composite score minus B's).

    Returns
    -------
    float
    """
    d = np.asarray(differences, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    sums = d[:, None] + d[None, :]
    iu = np.triu_indices(n)
    return float(np.median(sums[iu] / 2.0))


@dataclass(frozen=True)
class PairwiseComparison:
    """One variant-pair's paired test result over flights."""

    variant_a: str
    variant_b: str
    n_flights: int
    wilcoxon_stat: float
    p_value: float
    p_value_holm: float
    cliffs_delta: float
    hodges_lehmann_shift: float


def compare_variant_pairs(
    scores_by_variant: dict[str, NDArray[np.float64]],
) -> list[PairwiseComparison]:
    """
    Paired Wilcoxon signed-rank test (+ Holm-Bonferroni correction across all pairs)
    between every pair of variants, on one paired score per flight (FR-VAL-005).

    Parameters
    ----------
    scores_by_variant : dict[str, ndarray]
        `variant_name -> per-flight score` (e.g. composite score); all arrays must be
        the same length and flight-aligned (same flight at the same index).

    Returns
    -------
    list[PairwiseComparison]
        One entry per unordered pair, `p_value_holm` corrected across the whole list.
    """
    names = sorted(scores_by_variant)
    raw_results = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            a, b = scores_by_variant[name_a], scores_by_variant[name_b]
            differences = a - b
            if np.allclose(differences, 0.0):
                stat, p_value = float("nan"), 1.0
            else:
                stat, p_value = wilcoxon(a, b)
            raw_results.append(
                (name_a, name_b, len(a), float(stat), float(p_value), a, b, differences)
            )

    if not raw_results:
        return []
    p_values = [r[4] for r in raw_results]
    _, p_adjusted, _, _ = multipletests(p_values, method="holm")

    return [
        PairwiseComparison(
            variant_a=name_a,
            variant_b=name_b,
            n_flights=n,
            wilcoxon_stat=stat,
            p_value=p_value,
            p_value_holm=float(p_adj),
            cliffs_delta=cliffs_delta(a, b),
            hodges_lehmann_shift=hodges_lehmann_shift(differences),
        )
        for (name_a, name_b, n, stat, p_value, a, b, differences), p_adj in zip(
            raw_results, p_adjusted, strict=True
        )
    ]


def friedman_test(scores_by_variant: dict[str, NDArray[np.float64]]) -> tuple[float, float]:
    """
    Friedman test across all variants (>2) on paired per-flight scores.

    Returns
    -------
    tuple[float, float]
        `(statistic, p_value)`.
    """
    arrays = [scores_by_variant[name] for name in sorted(scores_by_variant)]
    stat, p_value = friedmanchisquare(*arrays)
    return float(stat), float(p_value)


def nemenyi_posthoc(scores_by_variant: dict[str, NDArray[np.float64]]) -> pd.DataFrame:
    """
    Nemenyi post-hoc test following a significant Friedman test.

    Returns
    -------
    pandas.DataFrame
        Square matrix of pairwise p-values, indexed/columned by variant name.
    """
    names = sorted(scores_by_variant)
    data = np.column_stack([scores_by_variant[name] for name in names])
    result: pd.DataFrame = sp.posthoc_nemenyi_friedman(data)
    result.index = names
    result.columns = names
    return result
