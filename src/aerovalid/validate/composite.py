"""Composite score and variant ranking (FR-VAL-006, analysis methods §5): weighted
normalised RMSE across signals, flight-level bootstrap ranking with CIs, and rank
agreement against a known ground-truth ordering (benchmark mode, S2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import kendalltau, spearmanr

# Default composite-score weights (analysis methods §5).
DEFAULT_WEIGHTS: dict[str, float] = {
    "alt_m": 0.2,
    "vel_air_mps": 0.2,
    "gamma_rad": 0.1,
    "chi_rad": 0.1,
    "nz_g": 0.15,
    "pitch_rad": 0.1,
    "roll_rad": 0.05,
}


def composite_score(
    metrics_by_flight_variant: pd.DataFrame,
    ref_variant: str,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Composite score per (flight, variant): `S = sum_s w_s * (RMSE_s,v / RMSE_s,ref)`.
    Lower is better.

    Parameters
    ----------
    metrics_by_flight_variant : pandas.DataFrame
        Must have `flight_id`, `variant`, `signal`, `rmse` columns (e.g. several
        `compute_metrics_table` calls, one per (flight, variant), concatenated).
    ref_variant : str
        Reference variant each RMSE is normalised against (e.g. `"V0"`).
    weights : dict[str, float], optional
        `signal -> weight`; defaults to `DEFAULT_WEIGHTS`. Signals without a weight
        entry are excluded from the sum.

    Returns
    -------
    pandas.DataFrame
        `flight_id, variant, composite_score`.
    """
    weights = weights or DEFAULT_WEIGHTS
    pivot = metrics_by_flight_variant.pivot_table(
        index=["flight_id", "signal"], columns="variant", values="rmse"
    )
    if ref_variant not in pivot.columns:
        raise ValueError(f"ref_variant {ref_variant!r} not found in metrics_by_flight_variant")
    normalised = pivot.div(pivot[ref_variant], axis=0)

    long = normalised.reset_index().melt(
        id_vars=["flight_id", "signal"], var_name="variant", value_name="rmse_ratio"
    )
    long["weight"] = long["signal"].map(weights).fillna(0.0)
    long["weighted"] = long["weight"] * long["rmse_ratio"]
    scores = long.groupby(["flight_id", "variant"], observed=True)["weighted"].sum()
    return scores.reset_index(name="composite_score")


def rank_variants(
    scores: pd.DataFrame, n_bootstrap: int = 2000, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Flight-level bootstrap ranking: resample flights with replacement, recompute each
    variant's mean score and rank, and report the rank distribution (FR-VAL-006).

    Parameters
    ----------
    scores : pandas.DataFrame
        `flight_id, variant, composite_score` (e.g. `composite_score`'s output).
    n_bootstrap : int, default 2000
    rng : numpy.random.Generator, optional

    Returns
    -------
    pandas.DataFrame
        One row per variant, sorted best-first (`mean_rank` ascending): `variant`,
        `mean_score`, `mean_rank`, `rank_ci_low`, `rank_ci_high`, `p_best` (bootstrap
        fraction where this variant has the lowest mean score).
    """
    rng = rng if rng is not None else np.random.default_rng()
    pivot = scores.pivot(index="flight_id", columns="variant", values="composite_score")
    variants = pivot.columns.tolist()
    n_flights = len(pivot)

    rank_samples = np.empty((n_bootstrap, len(variants)))
    best_counts = np.zeros(len(variants))
    values = pivot.to_numpy()
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n_flights, size=n_flights)
        mean_scores = values[sample_idx].mean(axis=0)
        rank_samples[i] = pd.Series(mean_scores).rank(method="average").to_numpy()
        best_counts[int(np.argmin(mean_scores))] += 1

    summary = pd.DataFrame(
        {
            "variant": variants,
            "mean_score": values.mean(axis=0),
            "mean_rank": rank_samples.mean(axis=0),
            "rank_ci_low": np.percentile(rank_samples, 2.5, axis=0),
            "rank_ci_high": np.percentile(rank_samples, 97.5, axis=0),
            "p_best": best_counts / n_bootstrap,
        }
    )
    return summary.sort_values("mean_rank").reset_index(drop=True)


def rank_agreement(observed_order: list[str], ground_truth_order: list[str]) -> tuple[float, float]:
    """
    Spearman rho and Kendall tau between an observed variant ranking and a known
    ground-truth fidelity order (benchmark mode, S2).

    Parameters
    ----------
    observed_order : list[str]
        Variant names, best-first, as recovered by `rank_variants`.
    ground_truth_order : list[str]
        Variant names, best-first, from `configs/benchmark_ground_truth.yaml`.

    Returns
    -------
    tuple[float, float]
        `(spearman_rho, kendall_tau)`.
    """
    gt_position = {name: i for i, name in enumerate(ground_truth_order)}
    observed_gt_positions: NDArray[np.float64] = np.array(
        [gt_position[name] for name in observed_order], dtype=np.float64
    )
    expected_positions = np.arange(len(observed_order), dtype=np.float64)
    rho, _ = spearmanr(observed_gt_positions, expected_positions)
    tau, _ = kendalltau(observed_gt_positions, expected_positions)
    return float(rho), float(tau)
