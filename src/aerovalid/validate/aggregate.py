"""Per-signal/phase/bin/horizon metric aggregation (FR-VAL-002), and flight-level
aggregation (analysis methods §4: "flights are the unit of replication for inference").
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from aerovalid.validate.metrics import MetricSet, compute_metrics_from_residual
from aerovalid.validate.tolerance import tolerance_compliance


def compute_metrics_table(
    aligned: pd.DataFrame,
    signals: list[str],
    tolerances: dict[str, float] | None = None,
    groupby: str | None = None,
) -> pd.DataFrame:
    """
    Compute every signal's `MetricSet` (+ tolerance compliance) over one or more
    aligned tables, optionally grouped by a column (e.g. `phase` or a `cond_*` band).

    Parameters
    ----------
    aligned : pandas.DataFrame
        One or more concatenated `prep/align.py` aligned tables (must carry
        `flight_id`); masked samples (`mask_<sig> == True`) are excluded.
    signals : list[str]
        Canonical signal names to compute metrics for.
    tolerances : dict[str, float], optional
        Per-signal engineering tolerance for compliance; signals without an entry
        get `tol_compliance = nan`.
    groupby : str, optional
        Column to group by before computing metrics (e.g. `"phase"`); `None` computes
        one row per signal over the whole table.

    Returns
    -------
    pandas.DataFrame
        One row per (signal, group); columns are `MetricSet`'s fields plus `signal`,
        `group`, and `tol_compliance`. Always has an `n_samples` column (FR-VAL-002).
    """
    tolerances = tolerances or {}
    groups: list[tuple[object, pd.DataFrame]] = (
        list(aligned.groupby(groupby, observed=True)) if groupby else [(None, aligned)]
    )

    rows: list[dict[str, object]] = []
    for group_value, group_df in groups:
        for signal in signals:
            obs_col, sim_col, res_col, mask_col = (
                f"obs_{signal}",
                f"sim_{signal}",
                f"res_{signal}",
                f"mask_{signal}",
            )
            if obs_col not in group_df.columns or sim_col not in group_df.columns:
                continue
            mask = (
                group_df[mask_col].to_numpy(dtype=bool)
                if mask_col in group_df.columns
                else np.zeros(len(group_df), dtype=bool)
            )
            valid = group_df.loc[~mask]
            obs = valid[obs_col].to_numpy(dtype=np.float64)
            sim = valid[sim_col].to_numpy(dtype=np.float64)
            residual = (
                valid[res_col].to_numpy(dtype=np.float64) if res_col in valid.columns else obs - sim
            )
            metric_set: MetricSet = compute_metrics_from_residual(obs, sim, residual)
            tol = tolerances.get(signal)
            compliance = tolerance_compliance(residual, tol) if tol is not None else float("nan")
            rows.append(
                {
                    "signal": signal,
                    "group": group_value,
                    **asdict(metric_set),
                    "tol_compliance": compliance,
                }
            )

    return pd.DataFrame(rows)


def aggregate_across_flights(metrics_table: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """
    Flight-level aggregation (mean, median, IQR) of a per-flight metrics table.

    Parameters
    ----------
    metrics_table : pandas.DataFrame
        Must have `flight_id`, `signal`, and (optionally) `group`, as produced by
        calling `compute_metrics_table` once per flight and concatenating.
    value_columns : list[str]
        Numeric metric columns to aggregate (e.g. `["rmse", "mae", "tol_compliance"]`).

    Returns
    -------
    pandas.DataFrame
        One row per (signal, [group]) with `<col>_mean`, `<col>_median`, `<col>_iqr`
        for each `value_columns` entry, plus `n_flights`.
    """
    group_cols = [c for c in ("signal", "group") if c in metrics_table.columns]
    grouped = metrics_table.groupby(group_cols, observed=True)

    summary = grouped.agg(n_flights=("flight_id", "nunique"))
    for col in value_columns:
        summary[f"{col}_mean"] = grouped[col].mean()
        summary[f"{col}_median"] = grouped[col].median()
        summary[f"{col}_iqr"] = grouped[col].quantile(0.75) - grouped[col].quantile(0.25)

    return summary.reset_index()
