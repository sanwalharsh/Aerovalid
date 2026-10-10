"""Machine-readable `summary.json` and the auto-generated executive summary
(FR-RPT-003/006): every sentence is built from a computed number already present in
`SummaryFacts` and carries the artifact path it came from -- never a hard-coded claim
(CLAUDE.md §2's "don't invent results").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd


@dataclass(frozen=True)
class SummaryFacts:
    """Numbers pulled from whichever phases' artifacts exist for this run; a `None`
    field means that phase wasn't run (or didn't apply), and its sentence is skipped."""

    run_id: str
    n_flights: int
    variants: list[str] = field(default_factory=list)
    best_variant: str | None = None
    best_variant_mean_rank: float | None = None
    predict_rmse_reduction_vs_sim_only_pct: float | None = None
    predict_interval_coverage: float | None = None
    detect_recall: float | None = None
    detect_far_per_hour: float | None = None
    detect_median_delay_s: float | None = None
    identify_top_parameter: str | None = None
    identify_top_parameter_change_pct: float | None = None
    identify_revalidation_rmse_reduction_pct: float | None = None


@dataclass(frozen=True)
class Statement:
    """One executive-summary sentence and the artifact it's computed from."""

    text: str
    evidence_path: str


def build_summary_json(facts: SummaryFacts) -> dict[str, object]:
    """`summary.json`'s contents: just `facts` as a plain dict (FR-RPT-003)."""
    return asdict(facts)


def build_executive_summary(facts: SummaryFacts) -> list[Statement]:
    """
    Auto-generate the executive summary (FR-RPT-006): one `Statement` per computed
    fact that's actually present, each carrying the artifact it came from.

    Parameters
    ----------
    facts : SummaryFacts

    Returns
    -------
    list[Statement]
    """
    statements: list[Statement] = []

    statements.append(
        Statement(
            f"Run {facts.run_id} processed {facts.n_flights} flight(s) across "
            f"{len(facts.variants)} variant(s): {', '.join(facts.variants)}.",
            evidence_path="aligned/index.json",
        )
    )

    if facts.best_variant is not None and facts.best_variant_mean_rank is not None:
        statements.append(
            Statement(
                f"{facts.best_variant} ranked best overall (mean rank "
                f"{facts.best_variant_mean_rank:.2f}).",
                evidence_path="validation/variant_ranking.parquet",
            )
        )

    if facts.predict_rmse_reduction_vs_sim_only_pct is not None:
        statements.append(
            Statement(
                f"The hybrid (sim + predicted residual) correction reduced held-out "
                f"RMSE by {facts.predict_rmse_reduction_vs_sim_only_pct:.1f}% vs "
                f"trusting the simulator alone.",
                evidence_path="predict/cv_results.parquet",
            )
        )

    if facts.predict_interval_coverage is not None:
        statements.append(
            Statement(
                f"Predicted residual intervals achieved "
                f"{100.0 * facts.predict_interval_coverage:.1f}% empirical coverage.",
                evidence_path="predict/intervals.parquet",
            )
        )

    if facts.detect_recall is not None and facts.detect_far_per_hour is not None:
        statements.append(
            Statement(
                f"The deviation detector recalled {100.0 * facts.detect_recall:.0f}% "
                f"of injected faults at {facts.detect_far_per_hour:.2f} false alarms "
                f"per flight-hour.",
                evidence_path="detect/detection_metrics.parquet",
            )
        )

    if facts.detect_median_delay_s is not None:
        statements.append(
            Statement(
                f"Median detection delay was {facts.detect_median_delay_s:.1f} s.",
                evidence_path="detect/detection_metrics.parquet",
            )
        )

    if (
        facts.identify_top_parameter is not None
        and facts.identify_top_parameter_change_pct is not None
    ):
        statements.append(
            Statement(
                f"{facts.identify_top_parameter} shifted "
                f"{facts.identify_top_parameter_change_pct:+.1f}% from its nominal "
                f"value in the output-error fit.",
                evidence_path="identify/fit_results.json",
            )
        )

    if facts.identify_revalidation_rmse_reduction_pct is not None:
        statements.append(
            Statement(
                f"Re-validating the refit variant on held-out flights changed RMSE "
                f"by {facts.identify_revalidation_rmse_reduction_pct:+.1f}% (positive "
                f"= improvement).",
                evidence_path="identify/revalidation.parquet",
            )
        )

    return statements


def _read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _read_json_list_or_empty(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text()) if path.exists() else []


def gather_summary_facts(run_id: str, index: dict[str, list[str]], run_dir: Path) -> SummaryFacts:
    """
    Read whichever phase artifacts exist under `run_dir` into `SummaryFacts`
    (FR-RPT-003/006). Shared by `report/builder.py` (the static HTML report) and
    `app/main.py` (the live dashboard's Overview page) so both tell the same story
    from the same numbers.

    Parameters
    ----------
    run_id : str
    index : dict[str, list[str]]
        `aligned/index.json`'s `{"flights": [...], "variants": [...]}`.
    run_dir : Path
        `runs/<run_id>/`.

    Returns
    -------
    SummaryFacts
    """
    ranking = _read_parquet_or_empty(run_dir / "validation" / "variant_ranking.parquet")
    best_variant = str(ranking.iloc[0]["variant"]) if not ranking.empty else None
    best_mean_rank = float(ranking.iloc[0]["mean_rank"]) if not ranking.empty else None

    cv_results = _read_parquet_or_empty(run_dir / "predict" / "cv_results.parquet")
    hybrid_rows = (
        cv_results[cv_results.get("task") == "hybrid"] if not cv_results.empty else cv_results
    )
    predict_reduction = (
        float(hybrid_rows["rmse_reduction_vs_sim_only_pct"].mean())
        if not hybrid_rows.empty and "rmse_reduction_vs_sim_only_pct" in hybrid_rows.columns
        else None
    )

    intervals = _read_parquet_or_empty(run_dir / "predict" / "intervals.parquet")
    predict_coverage = float(intervals["coverage"].mean()) if not intervals.empty else None

    detect_metrics = _read_parquet_or_empty(run_dir / "detect" / "detection_metrics.parquet")
    overall_detect = (
        detect_metrics[detect_metrics["fault_type"].isna()]
        if not detect_metrics.empty
        else detect_metrics
    )
    detect_recall = float(overall_detect["recall"].mean()) if not overall_detect.empty else None
    detect_far = float(overall_detect["far_per_hour"].mean()) if not overall_detect.empty else None
    detect_delay = (
        float(overall_detect["median_delay_s"].dropna().mean())
        if not overall_detect.empty and overall_detect["median_delay_s"].notna().any()
        else None
    )

    recommendations = _read_json_list_or_empty(run_dir / "identify" / "recommendations.json")
    top_param: str | None = None
    top_change: float | None = None
    if recommendations:
        top = recommendations[0]
        top_param = cast(str, top["parameter"])
        top_change = cast(float, top["estimated_change_pct"])

    revalidation = _read_parquet_or_empty(run_dir / "identify" / "revalidation.parquet")
    revalidation_reduction = (
        float(revalidation["rmse_reduction_pct"].mean()) if not revalidation.empty else None
    )

    return SummaryFacts(
        run_id=run_id,
        n_flights=len(index.get("flights", [])),
        variants=index.get("variants", []),
        best_variant=best_variant,
        best_variant_mean_rank=best_mean_rank,
        predict_rmse_reduction_vs_sim_only_pct=predict_reduction,
        predict_interval_coverage=predict_coverage,
        detect_recall=detect_recall,
        detect_far_per_hour=detect_far,
        detect_median_delay_s=detect_delay,
        identify_top_parameter=top_param,
        identify_top_parameter_change_pct=top_change,
        identify_revalidation_rmse_reduction_pct=revalidation_reduction,
    )
