"""Machine-readable `summary.json` and the auto-generated executive summary
(FR-RPT-003/006): every sentence is built from a computed number already present in
`SummaryFacts` and carries the artifact path it came from -- never a hard-coded claim
(CLAUDE.md §2's "don't invent results").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
