from __future__ import annotations

import re

import pytest

from aerovalid.report.summary import SummaryFacts, build_executive_summary, build_summary_json


@pytest.mark.req("FR-RPT-006")
def test_build_summary_json_round_trips_facts() -> None:
    facts = SummaryFacts(run_id="r1", n_flights=4, variants=["V0", "V1"], best_variant="V1")
    data = build_summary_json(facts)
    assert data["run_id"] == "r1"
    assert data["n_flights"] == 4
    assert data["variants"] == ["V0", "V1"]
    assert data["best_variant"] == "V1"


@pytest.mark.req("FR-RPT-006")
def test_executive_summary_always_includes_the_run_overview_statement() -> None:
    facts = SummaryFacts(run_id="r1", n_flights=4, variants=["V0", "V1"])
    statements = build_executive_summary(facts)
    assert len(statements) == 1
    assert "r1" in statements[0].text
    assert statements[0].evidence_path


@pytest.mark.req("FR-RPT-006")
def test_executive_summary_skips_statements_for_missing_facts() -> None:
    facts = SummaryFacts(run_id="r1", n_flights=4, variants=["V0"])
    statements = build_executive_summary(facts)
    texts = " ".join(s.text for s in statements)
    assert "ranked best" not in texts
    assert "RMSE" not in texts


@pytest.mark.req("FR-RPT-006")
def test_executive_summary_includes_statements_for_present_facts() -> None:
    facts = SummaryFacts(
        run_id="r1",
        n_flights=4,
        variants=["V0", "V1"],
        best_variant="V1",
        best_variant_mean_rank=1.2,
        predict_rmse_reduction_vs_sim_only_pct=42.5,
        predict_interval_coverage=0.9,
        detect_recall=0.95,
        detect_far_per_hour=0.5,
        detect_median_delay_s=3.2,
        identify_top_parameter="CD0",
        identify_top_parameter_change_pct=14.0,
        identify_revalidation_rmse_reduction_pct=35.0,
    )
    statements = build_executive_summary(facts)
    assert len(statements) == 8
    for statement in statements:
        assert statement.evidence_path


@pytest.mark.req("FR-RPT-006")
def test_every_statement_template_references_a_numeric_field() -> None:
    """FR-RPT-006's lint AC: every sentence must carry a computed number, not a
    hard-coded claim."""
    facts = SummaryFacts(
        run_id="r1",
        n_flights=4,
        variants=["V0", "V1"],
        best_variant="V1",
        best_variant_mean_rank=1.2,
        predict_rmse_reduction_vs_sim_only_pct=42.5,
        predict_interval_coverage=0.9,
        detect_recall=0.95,
        detect_far_per_hour=0.5,
        detect_median_delay_s=3.2,
        identify_top_parameter="CD0",
        identify_top_parameter_change_pct=14.0,
        identify_revalidation_rmse_reduction_pct=35.0,
    )
    for statement in build_executive_summary(facts):
        assert re.search(r"\d", statement.text), f"no numeric field in: {statement.text!r}"
