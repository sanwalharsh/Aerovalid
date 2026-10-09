from __future__ import annotations

import numpy as np
import pytest

from aerovalid.identify.fit import FitResult
from aerovalid.identify.jacobian import IdentifiabilityResult
from aerovalid.identify.recommendations import build_recommendations


def _fit_result(param_names: list[str], values: list[float]) -> FitResult:
    return FitResult(
        param_names=param_names,
        fitted_values=np.array(values),
        residuals=np.zeros(10),
        cost=0.1,
        success=True,
    )


def _identifiability(
    param_names: list[str], ci_95: dict[str, tuple[float, float]], poorly_identifiable: list[str]
) -> IdentifiabilityResult:
    n = len(param_names)
    return IdentifiabilityResult(
        param_names=param_names,
        jacobian=np.eye(n),
        covariance=np.eye(n),
        std_errors=np.ones(n),
        ci_95=ci_95,
        correlation=np.eye(n),
        condition_number=1.0,
        poorly_identifiable=poorly_identifiable,
    )


@pytest.mark.req("FR-IDN-005")
def test_recommendation_with_two_evidence_types_is_not_a_hypothesis() -> None:
    fit_result = _fit_result(["CD0"], [0.032])
    identifiability_result = _identifiability(["CD0"], {"CD0": (0.030, 0.034)}, [])

    recommendations = build_recommendations(
        target_variant="V0",
        nominal_values={"CD0": 0.028},
        fit_result=fit_result,
        identifiability_result=identifiability_result,
        sensitive_params={"CD0"},  # evidence type 2: sensitivity
        non_compliant_signals={"alt_m"},
        signal_to_fit_params={"alt_m": {"CD0"}},  # evidence type 3: condition
    )
    assert len(recommendations) == 1
    rec = recommendations[0]
    assert not rec.is_hypothesis
    assert set(rec.evidence) >= {"fit", "sensitivity", "condition"}
    assert rec.estimated_change_pct == pytest.approx(100.0 * (0.032 - 0.028) / 0.028)


@pytest.mark.req("FR-IDN-005")
def test_recommendation_with_fewer_than_two_evidence_types_is_a_hypothesis() -> None:
    fit_result = _fit_result(["CD0"], [0.032])
    identifiability_result = _identifiability(["CD0"], {"CD0": (0.030, 0.034)}, [])

    recommendations = build_recommendations(
        target_variant="V0",
        nominal_values={"CD0": 0.028},
        fit_result=fit_result,
        identifiability_result=identifiability_result,
        sensitive_params=set(),  # no sensitivity evidence
        non_compliant_signals=set(),  # no condition evidence
        signal_to_fit_params={},
    )
    assert recommendations[0].is_hypothesis
    assert recommendations[0].evidence == ["fit"]


@pytest.mark.req("FR-IDN-005")
def test_recommendation_ci_including_nominal_has_no_fit_evidence() -> None:
    fit_result = _fit_result(["CD0"], [0.0285])
    # CI spans the nominal value, so there's no significant fit-based evidence.
    identifiability_result = _identifiability(["CD0"], {"CD0": (0.020, 0.037)}, [])

    recommendations = build_recommendations(
        target_variant="V0",
        nominal_values={"CD0": 0.028},
        fit_result=fit_result,
        identifiability_result=identifiability_result,
        sensitive_params={"CD0"},
        non_compliant_signals={"alt_m"},
        signal_to_fit_params={"alt_m": {"CD0"}},
    )
    assert "fit" not in recommendations[0].evidence
    # Still has "sensitivity" + "condition" -- two independent evidence types remain.
    assert not recommendations[0].is_hypothesis


@pytest.mark.req("FR-IDN-005")
def test_poorly_identifiable_parameter_gets_a_caveat_but_can_still_have_support() -> None:
    fit_result = _fit_result(["CD0", "k"], [0.032, 0.05])
    identifiability_result = _identifiability(
        ["CD0", "k"],
        {"CD0": (0.030, 0.034), "k": (0.030, 0.034)},
        poorly_identifiable=["CD0", "k"],
    )

    recommendations = build_recommendations(
        target_variant="V0",
        nominal_values={"CD0": 0.028, "k": 0.047},
        fit_result=fit_result,
        identifiability_result=identifiability_result,
        sensitive_params={"CD0", "k"},
        non_compliant_signals={"alt_m"},
        signal_to_fit_params={"alt_m": {"CD0", "k"}},
    )
    for rec in recommendations:
        assert rec.caveats != ""
        assert not rec.is_hypothesis  # still has fit + sensitivity + condition support


@pytest.mark.req("FR-IDN-005")
def test_recommendations_sorted_by_expected_improvement_descending() -> None:
    fit_result = _fit_result(["CD0", "k"], [0.032, 0.05])
    identifiability_result = _identifiability(
        ["CD0", "k"], {"CD0": (0.030, 0.034), "k": (0.049, 0.051)}, []
    )

    recommendations = build_recommendations(
        target_variant="V0",
        nominal_values={"CD0": 0.028, "k": 0.047},
        fit_result=fit_result,
        identifiability_result=identifiability_result,
        sensitive_params={"CD0", "k"},
        non_compliant_signals={"alt_m"},
        signal_to_fit_params={"alt_m": {"CD0", "k"}},
        expected_improvement_pct={"CD0": 5.0, "k": 20.0},
    )
    assert [r.parameter for r in recommendations] == ["k", "CD0"]
