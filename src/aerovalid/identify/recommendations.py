"""Recommendation engine (FR-IDN-005, analysis methods §11): rule-based synthesis of
condition analysis, sensitivity, and identification results into ranked,
evidence-linked recommendations. Never emits a claim without >= 2 independent evidence
types; anything with fewer is labelled a `hypothesis`, never asserted as fact
(CLAUDE.md §2's "don't invent results").

**Scope note:** the spec's step 2 (`cosine > 0.6` alignment between a parameter's
Jacobian column and a signal's residual signature) is simplified here to "the
parameter was included in the fit that targeted this signal" -- a coarser but honest
proxy for the same idea, within this CLI's reduced scope (see DECISIONS.md ADR-019).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aerovalid.identify.fit import FitResult
from aerovalid.identify.jacobian import IdentifiabilityResult

_MIN_EVIDENCE_TYPES = 2


@dataclass(frozen=True)
class Recommendation:
    target_variant: str
    parameter: str
    regime: str | None
    estimated_change_pct: float
    estimated_change_ci_pct: tuple[float, float]
    expected_improvement_pct: float | None
    evidence: list[str] = field(default_factory=list)
    caveats: str = ""
    is_hypothesis: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "target_variant": self.target_variant,
            "parameter": self.parameter,
            "regime": self.regime,
            "estimated_change_pct": self.estimated_change_pct,
            "estimated_change_ci_pct": list(self.estimated_change_ci_pct),
            "expected_improvement_pct": self.expected_improvement_pct,
            "evidence": self.evidence,
            "caveats": self.caveats,
            "is_hypothesis": self.is_hypothesis,
        }


def build_recommendations(
    target_variant: str,
    nominal_values: dict[str, float],
    fit_result: FitResult,
    identifiability_result: IdentifiabilityResult,
    sensitive_params: set[str],
    non_compliant_signals: set[str],
    signal_to_fit_params: dict[str, set[str]],
    expected_improvement_pct: dict[str, float] | None = None,
) -> list[Recommendation]:
    """
    Combine fit + identifiability + sensitivity + condition evidence into
    recommendations (FR-IDN-005).

    Parameters
    ----------
    target_variant : str
    nominal_values : dict[str, float]
        Each fitted parameter's pre-fit (nominal) value, for the `estimated_change_pct`.
    fit_result : FitResult
    identifiability_result : IdentifiabilityResult
        Must come from the same fit (`ci_95` keyed by the same parameter names).
    sensitive_params : set[str]
        Parameter names ranked highly influential by `identify/sensitivity.py`.
    non_compliant_signals : set[str]
        Signals below their tolerance-compliance threshold (FR-VAL-004), i.e. where a
        correction is actually needed.
    signal_to_fit_params : dict[str, set[str]]
        Which fitted parameters were targeted at which signal (this ADR's simplified
        stand-in for a true Jacobian/residual cosine-alignment check).
    expected_improvement_pct : dict[str, float], optional
        Per-parameter expected held-out RMSE improvement (from `identify/
        revalidate.py`), if already computed.

    Returns
    -------
    list[Recommendation]
        Sorted by `expected_improvement_pct` (highest first; `None` last).
    """
    expected_improvement_pct = expected_improvement_pct or {}
    recommendations = []

    for param in fit_result.param_names:
        fitted = fit_result.params[param]
        nominal = nominal_values.get(param, fitted)
        if nominal == 0:
            continue
        change_pct = 100.0 * (fitted - nominal) / nominal
        ci_lower, ci_upper = identifiability_result.ci_95[param]
        ci_pct = (100.0 * (ci_lower - nominal) / nominal, 100.0 * (ci_upper - nominal) / nominal)
        ci_excludes_nominal = ci_lower > nominal or ci_upper < nominal

        targeted_signals = {
            signal
            for signal, params in signal_to_fit_params.items()
            if param in params and signal in non_compliant_signals
        }

        evidence = []
        if ci_excludes_nominal:
            evidence.append("fit")
        if param in sensitive_params:
            evidence.append("sensitivity")
        if targeted_signals:
            evidence.append("condition")
        if param in identifiability_result.poorly_identifiable:
            evidence.append("poorly_identifiable")  # a caveat flag, not independent support

        support_evidence = [e for e in evidence if e != "poorly_identifiable"]
        is_hypothesis = len(support_evidence) < _MIN_EVIDENCE_TYPES
        caveats = (
            "Not identifiable independently of other fitted parameters (high "
            "condition number or correlation); treat the point estimate with caution."
            if param in identifiability_result.poorly_identifiable
            else ""
        )

        recommendations.append(
            Recommendation(
                target_variant=target_variant,
                parameter=param,
                regime=None,
                estimated_change_pct=change_pct,
                estimated_change_ci_pct=ci_pct,
                expected_improvement_pct=expected_improvement_pct.get(param),
                evidence=evidence,
                caveats=caveats,
                is_hypothesis=is_hypothesis,
            )
        )

    def _sort_key(r: Recommendation) -> float:
        return r.expected_improvement_pct if r.expected_improvement_pct is not None else -1.0

    recommendations.sort(key=_sort_key, reverse=True)
    return recommendations
