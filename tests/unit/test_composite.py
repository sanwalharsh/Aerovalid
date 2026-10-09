from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerovalid.validate.composite import composite_score, rank_agreement, rank_variants


def _metrics_by_flight_variant() -> pd.DataFrame:
    # Two flights, two signals, three variants. V0 is the reference; V_better halves
    # every RMSE relative to V0, V_worse doubles it -> composite scores should rank
    # V_better < V0 < V_worse regardless of weights (as long as weights are positive).
    rows = []
    for flight_id in ("F0001", "F0002"):
        for signal, base_rmse in (("alt_m", 2.0), ("vel_air_mps", 1.0)):
            rows.append(
                {"flight_id": flight_id, "variant": "V0", "signal": signal, "rmse": base_rmse}
            )
            rows.append(
                {
                    "flight_id": flight_id,
                    "variant": "V_better",
                    "signal": signal,
                    "rmse": base_rmse * 0.5,
                }
            )
            rows.append(
                {
                    "flight_id": flight_id,
                    "variant": "V_worse",
                    "signal": signal,
                    "rmse": base_rmse * 2.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.req("FR-VAL-006")
def test_composite_score_reference_variant_is_always_one() -> None:
    scores = composite_score(_metrics_by_flight_variant(), ref_variant="V0")
    ref_scores = scores[scores["variant"] == "V0"]["composite_score"].to_numpy()
    weight_sum = sum({"alt_m": 0.2, "vel_air_mps": 0.2}.values())
    assert ref_scores == pytest.approx(weight_sum)


@pytest.mark.req("FR-VAL-006")
def test_composite_score_better_variant_scores_lower() -> None:
    scores = composite_score(_metrics_by_flight_variant(), ref_variant="V0")
    better = scores[scores["variant"] == "V_better"]["composite_score"].mean()
    worse = scores[scores["variant"] == "V_worse"]["composite_score"].mean()
    ref = scores[scores["variant"] == "V0"]["composite_score"].mean()
    assert better < ref < worse


@pytest.mark.req("FR-VAL-006")
def test_rank_variants_recovers_known_ordering() -> None:
    scores = composite_score(_metrics_by_flight_variant(), ref_variant="V0")
    ranking = rank_variants(scores, n_bootstrap=200, rng=np.random.default_rng(0))
    assert list(ranking["variant"]) == ["V_better", "V0", "V_worse"]
    assert ranking.iloc[0]["p_best"] > 0.9


@pytest.mark.req("FR-VAL-006")
def test_rank_agreement_perfect_match() -> None:
    ground_truth = ["V5", "V4", "V0", "V2", "V1", "V3"]
    rho, tau = rank_agreement(ground_truth, ground_truth)
    assert rho == pytest.approx(1.0)
    assert tau == pytest.approx(1.0)


@pytest.mark.req("FR-VAL-006")
def test_rank_agreement_reversed_order_is_negative() -> None:
    ground_truth = ["V5", "V4", "V0", "V2", "V1", "V3"]
    reversed_order = list(reversed(ground_truth))
    rho, tau = rank_agreement(reversed_order, ground_truth)
    assert rho == pytest.approx(-1.0)
    assert tau == pytest.approx(-1.0)
