from __future__ import annotations

import numpy as np
import pytest

from aerovalid.detect.evaluation import FlightRecord, evaluate_detections, window_level_auc


@pytest.mark.req("FR-DET-006")
def test_evaluate_detections_recall_and_delay_on_detected_fault() -> None:
    records = [
        FlightRecord(
            flight_id="F0001",
            is_fault=True,
            duration_s=300.0,
            fault_type="cd0_step",
            t_on_s=100.0,
            t_exc_s=110.0,
            t_alarm_s=103.0,
        ),
    ]
    metrics = evaluate_detections(records, max_delay_s=60.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.median_delay_s == pytest.approx(3.0)
    assert metrics.median_lead_time_s == pytest.approx(7.0)
    assert metrics.n_detected == 1
    assert metrics.n_faulty == 1


@pytest.mark.req("FR-DET-006")
def test_evaluate_detections_missed_fault_has_zero_recall() -> None:
    records = [
        FlightRecord(
            flight_id="F0001", is_fault=True, duration_s=300.0, t_on_s=100.0, t_alarm_s=None
        ),
    ]
    metrics = evaluate_detections(records)
    assert metrics.recall == pytest.approx(0.0)
    assert metrics.median_delay_s is None


@pytest.mark.req("FR-DET-006")
def test_evaluate_detections_false_alarm_on_nominal_flight_counts_toward_far() -> None:
    records = [
        FlightRecord(flight_id="F0001", is_fault=False, duration_s=3600.0, t_alarm_s=50.0),
        FlightRecord(flight_id="F0002", is_fault=False, duration_s=3600.0, t_alarm_s=None),
    ]
    metrics = evaluate_detections(records)
    assert metrics.far_per_hour == pytest.approx(1.0 / 2.0)  # 1 false alarm over 2 flight-hours
    assert metrics.n_false_alarms == 1


@pytest.mark.req("FR-DET-006")
def test_evaluate_detections_per_fault_type_breakdown() -> None:
    records = [
        FlightRecord(
            flight_id="F0001",
            is_fault=True,
            duration_s=300.0,
            fault_type="icing",
            t_on_s=50.0,
            t_alarm_s=55.0,
        ),
        FlightRecord(
            flight_id="F0002",
            is_fault=True,
            duration_s=300.0,
            fault_type="icing",
            t_on_s=50.0,
            t_alarm_s=None,
        ),
    ]
    metrics = evaluate_detections(records)
    assert metrics.per_fault_type["icing"]["recall"] == pytest.approx(0.5)
    assert metrics.per_fault_type["icing"]["n_flights"] == pytest.approx(2.0)


@pytest.mark.req("FR-DET-006")
def test_window_level_auc_separates_perfectly_separable_scores() -> None:
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([False, False, True, True])
    roc_auc, pr_auc = window_level_auc(scores, labels)
    assert roc_auc == pytest.approx(1.0)
    assert pr_auc == pytest.approx(1.0)


@pytest.mark.req("FR-DET-006")
def test_window_level_auc_returns_none_for_single_class_labels() -> None:
    roc_auc, pr_auc = window_level_auc(np.array([0.1, 0.5]), np.array([False, False]))
    assert roc_auc is None
    assert pr_auc is None
