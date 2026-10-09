"""Detector evaluation protocol (FR-DET-006, analysis methods §8.3): detection delay,
lead time, recall, precision, false-alarm rate per flight-hour, window-level ROC/PR
AUC, and a per-fault-type breakdown, evaluated per flight (one alarm-or-not outcome
each) against each flight's ground-truth fault event.

**Scope note:** a flight's alarm is "detected" only if its first alarm falls in
`[t_on, t_on + max_delay_s]`; an alarm before `t_on` on a faulty flight is treated the
same as a missed detection for recall (not as a separate "predictive early warning"
credit) -- `detect/early_warning.py`'s own lead-time framing is evaluated the same way,
via its `t_alarm_s` possibly preceding `t_on_s` while still landing in-window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class FlightRecord:
    """One flight's ground truth and detector outcome, ready for evaluation."""

    flight_id: str
    is_fault: bool
    duration_s: float
    fault_type: str | None = None
    t_on_s: float | None = None
    t_exc_s: float | None = None
    t_alarm_s: float | None = None


@dataclass
class DetectionMetrics:
    median_delay_s: float | None
    median_lead_time_s: float | None
    recall: float
    precision: float
    far_per_hour: float
    n_detected: int
    n_faulty: int
    n_false_alarms: int
    per_fault_type: dict[str, dict[str, float]] = field(default_factory=dict)


def _is_detected(record: FlightRecord, max_delay_s: float) -> bool:
    if record.t_alarm_s is None or record.t_on_s is None:
        return False
    return record.t_on_s <= record.t_alarm_s <= record.t_on_s + max_delay_s


def evaluate_detections(records: list[FlightRecord], max_delay_s: float = 60.0) -> DetectionMetrics:
    """
    Evaluate per-flight detector outcomes against ground-truth faults (FR-DET-006).

    Parameters
    ----------
    records : list[FlightRecord]
        One record per evaluated flight (faulty and nominal).
    max_delay_s : float, default 60.0
        A faulty flight's first alarm counts as "detected" only within
        `[t_on_s, t_on_s + max_delay_s]`.

    Returns
    -------
    DetectionMetrics
    """
    faulty = [r for r in records if r.is_fault]
    nominal = [r for r in records if not r.is_fault]

    delays: list[float] = []
    lead_times: list[float] = []
    n_detected = 0
    for record in faulty:
        if _is_detected(record, max_delay_s):
            n_detected += 1
            assert record.t_alarm_s is not None and record.t_on_s is not None
            delays.append(record.t_alarm_s - record.t_on_s)
            if record.t_exc_s is not None:
                lead_times.append(record.t_exc_s - record.t_alarm_s)

    n_false_alarms = sum(1 for r in nominal if r.t_alarm_s is not None)
    n_false_alarms += sum(
        1 for r in faulty if r.t_alarm_s is not None and not _is_detected(r, max_delay_s)
    )

    recall = n_detected / len(faulty) if faulty else float("nan")
    n_alarmed = n_detected + n_false_alarms
    precision = n_detected / n_alarmed if n_alarmed > 0 else float("nan")
    nominal_hours = sum(r.duration_s for r in nominal) / 3600.0
    far_per_hour = (
        sum(1 for r in nominal if r.t_alarm_s is not None) / nominal_hours
        if nominal_hours > 0
        else float("nan")
    )

    per_fault_type: dict[str, dict[str, float]] = {}
    fault_types = {r.fault_type for r in faulty if r.fault_type is not None}
    for fault_type in fault_types:
        subset = [r for r in faulty if r.fault_type == fault_type]
        detected_subset = sum(1 for r in subset if _is_detected(r, max_delay_s))
        per_fault_type[fault_type] = {
            "recall": detected_subset / len(subset) if subset else float("nan"),
            "n_flights": float(len(subset)),
        }

    return DetectionMetrics(
        median_delay_s=float(np.median(delays)) if delays else None,
        median_lead_time_s=float(np.median(lead_times)) if lead_times else None,
        recall=recall,
        precision=precision,
        far_per_hour=far_per_hour,
        n_detected=n_detected,
        n_faulty=len(faulty),
        n_false_alarms=n_false_alarms,
        per_fault_type=per_fault_type,
    )


def window_level_auc(
    scores: NDArray[np.float64], labels: NDArray[np.bool_]
) -> tuple[float | None, float | None]:
    """
    Window-level ROC-AUC / PR-AUC of a continuous detector score against binary
    "inside a fault window" labels (FR-DET-006).

    Returns
    -------
    tuple[float, float] or tuple[None, None]
        `(roc_auc, pr_auc)`; `(None, None)` if `labels` has only one class.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(labels)) < 2:
        return None, None
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )
