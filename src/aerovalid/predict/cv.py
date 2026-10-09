"""Cross-validation orchestration (FR-PRD-008): run any splitter over a
`WindowedDataset` and collect per-fold + overall metrics, plus a bounded nested CV for
hyper-parameter search (random search, <= 30 trials, searched only inside training folds).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aerovalid.predict.datasets import WindowedDataset, grouped_kfold
from aerovalid.predict.models import ResidualModel
from aerovalid.validate.metrics import MetricSet, compute_metrics_from_residual

_MAX_SEARCH_TRIALS = 30


@dataclass
class CVResult:
    """Per-fold and pooled-overall metrics from `cross_validate`."""

    fold_metrics: list[MetricSet]
    overall_metrics: MetricSet
    predictions: NDArray[np.float64]
    actuals: NDArray[np.float64]
    fold_id: NDArray[np.intp]


def cross_validate(
    model_factory: Callable[[], ResidualModel],
    dataset: WindowedDataset,
    splits: Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]],
) -> CVResult:
    """
    Run `model_factory()` over every `(train_idx, test_idx)` split, pooling
    out-of-fold predictions (FR-PRD-008).

    Parameters
    ----------
    model_factory : callable
        Returns a fresh, unfitted model each call (e.g. `lambda: RidgeModel()`).
    dataset : WindowedDataset
    splits : iterator of (train_idx, test_idx)
        E.g. `predict/datasets.py`'s `leave_one_flight_out`/`grouped_kfold`.

    Returns
    -------
    CVResult
    """
    fold_metrics: list[MetricSet] = []
    pred_chunks: list[NDArray[np.float64]] = []
    actual_chunks: list[NDArray[np.float64]] = []
    fold_id_chunks: list[NDArray[np.intp]] = []

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        model = model_factory()
        model.fit(
            dataset.features.iloc[train_idx],
            dataset.target[train_idx],
            dataset.flight_id[train_idx],
        )
        predicted = model.predict(dataset.features.iloc[test_idx])
        actual = dataset.target[test_idx]
        fold_metrics.append(compute_metrics_from_residual(actual, predicted, actual - predicted))
        pred_chunks.append(predicted)
        actual_chunks.append(actual)
        fold_id_chunks.append(np.full(len(test_idx), fold_id, dtype=np.intp))

    predictions = np.concatenate(pred_chunks)
    actuals = np.concatenate(actual_chunks)
    overall = compute_metrics_from_residual(actuals, predictions, actuals - predictions)

    return CVResult(
        fold_metrics=fold_metrics,
        overall_metrics=overall,
        predictions=predictions,
        actuals=actuals,
        fold_id=np.concatenate(fold_id_chunks),
    )


@dataclass
class SearchTrialResult:
    """One outer fold's nested hyper-parameter search outcome."""

    best_params: dict[str, object]
    inner_rmse: float
    outer_metrics: MetricSet


@dataclass
class NestedSearchResult:
    """Nested CV search results across all outer folds."""

    trials: list[SearchTrialResult] = field(default_factory=list)


def nested_cv_search(
    model_factory_from_params: Callable[[dict[str, object]], ResidualModel],
    param_grid: list[dict[str, object]],
    dataset: WindowedDataset,
    outer_splits: Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]],
    inner_n_splits: int = 3,
    rng: np.random.Generator | None = None,
    max_trials: int = _MAX_SEARCH_TRIALS,
) -> NestedSearchResult:
    """
    Nested CV: for each outer fold, search `param_grid` (random subset if larger than
    `max_trials`) via an *inner* grouped K-fold over the outer-training rows only, then
    refit the best params on the full outer-training split and score on the outer-test
    split (FR-PRD-008; no leakage -- the search never sees outer-test rows).

    Parameters
    ----------
    model_factory_from_params : callable
        `params -> ResidualModel`.
    param_grid : list[dict]
        Candidate hyper-parameter dicts.
    dataset : WindowedDataset
    outer_splits : iterator of (train_idx, test_idx)
    inner_n_splits : int, default 3
    rng : numpy.random.Generator, optional
    max_trials : int, default 30

    Returns
    -------
    NestedSearchResult
    """
    rng = rng if rng is not None else np.random.default_rng()
    result = NestedSearchResult()

    for outer_train_idx, outer_test_idx in outer_splits:
        if len(param_grid) > max_trials:
            chosen = rng.choice(len(param_grid), size=max_trials, replace=False)
            candidates = [param_grid[i] for i in chosen]
        else:
            candidates = param_grid

        inner_groups = dataset.flight_id[outer_train_idx]
        best_params: dict[str, object] | None = None
        best_score = np.inf
        for params in candidates:
            scores = []
            for inner_train_rel, inner_test_rel in grouped_kfold(
                inner_groups, n_splits=inner_n_splits
            ):
                inner_train_idx = outer_train_idx[inner_train_rel]
                inner_test_idx = outer_train_idx[inner_test_rel]
                model = model_factory_from_params(params)
                model.fit(dataset.features.iloc[inner_train_idx], dataset.target[inner_train_idx])
                predicted = model.predict(dataset.features.iloc[inner_test_idx])
                actual = dataset.target[inner_test_idx]
                scores.append(float(np.sqrt(np.mean((actual - predicted) ** 2))))
            mean_score = float(np.mean(scores)) if scores else np.inf
            if mean_score < best_score:
                best_score, best_params = mean_score, params

        assert best_params is not None
        final_model = model_factory_from_params(best_params)
        final_model.fit(dataset.features.iloc[outer_train_idx], dataset.target[outer_train_idx])
        predicted = final_model.predict(dataset.features.iloc[outer_test_idx])
        actual = dataset.target[outer_test_idx]
        outer_metrics = compute_metrics_from_residual(actual, predicted, actual - predicted)
        result.trials.append(
            SearchTrialResult(
                best_params=best_params, inner_rmse=best_score, outer_metrics=outer_metrics
            )
        )

    return result
