# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "scikit-learn==1.9.1"]
# ///
# How to run: imported by main.py.
"""Compute macro metrics, interpretable error counts, and paired fold tests."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from data import IntArray


@dataclass(frozen=True, slots=True)
class Scores:
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class PairTest:
    model_a: str
    model_b: str
    mean_difference: float
    t_statistic: float | None
    p_value: float | None
    holm_p: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class ErrorCounts:
    total_errors: int
    angle_only: int
    gesture_only: int
    both: int


def measure(truth: IntArray, predicted: IntArray) -> Scores:
    """Treat each of the eight classes equally for precision, recall, and F1."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return Scores(float(accuracy_score(truth, predicted)),
                  float(precision_score(truth, predicted, average="macro", zero_division=0)),
                  float(recall_score(truth, predicted, average="macro", zero_division=0)),
                  float(f1_score(truth, predicted, average="macro", zero_division=0)))


def categorize_errors(cm: IntArray, labels: Sequence[str]) -> ErrorCounts:
    """Read class names, rather than cell positions, to distinguish error axes."""
    angle_only = gesture_only = both = 0
    for actual, actual_name in enumerate(labels):
        angle_a, gesture_a = actual_name.split("_", 1)
        for predicted, predicted_name in enumerate(labels):
            if actual == predicted:
                continue
            angle_b, gesture_b = predicted_name.split("_", 1)
            count = int(cm[actual, predicted])
            if gesture_a == gesture_b:
                angle_only += count
            elif angle_a == angle_b:
                gesture_only += count
            else:
                both += count
    return ErrorCounts(int(cm.sum() - np.trace(cm)), angle_only, gesture_only, both)


def paired_tests(fold_accuracy: Mapping[str, Sequence[float]]) -> tuple[PairTest, ...]:
    """Pair the same five validation folds and additionally correct three p-values."""
    from scipy.stats import ttest_rel

    results: list[PairTest] = []
    for model_a, model_b in combinations(fold_accuracy, 2):
        a, b = np.array(fold_accuracy[model_a]), np.array(fold_accuracy[model_b])
        difference = a - b
        if np.std(difference, ddof=1) == 0:
            results.append(PairTest(model_a, model_b, float(difference.mean()), None,
                                    None, None, "폴드 차이의 분산이 0이라 검정하지 않음"))
            continue
        test = ttest_rel(a, b)
        finite = bool(np.isfinite(test.statistic) and np.isfinite(test.pvalue))
        results.append(PairTest(model_a, model_b, float(difference.mean()),
                                float(test.statistic) if finite else None,
                                float(test.pvalue) if finite else None, None,
                                "" if finite else "유한한 검정 통계량을 얻지 못함"))
    indices = sorted((i for i, item in enumerate(results) if item.p_value is not None),
                     key=lambda i: results[i].p_value)
    cumulative = 0.0
    for rank, index in enumerate(indices):
        p = results[index].p_value
        if p is not None:
            cumulative = min(1.0, max(cumulative, (len(indices) - rank) * p))
            results[index] = replace(results[index], holm_p=cumulative)
    return tuple(results)
