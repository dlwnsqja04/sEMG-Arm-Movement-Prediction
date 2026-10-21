# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
# How to run: imported by documentation.py and figures.py.
"""Read numeric evidence and derive confusion-matrix descriptions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from data import ROOT, IntArray
from evaluation import ErrorCounts, PairTest, Scores


@dataclass(frozen=True, slots=True)
class ModelResult:
    name: str
    scores: Scores
    train_seconds: float
    inference_ms: float
    batch_seconds: float
    cm: IntArray
    errors: ErrorCounts


@dataclass(frozen=True, slots=True)
class CVResult:
    name: str
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float


@dataclass(frozen=True, slots=True)
class FoldResult:
    name: str
    number: int
    accuracy: float
    f1: float
    train_trials: int
    validation_trials: int


@dataclass(frozen=True, slots=True)
class Results:
    models: tuple[ModelResult, ...]
    cv: tuple[CVResult, ...]
    folds: tuple[FoldResult, ...]
    tests: tuple[PairTest, ...]
    selected: str
    classes: tuple[str, ...]
    versions: Mapping[str, str]
    system: str
    architecture: str
    threads: int


@dataclass(frozen=True, slots=True)
class ConfusionAnalysis:
    best: tuple[str, ...]
    worst: tuple[str, ...]
    best_recall: float
    worst_recall: float
    actual: str
    predicted: str
    count: int


def read_results() -> Results:
    """Use the saved experiment, rather than manually transcribing metrics."""
    source = json.loads((ROOT / "results/experiment.json").read_text(encoding="utf-8"))
    models = tuple(ModelResult(row["model"], Scores(row["accuracy"], row["precision"], row["recall"], row["f1"]),
                               row["train_seconds"], row["inference_ms_per_window"], row["test_batch_seconds"],
                               np.array(row["confusion_matrix"], dtype=np.int64), ErrorCounts(**row["error_counts"]))
                   for row in source["holdout"])
    cv = tuple(CVResult(row["model"], row["accuracy_mean"], row["accuracy_std"], row["f1_mean"], row["f1_std"])
               for row in source["cv_summary"])
    folds = tuple(FoldResult(row["model"], row["fold"], row["accuracy"], row["f1"], row["train_trials"], row["validation_trials"])
                  for row in source["cv_folds"])
    return Results(models, cv, folds, tuple(PairTest(**row) for row in source["paired_tests"]),
                   source["selected_by_cv_f1"], tuple(source["class_names"]), source["versions"],
                   source["environment"]["os"], source["environment"]["architecture"], source["environment"]["threads"])


def analyze_matrix(model: ModelResult, classes: tuple[str, ...]) -> ConfusionAnalysis:
    """Define best/worst by class recall and major error by off-diagonal count."""
    cm = model.cm
    recall = np.diag(cm) / cm.sum(axis=1)
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    actual, predicted = np.unravel_index(int(off_diagonal.argmax()), cm.shape)
    return ConfusionAnalysis(tuple(classes[i] for i in np.flatnonzero(recall == recall.max())),
                             tuple(classes[i] for i in np.flatnonzero(recall == recall.min())),
                             float(recall.max()), float(recall.min()), classes[actual], classes[predicted],
                             int(off_diagonal[actual, predicted]))


def error_hypothesis(model: ModelResult) -> str:
    """Keep a possible mechanism separate from the observed error counts."""
    axes = (model.errors.angle_only, model.errors.gesture_only, model.errors.both)
    dominant = int(np.argmax(axes))
    reasons = (
        "같은 손동작의 각도만 바뀐 오류가 가장 많았다. 두 각도에서 비슷한 근활성 통계량이 나온 창을 구분하기 어려웠을 가능성이 있다.",
        "각도는 맞지만 손동작이 다른 오류가 가장 많았다. 한 창의 통계량만 사용하면 동작의 시간 순서 정보가 줄어 서로 다른 동작을 구분하기 어려울 수 있다.",
        "각도와 손동작을 모두 틀린 오류가 가장 많았다. 낮은 활성 구간이나 피험자별 진폭 차이 때문에 25차원 통계량만으로 조합을 구분하기 어려웠을 가능성이 있다.",
    )
    return reasons[dominant] + " 이는 오류 분포에 근거한 가설이며 원인을 분리한 실험으로 확인한 것은 아니다."


def p_text(value: float | None) -> str:
    return "검정 불가" if value is None else f"{value:.6g}"


def test_interpretation(test: PairTest) -> str:
    if test.p_value is None:
        return test.reason
    if test.holm_p is not None and test.holm_p < .05:
        return "원 p값과 Holm 보정 p값 모두 0.05 미만으로, 이번 다섯 폴드에서 평균 정확도 차이가 검출되었다."
    if test.p_value < .05:
        return "원 p값은 0.05 미만이지만 Holm 보정 후에는 0.05 이상이어서 보정 기준으로 차이가 검출되지 않았다."
    return "p값이 0.05 이상이므로 차이가 검출되지 않았다. 두 모델의 성능이 동등하다는 뜻은 아니다."
