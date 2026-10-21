# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scikit-learn==1.9.1"]
# ///
# How to run: imported by main.py; python main.py runs the complete experiment.
"""Validate the saved raw features and split CV folds by independent trials."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StrArray = NDArray[np.str_]
ROOT: Final = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class DatasetProblem(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class Dataset:
    Xtr: FloatArray
    ytr: IntArray
    Xte: FloatArray
    yte: IntArray
    groups_tr: StrArray
    groups_te: StrArray
    subjects_tr: StrArray
    subjects_te: StrArray
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Fold:
    number: int
    train_indices: IntArray
    val_indices: IntArray
    val_trials: tuple[str, ...]


def load_dataset(path: Path) -> Dataset:
    """Read unstandardized columns so every training fold fits its own scaler."""
    if not path.is_file():
        raise DatasetProblem(f"데이터셋이 없습니다: {path}. data/dataset.npz를 확인하세요")
    required = ("Xtr_raw", "Xte_raw", "ytr", "yte", "groups_tr", "groups_te",
                "subjects_tr", "subjects_te", "class_names", "feature_names")
    with np.load(path, allow_pickle=False) as stored:
        if not set(required).issubset(stored.files):
            raise DatasetProblem("표준화 전 특징과 시행 ID가 필요합니다. 포함된 dataset.npz를 사용하세요")
        data = Dataset(stored["Xtr_raw"], stored["ytr"], stored["Xte_raw"], stored["yte"],
                       stored["groups_tr"], stored["groups_te"], stored["subjects_tr"],
                       stored["subjects_te"], tuple(stored["class_names"].tolist()),
                       tuple(stored["feature_names"].tolist()))
    if len(data.class_names) != 8 or len(data.feature_names) != 25:
        raise DatasetProblem("8클래스·25개 특징 데이터셋이 아닙니다")
    if set(data.groups_tr) & set(data.groups_te):
        raise DatasetProblem("같은 시행이 학습셋과 테스트셋에 섞였습니다")
    for X, y, groups, subjects in ((data.Xtr, data.ytr, data.groups_tr, data.subjects_tr),
                                  (data.Xte, data.yte, data.groups_te, data.subjects_te)):
        if X.ndim != 2 or X.shape[1] != 25 or not np.isfinite(X).all():
            raise DatasetProblem("특징 모양 또는 NaN·무한값을 확인해야 합니다")
        if not (len(X) == len(y) == len(groups) == len(subjects)):
            raise DatasetProblem("특징·라벨·시행 ID 길이가 다릅니다")
        if not np.issubdtype(y.dtype, np.integer) or not np.array_equal(np.unique(y), np.arange(8)):
            raise DatasetProblem("라벨은 0~7 정수여야 합니다")
        unique, counts = np.unique(groups, return_counts=True)
        if not np.all(counts == 19):
            raise DatasetProblem("시행마다 19개 창이 있어야 합니다")
        for trial in unique:
            if len(np.unique(y[groups == trial])) != 1:
                raise DatasetProblem(f"시행 {trial}에 서로 다른 라벨이 섞였습니다")
    return data


def make_folds(data: Dataset) -> tuple[Fold, ...]:
    """Stratify 1,536 training trials; never split their 19 windows apart."""
    from sklearn.model_selection import StratifiedKFold

    trials, first = np.unique(data.groups_tr, return_index=True)
    labels = data.ytr[first]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds: list[Fold] = []
    coverage = np.zeros(len(data.Xtr), dtype=np.int64)
    for number, (train, val) in enumerate(splitter.split(trials, labels), 1):
        train_idx = np.flatnonzero(np.isin(data.groups_tr, trials[train]))
        val_idx = np.flatnonzero(np.isin(data.groups_tr, trials[val]))
        if set(data.groups_tr[train_idx]) & set(data.groups_tr[val_idx]):
            raise DatasetProblem("교차검증에서 시행 ID가 겹칩니다")
        coverage[val_idx] += 1
        folds.append(Fold(number, train_idx, val_idx, tuple(trials[val].tolist())))
    if not np.all(coverage == 1):
        raise DatasetProblem("학습 창은 검증 폴드에 정확히 한 번씩 포함되어야 합니다")
    return tuple(folds)
