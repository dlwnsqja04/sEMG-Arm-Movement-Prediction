#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "scikit-learn==1.9.1"]
# ///
# How to run from the project root: uv run preprocessing/build_dataset.py
# Existing environment: ../Step2_1주차/.venv/bin/python preprocessing/build_dataset.py
"""Split trials first, then filter, window, extract, scale, and save."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from signal_features import cut, make_windows, preprocess, td_features
from source_data import (CLASSES, FEATURE_NAMES, FS, HOP, LENGTH, MUSCLES,
                         ROOT, START, WIN, DataProblem, FloatArray, IntArray,
                         Trial, index_source)

SEED: Final = 42


@dataclass(frozen=True, slots=True)
class Partition:
    X: FloatArray
    y: IntArray
    groups: NDArrayStr
    subjects: NDArrayStr
    hashes: tuple[str, ...]


NDArrayStr = np.ndarray[tuple[int, ...], np.dtype[np.str_]]


def split_trials(trials: Sequence[Trial]) -> tuple[list[Trial], list[Trial]]:
    """Split the independent trial IDs BEFORE generating any windows."""
    labels = [trial.class_id for trial in trials]
    train_trials, test_trials = train_test_split(
        list(trials), test_size=0.2, stratify=labels, random_state=SEED
    )
    return train_trials, test_trials


def build_partition(trials: Sequence[Trial], recordings: Mapping[str, FloatArray]) -> Partition:
    """Filter each disjoint trial separately and keep all its windows together."""
    X: list[FloatArray] = []
    y: list[int] = []
    groups: list[str] = []
    subjects: list[str] = []
    hashes: list[str] = []
    for i, trial in enumerate(trials, 1):
        x = recordings[trial.filename][trial.start:trial.stop]
        hashes.append(hashlib.sha256(cut(x).tobytes()).hexdigest())
        segment = cut(preprocess(x))
        windows = make_windows(segment)
        if segment.shape != (2000, 5) or windows.shape != (19, 200, 5):
            raise DataProblem(trial.uid, "세그먼트 또는 윈도우 모양이 다릅니다")
        for window in windows:
            X.append(td_features(window))
            y.append(trial.class_id)
            groups.append(trial.uid)
            subjects.append(trial.subject)
        if i % 400 == 0:
            print(f"특징 추출 {i}/{len(trials)} 시행", flush=True)
    return Partition(np.stack(X), np.array(y, dtype=np.int64),
                     np.array(groups), np.array(subjects), tuple(hashes))


def main() -> None:
    """Write the dataset and inspectable evidence for the six submission items."""
    evidence = ROOT / "results"
    evidence.mkdir(parents=True, exist_ok=True)
    source = index_source()

    # 1. 시행 단위 분할이 필터·윈도우·특징 생성보다 먼저다.
    train_trials, test_trials = split_trials(source.trials)
    train_ids = {t.uid for t in train_trials}
    test_ids = {t.uid for t in test_trials}
    if train_ids & test_ids:
        raise DataProblem("split", "같은 시행이 양쪽 집합에 있습니다")
    with (evidence / "trial_split.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "trial_id", "subject", "class_id", "class_name", "file", "repeat", "start", "stop"])
        for split, trials in (("train", train_trials), ("test", test_trials)):
            writer.writerows((split, t.uid, t.subject, t.class_id, CLASSES[t.class_id],
                              t.filename, t.repeat, t.start, t.stop) for t in trials)

    # 2. 분할 이후 각 집합 안에서만 필터·윈도우·특징을 만든다.
    train = build_partition(train_trials, source.recordings)
    test = build_partition(test_trials, source.recordings)
    if set(train.hashes) & set(test.hashes):
        raise DataProblem("split", "동일한 원본 2초 신호가 양쪽 집합에 있습니다")
    if not np.isfinite(train.X).all() or not np.isfinite(test.X).all():
        raise DataProblem("features", "NaN 또는 무한값이 있습니다")

    # 3. 학습셋에서만 표준화 통계를 계산한다.
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(train.X)
    Xte_s = sc.transform(test.X)
    np.savez_compressed(
        ROOT / "dataset.npz", Xtr=Xtr_s, ytr=train.y, Xte=Xte_s, yte=test.y,
        Xtr_raw=train.X, Xte_raw=test.X, groups_tr=train.groups, groups_te=test.groups,
        subjects_tr=train.subjects, subjects_te=test.subjects,
        feature_names=np.array(FEATURE_NAMES), class_names=np.array(CLASSES),
        scaler_mean=sc.mean_, scaler_scale=sc.scale_, scaler_var=sc.var_,
        fs=FS, segment_start=START, segment_length=LENGTH, win=WIN, hop=HOP, seed=SEED,
    )
    with (evidence / "feature_columns.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "name", "channel", "muscle"])
        writer.writerows((i, name, i % 5 + 1, MUSCLES[i % 5]) for i, name in enumerate(FEATURE_NAMES))

    example = next(t for t in source.trials if t.filename == "a_30 fist.csv" and t.repeat == 1)
    x = source.recordings[example.filename][example.start:example.stop]
    xf = preprocess(x)
    segment = cut(xf)
    windows = make_windows(segment)
    feature = td_features(windows[0])
    np.savez_compressed(evidence / "example_arrays.npz", x=x, xf=xf, segment=segment,
                        windows=windows, feature=feature, zc_zero=td_features(windows[0], 0)[15:20])
    variable = sc.var_ > 0
    train_mean, train_std = Xtr_s.mean(0), Xtr_s.std(0)
    with np.load(ROOT / "dataset.npz", allow_pickle=False) as saved:
        for name, expected in (("Xtr", Xtr_s), ("Xte", Xte_s), ("ytr", train.y), ("yte", test.y)):
            if not np.array_equal(saved[name], expected):
                raise DataProblem("dataset.npz", f"{name} 저장 후 읽기 결과가 다릅니다")
    summary = {
        "versions": {"Python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "files": len(source.recordings), "trials": len(source.trials),
        "trial_length_min": min(t.stop - t.start for t in source.trials),
        "trial_length_max": max(t.stop - t.start for t in source.trials),
        "train_trials": len(train_trials), "test_trials": len(test_trials),
        "train_shape": list(train.X.shape), "test_shape": list(test.X.shape),
        "train_class_trials": np.bincount([t.class_id for t in train_trials], minlength=8).tolist(),
        "test_class_trials": np.bincount([t.class_id for t in test_trials], minlength=8).tolist(),
        "train_class_windows": np.bincount(train.y, minlength=8).tolist(),
        "test_class_windows": np.bincount(test.y, minlength=8).tolist(),
        "std_before": x.std(0).tolist(), "std_after": xf.std(0).tolist(),
        "example_shapes": {"trial": list(x.shape), "segment": list(segment.shape),
                           "windows": list(windows.shape), "feature": list(feature.shape)},
        "example_feature": feature.tolist(), "zc_zero": td_features(windows[0], 0)[15:20].tolist(),
        "train_mean": train_mean.tolist(), "train_std": train_std.tolist(),
        "train_mean_max_abs": float(np.max(np.abs(train_mean))),
        "train_std_variable_min": float(train_std[variable].min()),
        "train_std_variable_max": float(train_std[variable].max()),
        "constant_feature_names": [FEATURE_NAMES[i] for i in np.flatnonzero(~variable)],
        "test_mean_range": [float(Xte_s.mean(0).min()), float(Xte_s.mean(0).max())],
        "test_std_range": [float(Xte_s.std(0).min()), float(Xte_s.std(0).max())],
        "shared_trial_ids": len(train_ids & test_ids),
        "shared_segment_hashes": len(set(train.hashes) & set(test.hashes)),
        "finite": bool(np.isfinite(Xtr_s).all() and np.isfinite(Xte_s).all()),
        "dataset_bytes": (ROOT / "dataset.npz").stat().st_size,
        "file_hashes": dict(source.file_hashes),
    }
    (evidence / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"학습 시행 {len(train_trials)}, 테스트 시행 {len(test_trials)}", flush=True)
    print(f"학습 특징 {train.X.shape}, 테스트 특징 {test.X.shape}", flush=True)
    print(f"학습 평균 최대 절댓값 {summary['train_mean_max_abs']:.3e}", flush=True)
    print(f"학습 표준편차 {train_std.min():.12f}~{train_std.max():.12f}", flush=True)
    print("dataset.npz 저장 및 다시 읽기 확인 완료", flush=True)


if __name__ == "__main__":
    main()
