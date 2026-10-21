# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "scikit-learn==1.9.1", "xgboost==3.4.1"]
# ///
# How to run: imported by main.py.
"""Run the shared five folds, then evaluate the fixed hold-out once per model."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

from data import ROOT, Dataset, IntArray, make_folds
from evaluation import measure, categorize_errors, paired_tests
from model import JOBS, make_models


def run_experiment(data: Dataset) -> None:
    """Persist predictions and fold assignments so every reported number is auditable."""
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    models, folds = make_models(), make_folds(data)
    fold_rows = []
    oof = {spec.name: np.full(len(data.ytr), -1, dtype=np.int64) for spec in models}
    fold_ids = np.zeros(len(data.ytr), dtype=np.int64)
    with (output / "cv_trial_folds.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial_id", "validation_fold", "class_id"])
        for fold in folds:
            for trial in fold.val_trials:
                writer.writerow([trial, fold.number, int(data.ytr[np.flatnonzero(data.groups_tr == trial)[0]])])
    for fold in folds:
        train, val = fold.train_indices, fold.val_indices
        fold_ids[val] = fold.number
        for spec in models:
            print(f"CV {fold.number}/5 {spec.name}: 학습 시작", flush=True)
            pipeline = clone(spec.pipeline)
            start = time.perf_counter()
            with threadpool_limits(limits=JOBS):
                pipeline.fit(data.Xtr[train], data.ytr[train])
                elapsed = time.perf_counter() - start
                predicted = pipeline.predict(data.Xtr[val]).astype(np.int64)
            oof[spec.name][val] = predicted
            metrics = measure(data.ytr[val], predicted)
            fold_rows.append({"model": spec.name, "fold": fold.number,
                              **asdict(metrics), "train_seconds": elapsed,
                              "scaler_fit_samples": int(pipeline.named_steps["scaler"].n_samples_seen_),
                              "scaler_mean_max_error": float(np.max(np.abs(pipeline.named_steps["scaler"].mean_ - data.Xtr[train].mean(0)))),
                              "train_trials": len(np.unique(data.groups_tr[train])),
                              "validation_trials": len(fold.val_trials),
                              "train_windows": len(train), "validation_windows": len(val)})
            (output / "cv_folds.json").write_text(json.dumps(fold_rows, indent=2), encoding="utf-8")
            print(f"CV {fold.number}/5 {spec.name}: accuracy={metrics.accuracy:.4f}, F1={metrics.f1:.4f}, {elapsed:.2f}s", flush=True)
    with (output / "cv_folds.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    with (output / "cv_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "trial_id", "fold", "y_true", *(spec.name for spec in models)])
        writer.writerows((i, data.groups_tr[i], int(fold_ids[i]), int(data.ytr[i]),
                          *(int(oof[spec.name][i]) for spec in models)) for i in range(len(data.ytr)))
    cv_summary = []
    for spec in models:
        rows = [row for row in fold_rows if row["model"] == spec.name]
        cv_summary.append({"model": spec.name, **{
            f"{metric}_{statistic}": float(function([row[metric] for row in rows]))
            for metric in ("accuracy", "precision", "recall", "f1")
            for statistic, function in (("mean", np.mean), ("std", np.std))}})
    selected = max(cv_summary, key=lambda row: row["f1_mean"])["model"]
    print(f"학습셋 CV macro F1 기준 선택: {selected}", flush=True)
    tests = paired_tests({spec.name: [row["accuracy"] for row in fold_rows if row["model"] == spec.name] for spec in models})

    holdout = []
    predictions: dict[str, IntArray] = {}
    for spec in models:
        print(f"Hold-out {spec.name}: 전체 학습셋 학습 시작", flush=True)
        pipeline = clone(spec.pipeline)
        start = time.perf_counter()
        with threadpool_limits(limits=JOBS):
            pipeline.fit(data.Xtr, data.ytr)
            train_seconds = time.perf_counter() - start
            start = time.perf_counter()
            predicted = pipeline.predict(data.Xte).astype(np.int64)
            batch_seconds = time.perf_counter() - start
            for _ in range(10):
                pipeline.predict(data.Xte[:1])
            start = time.perf_counter()
            for _ in range(100):
                pipeline.predict(data.Xte[:1])
            infer_ms = (time.perf_counter() - start) / 100 * 1000
        predictions[spec.name] = predicted
        metrics = measure(data.yte, predicted)
        cm = confusion_matrix(data.yte, predicted, labels=np.arange(8))
        with (output / f"confusion_{spec.name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["actual / predicted", *data.class_names])
            writer.writerows([label, *row.tolist()] for label, row in zip(data.class_names, cm, strict=True))
        report = classification_report(data.yte, predicted, labels=np.arange(8), target_names=data.class_names,
                                       output_dict=True, zero_division=0)
        holdout.append({"model": spec.name, **asdict(metrics), "train_seconds": train_seconds,
                        "inference_ms_per_window": infer_ms, "test_batch_seconds": batch_seconds,
                        "scaler_fit_samples": int(pipeline.named_steps["scaler"].n_samples_seen_),
                        "confusion_matrix": cm.tolist(), "error_counts": asdict(categorize_errors(cm, data.class_names)),
                        "classification_report": report,
                        "model_params": {key: value for key, value in spec.pipeline.named_steps["model"].get_params().items()
                                         if key in {"kernel", "C", "gamma", "n_estimators", "max_depth", "learning_rate",
                                                    "subsample", "colsample_bytree", "random_state", "objective", "num_class",
                                                    "tree_method", "n_jobs", "cache_size", "decision_function_shape"}}})
        print(f"Hold-out {spec.name}: accuracy={metrics.accuracy:.4f}, F1={metrics.f1:.4f}, 학습 {train_seconds:.2f}s", flush=True)
    with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "trial_id", "subject", "y_true", *(spec.name for spec in models)])
        writer.writerows((i, data.groups_te[i], data.subjects_te[i], int(data.yte[i]),
                          *(int(predictions[spec.name][i]) for spec in models)) for i in range(len(data.yte)))
    fields = ["model", "accuracy", "precision", "recall", "f1", "train_seconds", "inference_ms_per_window", "test_batch_seconds"]
    with (output / "holdout_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in holdout)
    result = {
        "holdout": holdout, "cv_folds": fold_rows, "cv_summary": cv_summary,
        "paired_tests": [asdict(item) for item in tests], "selected_by_cv_f1": selected,
        "class_names": list(data.class_names), "feature_names": list(data.feature_names),
        "n_train_trials": len(np.unique(data.groups_tr)), "n_test_trials": len(np.unique(data.groups_te)),
        "n_train_windows": len(data.ytr), "n_test_windows": len(data.yte),
        "dataset_sha256": hashlib.sha256((ROOT / "data" / "dataset.npz").read_bytes()).hexdigest(),
        "versions": {"Python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "environment": {"os": platform.system(), "architecture": platform.machine(), "threads": JOBS},
        "std_ddof": 0, "cv_scope": "training trials only", "inference_warmup": 10,
        "inference_repetitions": 100, "timing_scope": "StandardScaler + estimator, perf_counter",
    }
    (output / "experiment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("전체 실험 완료", flush=True)
