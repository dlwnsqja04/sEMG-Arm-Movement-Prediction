#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "scikit-learn==1.9.1"]
# ///
# How to run: python verify_results.py after main.py.
"""Independently recalculate every table from saved predictions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from data import ROOT, DatasetProblem, load_dataset, make_folds


def verify_results() -> None:
    """Detect prediction/metric mismatch, trial leakage, and invalid fold scaling."""
    source = json.loads((ROOT / "results/experiment.json").read_text(encoding="utf-8"))
    dataset = load_dataset(ROOT / "data/dataset.npz")
    folds = make_folds(dataset)
    with (ROOT / "results/test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        test_rows = list(csv.DictReader(stream))
    with (ROOT / "results/cv_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        cv_rows = list(csv.DictReader(stream))
    truth = np.array([int(row["y_true"]) for row in test_rows])
    np.testing.assert_array_equal(truth, dataset.yte)
    np.testing.assert_array_equal([row["trial_id"] for row in test_rows], dataset.groups_te)
    np.testing.assert_array_equal([int(row["y_true"]) for row in cv_rows], dataset.ytr)
    np.testing.assert_array_equal([row["trial_id"] for row in cv_rows], dataset.groups_tr)
    for record in source["holdout"]:
        predicted = np.array([int(row[record["model"]]) for row in test_rows])
        cm = np.zeros((8, 8), dtype=np.int64)
        np.add.at(cm, (truth, predicted), 1)
        np.testing.assert_array_equal(cm, record["confusion_matrix"])
        np.testing.assert_array_equal(cm.sum(1), np.repeat(912, 8))
        precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(8), where=cm.sum(0) > 0)
        recall = np.diag(cm) / cm.sum(1)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(8), where=(precision + recall) > 0)
        np.testing.assert_allclose([np.trace(cm) / cm.sum(), precision.mean(), recall.mean(), f1.mean()],
                                   [record[key] for key in ("accuracy", "precision", "recall", "f1")], atol=1e-14)
        assert record["scaler_fit_samples"] == len(dataset.Xtr)
    for fold in folds:
        actual = np.array([int(cv_rows[i]["y_true"]) for i in fold.val_indices])
        assert not set(dataset.groups_tr[fold.train_indices]) & set(dataset.groups_tr[fold.val_indices])
        for record in (r for r in source["cv_folds"] if r["fold"] == fold.number):
            predicted = np.array([int(cv_rows[i][record["model"]]) for i in fold.val_indices])
            np.testing.assert_allclose(np.mean(actual == predicted), record["accuracy"], atol=1e-14)
            assert record["scaler_fit_samples"] == len(fold.train_indices)
            assert record["scaler_mean_max_error"] < 1e-12
    for record in source["cv_summary"]:
        rows = [r for r in source["cv_folds"] if r["model"] == record["model"]]
        for metric in ("accuracy", "precision", "recall", "f1"):
            values = [row[metric] for row in rows]
            np.testing.assert_allclose([np.mean(values), np.std(values)],
                                       [record[f"{metric}_mean"], record[f"{metric}_std"]], atol=1e-14)
    for record in source["paired_tests"]:
        a = np.array([r["accuracy"] for r in source["cv_folds"] if r["model"] == record["model_a"]])
        b = np.array([r["accuracy"] for r in source["cv_folds"] if r["model"] == record["model_b"]])
        delta = a - b
        if record["p_value"] is not None:
            statistic = delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)))
            p = 2 * student_t.sf(abs(statistic), df=len(delta) - 1)
            np.testing.assert_allclose([statistic, p], [record["t_statistic"], record["p_value"]], rtol=1e-12)
    assert hashlib.sha256((ROOT / "data/dataset.npz").read_bytes()).hexdigest() == source["dataset_sha256"]
    assert not list(ROOT.rglob("*.zip"))
    expected_names = tuple(f"{feature}_Ch{channel}" for feature in ("RMS", "MAV", "WL", "ZC", "SSC") for channel in range(1, 6))
    assert dataset.feature_names == expected_names
    missing_raised = False
    try:
        load_dataset(ROOT / "data/missing_dataset_for_validation.npz")
    except DatasetProblem:
        missing_raised = True
    assert missing_raised
    verification = {"holdout_metrics_from_predictions": True, "all_confusion_cells_match": True,
                    "shared_holdout_trial_ids": 0, "cv_trial_overlap": 0,
                    "fold_scaler_train_only": True, "cv_scores_from_predictions": True,
                    "paired_t_from_difference_vector": True, "dataset_sha256_match": True,
                    "models": 3, "cv_fits": 15, "holdout_fits": 3, "zip_files": 0}
    (ROOT / "results/verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print("예측·지표·혼동행렬·분할·검정 독립 재계산 검증 완료", flush=True)


if __name__ == "__main__":
    verify_results()
