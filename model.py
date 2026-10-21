# /// script
# requires-python = ">=3.12"
# dependencies = ["scikit-learn==1.9.1", "xgboost==3.4.1"]
# ///
# How to run: imported by main.py.
"""Define the lecture's three fixed model configurations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED: Final = 42
JOBS: Final = 4


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    pipeline: Pipeline


def make_models() -> tuple[ModelSpec, ...]:
    """Put scaler and estimator in one pipeline to prevent fold-level leakage."""
    return (
        ModelSpec("SVM", Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale",
                          decision_function_shape="ovr", cache_size=512)),
        ])),
        ModelSpec("RF", Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=JOBS)),
        ])),
        ModelSpec("XGBoost", Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=.05,
                                   subsample=.9, colsample_bytree=.8, random_state=SEED,
                                   objective="multi:softprob", num_class=8,
                                   eval_metric="mlogloss", tree_method="hist", n_jobs=JOBS)),
        ])),
    )
