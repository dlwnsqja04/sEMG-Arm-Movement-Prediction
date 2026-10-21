# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2"]
# ///
# How to run: python -c "from figures import render_figures; render_figures()"
"""Plot every confusion cell and compare the same five validation folds."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
from typing import Final

import numpy as np

from data import ROOT

CACHE: Final = ROOT / ".cache/mpl"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import read_results

PALETTE: Final = {"SVM": "#1F77B4", "RF": "#A66B00", "XGBoost": "#D95F02"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})


def render_figures() -> None:
    """Generate repository-ready PNGs directly from predictions and numeric results."""
    result = read_results()
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    for model in result.models:
        fig, ax = plt.subplots(figsize=(8.8, 7.4), layout="constrained")
        image = ax.imshow(model.cm, cmap="Blues", vmin=0, vmax=912)
        ax.set_xticks(range(8), result.classes, rotation=40, ha="right", fontsize=10)
        ax.set_yticks(range(8), result.classes, fontsize=10)
        ax.set(xlabel="Predicted class", ylabel="Actual class",
               title=f"{model.name} | Hold-out confusion matrix\n7,296 windows / 384 trials / 912 windows per class")
        for row in range(8):
            for col in range(8):
                value = int(model.cm[row, col])
                ax.text(col, row, str(value), ha="center", va="center", fontsize=10,
                        color="white" if value > 456 else "#222222")
        fig.colorbar(image, ax=ax, shrink=.78, label="Number of windows")
        fig.savefig(folder / f"confusion_{model.name}.png")
        plt.close(fig)
    shutil.copyfile(folder / f"confusion_{result.selected}.png", folder / "confusion8.png")
    fig, ax = plt.subplots(figsize=(8.8, 4.3), layout="constrained")
    for name, marker, style in zip(PALETTE, ("o", "s", "D"), ("-", "--", ":"), strict=True):
        rows = sorted((row for row in result.folds if row.name == name), key=lambda row: row.number)
        ax.plot([row.number for row in rows], [row.accuracy * 100 for row in rows], marker=marker,
                linestyle=style, color=PALETTE[name], label=name, linewidth=1.6)
    ax.set(xticks=range(1, 6), xlabel="Shared validation fold", ylabel="Window accuracy (%)",
           title="Five-fold CV within training trials | Same splits for all models")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(folder / "cv_accuracy.png")
    plt.close(fig)

    with (ROOT / "results/test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(ROOT / "data/dataset.npz", allow_pickle=False) as data:
        rms = np.sqrt(np.mean(data["Xte_raw"][:, :5] ** 2, axis=1))
    window_rows = []
    counts: dict[str, int] = {}
    position = []
    for row in rows:
        trial = row["trial_id"]
        position.append(counts.get(trial, 0))
        counts[trial] = counts.get(trial, 0) + 1
    positions = np.array(position)
    for index in range(19):
        chosen = np.flatnonzero(positions == index)
        item = {"window": index + 1, "start_s": 1.5 + index * .1,
                "stop_s": 1.7 + index * .1, "n": len(chosen), "median_pooled_rms": float(np.median(rms[chosen]))}
        for model in result.models:
            item[model.name] = float(np.mean([rows[i]["y_true"] == rows[i][model.name] for i in chosen]))
        window_rows.append(item)
    with (ROOT / "results/window_accuracy.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True, layout="constrained")
    centers = [row["start_s"] + .1 for row in window_rows]
    for name, style in zip(PALETTE, ("-", "--", ":"), strict=True):
        axes[0].plot(centers, [row[name] * 100 for row in window_rows], color=PALETTE[name],
                     linestyle=style, marker=".", label=name)
    axes[0].set(ylabel="Accuracy (%)", ylim=(0, 100), title="Hold-out diagnosis by window position | 384 windows at each position")
    axes[0].legend(frameon=False, ncols=3)
    axes[1].plot(centers, [row["median_pooled_rms"] for row in window_rows], color="#333333", marker=".")
    axes[1].set(xlabel="Window center from trial mark (s)", ylabel="Median pooled RMS\n(CSV amplitude)", ylim=(0, None))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(folder / "window_diagnosis.png")
    plt.close(fig)
    print("모델별 혼동행렬·교차검증·시간 구간별 진단 그림 저장 완료", flush=True)
