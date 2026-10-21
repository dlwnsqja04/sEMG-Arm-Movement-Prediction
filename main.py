#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "scikit-learn==1.9.1", "xgboost==3.4.1", "matplotlib==3.11.2"]
# ///
# How to run: python3 main.py (installed environment), or uv run main.py.
"""Run all fixed experiments and regenerate their numeric results and figures."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def prepare_runtime() -> None:
    """Use an already bundled OpenMP library on macOS without installing software."""
    if sys.platform == "darwin" and os.environ.get("SEMG_OPENMP_READY") != "1":
        package = importlib.util.find_spec("sklearn")
        if package is not None and package.origin is not None:
            folder = Path(package.origin).parent / ".dylibs"
            if (folder / "libomp.dylib").is_file():
                environment = os.environ.copy()
                existing = environment.get("DYLD_LIBRARY_PATH", "")
                environment["DYLD_LIBRARY_PATH"] = str(folder) + (os.pathsep + existing if existing else "")
                environment["SEMG_OPENMP_READY"] = "1"
                os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def main() -> None:
    """Train and evaluate the three models, then render the submitted evidence."""
    prepare_runtime()
    from data import ROOT, load_dataset
    from experiment import run_experiment

    run_experiment(load_dataset(ROOT / "data" / "dataset.npz"))
    from figures import render_figures

    render_figures()
    from verify_results import verify_results

    verify_results()
    print("결과를 results/, figures/에 저장했습니다")


if __name__ == "__main__":
    main()
