# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
# How to run: imported by build_dataset.py; use the existing coursework Python.
"""Read original CSV recordings into a portable output folder."""
from __future__ import annotations

import csv
import os
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ROOT: Final = Path(os.environ.get("SEMG_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "data")))
SOURCE: Final = Path(os.environ.get("SEMG_RAW_DIR", str(ROOT / "raw")))
FS: Final = 1000
START: Final = 1500
LENGTH: Final = 2000
WIN: Final = 200
HOP: Final = 100
MUSCLES: Final = ("ECR", "ED", "FDS", "Biceps", "Triceps")
GESTURES: Final = {"fist": "fist", "open": "open", "pro": "pronation", "extor": "supination"}
CLASSES: Final = tuple(f"{a}_{g}" for a in (30, 60) for g in ("fist", "open", "pronation", "supination"))
FEATURE_NAMES: Final = tuple(f"{f}_Ch{c}" for f in ("RMS", "MAV", "WL", "ZC", "SSC") for c in range(1, 6))


@dataclass(frozen=True, slots=True)
class DataProblem(Exception):
    filename: str
    detail: str

    def __str__(self) -> str:
        return f"{self.filename}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Trial:
    uid: str
    filename: str
    subject: str
    class_id: int
    repeat: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class SourceIndex:
    trials: tuple[Trial, ...]
    recordings: Mapping[str, FloatArray]
    file_hashes: Mapping[str, str]


def index_source() -> SourceIndex:
    """Parse all 64 CSVs; mark boundaries define nonoverlapping trials."""
    paths = sorted(SOURCE.glob("*.csv"))
    if len(paths) != 64:
        raise DataProblem(str(SOURCE), f"CSV 64개가 필요합니다. 현재 {len(paths)}개입니다")
    trials: list[Trial] = []
    recordings: dict[str, FloatArray] = {}
    hashes: dict[str, str] = {}
    header_expected = [f"bandpass {c}" for c in range(1, 6)] + ["time", "onset", "mark"]
    for i, path in enumerate(paths, 1):
        match = re.fullmatch(r"([a-h])_(30|60) (fist|open|pro|extor)\.csv", path.name)
        if match is None:
            raise DataProblem(path.name, "파일명을 확인해야 합니다")
        subject, angle, gesture = match.groups()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream))
        if header != header_expected:
            raise DataProblem(path.name, "예상한 8개 열과 순서가 다릅니다")
        raw = np.loadtxt(path, delimiter=",", skiprows=1)
        if raw.ndim != 2 or raw.shape[1] != 8 or not np.isfinite(raw).all():
            raise DataProblem(path.name, "배열 모양 또는 NaN·무한값을 확인해야 합니다")
        if not np.allclose(np.diff(raw[:, 5]), 1 / FS, rtol=0, atol=1e-7):
            raise DataProblem(path.name, "시간 간격이 1ms가 아닙니다")
        if not np.isin(raw[:, 6:], (0, 1)).all():
            raise DataProblem(path.name, "이벤트 열은 0과 1이어야 합니다")
        starts = np.flatnonzero(raw[:, 7] == 1)
        if len(starts) != 30 or not np.all(np.abs(np.diff(starts) - 5000) <= 1):
            raise DataProblem(path.name, "30회 시행과 약 5초 간격을 확인해야 합니다")
        stops = np.r_[starts[1:], starts[-1] + 5000]
        if stops[-1] > len(raw):
            raise DataProblem(path.name, "마지막 시행이 5초보다 짧습니다")
        class_id = CLASSES.index(f"{angle}_{GESTURES[gesture]}")
        for repeat, (start, stop) in enumerate(zip(starts, stops, strict=True), 1):
            trials.append(Trial(f"{path.stem}::r{repeat:02d}", path.name, subject,
                                class_id, repeat, int(start), int(stop)))
        recordings[path.name] = np.ascontiguousarray(raw[:, :5])
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if i % 16 == 0:
            print(f"원자료 확인 {i}/64", flush=True)
    return SourceIndex(tuple(trials), recordings, hashes)
