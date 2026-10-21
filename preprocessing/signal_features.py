# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1"]
# ///
# How to run: imported by build_dataset.py.
"""Implement the lecture's filters, windows, and 25 time-domain features."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from source_data import FS, HOP, LENGTH, START, WIN, FloatArray


def preprocess(x: FloatArray, fs: int = FS) -> FloatArray:
    """Apply 60-Hz notch then fourth-order 20-450-Hz bandpass in time."""
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20 / (fs / 2), 450 / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def cut(x: FloatArray) -> FloatArray:
    """Take [1.5, 3.5) seconds relative to this trial's mark."""
    return x[START:START + LENGTH]


def make_windows(x: FloatArray) -> FloatArray:
    """Return 19 overlapping windows in their original temporal order."""
    n = (len(x) - WIN) // HOP + 1
    return np.stack([x[i * HOP:i * HOP + WIN] for i in range(n)])


def td_features(w: FloatArray, th: float = 0.01) -> FloatArray:
    """(200, 5) -> (25,): RMS, MAV, WL, ZC, SSC; Ch1-5 in each."""
    rms = np.sqrt(np.mean(w**2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum(((w[:-1] * w[1:]) < 0) &
                (np.abs(d) >= th), axis=0)
    ssc = np.sum((d[:-1] * d[1:]) < 0, axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])
