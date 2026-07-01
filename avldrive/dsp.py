"""Digital signal processing primitives using AVL-DRIVE filter notation.

AVL filtered-channel naming: ``SMO(x)`` smoothing, ``LP(x)`` low-pass, ``HP(x)``
high-pass, ``BP(x,y)`` band-pass. All frequency filters are zero-phase
(Butterworth + ``filtfilt``) on the 100 Hz analysis grid.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from .config import FS


def _butter_filt(data, btype, cut, fs: float = FS, order: int = 3):
    x = np.asarray(data, dtype=float)
    nyq = 0.5 * fs
    if btype == "band":
        lo, hi = max(cut[0] / nyq, 1e-4), min(cut[1] / nyq, 0.999)
        if lo >= hi:
            return x
        b, a = butter(order, [lo, hi], btype="band")
    elif btype == "low":
        b, a = butter(order, min(cut / nyq, 0.999), btype="low")
    else:  # high
        b, a = butter(order, max(cut / nyq, 1e-4), btype="high")
    if len(x) <= 3 * max(len(a), len(b)):
        return x
    return filtfilt(b, a, x)


def LP(x, cutoff: float, fs: float = FS):
    """Low-pass: passes frequencies below ``cutoff`` Hz."""
    return _butter_filt(x, "low", cutoff, fs)


def HP(x, cutoff: float, fs: float = FS):
    """High-pass: passes frequencies above ``cutoff`` Hz."""
    return _butter_filt(x, "high", cutoff, fs)


def BP(x, low: float, high: float, fs: float = FS):
    """Band-pass: passes frequencies between ``low`` and ``high`` Hz."""
    return _butter_filt(x, "band", (low, high), fs)


def SMO(x, n: int):
    """AVL smoothing filter: triangular weighted moving average over ``n`` points."""
    x = np.asarray(x, dtype=float)
    n = max(1, int(n))
    if n <= 1 or len(x) < n:
        return x
    w = np.concatenate([np.arange(1, n // 2 + 2), np.arange((n + 1) // 2, 0, -1)])[:n].astype(float)
    w /= w.sum()
    return np.convolve(x, w, mode="same")


def rms(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def crest_factor(x) -> float:
    """Peak-to-RMS ratio (AVL: high value indicates occasional bumps)."""
    r = rms(x)
    return float(np.max(np.abs(np.asarray(x, dtype=float))) / r) if r > 1e-9 else 0.0
