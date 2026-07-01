"""Welch power-spectral-density mapping and dominant-frequency interpretation."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .config import FS


def compute_band_spectrum(signal_data, band, fs: float = FS):
    """Welch PSD with the dominant peak isolated inside ``band``.
    Returns a dict (or ``None`` if the slice is too short)."""
    signal = np.asarray(signal_data, dtype=float)
    signal = signal[np.isfinite(signal)]
    nperseg = int(min(len(signal), 1024))
    if nperseg < 16:
        return None
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return None
    bf, bp_ = freqs[mask], psd[mask]
    i = int(np.argmax(bp_))
    return {"freqs": freqs, "psd": psd, "band": band,
            "dominant_freq": float(bf[i]), "dominant_power": float(bp_[i])}


def interpret_surge_source(freq: float) -> str:
    if freq < 1.0:
        return "Low-frequency chugging (< 1.0 Hz) — engine combustion / lugging or driveline lash."
    if freq < 1.5:
        return "Mid-band surge (1.0-1.5 Hz) — mixed powertrain source."
    return "Faster surge (> 1.5 Hz) — electric motor / e-drive controller damping loop."
