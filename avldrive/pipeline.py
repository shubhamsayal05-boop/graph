"""High-level orchestration: measurement import and full assessment.

Keeps Streamlit out of the core so the pipeline is testable and reusable.
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

import pandas as pd
from asammdf import MDF

from .assessment import assess
from .channels import build_calculated_channels, resolve_channels
from .config import FS, REQUIRED_LOGICAL
from .operation_modes import detect_events


class MissingChannelsError(RuntimeError):
    """Raised when the measurement lacks the required DRIVE channels.

    ``available`` holds every raw channel name in the file so the UI can offer a
    manual mapping.
    """

    def __init__(self, missing, available):
        self.missing = missing
        self.available = list(available)
        super().__init__(
            f"Missing required DRIVE channels: {missing}. "
            f"The file has {len(self.available)} channels — map them manually.")


def _close_quietly(obj):
    """Close an asammdf MDF (or similar) if it exposes ``close``, ignoring errors."""
    try:
        if obj is not None and hasattr(obj, "close"):
            obj.close()
    except Exception:
        pass


def _remove_quietly(path: str, retries: int = 5, delay: float = 0.2):
    """Best-effort file removal. On Windows a handle may linger briefly after
    close, so retry a few times before giving up (never raises)."""
    for _ in range(retries):
        try:
            if not os.path.exists(path):
                return
            os.remove(path)
            return
        except (PermissionError, OSError):
            time.sleep(delay)


def list_channels(path: str) -> list[str]:
    """Return every raw channel name in a measurement file (sorted, unique)."""
    mdf = MDF(path)
    try:
        return sorted(set(mdf.channels_db.keys()))
    finally:
        _close_quietly(mdf)


def list_channels_from_bytes(data: bytes, suffix: str) -> list[str]:
    fd, tmp_path = tempfile.mkstemp(suffix=suffix or ".mf4")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        return list_channels(tmp_path)
    finally:
        _remove_quietly(tmp_path)


def load_measurement(path: str, cfg: dict, mapping: dict | None = None):
    """Import an .mf4/.dat file, resample to the 100 Hz grid, rename to AVL
    logical channels and build calculated channels. Returns ``(df, found)``.

    ``mapping`` optionally overrides channel resolution (logical -> raw); only
    entries whose raw channel actually exists are kept.

    The source ``MDF`` (and intermediate filtered/resampled objects) are always
    closed before returning so the underlying file is released — critical on
    Windows, where an open handle blocks deletion (WinError 32).
    """
    mdf = MDF(path)
    filtered = res = None
    try:
        found = dict(mapping) if mapping else resolve_channels(mdf)
        # Keep only mappings whose raw channel is present in the file.
        found = {lg: raw for lg, raw in found.items() if raw in mdf.channels_db}
        missing = [c for c in REQUIRED_LOGICAL if c not in found]
        if missing:
            raise MissingChannelsError(missing, sorted(mdf.channels_db.keys()))
        filtered = mdf.filter(list(found.values()))
        res = filtered.resample(raster=1.0 / FS)
        # ``.copy()`` materializes the data so it stays valid after the MDF (and
        # any backing memory-maps) are closed below.
        df = res.to_dataframe().rename(columns={raw: lg for lg, raw in found.items()}).copy()
    finally:
        _close_quietly(res)
        _close_quietly(filtered)
        _close_quietly(mdf)

    df = df.reset_index().rename(columns={"index": "timestamp", "time": "timestamp"})
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df = build_calculated_channels(df, cfg)
    return df, found


def load_measurement_from_bytes(data: bytes, suffix: str, cfg: dict, mapping: dict | None = None):
    """Convenience wrapper for in-memory uploads. Writes a temp file, imports it
    and always cleans up (best-effort, Windows-safe)."""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix or ".mf4")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        return load_measurement(tmp_path, cfg, mapping)
    finally:
        _remove_quietly(tmp_path)


@dataclass
class AssessmentResult:
    overall: float | None
    mode_results: dict
    events: list
    found: dict
    duration_s: float
    n_samples: int


def run_assessment(df, enabled_modes, cfg: dict, dna: dict,
                   mode_weights=None, criteria_weights=None) -> AssessmentResult:
    """Detect operation modes and compute the DRIVE-Rating tree."""
    events = detect_events(df, enabled_modes, cfg)
    overall, mode_results = assess(df, events, dna, mode_weights, criteria_weights)
    duration = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) else 0.0
    return AssessmentResult(overall=overall, mode_results=mode_results, events=events,
                            found={}, duration_s=duration, n_samples=len(df))
