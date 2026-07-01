"""Simplified AVL-DRIVE trigger engine: operation-mode detection.

Detects transient events (Tip in/out, Drive away, Gear shift, Engine start/shut
off) and segment-based modes (Acceleration, Constant speed, Deceleration,
Sailing, Recuperation, Idle, Vehicle stationary, Maneuvering). Only modes
enabled for the selected transmission are produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FS


def _segments(t, mask, min_dur=0.5, merge_gap=0.3, pre=0.0, post=0.0):
    """Convert a boolean mask into (start, end) time segments."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    segs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if t[i] - t[prev] > merge_gap:
            segs.append((t[start], t[prev]))
            start = i
        prev = i
    segs.append((t[start], t[prev]))
    return [(max(t[0], s - pre), min(t[-1], e + post)) for s, e in segs if (e - s) >= min_dur]


def _clusters(t, idxs, gap=3.0, pre=0.5, post=2.5):
    """Cluster trigger indices into event windows."""
    idxs = np.asarray(idxs, dtype=int)
    if idxs.size == 0:
        return []
    starts = [t[idxs[0]]]
    for i in idxs[1:]:
        if t[i] - starts[-1] > gap:
            starts.append(t[i])
    return [(s - pre, s + post) for s in starts]


def detect_events(df, enabled_modes, cfg: dict):
    """Return a time-sorted list of ``{mode, t_start, t_end}`` for enabled modes."""
    t = df["timestamp"].to_numpy(float)
    pedal = df["AcceleratorPedal"].to_numpy(float)
    ax = df["AccelerationChassisCompensated"].to_numpy(float)
    speed = df["VehicleSpeed"].to_numpy(float) if "VehicleSpeed" in df else np.zeros_like(t)
    brake = df["BrakePosition"].to_numpy(float) if "BrakePosition" in df else np.zeros_like(t)
    espeed = df["EngineSpeed"].to_numpy(float) if "EngineSpeed" in df else None
    gear = df["GearEngaged"].to_numpy(float) if "GearEngaged" in df else None
    prate = np.gradient(pedal, 1.0 / FS)
    events: list[dict] = []

    def add(mode, segs):
        for s, e in segs:
            events.append({"mode": mode, "t_start": float(s), "t_end": float(e)})

    if "Tip in" in enabled_modes:
        add("Tip in", _clusters(t, np.where((prate > 40) & (pedal > 10))[0]))
    if "Tip out" in enabled_modes:
        add("Tip out", _clusters(t, np.where((prate < -40) & (pedal < 60))[0], post=2.0))
    if "Drive away" in enabled_modes and "VehicleSpeed" in df:
        cross = np.where((speed[:-1] < 5.0) & (speed[1:] >= 5.0))[0] + 1
        win = int(1.5 * FS)
        launch = [i for i in cross if speed[max(0, i - win):i].min() < 3.0]
        add("Drive away", _clusters(t, launch, gap=5.0, pre=0.5, post=4.0))
    if "Acceleration" in enabled_modes:
        add("Acceleration", _segments(t, (ax > 0.4) & (pedal > 10) & (speed > 5), min_dur=1.0, post=0.3))
    if "Constant speed" in enabled_modes and "VehicleSpeed" in df:
        std = pd.Series(speed).rolling(int(2 * FS), center=True, min_periods=int(FS)).std().to_numpy()
        add("Constant speed", _segments(t, (speed > 10) & (std < 0.6) & (np.abs(prate) < 8), min_dur=2.0))
    if "Deceleration" in enabled_modes:
        add("Deceleration", _segments(t, (ax < -0.4) | (brake > 5), min_dur=0.7, post=0.2))
    if "Recuperation" in enabled_modes:
        add("Recuperation", _segments(t, (ax < -0.3) & (brake < 5) & (pedal < 5) & (speed > 5), min_dur=0.7))
    if "Sailing" in enabled_modes:
        add("Sailing", _segments(t, (pedal < 3) & (np.abs(ax) < 0.3) & (speed > 20), min_dur=1.5))
    if "Maneuvering" in enabled_modes:
        add("Maneuvering", _segments(t, (speed > 0.5) & (speed < 10), min_dur=1.0))
    if "Vehicle stationary" in enabled_modes:
        add("Vehicle stationary", _segments(t, speed < 1.0, min_dur=1.0))
    if "Idle" in enabled_modes:
        idle_mask = speed < 1.0
        if espeed is not None:
            idle_mask = idle_mask & (espeed > 300)
        add("Idle", _segments(t, idle_mask, min_dur=1.0))
    if "Gear shift" in enabled_modes and gear is not None:
        gchg = np.where(np.abs(np.diff(gear)) > 0.5)[0] + 1
        add("Gear shift", _clusters(t, gchg, gap=1.0, pre=0.3, post=1.0))
    if espeed is not None:
        if "Engine start" in enabled_modes:
            add("Engine start", _clusters(t, np.where((espeed[:-1] < 400) & (espeed[1:] >= 400))[0] + 1, gap=2.0, pre=0.3, post=1.5))
        if "Engine shut off" in enabled_modes:
            add("Engine shut off", _clusters(t, np.where((espeed[:-1] >= 400) & (espeed[1:] < 400))[0] + 1, gap=2.0, pre=0.3, post=1.5))

    events.sort(key=lambda ev: ev["t_start"])
    return events
