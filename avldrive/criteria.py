"""AVL-DRIVE criteria computation (1-10 rating, 10 = best).

Assessment direction always follows AVL: the higher the disturbance / the longer
the delay / the worse the correlation, the lower the rating.
"""
from __future__ import annotations

import numpy as np

from .config import FS, MODE_CRITERIA
from .dsp import BP, HP, LP, SMO, crest_factor, rms


def rate(value: float, sensitivity: float, offset: float = 0.0) -> float:
    """Convert a fault metric into a 1-10 rating."""
    return float(np.clip(10.0 - max(0.0, value - offset) * sensitivity, 1.0, 10.0))


def disturbance_metrics(accel_comp) -> dict:
    """Acceleration-disturbance family on AccelerationChassisCompensated.

    Bands per AVL: total 2-50 Hz, LF 2-10 Hz, HF >10 Hz, crest factor in HF.
    Returns ``{key: (metric, signal)}``.
    """
    total = BP(accel_comp, 2.0, 49.5)
    lf = BP(accel_comp, 2.0, 10.0)
    hf = HP(accel_comp, 10.0)
    return {"total": (rms(total), total), "lf": (rms(lf), lf),
            "hf": (rms(hf), hf), "crest": (crest_factor(hf), hf)}


def correlation_metric(pedal, accel_comp):
    """Surge/correlation: pedal↔acceleration correlation at <2 Hz.
    Returns ``(fault_parameter, pearson_r)`` where parameter = 1 - max(0, r)."""
    p = LP(np.asarray(pedal, float), 2.0)
    a = LP(np.asarray(accel_comp, float), 2.0)
    active = np.asarray(pedal, float) > 5.0
    if active.sum() > 50:
        p, a = p[active], a[active]
    if np.std(p) < 1e-6 or np.std(a) < 1e-6:
        return 0.0, 1.0
    corr = float(np.corrcoef(p, a)[0, 1])
    return 1.0 - max(0.0, corr), corr


def response_delay(window):
    """Response delay [s]: dead-time from accelerator tip-in (>5%) to acceleration
    onset, using AcceleratorPedal vs AccelerationChassis_SMO(20)."""
    ev = window.reset_index(drop=True)
    pedal = ev["AcceleratorPedal"].to_numpy(float)
    if not (pedal > 5.0).any():
        return None
    accel_smo = SMO(ev["AccelerationChassisCompensated"].to_numpy(float), 20)
    t = ev["timestamp"].to_numpy(float)
    trig = int(np.argmax(pedal > 5.0))
    jerk = np.gradient(accel_smo, 1.0 / FS)
    post = np.arange(trig, len(t))
    onset = post[jerk[post] > 0.5]
    if onset.size == 0:
        return None
    return max(0.0, float(t[onset[0]] - t[trig]))


def event_criteria(window, mode: str, dna: dict) -> dict:
    """Compute the criteria applicable to ``mode`` over one event window.
    Returns ``{criterion: {rating, metric, unit}}``."""
    ac = window["AccelerationChassisCompensated"].to_numpy(float)
    wanted = MODE_CRITERIA.get(mode, ["disturbances", "disturbances_lf", "disturbances_hf"])
    dm = disturbance_metrics(ac)
    res: dict[str, dict] = {}
    if "disturbances" in wanted:
        res["disturbances"] = {"rating": rate(dm["total"][0], dna["disturbances"]), "metric": dm["total"][0], "unit": "m/s²"}
    if "disturbances_lf" in wanted:
        res["disturbances_lf"] = {"rating": rate(dm["lf"][0], dna["lf"]), "metric": dm["lf"][0], "unit": "m/s²"}
    if "disturbances_hf" in wanted:
        res["disturbances_hf"] = {"rating": rate(dm["hf"][0], dna["hf"]), "metric": dm["hf"][0], "unit": "m/s²"}
    if "crest_factor" in wanted:
        res["crest_factor"] = {"rating": rate(dm["crest"][0], dna["crest"], offset=3.0), "metric": dm["crest"][0], "unit": "-"}
    if "correlation" in wanted:
        param, corr = correlation_metric(window["AcceleratorPedal"], ac)
        res["correlation"] = {"rating": rate(param, dna["correlation"]), "metric": corr, "unit": "r"}
    if "response_delay" in wanted:
        d = response_delay(window)
        if d is not None:
            res["response_delay"] = {"rating": rate(d, dna["delay"]), "metric": d, "unit": "s"}
    if "shift_shock" in wanted:
        pp = float(np.ptp(BP(ac, 3.0, 20.0)))
        res["shift_shock"] = {"rating": rate(pp, dna["hf"]), "metric": pp, "unit": "m/s²"}
    return res
