"""AVL-DRIVE-style objective drivability assessment tool.

Modelled as closely as practical on AVL-DRIVE(TM) 4.6 SR1. The user first selects
the transmission / powertrain architecture on the home page; the tool then
re-assigns itself (enabled operation modes, applicable criteria, relevant
channels, propulsion) to that selection. Ratings follow AVL's DRIVE Rating (DR,
1-10, 10 = best) aggregated criteria -> sub/main operation mode -> overall via a
weight tree plus extreme-value weighting.

Independent re-implementation for demonstration; it does not reproduce AVL's
proprietary calibration data or exact algorithms.
"""
import streamlit as st
from asammdf import MDF
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
import plotly.graph_objects as go
import os
from io import BytesIO
from datetime import datetime

FS = 100.0  # AVL-DRIVE analysis grid: 100 Hz (Nyquist 50 Hz).
G = 9.81

# -----------------------------------------------------------------------------
# 1. TRANSMISSION / POWERTRAIN CONFIGURATION
# -----------------------------------------------------------------------------
# Selecting a transmission re-assigns the enabled operation modes, propulsion,
# gearbox character and relevant channels. Modes are drawn from the AVL-DRIVE
# Function Descriptions (AT/CVT/DCT/DHT) and the .ect template families.
TRANSMISSION_CONFIG = {
    "AT": {
        "label": "Automatic (torque converter)", "propulsion": "ICE", "gearbox": "Stepped",
        "features": ["Turbine speed", "Torque-converter lock-up (TCC)", "Stepped gear shifts"],
        "modes": ["Drive away", "Tip in", "Tip out", "Acceleration", "Constant speed",
                  "Deceleration", "Gear shift", "Engine start", "Engine shut off",
                  "Idle", "Vehicle stationary", "Maneuvering"],
    },
    "CVT": {
        "label": "Continuously Variable", "propulsion": "ICE", "gearbox": "Continuous",
        "features": ["Turbine speed", "TCC", "Sailing"],
        "modes": ["Drive away", "Tip in", "Tip out", "Acceleration", "Constant speed",
                  "Deceleration", "Sailing", "Engine start", "Engine shut off",
                  "Idle", "Vehicle stationary", "Maneuvering"],
    },
    "eCVT": {
        "label": "Electronic CVT (power-split hybrid)", "propulsion": "Hybrid",
        "gearbox": "Power-split", "features": ["Electric launch", "Sailing",
                                               "Recuperation", "Engine start/stop"],
        "modes": ["Drive away", "Tip in", "Tip out", "Acceleration", "Constant speed",
                  "Deceleration", "Sailing", "Recuperation", "Engine start",
                  "Engine shut off", "Vehicle stationary", "Maneuvering"],
    },
    "DCT": {
        "label": "Dual-Clutch", "propulsion": "ICE", "gearbox": "Stepped",
        "features": ["Dual-clutch shifts", "Sailing"],
        "modes": ["Drive away", "Tip in", "Tip out", "Acceleration", "Constant speed",
                  "Deceleration", "Gear shift", "Sailing", "Engine start",
                  "Engine shut off", "Idle", "Vehicle stationary", "Maneuvering"],
    },
    "DHT": {
        "label": "Dedicated Hybrid Transmission", "propulsion": "Hybrid", "gearbox": "Hybrid",
        "features": ["Series/parallel modes", "Recuperation", "Engine start/stop"],
        "modes": ["Drive away", "Tip in", "Tip out", "Acceleration", "Constant speed",
                  "Deceleration", "Recuperation", "Engine start", "Engine shut off",
                  "Vehicle stationary", "Maneuvering"],
    },
    "MT": {
        "label": "Manual", "propulsion": "ICE", "gearbox": "Manual",
        "features": ["Clutch characteristics", "Manual gearshift", "Engine start/stop"],
        "modes": ["Drive away", "Tip in", "Tip out", "Acceleration", "Constant speed",
                  "Deceleration", "Gear shift", "Engine start", "Engine shut off",
                  "Idle", "Vehicle stationary", "Maneuvering"],
    },
    "BEV": {
        "label": "Battery Electric", "propulsion": "BEV", "gearbox": "Single-speed",
        "features": ["Single-speed reducer", "Recuperation", "No engine / idle"],
        "modes": ["Drive away", "Tip in", "Tip out", "Acceleration", "Constant speed",
                  "Deceleration", "Recuperation", "Vehicle stationary", "Maneuvering"],
    },
}

TRANSIENT_MODES = {"Tip in", "Tip out", "Drive away", "Gear shift",
                   "Engine start", "Engine shut off"}

# Criteria assessed per operation mode (subset of AVL criteria lists).
MODE_CRITERIA = {
    "Tip in": ["response_delay", "correlation", "disturbances", "disturbances_lf", "disturbances_hf"],
    "Drive away": ["response_delay", "disturbances", "disturbances_lf", "disturbances_hf"],
    "Tip out": ["disturbances", "disturbances_lf", "disturbances_hf"],
    "Acceleration": ["disturbances", "disturbances_lf", "disturbances_hf", "crest_factor"],
    "Constant speed": ["disturbances", "disturbances_lf", "disturbances_hf", "crest_factor"],
    "Deceleration": ["disturbances", "disturbances_lf", "disturbances_hf"],
    "Gear shift": ["shift_shock", "disturbances_lf", "disturbances_hf"],
    "Sailing": ["disturbances", "disturbances_lf"],
    "Recuperation": ["disturbances", "disturbances_lf", "disturbances_hf"],
    "Idle": ["disturbances_hf", "crest_factor"],
    "Vehicle stationary": ["disturbances_hf"],
    "Engine start": ["disturbances", "disturbances_hf"],
    "Engine shut off": ["disturbances", "disturbances_hf"],
    "Maneuvering": ["disturbances_lf"],
}

# AVL weight tree defaults: criteria weights and main-operation-mode weights (1-5).
CRITERIA_META = {
    "disturbances": {"label": "Acceleration disturbances (2-50 Hz)", "weight": 4, "unit": "m/s²", "kind": "dna_disturbances"},
    "disturbances_lf": {"label": "Disturbances LF (2-10 Hz)", "weight": 4, "unit": "m/s²", "kind": "dna_lf"},
    "disturbances_hf": {"label": "Disturbances HF (>10 Hz)", "weight": 3, "unit": "m/s²", "kind": "dna_hf"},
    "crest_factor": {"label": "Crest factor (HF)", "weight": 2, "unit": "-", "kind": "dna_crest"},
    "correlation": {"label": "Surge / correlation (<2 Hz)", "weight": 3, "unit": "-", "kind": "dna_correlation"},
    "response_delay": {"label": "Response delay", "weight": 5, "unit": "s", "kind": "dna_delay"},
    "shift_shock": {"label": "Shift shock", "weight": 4, "unit": "m/s²", "kind": "dna_hf"},
}
MODE_WEIGHTS = {
    "Tip in": 5, "Drive away": 5, "Tip out": 4, "Acceleration": 4, "Gear shift": 4,
    "Constant speed": 3, "Deceleration": 3, "Recuperation": 3, "Sailing": 2,
    "Idle": 2, "Engine start": 2, "Engine shut off": 2, "Vehicle stationary": 1,
    "Maneuvering": 2,
}

# Brand-DNA targets: how strictly each criterion is scored (higher = stricter).
BRAND_DNA = {
    "Luxury Sedan": {"disturbances": 3.0, "lf": 3.5, "hf": 2.5, "crest": 0.7, "correlation": 5.0, "delay": 12.0},
    "Sports Car": {"disturbances": 4.5, "lf": 5.0, "hf": 4.0, "crest": 1.0, "correlation": 7.0, "delay": 26.0},
    "Eco EV": {"disturbances": 2.5, "lf": 3.0, "hf": 2.5, "crest": 0.6, "correlation": 4.0, "delay": 6.0},
}

# AVL DRIVE channels: logical name -> candidate raw channel names (first present wins).
CHANNEL_CANDIDATES = {
    "AccelerationChassis": ["AccelerationChassis", "AccelerationChassisComp", "AccelChassis", "Accel_Filt_X", "Ax_Sensor_g", "Ax"],
    "AccelerationVertical": ["AccelerationVertical", "AccelerationWheel_Z", "Az_Sensor_g", "Az"],
    "AccelerationLateral": ["AccelerationLateral", "Ay_Sensor_g", "Ay"],
    "AcceleratorPedal": ["AcceleratorPedal", "Acc_Pedal_Pct", "AccPedal", "PedalPosition", "iv_dki"],
    "BrakePosition": ["BrakePosition", "BrakePdlPosn", "Brake", "iv_BrakePdlPosn"],
    "ClutchPedal": ["ClutchPedal", "ClutchPosition"],
    "EngineSpeed": ["EngineSpeed", "Engine_RPM", "EngSpeed", "nmot", "Nmot"],
    "VehicleSpeed": ["VehicleSpeed", "veh_Spd_Kph", "VehicleSpeedMPH", "VehSpeed"],
    "TurbineSpeed": ["TurbineSpeed"],
    "GearEngaged": ["GearEngaged", "Gear Engaged", "Current_Gear", "Gear", "GearDMU"],
    "SelectorLeverDMU": ["SelectorLeverDMU", "SelectorLever"],
    "TCC_State": ["TCC_State", "TCC"],
    "EngineTorque": ["Engine_Torque", "EngineTorque", "Torque"],
    "WheelSpeed_FL": ["WheelSpeed_FL", "WheelSpeedFL"],
    "WheelSpeed_FR": ["WheelSpeed_FR", "WheelSpeedFR"],
    "WheelSpeed_RL": ["WheelSpeed_RL", "WheelSpeedRL"],
    "WheelSpeed_RR": ["WheelSpeed_RR", "WheelSpeedRR"],
}
REQUIRED_LOGICAL = ["AcceleratorPedal", "AccelerationChassis"]

def relevant_channels(tx):
    """Channels relevant to the selected transmission (for the assignment summary)."""
    base = ["AccelerationChassis", "AccelerationVertical", "AcceleratorPedal",
            "BrakePosition", "VehicleSpeed"]
    cfg = TRANSMISSION_CONFIG[tx]
    if cfg["propulsion"] != "BEV":
        base += ["EngineSpeed"]
    if tx in ("AT", "CVT"):
        base += ["TurbineSpeed", "TCC_State"]
    if cfg["gearbox"] in ("Stepped", "Manual"):
        base += ["GearEngaged", "SelectorLeverDMU"]
    if tx == "MT":
        base += ["ClutchPedal"]
    base += ["WheelSpeed_FL", "WheelSpeed_FR", "WheelSpeed_RL", "WheelSpeed_RR"]
    return base

# -----------------------------------------------------------------------------
# 2. SIGNAL PROCESSING (AVL filter notation: SMO / LP / HP / BP)
# -----------------------------------------------------------------------------
def _butter_filt(data, btype, cut, fs=FS, order=3):
    x = np.asarray(data, dtype=float)
    nyq = 0.5 * fs
    if btype == "band":
        lo, hi = max(cut[0] / nyq, 1e-4), min(cut[1] / nyq, 0.999)
        if lo >= hi:
            return x
        b, a = butter(order, [lo, hi], btype="band")
    elif btype == "low":
        b, a = butter(order, min(cut / nyq, 0.999), btype="low")
    else:
        b, a = butter(order, max(cut / nyq, 1e-4), btype="high")
    if len(x) <= 3 * max(len(a), len(b)):
        return x
    return filtfilt(b, a, x)

def LP(x, c, fs=FS):
    return _butter_filt(x, "low", c, fs)

def HP(x, c, fs=FS):
    return _butter_filt(x, "high", c, fs)

def BP(x, lo, hi, fs=FS):
    return _butter_filt(x, "band", (lo, hi), fs)

def SMO(x, n):
    x = np.asarray(x, dtype=float)
    n = max(1, int(n))
    if n <= 1 or len(x) < n:
        return x
    w = np.concatenate([np.arange(1, n // 2 + 2), np.arange((n + 1) // 2, 0, -1)])[:n].astype(float)
    w /= w.sum()
    return np.convolve(x, w, mode="same")

def rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

def crest_factor(x):
    r = rms(x)
    return float(np.max(np.abs(np.asarray(x, dtype=float))) / r) if r > 1e-9 else 0.0

# -----------------------------------------------------------------------------
# 3. RATING PRIMITIVES + DR AGGREGATION (weight tree + extreme value)
# -----------------------------------------------------------------------------
def rate(value, sensitivity, offset=0.0):
    """Fault metric -> 1-10 rating (higher fault -> lower rating; AVL direction)."""
    return float(np.clip(10.0 - max(0.0, value - offset) * sensitivity, 1.0, 10.0))

def aggregate_dr(ratings, weights=None, extreme_p=2.0):
    """Combine ratings into a higher-level DRIVE Rating using the weight tree and
    AVL extreme-value weighting (worse ratings carry disproportionately more
    impact)."""
    r = np.asarray([x for x in ratings if x is not None], dtype=float)
    if r.size == 0:
        return None
    if weights is None:
        w = np.ones_like(r)
    else:
        w = np.asarray([wi for wi, x in zip(weights, ratings) if x is not None], dtype=float)
    ev = np.power(np.clip(11.0 - r, 1e-6, None), extreme_p)  # bad ratings weigh more
    W = w * ev
    if W.sum() <= 0:
        return float(np.mean(r))
    return float(np.sum(W * r) / np.sum(W))

# -----------------------------------------------------------------------------
# 4. CALCULATED CHANNELS (AVL Function Description, section 1.3)
# -----------------------------------------------------------------------------
def build_calculated_channels(df, cfg):
    ax = df["AccelerationChassis"].to_numpy(dtype=float)
    df["AccelerationChassisCompensated"] = ax - LP(ax, 0.3)

    if "VehicleSpeed" in df:
        v_ms = df["VehicleSpeed"].to_numpy(dtype=float) / 3.6
        dvdt = np.gradient(v_ms, 1.0 / FS)
        # RoadGradient: gravity component = measured accel - vehicle accel.
        grade = np.clip((ax - dvdt) / G, -0.4, 0.4)
        df["RoadGradient"] = np.degrees(np.arcsin(grade))
        # Road-load force F = A0 + B0*v + C0*v^2 + m*g*sin(alpha).
        m = cfg["mass"]
        spd = df["VehicleSpeed"].to_numpy(dtype=float)
        f_res = cfg["A0"] + cfg["B0"] * spd + cfg["C0"] * spd ** 2
        f_grav = m * G * np.sin(np.radians(df["RoadGradient"].to_numpy(float)))
        km = 1.05  # rotational mass factor
        f_tractive = m * km * dvdt + f_res + f_grav
        df["TractiveForce"] = f_tractive
        df["WheelTorque"] = f_tractive * cfg["wheel_radius"]
        if "EngineSpeed" in df and cfg["propulsion"] != "BEV":
            eng_w = df["EngineSpeed"].to_numpy(float) * 2.0 * np.pi / 60.0
            wheel_w = np.where(v_ms > 0.1, v_ms / cfg["wheel_radius"], np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                i_tot = eng_w / wheel_w
                df["EngineTorqueEstimated"] = np.where(np.isfinite(i_tot) & (np.abs(i_tot) > 0.1),
                                                       df["WheelTorque"].to_numpy(float) / i_tot, np.nan)

    if "EngineSpeed" in df and "VehicleSpeed" in df:
        v = df["VehicleSpeed"].to_numpy(dtype=float)
        n = df["EngineSpeed"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["GearRatio"] = np.where(np.abs(v) > 1.0, n / np.where(v == 0, np.nan, v), np.nan)

    fl, fr = df.get("WheelSpeed_FL"), df.get("WheelSpeed_FR")
    rl, rr = df.get("WheelSpeed_RL"), df.get("WheelSpeed_RR")
    if all(w is not None for w in (fl, fr, rl, rr)):
        front = (fl.to_numpy(float) + fr.to_numpy(float)) / 2.0
        rear = (rl.to_numpy(float) + rr.to_numpy(float)) / 2.0
        driven, non_driven = (rear, front) if cfg.get("drive") == "RWD" else (front, rear)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["WheelSlip"] = np.where(np.abs(non_driven) > 1.0,
                                       (driven - non_driven) / non_driven * 100.0, 0.0)
    return df

# -----------------------------------------------------------------------------
# 5. CRITERIA COMPUTATION
# -----------------------------------------------------------------------------
def disturbance_metrics(accel_comp):
    total = BP(accel_comp, 2.0, 49.5)
    lf = BP(accel_comp, 2.0, 10.0)
    hf = HP(accel_comp, 10.0)
    return {"total": (rms(total), total), "lf": (rms(lf), lf),
            "hf": (rms(hf), hf), "crest": (crest_factor(hf), hf)}

def correlation_metric(pedal, accel_comp):
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

def event_criteria(window, mode, dna):
    """Compute the criteria applicable to a mode over one event window.
    Returns {criterion: {rating, metric, unit}}."""
    ac = window["AccelerationChassisCompensated"].to_numpy(float)
    wanted = MODE_CRITERIA.get(mode, ["disturbances", "disturbances_lf", "disturbances_hf"])
    dm = disturbance_metrics(ac)
    res = {}
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

# -----------------------------------------------------------------------------
# 6. TRIGGER ENGINE / OPERATION MODE DETECTION
# -----------------------------------------------------------------------------
def _segments(t, mask, min_dur=0.5, merge_gap=0.3, pre=0.0, post=0.0):
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
    if len(idxs) == 0:
        return []
    starts = [t[idxs[0]]]
    for i in idxs[1:]:
        if t[i] - starts[-1] > gap:
            starts.append(t[i])
    return [(s - pre, s + post) for s in starts]

def detect_events(df, enabled_modes, cfg):
    """Detects all enabled operation modes. Returns list of {mode, t_start, t_end}."""
    t = df["timestamp"].to_numpy(float)
    pedal = df["AcceleratorPedal"].to_numpy(float)
    ax = df["AccelerationChassisCompensated"].to_numpy(float)
    speed = df["VehicleSpeed"].to_numpy(float) if "VehicleSpeed" in df else np.zeros_like(t)
    brake = df["BrakePosition"].to_numpy(float) if "BrakePosition" in df else np.zeros_like(t)
    espeed = df["EngineSpeed"].to_numpy(float) if "EngineSpeed" in df else None
    gear = df["GearEngaged"].to_numpy(float) if "GearEngaged" in df else None
    prate = np.gradient(pedal, 1.0 / FS)
    events = []

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
        add("Drive away", _clusters(t, np.array(launch, dtype=int), gap=5.0, pre=0.5, post=4.0))
    if "Acceleration" in enabled_modes:
        add("Acceleration", _segments(t, (ax > 0.4) & (pedal > 10) & (speed > 5), min_dur=1.0, post=0.3))
    if "Constant speed" in enabled_modes and "VehicleSpeed" in df:
        sp = pd.Series(speed)
        std = sp.rolling(int(2 * FS), center=True, min_periods=int(FS)).std().to_numpy()
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
            st_idx = np.where((espeed[:-1] < 400) & (espeed[1:] >= 400))[0] + 1
            add("Engine start", _clusters(t, st_idx, gap=2.0, pre=0.3, post=1.5))
        if "Engine shut off" in enabled_modes:
            so_idx = np.where((espeed[:-1] >= 400) & (espeed[1:] < 400))[0] + 1
            add("Engine shut off", _clusters(t, so_idx, gap=2.0, pre=0.3, post=1.5))

    events.sort(key=lambda ev: ev["t_start"])
    return events

# -----------------------------------------------------------------------------
# 7. ASSESSMENT: events -> criteria -> mode DR -> overall DR
# -----------------------------------------------------------------------------
def assess(df, events, dna):
    """Builds the AVL-style rating tree. Returns (overall_dr, mode_results) where
    mode_results maps mode -> {dr, n_events, criteria:{crit:{rating,metric,unit,weight}}}."""
    by_mode = {}
    for ev in events:
        seg = df[(df["timestamp"] >= ev["t_start"]) & (df["timestamp"] <= ev["t_end"])]
        if len(seg) < int(0.3 * FS):
            continue
        crit = event_criteria(seg, ev["mode"], dna)
        if not crit:
            continue
        by_mode.setdefault(ev["mode"], []).append(crit)

    mode_results = {}
    for mode, occ in by_mode.items():
        # Aggregate each criterion across occurrences (extreme value), then across
        # criteria (criteria weights + extreme value) -> mode DR.
        crit_names = MODE_CRITERIA.get(mode, [])
        crit_summary = {}
        crit_ratings, crit_weights = [], []
        for c in crit_names:
            ratings = [o[c]["rating"] for o in occ if c in o]
            if not ratings:
                continue
            metrics = [o[c]["metric"] for o in occ if c in o]
            unit = next(o[c]["unit"] for o in occ if c in o)
            cr = aggregate_dr(ratings)
            crit_summary[c] = {"rating": cr, "metric": float(np.mean(metrics)), "unit": unit,
                               "weight": CRITERIA_META[c]["weight"], "label": CRITERIA_META[c]["label"]}
            crit_ratings.append(cr)
            crit_weights.append(CRITERIA_META[c]["weight"])
        mode_dr = aggregate_dr(crit_ratings, crit_weights)
        mode_results[mode] = {"dr": mode_dr, "n_events": len(occ), "criteria": crit_summary}

    mode_names = list(mode_results.keys())
    overall = aggregate_dr([mode_results[m]["dr"] for m in mode_names],
                           [MODE_WEIGHTS.get(m, 3) for m in mode_names])
    return overall, mode_results

# -----------------------------------------------------------------------------
# 8. SPECTRUM MAPPING (Welch PSD)
# -----------------------------------------------------------------------------
def compute_band_spectrum(signal_data, band, fs=FS):
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
    return {"freqs": freqs, "psd": psd, "band": band, "dominant_freq": float(bf[i]), "dominant_power": float(bp_[i])}

def interpret_surge_source(f):
    if f < 1.0:
        return "Low-frequency chugging (< 1.0 Hz) — engine combustion / lugging or driveline lash."
    if f < 1.5:
        return "Mid-band surge (1.0-1.5 Hz) — mixed powertrain source."
    return "Faster surge (> 1.5 Hz) — electric motor / e-drive controller damping loop."

def render_psd_block(container, signal_data, band, title, color, caption_fn=None):
    spec = compute_band_spectrum(signal_data, band)
    low, high = band
    if spec is None:
        container.info(f"{title}: slice too short to resolve a dominant frequency peak.")
        return
    dom = spec["dominant_freq"]
    container.markdown(f"**{title} — dominant frequency: `{dom:.2f} Hz`** "
                       f"(band {low:g}–{high:g} Hz, PSD peak {spec['dominant_power']:.3e} (m/s²)²/Hz)")
    if caption_fn is not None:
        container.caption(caption_fn(dom))
    mask = (spec["freqs"] >= low) & (spec["freqs"] <= high)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spec["freqs"][mask], y=spec["psd"][mask], name=f"{title} PSD",
                             line=dict(color=color), fill="tozeroy"))
    fig.add_vline(x=dom, line=dict(color="crimson", dash="dash"),
                  annotation_text=f"{dom:.2f} Hz", annotation_position="top")
    fig.update_layout(title=f"{title} PSD (Welch)", xaxis_title="Frequency (Hz)",
                      yaxis_title="PSD (m/s²)²/Hz", height=320)
    container.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 9. REPORT EXPORT
# -----------------------------------------------------------------------------
def build_summary_report(meta, mode_rows):
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=2 * cm, bottomMargin=2 * cm)
        styles = getSampleStyleSheet()
        head = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white])]
        story = [Paragraph("AVL-DRIVE-Style Executive Drivability Summary", styles["Title"]),
                 Spacer(1, 6), Paragraph(f"Generated: {generated}", styles["Normal"]),
                 Spacer(1, 12), Paragraph("Configuration &amp; File Metadata", styles["Heading2"])]
        mt = Table([["Field", "Value"]] + [[k, str(v)] for k, v in meta.items()], colWidths=[6 * cm, 9 * cm])
        mt.setStyle(TableStyle(head))
        story += [mt, Spacer(1, 16), Paragraph("Operation-Mode DRIVE Ratings (1-10)", styles["Heading2"])]
        st_tbl = Table([["Operation Mode", "DR", "Events", "Weight"]] + mode_rows, colWidths=[7 * cm, 3 * cm, 3 * cm, 2 * cm])
        st_tbl.setStyle(TableStyle(head))
        story.append(st_tbl)
        doc.build(story)
        return buf.getvalue(), "application/pdf", "pdf"
    except Exception:
        lines = ["AVL-DRIVE-STYLE EXECUTIVE DRIVABILITY SUMMARY", "=" * 46, f"Generated: {generated}",
                 "", "CONFIGURATION & FILE METADATA", "-" * 46]
        lines += [f"{k:<26}: {v}" for k, v in meta.items()]
        lines += ["", "OPERATION-MODE DRIVE RATINGS (1-10)", "-" * 46]
        lines += [f"{r[0]:<26}: DR {r[1]:>5}  events {r[2]:>3}  weight {r[3]}" for r in mode_rows]
        return ("\n".join(lines) + "\n").encode("utf-8"), "text/plain", "txt"

def resolve_channels(mdf):
    available = list(mdf.channels_db.keys())
    lower = {name.lower(): name for name in available}
    found = {}
    for logical, cands in CHANNEL_CANDIDATES.items():
        for c in cands:
            if c in mdf.channels_db:
                found[logical] = c
                break
            if c.lower() in lower:
                found[logical] = lower[c.lower()]
                break
    return found

# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="AVL-DRIVE-Style Drivability Lab", layout="wide")
st.title("⚙️ AVL-DRIVE-Style Objective Drivability Assessment")
st.caption("Independent re-implementation of AVL-DRIVE™ 4.6 SR1 methodology.")

# --- HOME-PAGE TRANSMISSION SELECTOR (re-assigns the whole tool) --------------
st.markdown("### 1️⃣ Select transmission / powertrain architecture")
transmission = st.selectbox(
    "The tool re-assigns its operation modes, criteria, propulsion and relevant "
    "channels to the selected architecture.",
    options=list(TRANSMISSION_CONFIG.keys()),
    format_func=lambda k: f"{k} — {TRANSMISSION_CONFIG[k]['label']}",
    key="transmission")
tx = TRANSMISSION_CONFIG[transmission]
enabled_modes = tx["modes"]
propulsion = tx["propulsion"]

a, b, c = st.columns(3)
a.metric("Propulsion", propulsion)
b.metric("Gearbox", tx["gearbox"])
c.metric("Operation modes", len(enabled_modes))
with st.expander(f"🔧 Assigned configuration for **{transmission} — {tx['label']}**", expanded=True):
    cc1, cc2, cc3 = st.columns(3)
    cc1.markdown("**Enabled operation modes**\n\n" + "\n".join(f"- {m}" for m in enabled_modes))
    cc2.markdown("**Powertrain features**\n\n" + "\n".join(f"- {f}" for f in tx["features"]))
    cc3.markdown("**Relevant DRIVE channels**\n\n" + "\n".join(f"- {ch}" for ch in relevant_channels(transmission)))

# --- SIDEBAR: configuration data + brand-DNA + upload -------------------------
st.sidebar.header("🔧 Configuration Mode")
st.sidebar.caption(f"Active architecture: **{transmission} — {tx['label']}** ({propulsion})")
brand = st.sidebar.selectbox("Brand-DNA target profile", list(BRAND_DNA.keys()),
                             index=list(BRAND_DNA.keys()).index("Eco EV" if propulsion == "BEV" else "Luxury Sedan"))
dna = BRAND_DNA[brand]
drive_layout = st.sidebar.selectbox("Driven axle", ["FWD", "RWD", "AWD"], index=0)
st.sidebar.markdown("**Vehicle specific data**")
mass = st.sidebar.number_input("Vehicle mass [kg]", 500.0, 4000.0, 1600.0, 10.0)
wheel_radius = st.sidebar.number_input("Dynamic wheel radius [m]", 0.20, 0.60, 0.32, 0.01)
with st.sidebar.expander("Road-load coefficients (F = A0 + B0·v + C0·v²)"):
    A0 = st.number_input("A0 [N]", 0.0, 1000.0, 120.0, 5.0)
    B0 = st.number_input("B0 [N/(km/h)]", 0.0, 50.0, 0.0, 0.5)
    C0 = st.number_input("C0 [N/(km/h)²]", 0.0, 1.0, 0.045, 0.005, format="%.3f")
clutch_points = None
if transmission == "MT":
    with st.sidebar.expander("Clutch pedal operating points [%]"):
        clutch_points = {
            "disengaged": st.number_input("Disengaged", 0.0, 100.0, 80.0, 1.0),
            "touchpoint": st.number_input("Touchpoint", 0.0, 100.0, 45.0, 1.0),
            "engaged": st.number_input("Engaged", 0.0, 100.0, 20.0, 1.0)}

cfg = {"transmission": transmission, "propulsion": propulsion, "drive": drive_layout,
       "brand": brand, "mass": mass, "wheel_radius": wheel_radius,
       "A0": A0, "B0": B0, "C0": C0, "clutch": clutch_points}

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Import measurement (.mf4, .dat)", type=["mf4", "dat"])

st.markdown("### 2️⃣ Import a measurement & assess")
if uploaded_file is None:
    st.info("⬅️ Import a `.mf4` / `.dat` measurement in the sidebar to run the assessment "
            f"for the **{transmission}** architecture.")
else:
    temp_filename = f"temp_{uploaded_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        with st.spinner("Importing channels and building the 100 Hz analysis grid..."):
            mdf = MDF(temp_filename)
            found = resolve_channels(mdf)
            missing = [c for c in REQUIRED_LOGICAL if c not in found]
            if missing:
                st.error("Missing required DRIVE channels: " + ", ".join(missing) +
                         f". Detected: {sorted(found.keys())}")
                st.stop()
            mdf_res = mdf.filter(list(found.values())).resample(raster=1.0 / FS)
            df = mdf_res.to_dataframe().rename(columns={raw: lg for lg, raw in found.items()})
            df = df.reset_index().rename(columns={"index": "timestamp", "time": "timestamp"})
            if "timestamp" not in df.columns:
                df = df.rename(columns={df.columns[0]: "timestamp"})
            df = build_calculated_channels(df, cfg)

        st.sidebar.success(f"Imported {len(found)} DRIVE channels.")
        with st.sidebar.expander("Resolved channel mapping"):
            st.write(found)

        events = detect_events(df, enabled_modes, cfg)
        overall, mode_results = assess(df, events, dna)

        # ---- Scoreboard ------------------------------------------------------
        st.markdown(f"## 🏁 AVL-DRIVE Rating — *{brand}* on **{transmission} ({propulsion})**")
        st.metric("Overall AVL-DRIVE Rating (weighted + extreme-value)",
                  f"{round(overall, 1) if overall else '—'} / 10")

        if mode_results:
            st.markdown("### 📊 Main Operation-Mode DRIVE Ratings")
            rated = [(m, r) for m, r in mode_results.items() if r["dr"] is not None]
            rated.sort(key=lambda x: x[1]["dr"])
            cols = st.columns(min(4, len(rated)) or 1)
            for i, (m, r) in enumerate(rated):
                cols[i % len(cols)].metric(m, f"{round(r['dr'], 1)} / 10",
                                           f"{r['n_events']} event(s) · w{MODE_WEIGHTS.get(m, 3)}")

            om_df = pd.DataFrame([
                {"Operation Mode": m, "DRIVE Rating": round(r["dr"], 1),
                 "Events": r["n_events"], "Mode Weight": MODE_WEIGHTS.get(m, 3)}
                for m, r in rated])
            st.dataframe(om_df, use_container_width=True, hide_index=True)

            st.markdown("### 🔬 Criteria Breakdown per Operation Mode")
            sel_mode = st.selectbox("Operation mode", [m for m, _ in rated])
            crit = mode_results[sel_mode]["criteria"]
            cb_df = pd.DataFrame([
                {"Criterion": v["label"], "DRIVE Rating": round(v["rating"], 1),
                 "Metric": f"{v['metric']:.3f} {v['unit']}", "Weight": v["weight"]}
                for v in crit.values()])
            st.dataframe(cb_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No enabled operation modes were triggered in this measurement.")

        # ---- Operation-mode timeline ----------------------------------------
        st.markdown("### 🧭 Detected Operation-Mode Timeline")
        if events:
            st.dataframe(pd.DataFrame([
                {"Operation Mode": ev["mode"], "Start [s]": round(ev["t_start"], 2),
                 "End [s]": round(ev["t_end"], 2), "Duration [s]": round(ev["t_end"] - ev["t_start"], 2)}
                for ev in events]), use_container_width=True, hide_index=True)
        else:
            st.info("No operation-mode events triggered.")

        # ---- Report ----------------------------------------------------------
        duration_s = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) else 0.0
        meta = {"Source File": uploaded_file.name,
                "Architecture": f"{transmission} — {tx['label']}",
                "Propulsion / Gearbox": f"{propulsion} / {tx['gearbox']}",
                "Brand-DNA Target": brand,
                "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Log Duration (s)": f"{duration_s:.1f}", "Samples (100 Hz)": len(df),
                "Overall AVL-DRIVE Rating": f"{round(overall, 1) if overall else '—'} / 10",
                "Operation Modes Triggered": len(mode_results)}
        mode_rows = [[m, round(r["dr"], 1) if r["dr"] else "—", r["n_events"], MODE_WEIGHTS.get(m, 3)]
                     for m, r in mode_results.items()]
        rb, mime, ext = build_summary_report(meta, mode_rows)
        st.sidebar.markdown("---")
        st.sidebar.download_button("📥 Export AVL-Style Executive Summary Report", data=rb,
                                   file_name=f"drivability_{transmission}_{os.path.splitext(uploaded_file.name)[0]}.{ext}",
                                   mime=mime, use_container_width=True)

        # ---- Diagnostic tabs -------------------------------------------------
        st.markdown("### 🎛️ Diagnostic Analytics")
        accel_comp = df["AccelerationChassisCompensated"]
        dm = disturbance_metrics(accel_comp.to_numpy(float))
        tab1, tab2, tab3 = st.tabs(["Disturbance Traces", "Frequency Mapping (PSD)", "Transient Microscope"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=dm["lf"][1], name="Disturbances LF (2–10 Hz)", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df["timestamp"], y=dm["hf"][1], name="Disturbances HF (>10 Hz)", line=dict(color="crimson")))
            fig.update_layout(title="Isolated Acceleration Disturbances (AccelerationChassisCompensated)",
                              xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)", height=420)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            c1, c2 = st.columns(2)
            render_psd_block(c1, LP(accel_comp.to_numpy(float), 2.0), (0.3, 2.0), "Surge (<2 Hz)", "orange", interpret_surge_source)
            render_psd_block(c2, dm["lf"][1], (2.0, 10.0), "Disturbances LF (2–10 Hz)", "blue")
            render_psd_block(st, dm["hf"][1], (10.0, 50.0), "Disturbances HF (10–50 Hz)", "crimson")
        with tab3:
            tip_events = [ev for ev in events if ev["mode"] in ("Tip in", "Drive away")]
            if tip_events:
                labels = [f"{ev['mode']} @ {ev['t_start'] + 0.5:.2f}s" for ev in tip_events]
                sel = st.selectbox("Select transient event", list(range(len(labels))), format_func=lambda i: labels[i])
                ev = tip_events[sel]
                w = df[(df["timestamp"] >= ev["t_start"]) & (df["timestamp"] <= ev["t_end"] + 1.0)]
                ft = go.Figure()
                ft.add_trace(go.Scatter(x=w["timestamp"], y=w["AcceleratorPedal"], name="AcceleratorPedal (%)",
                                        line=dict(color="green"), yaxis="y2"))
                ft.add_trace(go.Scatter(x=w["timestamp"], y=SMO(w["AccelerationChassisCompensated"].to_numpy(float), 20),
                                        name="AccelerationChassis_SMO(20) (m/s²)", line=dict(color="black", width=2)))
                ft.update_layout(title="Pedal Tip-In vs Acceleration Response", xaxis_title="Time (s)",
                                 yaxis=dict(title="Acceleration (m/s²)"),
                                 yaxis2=dict(title="Pedal (%)", overlaying="y", side="right", range=[0, 100]), height=450)
                st.plotly_chart(ft, use_container_width=True)
            else:
                st.info("No transient (Tip-in / Drive-away) events available for microscope view.")

    except Exception as e:
        st.error(f"Execution Error: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
