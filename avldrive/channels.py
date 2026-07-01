"""Channel resolution and AVL calculated channels (Function Description §1.3)."""
from __future__ import annotations

import numpy as np

from .config import CHANNEL_CANDIDATES, FS, G
from .dsp import LP


def resolve_channels(mdf) -> dict[str, str]:
    """Map available raw channels to AVL logical channels using candidate lists."""
    available = list(mdf.channels_db.keys())
    lower = {name.lower(): name for name in available}
    found: dict[str, str] = {}
    for logical, cands in CHANNEL_CANDIDATES.items():
        for c in cands:
            if c in mdf.channels_db:
                found[logical] = c
                break
            if c.lower() in lower:
                found[logical] = lower[c.lower()]
                break
    return found


def build_calculated_channels(df, cfg: dict):
    """Derive AVL calculated channels from the imported raw channels.

    ``cfg`` is a dict-like with keys: mass, wheel_radius, A0, B0, C0, propulsion,
    drive.
    """
    ax = df["AccelerationChassis"].to_numpy(dtype=float)
    # Chassis acceleration with the slow road-gradient / gravity drift removed.
    df["AccelerationChassisCompensated"] = ax - LP(ax, 0.3)

    if "VehicleSpeed" in df:
        v_ms = df["VehicleSpeed"].to_numpy(dtype=float) / 3.6
        dvdt = np.gradient(v_ms, 1.0 / FS)
        grade = np.clip((ax - dvdt) / G, -0.4, 0.4)
        df["RoadGradient"] = np.degrees(np.arcsin(grade))

        m = cfg["mass"]
        spd = df["VehicleSpeed"].to_numpy(dtype=float)
        f_res = cfg["A0"] + cfg["B0"] * spd + cfg["C0"] * spd ** 2
        f_grav = m * G * np.sin(np.radians(df["RoadGradient"].to_numpy(float)))
        km = 1.05  # rotational-mass factor
        f_tractive = m * km * dvdt + f_res + f_grav
        df["TractiveForce"] = f_tractive
        df["WheelTorque"] = f_tractive * cfg["wheel_radius"]

        if "EngineSpeed" in df and cfg.get("propulsion") != "BEV":
            eng_w = df["EngineSpeed"].to_numpy(float) * 2.0 * np.pi / 60.0
            wheel_w = np.where(v_ms > 0.1, v_ms / cfg["wheel_radius"], np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                i_tot = eng_w / wheel_w
                df["EngineTorqueEstimated"] = np.where(
                    np.isfinite(i_tot) & (np.abs(i_tot) > 0.1),
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
