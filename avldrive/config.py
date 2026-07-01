"""Configuration constants for the AVL-DRIVE-style drivability assessment.

Terminology, channels, operation modes, criteria and the DRIVE-Rating weight
tree follow the AVL-DRIVE(TM) 4.6 SR1 Function Descriptions (AT/CVT/DCT/DHT) and
Product Guide. This is an independent re-implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# Analysis grid: AVL-DRIVE uses 100 Hz (Nyquist 50 Hz).
FS: float = 100.0
G: float = 9.81

# -----------------------------------------------------------------------------
# Transmission / powertrain architectures
# -----------------------------------------------------------------------------
TRANSMISSION_CONFIG: dict[str, dict] = {
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
        "gearbox": "Power-split",
        "features": ["Electric launch", "Sailing", "Recuperation", "Engine start/stop"],
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
MODE_CRITERIA: dict[str, list[str]] = {
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

# Criterion metadata; ``dna_key`` selects the brand-DNA sensitivity to apply.
CRITERIA_META: dict[str, dict] = {
    "disturbances": {"label": "Acceleration disturbances (2-50 Hz)", "weight": 4, "unit": "m/s²", "dna_key": "disturbances"},
    "disturbances_lf": {"label": "Disturbances LF (2-10 Hz)", "weight": 4, "unit": "m/s²", "dna_key": "lf"},
    "disturbances_hf": {"label": "Disturbances HF (>10 Hz)", "weight": 3, "unit": "m/s²", "dna_key": "hf"},
    "crest_factor": {"label": "Crest factor (HF)", "weight": 2, "unit": "-", "dna_key": "crest"},
    "correlation": {"label": "Surge / correlation (<2 Hz)", "weight": 3, "unit": "r", "dna_key": "correlation"},
    "response_delay": {"label": "Response delay", "weight": 5, "unit": "s", "dna_key": "delay"},
    "shift_shock": {"label": "Shift shock", "weight": 4, "unit": "m/s²", "dna_key": "hf"},
}

# Default main-operation-mode weights (AVL weight tree, 1-5).
MODE_WEIGHTS: dict[str, int] = {
    "Tip in": 5, "Drive away": 5, "Tip out": 4, "Acceleration": 4, "Gear shift": 4,
    "Constant speed": 3, "Deceleration": 3, "Recuperation": 3, "Sailing": 2,
    "Idle": 2, "Engine start": 2, "Engine shut off": 2, "Vehicle stationary": 1,
    "Maneuvering": 2,
}

# Brand-DNA targets: per-criterion scoring sensitivity (higher = stricter).
BRAND_DNA: dict[str, dict] = {
    "Luxury Sedan": {"disturbances": 3.0, "lf": 3.5, "hf": 2.5, "crest": 0.7, "correlation": 5.0, "delay": 12.0},
    "Sports Car": {"disturbances": 4.5, "lf": 5.0, "hf": 4.0, "crest": 1.0, "correlation": 7.0, "delay": 26.0},
    "Eco EV": {"disturbances": 2.5, "lf": 3.0, "hf": 2.5, "crest": 0.6, "correlation": 4.0, "delay": 6.0},
}

# AVL DRIVE channels: logical name -> candidate raw channel names (first present wins).
CHANNEL_CANDIDATES: dict[str, list[str]] = {
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


def relevant_channels(transmission: str) -> list[str]:
    """Channels relevant to a transmission (for the assignment summary)."""
    cfg = TRANSMISSION_CONFIG[transmission]
    base = ["AccelerationChassis", "AccelerationVertical", "AcceleratorPedal",
            "BrakePosition", "VehicleSpeed"]
    if cfg["propulsion"] != "BEV":
        base.append("EngineSpeed")
    if transmission in ("AT", "CVT"):
        base += ["TurbineSpeed", "TCC_State"]
    if cfg["gearbox"] in ("Stepped", "Manual"):
        base += ["GearEngaged", "SelectorLeverDMU"]
    if transmission == "MT":
        base.append("ClutchPedal")
    base += ["WheelSpeed_FL", "WheelSpeed_FR", "WheelSpeed_RL", "WheelSpeed_RR"]
    return base


@dataclass
class VehicleConfig:
    """Configuration-Mode vehicle data feeding calculated channels."""
    transmission: str = "AT"
    propulsion: str = "ICE"
    drive: str = "FWD"
    brand: str = "Luxury Sedan"
    mass: float = 1600.0            # kg
    wheel_radius: float = 0.32      # m (dynamic)
    A0: float = 120.0               # N          (road-load constant)
    B0: float = 0.0                 # N/(km/h)   (road-load linear)
    C0: float = 0.045               # N/(km/h)^2 (road-load quadratic)
    clutch: Optional[dict] = None   # MT clutch operating points [%]

    def as_dict(self) -> dict:
        return asdict(self)


def default_mode_weights() -> dict[str, int]:
    return dict(MODE_WEIGHTS)


def default_criteria_weights() -> dict[str, int]:
    return {k: v["weight"] for k, v in CRITERIA_META.items()}
