"""Per-transmission operation-mode template catalog.

AVL-DRIVE ships a specific set of ``.ect`` event-configuration templates for each
transmission family (e.g. ``AT_V46_TITO_AD.ect``, ``MT_V46_Gearshift.ect``).
Selecting a transmission loads *that* family's templates, which drive the
weight-tree editor and roll up into the operation-mode weights used for the
DRIVE-Rating aggregation.

The built-in catalog reproduces the template families supplied in the repository
for AT/CVT/MT/BEV; architectures without provided ``.ect`` files (eCVT/DCT/DHT)
are synthesized from their operation modes. A directory of real ``.ect`` files
can be scanned at runtime to rebuild the catalog (:func:`load_catalog_from_dir`).
"""
from __future__ import annotations

import os
import re

from .config import MODE_WEIGHTS, TRANSMISSION_CONFIG, default_mode_weights

# Maneuver token (from the .ect filename) -> (base operation mode, human label).
MANEUVER_MAP: dict[str, tuple[str, str]] = {
    "Deceleration": ("Deceleration", "Deceleration"),
    "DriveAway": ("Drive away", "Drive away"),
    "DriveAway_ESS": ("Drive away", "Drive away – engine start/stop"),
    "DriveAway_RL": ("Drive away", "Drive away – rolling start"),
    "Garage_Shift": ("Gear shift", "Garage shift (P-R-N-D)"),
    "Maneuvering": ("Maneuvering", "Maneuvering (low speed)"),
    "Manual_Power_Off_Down": ("Gear shift", "Manual downshift – power off"),
    "Manual_Power_On_Down": ("Gear shift", "Manual downshift – power on"),
    "Manual_Power_On_Up": ("Gear shift", "Manual upshift – power on"),
    "Gearshift": ("Gear shift", "Gear shift"),
    "ESS": ("Engine start", "Engine start/stop"),
    "TITO_ACS": ("Tip in", "Tip-in/Tip-out – Adaptive Comfort/Sport"),
    "TITO_AD": ("Tip in", "Tip-in/Tip-out – Agility Drive"),
    "TITO_ACS_Brazil": ("Tip in", "Tip-in/Tip-out – ACS (Brazil)"),
    "TITO_AD_Brazil": ("Tip in", "Tip-in/Tip-out – AD (Brazil)"),
    "TIPin_AD": ("Tip in", "Tip-in – Agility Drive"),
    "TipIn_Agility_Light": ("Tip in", "Tip-in – Agility (light)"),
    "TipIn_FWON_Allspeeds": ("Tip in", "Tip-in – full-warm engine-on, all speeds"),
    "TipIn_FWON_COM_Allspeeds": ("Tip in", "Tip-in – FWON comfort, all speeds"),
    "TIPout_AA": ("Tip out", "Tip-out – anti-alarm"),
}

# Template families as supplied in the repository (.ect filenames without prefix).
RAW_TEMPLATES: dict[str, list[str]] = {
    "AT": ["Deceleration", "DriveAway_ESS", "DriveAway_RL", "Garage_Shift", "Maneuvering",
           "Manual_Power_Off_Down", "Manual_Power_On_Down", "Manual_Power_On_Up",
           "TITO_ACS", "TITO_AD", "TipIn_Agility_Light", "TipIn_FWON_Allspeeds",
           "TipIn_FWON_COM_Allspeeds"],
    "CVT": ["Deceleration", "DriveAway_ESS", "DriveAway_RL", "Garage_Shift", "Maneuvering",
            "Manual_Power_Off_Down", "Manual_Power_On_Down", "Manual_Power_On_Up",
            "TITO_ACS", "TITO_AD", "TipIn_Agility_Light"],
    "MT": ["DriveAway", "ESS", "Gearshift", "TIPin_AD", "TIPout_AA", "TITO_ACS",
           "TITO_ACS_Brazil", "TITO_AD", "TITO_AD_Brazil", "TipIn_Agility_Light"],
    "BEV": ["Deceleration", "DriveAway", "DriveAway_RL", "Garage_Shift", "Maneuvering",
            "TIPin_AD", "TIPout_AA", "TITO_ACS", "TITO_AD", "TipIn_Agility_Light"],
}


def _entry(transmission: str, token: str) -> dict:
    mode, label = MANEUVER_MAP.get(token, ("Maneuvering", token.replace("_", " ")))
    return {"template": f"{transmission}_V46_{token}", "token": token, "label": label,
            "mode": mode, "weight": int(MODE_WEIGHTS.get(mode, 3))}


def _build_catalog() -> dict[str, list[dict]]:
    catalog: dict[str, list[dict]] = {}
    for tx, tokens in RAW_TEMPLATES.items():
        catalog[tx] = [_entry(tx, tok) for tok in tokens]
    # Synthesize catalogs for architectures without supplied .ect files.
    for tx, cfg in TRANSMISSION_CONFIG.items():
        if tx in catalog:
            continue
        catalog[tx] = [{"template": f"{tx}_{m.replace(' ', '_')}", "token": m, "label": m,
                        "mode": m, "weight": int(MODE_WEIGHTS.get(m, 3))}
                       for m in cfg["modes"]]
    return catalog


TEMPLATE_CATALOG: dict[str, list[dict]] = _build_catalog()


def templates_for(transmission: str) -> list[dict]:
    """Operation-mode templates loaded for the selected transmission."""
    return TEMPLATE_CATALOG.get(transmission, [])


def default_template_weights(transmission: str) -> dict[str, int]:
    return {e["template"]: e["weight"] for e in templates_for(transmission)}


def mode_weights_from_templates(transmission: str, template_weights: dict | None = None) -> dict[str, int]:
    """Roll per-template weights up to base operation-mode weights.

    Each base operation mode takes the **strongest** (max) weight among its
    templates; modes without a template keep their AVL default weight.
    """
    template_weights = template_weights or {}
    weights = default_mode_weights()
    grouped: dict[str, list[int]] = {}
    for e in templates_for(transmission):
        w = int(template_weights.get(e["template"], e["weight"]))
        grouped.setdefault(e["mode"], []).append(w)
    for mode, ws in grouped.items():
        weights[mode] = max(ws)
    return weights


def parse_ect_filename(filename: str):
    """Parse ``<TX>_V<ver>_<token>.ect`` -> ``(transmission, token)`` or ``None``."""
    stem = os.path.splitext(os.path.basename(filename))[0].strip()
    m = re.match(r"([A-Za-z]+)_V\d+_(.+)", stem)
    if not m:
        return None
    return m.group(1), m.group(2)


def load_catalog_from_dir(path: str) -> dict[str, list[dict]]:
    """Scan a directory of ``.ect`` files and build a catalog from real filenames.

    Falls back to the maneuver map for labels/modes; unknown tokens map to the
    Maneuvering mode. Returns a catalog dict (does not mutate the built-in one).
    """
    catalog: dict[str, list[dict]] = {}
    if not os.path.isdir(path):
        return catalog
    for name in sorted(os.listdir(path)):
        if not name.lower().endswith(".ect"):
            continue
        parsed = parse_ect_filename(name)
        if not parsed:
            continue
        tx, token = parsed
        catalog.setdefault(tx, []).append(_entry(tx, token))
    return catalog
