"""Calibration Advisor: turn ratings into a prioritized, actionable work list.

Given an assessment, this module answers the calibration engineer's questions:
  * Which criterion, if improved, raises the overall DRIVE Rating the most?
  * How far is each criterion from its target, and by how much must the
    underlying physical metric change to get there?
  * What calibration levers typically address that criterion?
"""
from __future__ import annotations

import numpy as np

from .assessment import aggregate_dr
from .config import CRITERIA_META

# Root-cause -> calibration-lever knowledge base (independent engineering
# guidance, not AVL proprietary content).
CALIBRATION_KB: dict[str, str] = {
    "response_delay": (
        "Reduce powertrain dead-time: raise initial torque-ramp gain / pedal-to-"
        "torque responsiveness, cut request-path filtering lag, advance converter "
        "lock-up or clutch engagement, and check e-motor torque rise-time."),
    "correlation": (
        "Improve pedal↔acceleration tracking (surge/chugging < 2 Hz): linearize "
        "the pedal map, add/tune anti-surge (anti-lug) damping, and refine creep / "
        "lock-up control so acceleration follows driver intent."),
    "disturbances": (
        "Broadband disturbances (2–50 Hz): review overall torque smoothness, "
        "engine-mount tuning and combustion / e-drive torque delivery."),
    "disturbances_lf": (
        "Low-frequency driveline shuffle (2–10 Hz): add or retune active anti-jerk "
        "(lash) compensation, shape backlash traversal, and damp the driveline "
        "first torsional mode."),
    "disturbances_hf": (
        "High-frequency harshness (> 10 Hz): check combustion roughness / misfire, "
        "injection & ignition dithering, e-motor current ripple / PWM, and mount "
        "isolation."),
    "crest_factor": (
        "Occasional impacts (high crest factor): target lash impacts, tip-in/out "
        "steps and shift / clutch engagement shocks rather than steady-state noise."),
    "shift_shock": (
        "Gear-shift shock: refine clutch-to-clutch handover timing, apply torque "
        "reduction during the torque phase, and tune inertia-phase speed control."),
}


def _mode_dr_from_criteria(crit: dict) -> float | None:
    return aggregate_dr([v["rating"] for v in crit.values()],
                        [v["weight"] for v in crit.values()])


def improvement_opportunities(mode_results: dict, mode_weights: dict,
                              target: float = 9.0):
    """Rank calibration opportunities by their impact on the overall DRIVE Rating.

    For each under-target criterion, we lift its rating to ``target`` and
    recompute the overall DR through the weight tree; the resulting increase is
    that criterion's *improvement potential*. Returns ``(base_overall, opps)``
    with ``opps`` sorted by potential gain (largest first).
    """
    modes = list(mode_results.keys())
    mode_drs = {m: mode_results[m]["dr"] for m in modes}
    weights = [mode_weights.get(m, 3) for m in modes]
    base_overall = aggregate_dr([mode_drs[m] for m in modes], weights)

    opps = []
    for m in modes:
        crit = mode_results[m]["criteria"]
        for c, v in crit.items():
            if v["rating"] is None or v["rating"] >= target:
                continue
            lifted = {k: dict(val) for k, val in crit.items()}
            lifted[c]["rating"] = target
            new_mode_dr = _mode_dr_from_criteria(lifted)
            new_drs = dict(mode_drs)
            new_drs[m] = new_mode_dr
            new_overall = aggregate_dr([new_drs[x] for x in modes], weights)
            gain = (new_overall or 0.0) - (base_overall or 0.0)
            opps.append({
                "mode": m, "criterion": c, "label": v["label"],
                "current_rating": round(v["rating"], 1), "metric": v["metric"],
                "unit": v["unit"], "mode_weight": mode_weights.get(m, 3),
                "criterion_weight": v["weight"], "potential_gain": round(max(0.0, gain), 2),
                "hint": CALIBRATION_KB.get(c, "Review the relevant control calibration."),
            })
    opps.sort(key=lambda o: o["potential_gain"], reverse=True)
    return base_overall, opps


def target_gaps(mode_results: dict, target_dr: float = 8.0):
    """Per-criterion gap to a target DRIVE Rating (worst first)."""
    rows = []
    for m, r in mode_results.items():
        for c, v in r["criteria"].items():
            if v["rating"] is None:
                continue
            rows.append({"mode": m, "criterion": v["label"],
                         "rating": round(v["rating"], 1),
                         "gap_to_target": round(max(0.0, target_dr - v["rating"]), 1),
                         "metric": v["metric"], "unit": v["unit"]})
    rows.sort(key=lambda x: x["gap_to_target"], reverse=True)
    return rows
