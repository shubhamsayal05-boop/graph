"""A/B calibration comparison and regression detection.

Compares two assessments (e.g. calibration A vs B) at overall, operation-mode
and criterion level, flagging improvements and regressions beyond a threshold.
"""
from __future__ import annotations


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(b - a, 2)


def _verdict(delta, threshold):
    if delta is None:
        return "n/a"
    if delta >= threshold:
        return "improved"
    if delta <= -threshold:
        return "regressed"
    return "unchanged"


def compare_results(result_a, result_b, threshold: float = 0.5):
    """Compare two AssessmentResult objects (A = baseline, B = candidate).

    Returns a dict with the overall delta and per-mode / per-criterion rows,
    plus counts of regressions and improvements (mode level).
    """
    overall_delta = _delta(result_a.overall, result_b.overall)

    modes = sorted(set(result_a.mode_results) | set(result_b.mode_results))
    mode_rows, regressions, improvements = [], 0, 0
    for m in modes:
        a = result_a.mode_results.get(m, {}).get("dr")
        b = result_b.mode_results.get(m, {}).get("dr")
        d = _delta(a, b)
        v = _verdict(d, threshold)
        regressions += v == "regressed"
        improvements += v == "improved"
        mode_rows.append({"mode": m, "a": None if a is None else round(a, 1),
                          "b": None if b is None else round(b, 1), "delta": d, "verdict": v})

    crit_rows = []
    for m in modes:
        ca = result_a.mode_results.get(m, {}).get("criteria", {})
        cb = result_b.mode_results.get(m, {}).get("criteria", {})
        for c in sorted(set(ca) | set(cb)):
            a = ca.get(c, {}).get("rating")
            b = cb.get(c, {}).get("rating")
            d = _delta(a, b)
            label = (ca.get(c) or cb.get(c) or {}).get("label", c)
            crit_rows.append({"mode": m, "criterion": label,
                              "a": None if a is None else round(a, 1),
                              "b": None if b is None else round(b, 1),
                              "delta": d, "verdict": _verdict(d, threshold)})

    return {"overall_delta": overall_delta, "overall_verdict": _verdict(overall_delta, threshold),
            "mode_rows": mode_rows, "criterion_rows": crit_rows,
            "n_regressions": regressions, "n_improvements": improvements}
