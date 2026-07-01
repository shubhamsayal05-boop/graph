"""Drivability verification: acceptance specs and PASS/FAIL sign-off gate.

Mirrors AVL-DRIVE's use as an "acceptance tool for quality assurance of
attributes (milestone planning)": define target DRIVE Ratings and check a
measurement against them, producing a verdict, a requirement table and a
worst-event issue log.
"""
from __future__ import annotations

# Ready-made target specs for common development gates.
VERIFICATION_PRESETS: dict[str, dict] = {
    "Production sign-off": {"overall_min": 8.0, "mode_min": 7.5, "criterion_min": 7.0},
    "Development milestone": {"overall_min": 7.0, "mode_min": 6.0, "criterion_min": 5.5},
    "Prototype baseline": {"overall_min": 6.0, "mode_min": 5.0, "criterion_min": 4.0},
}


def _item(requirement, actual, target, level):
    ok = actual is not None and actual >= target
    return {"requirement": requirement, "level": level,
            "actual": None if actual is None else round(actual, 1),
            "target": target, "pass": bool(ok)}


def verify(overall, mode_results: dict, spec: dict):
    """Evaluate a measurement against an acceptance spec.

    Returns ``{passed, items, n_fail, n_checks}`` where ``items`` covers the
    overall rating, each operation mode and each criterion.
    """
    items = [_item("Overall AVL-DRIVE Rating", overall, spec["overall_min"], "overall")]
    for m, r in mode_results.items():
        items.append(_item(f"Mode · {m}", r["dr"], spec["mode_min"], "mode"))
        for c, v in r["criteria"].items():
            items.append(_item(f"{m} · {v['label']}", v["rating"], spec["criterion_min"], "criterion"))
    checks = [i for i in items if i["actual"] is not None]
    n_fail = sum(1 for i in checks if not i["pass"])
    return {"passed": n_fail == 0 and len(checks) > 0, "items": items,
            "n_fail": n_fail, "n_checks": len(checks)}


def issue_log(df, events, dna, criterion_min: float = 5.5, top_n: int = 20):
    """Worst individual events below the criterion threshold, for inspection.

    Returns a list sorted by severity (lowest event DR first) with the timestamp
    and the driving criterion, so an engineer can jump straight to the problem.
    """
    from .assessment import aggregate_dr
    from .criteria import event_criteria

    rows = []
    for ev in events:
        seg = df[(df["timestamp"] >= ev["t_start"]) & (df["timestamp"] <= ev["t_end"])]
        crit = event_criteria(seg, ev["mode"], dna)
        if not crit:
            continue
        ev_dr = aggregate_dr([v["rating"] for v in crit.values()])
        if ev_dr is None or ev_dr >= criterion_min:
            continue
        worst_c = min(crit.items(), key=lambda kv: kv[1]["rating"])
        rows.append({
            "mode": ev["mode"], "t_start": round(ev["t_start"], 2),
            "t_end": round(ev["t_end"], 2), "event_dr": round(ev_dr, 1),
            "worst_criterion": worst_c[1] if False else worst_c[0],
            "worst_rating": round(worst_c[1]["rating"], 1),
        })
    rows.sort(key=lambda r: r["event_dr"])
    return rows[:top_n]
