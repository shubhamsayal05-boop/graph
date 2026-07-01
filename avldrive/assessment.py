"""DRIVE-Rating aggregation: criteria -> operation mode -> overall.

Uses the AVL weight tree (criteria and mode weights) plus extreme-value
weighting: from a subjective point of view, a few very bad occurrences carry
disproportionately more impact than many good ones.
"""
from __future__ import annotations

import numpy as np

from .config import CRITERIA_META, FS, MODE_CRITERIA, MODE_WEIGHTS
from .criteria import event_criteria


def aggregate_dr(ratings, weights=None, extreme_p: float = 2.0):
    """Combine child ratings into a parent DRIVE Rating.

    ``weights`` are the static weight-tree weights; extreme-value weighting adds
    an additional ``(11 - rating) ** extreme_p`` factor so that worse ratings
    dominate. Returns ``None`` if there are no ratings.
    """
    pairs = [(r, (1.0 if weights is None else w))
             for r, w in zip(ratings, [None] * len(ratings) if weights is None else weights)
             if r is not None]
    if not pairs:
        return None
    r = np.asarray([p[0] for p in pairs], dtype=float)
    w = np.asarray([p[1] for p in pairs], dtype=float)
    ev = np.power(np.clip(11.0 - r, 1e-6, None), extreme_p)
    W = w * ev
    if W.sum() <= 0:
        return float(np.mean(r))
    return float(np.sum(W * r) / np.sum(W))


def assess(df, events, dna: dict, mode_weights=None, criteria_weights=None):
    """Build the rating tree.

    Returns ``(overall_dr, mode_results)`` where ``mode_results`` maps
    ``mode -> {dr, n_events, criteria:{crit:{rating, metric, unit, weight, label}}}``.
    """
    mode_weights = mode_weights or MODE_WEIGHTS
    criteria_weights = criteria_weights or {k: v["weight"] for k, v in CRITERIA_META.items()}

    by_mode: dict[str, list] = {}
    for ev in events:
        seg = df[(df["timestamp"] >= ev["t_start"]) & (df["timestamp"] <= ev["t_end"])]
        if len(seg) < int(0.3 * FS):
            continue
        crit = event_criteria(seg, ev["mode"], dna)
        if crit:
            by_mode.setdefault(ev["mode"], []).append(crit)

    mode_results: dict[str, dict] = {}
    for mode, occ in by_mode.items():
        crit_summary, crit_ratings, crit_weights = {}, [], []
        for c in MODE_CRITERIA.get(mode, []):
            ratings = [o[c]["rating"] for o in occ if c in o]
            if not ratings:
                continue
            metrics = [o[c]["metric"] for o in occ if c in o]
            unit = next(o[c]["unit"] for o in occ if c in o)
            weight = criteria_weights.get(c, CRITERIA_META[c]["weight"])
            cr = aggregate_dr(ratings)
            crit_summary[c] = {"rating": cr, "metric": float(np.mean(metrics)), "unit": unit,
                               "weight": weight, "label": CRITERIA_META[c]["label"]}
            crit_ratings.append(cr)
            crit_weights.append(weight)
        mode_dr = aggregate_dr(crit_ratings, crit_weights)
        mode_results[mode] = {"dr": mode_dr, "n_events": len(occ), "criteria": crit_summary}

    modes = list(mode_results.keys())
    overall = aggregate_dr([mode_results[m]["dr"] for m in modes],
                           [mode_weights.get(m, 3) for m in modes])
    return overall, mode_results
