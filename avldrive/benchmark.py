"""Benchmark library: store reference (e.g. best-in-class) assessments and
compare measurements against them.

References are plain dicts so they serialize to JSON for a persistent benchmark
library, matching AVL-DRIVE's benchmark-library workflow.
"""
from __future__ import annotations


def result_to_reference(name: str, transmission: str, brand: str, result) -> dict:
    """Snapshot an AssessmentResult into a serializable reference record."""
    return {
        "name": name,
        "transmission": transmission,
        "brand": brand,
        "overall": None if result.overall is None else round(result.overall, 2),
        "mode_drs": {m: (None if r["dr"] is None else round(r["dr"], 2))
                     for m, r in result.mode_results.items()},
    }


def library_to_json(library: list[dict]) -> str:
    import json
    return json.dumps({"version": 1, "references": library}, indent=2)


def library_from_json(text: str) -> list[dict]:
    import json
    data = json.loads(text)
    refs = data.get("references", data if isinstance(data, list) else [])
    return [r for r in refs if isinstance(r, dict) and "name" in r]


def ranking(references: list[dict]):
    """Rank references by overall DRIVE Rating (best first)."""
    ranked = sorted([r for r in references if r.get("overall") is not None],
                    key=lambda r: r["overall"], reverse=True)
    return [{"Rank": i + 1, "Reference": r["name"], "Architecture": r["transmission"],
             "Brand-DNA": r["brand"], "Overall DR": r["overall"]}
            for i, r in enumerate(ranked)]


def fingerprint(references: list[dict]):
    """Build radar-fingerprint data across the union of operation modes.

    Returns ``(modes, series)`` where ``series`` is ``{name: [dr per mode]}``
    (missing modes -> None), suitable for a plotly radar chart.
    """
    modes = sorted({m for r in references for m in r.get("mode_drs", {})})
    series = {r["name"]: [r.get("mode_drs", {}).get(m) for m in modes] for r in references}
    return modes, series
