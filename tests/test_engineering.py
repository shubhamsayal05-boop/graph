"""Tests for the calibration/verification/benchmark/compare features."""
from types import SimpleNamespace

from avldrive import (compare_results, fingerprint, improvement_opportunities,
                      issue_log, library_from_json, library_to_json, ranking,
                      result_to_reference, target_gaps, verify)


def _mode_results():
    return {
        "Tip in": {"dr": 5.0, "n_events": 2, "criteria": {
            "response_delay": {"rating": 4.0, "metric": 0.4, "unit": "s", "weight": 5, "label": "Response delay"},
            "disturbances_lf": {"rating": 6.0, "metric": 0.3, "unit": "m/s²", "weight": 4, "label": "Disturbances LF (2-10 Hz)"},
        }},
        "Constant speed": {"dr": 9.0, "n_events": 1, "criteria": {
            "disturbances_hf": {"rating": 9.0, "metric": 0.02, "unit": "m/s²", "weight": 3, "label": "Disturbances HF (>10 Hz)"},
        }},
    }


def _result(mr, overall):
    return SimpleNamespace(overall=overall, mode_results=mr, events=[])


def test_improvement_opportunities_ranks_high_impact_first():
    mr = _mode_results()
    base, opps = improvement_opportunities(mr, {"Tip in": 5, "Constant speed": 3})
    assert base is not None and opps
    # The worst, highest-weight criterion (response delay in Tip in) should top the list.
    assert opps[0]["criterion"] == "response_delay"
    assert opps[0]["potential_gain"] >= opps[-1]["potential_gain"]


def test_target_gaps_sorted_worst_first():
    gaps = target_gaps(_mode_results(), target_dr=8.0)
    assert gaps[0]["gap_to_target"] >= gaps[-1]["gap_to_target"]
    assert gaps[0]["gap_to_target"] > 0


def test_verify_pass_and_fail():
    mr = _mode_results()
    fail = verify(6.0, mr, {"overall_min": 8.0, "mode_min": 7.5, "criterion_min": 7.0})
    assert fail["passed"] is False and fail["n_fail"] > 0
    lenient = verify(9.5, {"Constant speed": mr["Constant speed"]},
                     {"overall_min": 5.0, "mode_min": 5.0, "criterion_min": 5.0})
    assert lenient["passed"] is True


def test_benchmark_reference_roundtrip_and_ranking():
    r = _result(_mode_results(), 6.2)
    ref = result_to_reference("Baseline A", "AT", "Luxury Sedan", r)
    assert ref["overall"] == 6.2 and "Tip in" in ref["mode_drs"]
    ref2 = result_to_reference("Cal B", "AT", "Luxury Sedan", _result(_mode_results(), 7.8))
    text = library_to_json([ref, ref2])
    back = library_from_json(text)
    assert len(back) == 2
    rk = ranking(back)
    assert rk[0]["Reference"] == "Cal B"  # higher overall ranks first
    modes, series = fingerprint(back)
    assert "Tip in" in modes and "Baseline A" in series


def test_compare_flags_regression_and_improvement():
    a = _result(_mode_results(), 6.0)
    better = _mode_results()
    better["Tip in"]["dr"] = 7.5  # improved
    worse = _mode_results()
    worse["Constant speed"]["dr"] = 6.0  # regressed
    b = _result({**better, "Constant speed": worse["Constant speed"]}, 6.5)
    cmp = compare_results(a, b, threshold=0.5)
    assert cmp["overall_delta"] == 0.5
    verdicts = {row["mode"]: row["verdict"] for row in cmp["mode_rows"]}
    assert verdicts["Tip in"] == "improved"
    assert verdicts["Constant speed"] == "regressed"
