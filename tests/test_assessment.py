import numpy as np

from avldrive.assessment import aggregate_dr
from avldrive.criteria import rate
from avldrive.config import BRAND_DNA


def test_rate_direction_and_clamp():
    # Zero fault -> best rating; large fault -> clamped to 1.
    assert rate(0.0, 5.0) == 10.0
    assert rate(100.0, 5.0) == 1.0
    # Higher fault -> lower rating.
    assert rate(0.2, 5.0) > rate(0.5, 5.0)


def test_delay_rating_matches_brand_dna_spec():
    # 200 ms delay: Luxury Sedan 7.6, Sports Car 4.8, Eco EV 8.8.
    assert round(rate(0.2, BRAND_DNA["Luxury Sedan"]["delay"]), 1) == 7.6
    assert round(rate(0.2, BRAND_DNA["Sports Car"]["delay"]), 1) == 4.8
    assert round(rate(0.2, BRAND_DNA["Eco EV"]["delay"]), 1) == 8.8


def test_aggregate_extreme_value_bias():
    # Extreme-value weighting pulls the aggregate below the plain mean.
    ratings = [9.0, 9.0, 9.0, 2.0]
    agg = aggregate_dr(ratings)
    assert agg < np.mean(ratings)
    assert 2.0 < agg < 9.0


def test_aggregate_respects_weights():
    # A heavily weighted good criterion lifts the aggregate.
    low_weight_bad = aggregate_dr([9.0, 3.0], [5, 1])
    high_weight_bad = aggregate_dr([9.0, 3.0], [1, 5])
    assert low_weight_bad > high_weight_bad


def test_aggregate_handles_empty_and_none():
    assert aggregate_dr([]) is None
    assert aggregate_dr([None, None]) is None
    assert aggregate_dr([None, 8.0]) == 8.0
