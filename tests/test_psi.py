"""Tests for the hand-written PSI (monitoring/drift.py, contract 4).

PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%), each share floored at
EPSILON. Numeric bins are intervals: each edge sits halfway between a
deduplicated reference quantile and the next distinct reference value, so no
reference value lies on an edge and a tiny jitter never changes a bin. Two
open-ended outer bins (half a gap beyond the reference min and max) and a NaN
bin complete them.
"""

import math

import numpy as np
import pandas as pd
import pytest

from crediscore.monitoring.drift import EPSILON, _numeric_edges, psi


def term(cur: float, ref: float) -> float:
    """One PSI term, with the EPSILON floor written out."""
    cur, ref = max(cur, EPSILON), max(ref, EPSILON)
    return (cur - ref) * math.log(cur / ref)


def test_epsilon_is_documented_value() -> None:
    assert EPSILON == 1e-4


def test_identical_numeric_data_gives_zero() -> None:
    values = pd.Series(np.random.default_rng(0).normal(size=1000))
    assert psi(values, values.copy()) == 0.0


def test_identical_categorical_data_gives_zero() -> None:
    values = pd.Series(["a", "b", "b", None, "c"] * 20)
    assert psi(values, values.copy()) == 0.0


def test_known_numeric_value_by_hand() -> None:
    # Reference 1, 2, 3, 4 with bins=2: the inner quantile ("lower") is 2 and
    # the next distinct value 3, so one edge at 2.5. Outer edges: half a gap
    # beyond the min and the max, 0.5 and 4.5.
    # Bins:       <0.5 | [0.5, 2.5) | [2.5, 4.5) | >=4.5 | NaN
    # Reference:     0 |         .5 |         .5 |     0 |   0
    ref = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert list(_numeric_edges(ref, 2)) == [0.5, 2.5, 4.5]
    # Current 0, 2, 2, 5:  .25 | .5 | 0 | .25 | 0
    cur = pd.Series([0.0, 2.0, 2.0, 5.0])
    expected = (
        term(0.25, 0)  # below the reference
        + term(0.5, 0.5)  # [0.5, 2.5)
        + term(0, 0.5)  # [2.5, 4.5)
        + term(0.25, 0)  # above the reference
    )
    assert psi(ref, cur, bins=2) == pytest.approx(expected)
    # 2 * (.25 - 1e-4) * ln(2500) + (.5 - 1e-4) * ln(5000)
    assert expected == pytest.approx(
        2 * 0.2499 * math.log(2500) + 0.4999 * math.log(5000)
    )


def test_known_categorical_value_by_hand() -> None:
    ref = pd.Series(["a", "a", "b", "b"])
    cur = pd.Series(["a", "a", "a", "b"])
    expected = term(0.75, 0.5) + term(0.25, 0.5)
    assert psi(ref, cur) == pytest.approx(expected)
    assert expected == pytest.approx(0.2747, abs=1e-4)


def test_reference_nan_never_enters_the_edges() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    with_nan = pd.concat([values, pd.Series([np.nan] * 100)], ignore_index=True)
    assert list(_numeric_edges(with_nan, 2)) == list(_numeric_edges(values, 2))


def test_heaped_column_ignores_a_tiny_jitter() -> None:
    # Most of the mass sits exactly on a few values, as for incomes.
    ref = pd.Series(np.repeat([90_000.0, 135_000.0, 180_000.0, 225_000.0], 250))
    for factor in (1 + 1e-9, 1 - 1e-9):
        assert psi(ref, ref * factor) == 0.0


def test_one_unit_on_a_large_column_gives_almost_zero() -> None:
    rng = np.random.default_rng(0)
    heaps = rng.choice([90_000.0, 135_000.0, 180_000.0], size=5000)
    ref = pd.Series(np.r_[heaps, rng.lognormal(12, 0.5, 5000)])
    assert psi(ref, ref + 1) < 0.01


def test_float32_round_trip_gives_almost_zero() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(np.r_[rng.lognormal(12, 0.5, 5000), np.full(5000, 135_000.0)])
    assert psi(ref, ref.astype("float32").astype("float64")) < 0.01


def test_crisis_like_scale_shift_gives_a_moderate_psi() -> None:
    # Incomes x 0.8: a shift of 0.45 standard deviation in log scale.
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.lognormal(12, 0.5, 10_000))
    assert 0.1 < psi(ref, ref * 0.8) < 0.5


def test_direction_matters_but_stays_bounded() -> None:
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=10_000))
    b = pd.Series(rng.normal(loc=0.5, size=10_000))
    forward, backward = psi(a, b), psi(b, a)
    assert forward != backward  # the reference sets the bins
    assert 0.5 < forward / backward < 2


def test_empty_bins_stay_finite() -> None:
    ref = pd.Series(np.arange(100, dtype=float))
    cur = pd.Series(np.full(100, 1000.0))  # every row above the reference max
    value = psi(ref, cur)
    assert math.isfinite(value)
    assert value > 1


def test_constant_column() -> None:
    ref = pd.Series(np.full(50, 3.0))
    assert psi(ref, ref.copy()) == 0.0
    # A shift above OR below the constant lands in an outer bin; a 1e-9
    # jitter stays inside (relative pad 1e-6).
    assert psi(ref, pd.Series(np.full(50, 4.0))) > 1
    assert psi(ref, pd.Series(np.full(50, 2.0))) > 1
    assert psi(ref, ref * (1 + 1e-9)) == 0.0


def test_binary_column_keeps_two_bins() -> None:
    # The deciles of a 0/1 column collapse to 0 and 1: one edge at 0.5.
    ref = pd.Series([0.0] * 80 + [1.0] * 20)
    assert list(_numeric_edges(ref, 10)) == [-0.5, 0.5, 1.5]
    cur = pd.Series([0.0] * 50 + [1.0] * 50)
    expected = term(0.5, 0.8) + term(0.5, 0.2)
    assert psi(ref, cur) == pytest.approx(expected)


def test_nan_is_its_own_bin() -> None:
    ref = pd.Series(np.repeat([1.0, 2.0, 3.0, 4.0], 24))
    cur = ref.copy()
    cur[::2] = np.nan  # half the rows of each value lose it
    # Non-missing current rows keep the reference shape, halved; NaN 0 -> .5.
    expected = 4 * term(0.125, 0.25) + term(0.5, 0)
    assert psi(ref, cur, bins=4) == pytest.approx(expected)


def test_nan_only_in_reference_also_counts() -> None:
    ref = pd.Series([1.0, np.nan] * 50)
    cur = pd.Series([1.0] * 100)
    assert psi(ref, cur) == pytest.approx(term(1.0, 0.5) + term(0, 0.5))


def test_all_nan_numeric_column() -> None:
    nan = pd.Series([np.nan] * 10)
    assert psi(nan, nan.copy()) == 0.0
    assert psi(nan, pd.Series([1.0] * 10)) > 1


def test_categorical_missing_is_its_own_category() -> None:
    ref = pd.Series(["a", "b"] * 50)
    cur = pd.Series(["a", None] * 50)
    expected = term(0.5, 0.5) + term(0, 0.5) + term(0.5, 0)
    assert psi(ref, cur) == pytest.approx(expected)


def test_unseen_category_is_its_own_bin() -> None:
    ref = pd.Series(["a", "b"] * 50)
    cur = pd.Series(["a", "b", "z", "z"] * 25)
    expected = term(0.25, 0.5) * 2 + term(0.5, 0)
    assert psi(ref, cur) == pytest.approx(expected)


def test_bins_sets_the_number_of_reference_quantiles() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.uniform(0, 1, 10_000))
    cur = pd.Series(rng.uniform(0, 0.5, 10_000))  # all below the median
    # bins=2: halves of .5 each, current 1 / 0.
    assert psi(ref, cur, bins=2) == pytest.approx(term(1, 0.5) + term(0, 0.5), rel=0.02)
    # bins=10: deciles of .1 each, current .2 in the low five, 0 above.
    assert psi(ref, cur) == pytest.approx(
        5 * term(0.2, 0.1) + 5 * term(0, 0.1), rel=0.02
    )


def test_shifted_normal_is_large_and_same_draw_is_small() -> None:
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(size=5000))
    same = pd.Series(rng.normal(size=5000))
    shifted = pd.Series(rng.normal(loc=0.5, size=5000))
    assert psi(ref, same) < 0.02
    assert psi(ref, shifted) > 0.2


def test_inputs_are_not_modified() -> None:
    ref = pd.Series([1.0, np.nan, 3.0])
    cur = pd.Series([2.0, 2.0, np.nan])
    ref_before, cur_before = ref.copy(), cur.copy()
    psi(ref, cur)
    pd.testing.assert_series_equal(ref, ref_before)
    pd.testing.assert_series_equal(cur, cur_before)


def test_bins_below_two_are_rejected() -> None:
    with pytest.raises(ValueError, match="bins"):
        psi(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), bins=1)


def test_constant_zero_column_ignores_a_tiny_jitter() -> None:
    # The pad is relative to max(|c|, 1): a column of zeros still gets one.
    ref = pd.Series(np.zeros(50))
    for jitter in (1e-9, -1e-9):
        assert psi(ref, pd.Series(np.full(50, jitter))) == 0.0


def test_a_current_value_on_an_edge_goes_to_the_upper_bin() -> None:
    ref = pd.Series([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    # bins=4: inner quantiles 1, 1, 2 -> edges 0.5 | 1.5 | 2.5 | 4.5
    assert list(_numeric_edges(ref, 4)) == [0.5, 1.5, 2.5, 4.5]
    cur = pd.Series([2.5] * 6)  # lands in [2.5, 4.5), not in [1.5, 2.5)
    expected = term(0, 0.5) + term(0, 1 / 6) + term(1, 2 / 6)
    assert psi(ref, cur, bins=4) == pytest.approx(expected)
