"""Tests for the hand-written PSI (monitoring/drift.py, contract 4).

PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%), each share floored at
EPSILON. Numeric bins come from the reference quantiles (deduplicated): each
quantile value is its own bin, and so is each open interval around them.
"""

import math

import numpy as np
import pandas as pd
import pytest

from crediscore.monitoring.drift import EPSILON, psi


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
    # Reference 1, 2, 3, 4 with bins=2: quantiles 0, .5, 1 = 1, 2.5, 4.
    # Bins: <1 | =1 | (1, 2.5) | =2.5 | (2.5, 4) | =4 | >4 | NaN
    # Reference shares:  0 | .25 | .25 | 0 | .25 | .25 | 0 | 0
    ref = pd.Series([1.0, 2.0, 3.0, 4.0])
    # Current 0, 2, 2, 5:  .25 | 0 | .5 | 0 | 0 | 0 | .25 | 0
    cur = pd.Series([0.0, 2.0, 2.0, 5.0])
    expected = (
        term(0.25, 0)  # <1: below the reference minimum
        + term(0, 0.25)  # =1
        + term(0.5, 0.25)  # (1, 2.5)
        + term(0, 0.25)  # (2.5, 4)
        + term(0, 0.25)  # =4
        + term(0.25, 0)  # >4: above the reference maximum
    )
    assert psi(ref, cur, bins=2) == pytest.approx(expected)
    # 5 empty-vs-.25 terms of (.25 - 1e-4) * ln(2500) + .25 * ln(2)
    assert expected == pytest.approx(5 * 0.2499 * math.log(2500) + 0.25 * math.log(2))
    assert expected == pytest.approx(9.9494, abs=1e-4)


def test_known_categorical_value_by_hand() -> None:
    ref = pd.Series(["a", "a", "b", "b"])
    cur = pd.Series(["a", "a", "a", "b"])
    expected = term(0.75, 0.5) + term(0.25, 0.5)
    assert psi(ref, cur) == pytest.approx(expected)
    assert expected == pytest.approx(0.2747, abs=1e-4)


def test_empty_bins_stay_finite() -> None:
    ref = pd.Series(np.arange(100, dtype=float))
    cur = pd.Series(np.full(100, 1000.0))  # every row above the reference max
    value = psi(ref, cur)
    assert math.isfinite(value)
    assert value > 1


def test_constant_column() -> None:
    ref = pd.Series(np.full(50, 3.0))
    assert psi(ref, ref.copy()) == 0.0
    # A shift above OR below the constant lands in its own bin.
    assert psi(ref, pd.Series(np.full(50, 4.0))) > 1
    assert psi(ref, pd.Series(np.full(50, 2.0))) > 1


def test_binary_column_keeps_two_bins() -> None:
    # Quantiles of a 0/1 column collapse to 0 and 1: deduplicated, not lost.
    ref = pd.Series([0.0] * 80 + [1.0] * 20)
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


def test_bins_come_from_the_reference() -> None:
    # PSI is not symmetric: the reference sets the bins.
    wide = pd.Series(np.linspace(0, 100, 101))
    narrow = pd.Series(np.linspace(40, 60, 101))
    forward = psi(wide, narrow)
    backward = psi(narrow, wide)
    # From the narrow reference, 40 % of `wide` falls in each open outer bin,
    # empty in the reference: the backward PSI is the larger one.
    assert forward == pytest.approx(6.5382, abs=1e-4)
    assert backward == pytest.approx(8.3004, abs=1e-4)


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
