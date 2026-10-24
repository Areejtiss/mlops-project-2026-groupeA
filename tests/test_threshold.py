"""Tests for the 3-zone decision rule and the exact threshold search (D3, D4, D8)."""

import itertools
import json

import numpy as np
import numpy.typing as npt
import pytest

from crediscore.models.cost import ACCEPT, REFUSE, REVIEW, business_cost
from crediscore.models.threshold import (
    NEVER,
    decide,
    optimize_thresholds,
    zone_metrics,
)

COSTS = {"cost_fn": 5.0, "cost_fp": 1.0, "cost_review": 0.3}
FREE = {"min_recall": 0.0, "max_review_share": 1.0}
D3 = {"min_recall": 0.65, "max_review_share": 0.15}


def recall(y: npt.ArrayLike, decisions: npt.ArrayLike) -> float:
    """D8 recall, written independently: review counts as detected."""
    y, d = np.asarray(y), np.asarray(decisions)
    return float(np.mean(d[y == 1] != ACCEPT))


def brute_force(
    y: np.ndarray, p: np.ndarray, *, min_recall: float, max_review_share: float
) -> float:
    """Lowest cost over every candidate pair low <= high, scored with decide()."""
    cuts = [*np.unique(p), NEVER]
    best = np.inf
    for low, high in itertools.combinations_with_replacement(cuts, 2):
        d = decide(p, low, high)
        if recall(y, d) >= min_recall and np.mean(d == REVIEW) <= max_review_share:
            best = min(best, business_cost(y, d, **COSTS))
    return float(best)


def tied_case(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic-like scores: few distinct values, many ties, ~10 % positives."""
    rng = np.random.default_rng(seed)
    steps = np.sort(rng.uniform(0.01, 0.6, size=12))
    p = rng.choice(steps, size=n)
    y = (rng.uniform(size=n) < p * 0.4).astype(int)
    y[np.argmax(p)] = 1  # at least one positive
    return y, p


# --- decide -----------------------------------------------------------------


def test_decide_boundaries() -> None:
    p = np.array([0.0, 0.0999, 0.1, 0.2, 0.2999, 0.3, 1.0])
    d = decide(p, 0.1, 0.3)
    assert d.tolist() == [ACCEPT, ACCEPT, REVIEW, REVIEW, REVIEW, REFUSE, REFUSE]


def test_decide_low_equal_high_gives_two_zones() -> None:
    assert decide([0.1, 0.2, 0.3], 0.2, 0.2).tolist() == [ACCEPT, REFUSE, REFUSE]


def test_decide_never_refuses_above_one() -> None:
    assert decide([1.0], 0.5, NEVER).tolist() == [REVIEW]
    assert NEVER > 1.0


def test_decide_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="low must be <= high"):
        decide([0.5], 0.6, 0.3)


def test_decide_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        decide([0.5, np.nan], 0.1, 0.3)


# --- zone_metrics -------------------------------------------------------------


def test_zone_metrics_hand_computed() -> None:
    y = [0, 1, 0, 1, 1]
    p = [0.05, 0.05, 0.2, 0.2, 0.9]  # ACCEPT, ACCEPT, REVIEW, REVIEW, REFUSE
    m = zone_metrics(y, p, 0.1, 0.5, **COSTS)
    assert m["cost"] == pytest.approx(5 + 0.6)
    assert m["cost_per_file"] == pytest.approx(5.6 / 5)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["review_share"] == pytest.approx(0.4)
    assert m["accept_share"] == pytest.approx(0.4)
    assert m["refuse_share"] == pytest.approx(0.2)


# --- optimize_thresholds ----------------------------------------------------------

# Review the 0.3 step (2 files, 0.6) beats refusing it (1 FP) or accepting it
# (1 FN = 5). Cuts are midpoints: low (0.2+0.3)/2, high (0.3+0.4)/2.
TOY_Y = [0, 0, 1, 0, 1]
TOY_P = [0.1, 0.2, 0.3, 0.3, 0.4]


def test_hand_computed_toy_case() -> None:
    low, high = optimize_thresholds(TOY_Y, TOY_P, **COSTS, **FREE)
    assert (low, high) == pytest.approx((0.25, 0.35))
    assert business_cost(TOY_Y, decide(TOY_P, low, high), **COSTS) == pytest.approx(0.6)


def test_review_cap_is_inclusive() -> None:
    # The optimum reviews exactly 2 of 5 files = 0.4: allowed at a 0.4 cap.
    at_cap = optimize_thresholds(
        TOY_Y, TOY_P, **COSTS, min_recall=0, max_review_share=0.4
    )
    assert at_cap == pytest.approx((0.25, 0.35))
    below = optimize_thresholds(
        TOY_Y, TOY_P, **COSTS, min_recall=0, max_review_share=0.39
    )
    d = decide(TOY_P, *below)
    assert np.mean(d == REVIEW) <= 0.39
    assert business_cost(TOY_Y, d, **COSTS) == pytest.approx(1.0)  # refuse the step


# Unconstrained, accepting the 0.1 step (1 default: 5) beats reviewing its
# 20 files (6) or refusing its 19 good clients (19).
REC_Y = [0] + [1] + [0] * 19 + [1]
REC_P = [0.05] + [0.1] * 20 + [0.9]


def test_recall_constraint_forces_a_lower_low() -> None:
    free = optimize_thresholds(REC_Y, REC_P, **COSTS, **FREE)
    assert recall(REC_Y, decide(REC_P, *free)) == 0.5
    low, high = optimize_thresholds(
        REC_Y, REC_P, **COSTS, min_recall=1.0, max_review_share=1.0
    )
    assert low <= 0.1
    assert recall(REC_Y, decide(REC_P, low, high)) == 1.0


def test_recall_constraint_is_inclusive() -> None:
    # The free optimum has recall exactly 0.5: min_recall=0.5 keeps it.
    free = optimize_thresholds(REC_Y, REC_P, **COSTS, **FREE)
    at = optimize_thresholds(REC_Y, REC_P, **COSTS, min_recall=0.5, max_review_share=1)
    assert at == free


# Exactly representable review cost, so that tied costs are equal floats.
QUARTER = {**COSTS, "cost_review": 0.25}


def test_cost_ties_keep_the_lowest_high() -> None:
    # 4 files on one step, 1 good client: refuse (1) == review (4 * 0.25).
    y, p = [1, 1, 1, 0], [0.5] * 4
    low, high = optimize_thresholds(y, p, **QUARTER, **FREE)
    assert set(decide(p, low, high).tolist()) == {REFUSE}


def test_cost_ties_keep_the_lowest_low() -> None:
    # 20 files on one step, 1 default: review (20 * 0.25) == accept (5).
    y, p = [1] + [0] * 19, [0.5] * 20
    low, high = optimize_thresholds(y, p, **QUARTER, **FREE)
    assert set(decide(p, low, high).tolist()) == {REVIEW}


@pytest.mark.parametrize("step", [0.5, 1.0])  # isotonic can output exactly 1.0
def test_accept_all_is_a_candidate(step: float) -> None:
    # One default tied with 20 good clients: accepting all (5) beats
    # reviewing all (6.3) or refusing all (20).
    y, p = [1] + [0] * 20, [step] * 21
    low, high = optimize_thresholds(y, p, **COSTS, **FREE)
    assert set(decide(p, low, high).tolist()) == {ACCEPT}
    assert low == high > step


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "constraints", [FREE, D3, {"min_recall": 0.9, "max_review_share": 0.3}]
)
def test_exact_optimum_matches_brute_force_with_ties(
    seed: int, constraints: dict[str, float]
) -> None:
    y, p = tied_case(seed)
    low, high = optimize_thresholds(y, p, **COSTS, **constraints)
    d = decide(p, low, high)
    assert low <= high
    assert recall(y, d) >= constraints["min_recall"]
    assert np.mean(d == REVIEW) <= constraints["max_review_share"]
    assert business_cost(y, d, **COSTS) == pytest.approx(
        brute_force(y, p, **constraints)
    )


@pytest.mark.parametrize("seed", range(8))
def test_cuts_are_midpoints_between_steps(seed: int) -> None:
    y, p = tied_case(seed)
    steps = set(np.unique(p).tolist())
    for cut in optimize_thresholds(y, p, **COSTS, **D3):
        # Never on a step, unless it is the lowest step (nothing below it).
        assert cut not in steps or cut == min(steps)


@pytest.mark.parametrize("seed", range(8))
def test_decisions_survive_json_round_trip_and_float32(seed: int) -> None:
    y, p = tied_case(seed)
    low, high = optimize_thresholds(y, p, **COSTS, **D3)
    ref = decide(p, low, high)
    back = json.loads(json.dumps({"low": low, "high": high}))
    np.testing.assert_array_equal(decide(p, back["low"], back["high"]), ref)
    np.testing.assert_array_equal(decide(p.astype(np.float32), low, high), ref)


@pytest.mark.parametrize("seed", range(5))
def test_cost_not_worse_than_accept_all_or_refuse_all(seed: int) -> None:
    y, p = tied_case(seed, n=1000)
    low, high = optimize_thresholds(y, p, **COSTS, **FREE)
    cost = business_cost(y, decide(p, low, high), **COSTS)
    assert cost <= business_cost(y, [ACCEPT] * len(y), **COSTS)
    assert cost <= business_cost(y, [REFUSE] * len(y), **COSTS)


def test_continuous_scores() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    y = (rng.uniform(size=200) < p * 0.3).astype(int)
    low, high = optimize_thresholds(y, p, **COSTS, **D3)
    assert business_cost(y, decide(p, low, high), **COSTS) == pytest.approx(
        brute_force(y, p, **D3)
    )


def test_tie_exactly_at_a_step_moves_the_whole_step() -> None:
    # 4 files share 0.2: they are all on the same side of every threshold.
    y = np.array([0, 1, 0, 1, 1, 0, 1])
    p = np.array([0.05, 0.2, 0.2, 0.2, 0.2, 0.6, 0.6])
    low, high = optimize_thresholds(y, p, **COSTS, **FREE)
    d = decide(p, low, high)
    assert len(set(d[p == 0.2].tolist())) == 1
    assert 0.2 not in (low, high)


def test_infeasible_constraints_raise() -> None:
    with pytest.raises(ValueError, match="no thresholds satisfy"):
        optimize_thresholds(
            [0, 1], [0.2, 0.8], **COSTS, min_recall=1.01, max_review_share=1
        )


def test_no_positive_raises() -> None:
    with pytest.raises(ValueError, match="at least one default"):
        optimize_thresholds([0, 0], [0.1, 0.2], **COSTS, **FREE)


@pytest.mark.parametrize(
    ("y", "p"),
    [([0, 1], [0.1]), ([0, 2], [0.1, 0.2]), ([0, 1], [0.1, np.nan]), ([], [])],
)
def test_bad_inputs_raise(y: list[float], p: list[float]) -> None:
    with pytest.raises(ValueError):
        optimize_thresholds(y, p, **COSTS, **FREE)
