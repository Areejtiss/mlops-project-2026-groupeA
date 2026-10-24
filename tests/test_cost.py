"""Tests for the business cost of credit decisions (D3, D4)."""

import numpy as np
import pytest

from crediscore.models.cost import business_cost

COSTS = {"cost_fn": 5.0, "cost_fp": 1.0, "cost_review": 0.3}


def test_all_correct_costs_nothing() -> None:
    y = [0, 1, 0, 1]
    decisions = ["ACCEPTE", "REFUSE", "ACCEPTE", "REFUSE"]
    assert business_cost(y, decisions, **COSTS) == 0


def test_one_false_negative_costs_cost_fn() -> None:
    assert business_cost([1, 0], ["ACCEPTE", "ACCEPTE"], **COSTS) == 5.0


def test_one_false_positive_costs_cost_fp() -> None:
    assert business_cost([0, 1], ["REFUSE", "REFUSE"], **COSTS) == 1.0


def test_mixed_counts() -> None:
    # 2 FN, 3 FP, 1 TP, 1 TN -> 2 * 5 + 3 * 1
    y = [1, 1, 0, 0, 0, 1, 0]
    decisions = ["ACCEPTE"] * 2 + ["REFUSE"] * 4 + ["ACCEPTE"]
    assert business_cost(y, decisions, **COSTS) == 13.0


def test_review_costs_cost_review_per_file_whatever_the_label() -> None:
    y = [0, 1, 1, 0]
    decisions = ["A_ETUDIER", "A_ETUDIER", "A_ETUDIER", "ACCEPTE"]
    assert business_cost(y, decisions, **COSTS) == pytest.approx(0.9)


def test_three_zones_mixed() -> None:
    # 1 FN (5) + 1 FP (1) + 2 reviews (0.6)
    y = np.array([1, 0, 1, 0, 0, 1])
    decisions = np.array(
        ["ACCEPTE", "REFUSE", "A_ETUDIER", "A_ETUDIER", "ACCEPTE", "REFUSE"]
    )
    assert business_cost(y, decisions, **COSTS) == pytest.approx(6.6)


def test_binary_predictions_mean_refuse_when_one() -> None:
    # 1 = predicted default = REFUSE; 0 = ACCEPTE. 1 FN + 1 FP.
    assert business_cost([1, 0, 1, 0], [0, 1, 1, 0], **COSTS) == 6.0


def test_costs_are_taken_from_arguments() -> None:
    cost = business_cost(
        [1, 0, 0],
        ["ACCEPTE", "REFUSE", "A_ETUDIER"],
        cost_fn=10,
        cost_fp=2,
        cost_review=1,
    )
    assert cost == 13.0


def test_empty_input_costs_nothing() -> None:
    assert business_cost([], [], **COSTS) == 0


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="length"):
        business_cost([0, 1], ["ACCEPTE"], **COSTS)


def test_two_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="length"):
        business_cost([[0, 1]], [["ACCEPTE", "REFUSE"]], **COSTS)


@pytest.mark.parametrize("y", [[0, 2], [0, -1], [0.5, 1], ["0", "1"]])
def test_non_binary_labels_are_rejected(y: list[object]) -> None:
    with pytest.raises(ValueError, match="binary"):
        business_cost(y, ["ACCEPTE", "ACCEPTE"], **COSTS)


@pytest.mark.parametrize(
    "decisions", [["ACCEPTE", "ACCEPTÉ"], ["ACCEPTE", "A ETUDIER"], [0, 2]]
)
def test_unknown_decision_is_rejected(decisions: list[object]) -> None:
    with pytest.raises(ValueError, match="decision"):
        business_cost([0, 1], decisions, **COSTS)
