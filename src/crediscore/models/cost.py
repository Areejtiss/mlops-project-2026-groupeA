"""Business cost of credit decisions, for 2 or 3 decision zones (D3, D4).

A missed default (FN: TARGET=1 accepted) costs `cost_fn`, a good client refused
(FP: TARGET=0 refused) costs `cost_fp`, and every file sent to manual review
costs `cost_review`, whatever its label: the analyst is assumed to decide it
correctly. Costs are passed in (from params.yaml by the caller).

Usage:
    from crediscore.models.cost import business_cost

    business_cost([1, 0, 0], ["ACCEPTE", "A_ETUDIER", "REFUSE"],
                  cost_fn=5, cost_fp=1, cost_review=0.3)   # 5 + 0.3 + 1 = 6.3
    business_cost([1, 0], [1, 0], cost_fn=5, cost_fp=1, cost_review=0.3)  # 0.0
"""

import numpy as np
import numpy.typing as npt

ACCEPT, REVIEW, REFUSE = "ACCEPTE", "A_ETUDIER", "REFUSE"
DECISIONS = (ACCEPT, REVIEW, REFUSE)


def business_cost(
    y_true: npt.ArrayLike,
    decisions: npt.ArrayLike,
    *,
    cost_fn: float,
    cost_fp: float,
    cost_review: float,
) -> float:
    """Total business cost of a set of decisions.

    Args:
        y_true: True labels, 0 (repaid) or 1 (default).
        decisions: One decision per file: a D4 code ("ACCEPTE", "A_ETUDIER",
            "REFUSE"), or a binary prediction (1 = default = REFUSE, 0 = ACCEPTE).
        cost_fn: Cost of one accepted default.
        cost_fp: Cost of one refused good client.
        cost_review: Cost of one file sent to manual review.

    Returns:
        The summed cost (0.0 for empty inputs).

    Raises:
        ValueError: If lengths differ, a label is not 0/1, or a decision is
            not a known code or binary value.
    """
    y = np.asarray(y_true)
    d = np.asarray(decisions)
    if y.shape != d.shape or y.ndim != 1:
        raise ValueError(f"y_true and decisions length differ: {y.shape} vs {d.shape}")
    if y.dtype.kind not in "biuf" or not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary (0/1)")
    if d.dtype.kind in "biu":
        if not np.isin(d, (0, 1)).all():
            raise ValueError("binary decisions must be 0 or 1")
        d = np.where(d == 1, REFUSE, ACCEPT)
    unknown = set(d.tolist()) - set(DECISIONS)
    if unknown:
        raise ValueError(f"unknown decision codes: {sorted(map(str, unknown))}")
    fn = np.sum((y == 1) & (d == ACCEPT))
    fp = np.sum((y == 0) & (d == REFUSE))
    review = np.sum(d == REVIEW)
    return float(cost_fn * fn + cost_fp * fp + cost_review * review)
