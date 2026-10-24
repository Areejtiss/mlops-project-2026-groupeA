"""3-zone decision rule and exact, constrained threshold search (D3, D4, D8, D9).

`decide` is the rule the API applies: `proba < low` -> ACCEPTE,
`low <= proba < high` -> A_ETUDIER, `proba >= high` -> REFUSE. `low == high`
gives 2 zones. It must receive the unrounded float64 probability.

`optimize_thresholds` minimizes `business_cost` on validation over EVERY
distinct cut-off (exact, not a grid), under recall >= min_recall (review counts
as detected, D8) and review share <= max_review_share. Calibrated scores come in
steps (~69 values), so a cut sitting exactly on a step would make decisions
depend on float equality: each returned cut is moved to the midpoint between
the chosen step and the next lower one (same decisions on the search data,
robust to rounding and float32).

Usage:
    from crediscore.models.threshold import decide, optimize_thresholds

    low, high = optimize_thresholds(y_val, p_val, cost_fn=5, cost_fp=1,
                                    cost_review=0.3, min_recall=0.65,
                                    max_review_share=0.15)
    decide(p_new, low, high)        # array of "ACCEPTE" / "A_ETUDIER" / "REFUSE"
"""

import numpy as np
import numpy.typing as npt

from crediscore.models.cost import ACCEPT, REFUSE, REVIEW, business_cost

# A cut above every probability: `high = NEVER` never refuses.
NEVER = float(np.nextafter(1.0, 2.0))


def decide(proba: npt.ArrayLike, low: float, high: float) -> npt.NDArray[np.str_]:
    """Map probabilities to D4 decision codes.

    Args:
        proba: Calibrated default probabilities (unrounded).
        low: Cut between ACCEPTE and A_ETUDIER.
        high: Cut between A_ETUDIER and REFUSE (>= low).

    Returns:
        One code per probability.

    Raises:
        ValueError: If a probability is NaN or outside [0, 1], a threshold is
            outside [0, 1] (`NEVER` allowed), or low > high.
    """
    p = _check_proba(proba)
    for cut in (low, high):
        if not (0.0 <= cut <= 1.0 or cut == NEVER):  # also False for NaN
            raise ValueError(f"low and high must be in [0, 1] or NEVER: {low}, {high}")
    if low > high:
        raise ValueError(f"low must be <= high, got {low} > {high}")
    return np.where(p < low, ACCEPT, np.where(p < high, REVIEW, REFUSE))


def _check_proba(proba: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Probabilities as float64, rejecting NaN and values outside [0, 1]."""
    p = np.asarray(proba, dtype=float)
    if np.isnan(p).any():
        raise ValueError("proba contains NaN")
    if ((p < 0) | (p > 1)).any():  # inf included
        raise ValueError("proba must be in [0, 1]")
    return p


def zone_metrics(
    y_true: npt.ArrayLike,
    proba: npt.ArrayLike,
    low: float,
    high: float,
    *,
    cost_fn: float,
    cost_fp: float,
    cost_review: float,
) -> dict[str, float]:
    """Cost and shares of the decisions taken with (low, high).

    Args:
        y_true: True labels (0/1).
        proba: Calibrated default probabilities.
        low: Lower cut.
        high: Upper cut.
        cost_fn: Cost of one accepted default.
        cost_fp: Cost of one refused good client.
        cost_review: Cost of one reviewed file.

    Returns:
        cost, cost_per_file, recall (D8), review_share, accept_share,
        refuse_share.
    """
    y = np.asarray(y_true)
    d = decide(proba, low, high)
    cost = business_cost(
        y, d, cost_fn=cost_fn, cost_fp=cost_fp, cost_review=cost_review
    )
    return {
        "cost": cost,
        "cost_per_file": cost / len(y),
        "recall": float(np.mean(d[y == 1] != ACCEPT)),
        "review_share": float(np.mean(d == REVIEW)),
        "accept_share": float(np.mean(d == ACCEPT)),
        "refuse_share": float(np.mean(d == REFUSE)),
    }


def _midpoint(cut: float, steps: npt.NDArray[np.float64]) -> float:
    """Midpoint between `cut` and the next lower step (`cut / 2` if none).

    `NEVER` is not a step and is kept. When `cut` and the lower step are
    neighbouring doubles, the midpoint rounds onto one of them; the cut is
    then kept, since moving onto the lower step would change decisions.
    """
    if cut == NEVER:
        return cut
    lower = steps[steps < cut]
    if lower.size == 0:
        return cut / 2
    mid = float((cut + lower[-1]) / 2)
    return cut if mid <= lower[-1] else mid


def optimize_thresholds(
    y_true: npt.ArrayLike,
    proba: npt.ArrayLike,
    *,
    cost_fn: float,
    cost_fp: float,
    cost_review: float,
    min_recall: float,
    max_review_share: float,
) -> tuple[float, float]:
    """Cost-optimal (low, high) under the recall and review-share constraints.

    Every pair of cuts low <= high taken from the distinct probabilities (plus
    `NEVER`, so "accept all" is a candidate) is scored from counts. On cost
    ties the first optimum wins (lowest low, then lowest high).

    Args:
        y_true: Validation labels (0/1), at least one default.
        proba: Calibrated validation probabilities.
        cost_fn: Cost of one accepted default.
        cost_fp: Cost of one refused good client.
        cost_review: Cost of one reviewed file.
        min_recall: Minimum share of defaults not accepted (D8), inclusive.
        max_review_share: Maximum share of files sent to review, inclusive.

    Returns:
        (low, high), full float64 precision, each at a midpoint between steps.

    Raises:
        ValueError: On empty, non-1-D, mismatched, non-binary, NaN or
            out-of-[0, 1] inputs,
            no default in `y_true`, or no pair satisfying the constraints.
    """
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    if y.ndim != 1 or p.ndim != 1:
        raise ValueError(f"1-D inputs expected, got shapes {y.shape} and {p.shape}")
    if y.size == 0:
        raise ValueError("empty inputs")
    if y.shape != p.shape:
        raise ValueError(
            f"y_true and proba must have the same length: {y.shape}, {p.shape}"
        )
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary (0/1)")
    p = _check_proba(p)
    n, n_pos = y.size, int(y.sum())
    if n_pos == 0:
        raise ValueError("y_true needs at least one default to measure recall")
    n_neg = n - n_pos
    steps = np.unique(p)
    cuts = np.r_[steps, NEVER]
    below = np.asarray(np.searchsorted(np.sort(p), cuts))  # files with p < cut
    pos_below = np.asarray(np.searchsorted(np.sort(p[y == 1]), cuts))

    best_cost, best = np.inf, None
    for i in range(cuts.size):  # low = cuts[i]
        if (n_pos - pos_below[i]) / n_pos < min_recall:
            continue
        j = np.arange(i, cuts.size)  # high = cuts[j] >= low
        review = below[j] - below[i]
        refused_neg = n_neg - (below[j] - pos_below[j])
        cost = cost_fn * pos_below[i] + cost_fp * refused_neg + cost_review * review
        cost = np.where(review / n <= max_review_share, cost, np.inf)
        k = int(np.argmin(cost))
        if cost[k] < best_cost:
            best_cost, best = float(cost[k]), (float(cuts[i]), float(cuts[j[k]]))
    if best is None:
        raise ValueError(
            f"no thresholds satisfy min_recall={min_recall} and "
            f"max_review_share={max_review_share}"
        )
    low, high = _midpoint(best[0], steps), _midpoint(best[1], steps)
    # Internal cross-checks: same decisions as the step cuts, same cost as cost.py.
    if not np.array_equal(decide(p, *best), decide(p, low, high)):
        raise RuntimeError("midpoint cuts changed decisions on the search data")
    costs = {"cost_fn": cost_fn, "cost_fp": cost_fp, "cost_review": cost_review}
    check = zone_metrics(y, p, low, high, **costs)["cost"]
    if not np.isclose(check, best_cost):
        raise RuntimeError(f"count-based cost {best_cost} != business_cost {check}")
    return low, high
