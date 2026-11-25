"""Fairness audit: acceptance gaps per gender and age band, and the D5 checks.

The single home of the fairness numbers used by `evaluate` and `promote`.
Only ACCEPTE counts as accepted; a default sent to review or refused counts as
detected (D8). Per group:
- acceptance rate (raw);
- equal-opportunity rate: acceptance among good payers (TARGET = 0);
- recall: share of defaults not accepted.
Each gap is max - min over the groups, in points; 0 with fewer than 2 groups.
Gender rows with XNA (or no value) and rows without an age are left out and
counted. Age bands from AGE_YEARS: <30, 30-50, >=50 (left-closed).

D5 (decided 2026-10-25): gender raw and equal-opportunity gaps <= a fixed
limit; age equal-opportunity gap <= the baseline's (anchor) + a margin. The
tolerances come from `params.promotion`.

Usage:
    from crediscore.fairness.audit import audit, d5_checks

    fair = audit(test_df, decisions, target="TARGET")
    fair["gender"]["eo_gap_pts"], fair["age"]["acceptance_by_group"]
    checks = d5_checks(fair, age_anchor_pts=38.7, max_gender_gap_pts=10,
                       age_gap_margin_pts=2)   # {"gender_raw_gap": {...}, ...}
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from crediscore.models.cost import ACCEPT

GENDER_COL, GENDER_EXCLUDED = "CODE_GENDER", "XNA"
AGE_COL = "AGE_YEARS"
AGE_BINS, AGE_LABELS = [0.0, 30.0, 50.0, np.inf], ["<30", "30-50", "≥50"]
# Gaps are compared at this precision: 100 * (0.4 - 0.3) is still 10 points.
_DIGITS = 9


def gap_pts(rates: pd.Series) -> float:
    """Max - min of the group rates, in points (0 with fewer than 2 groups)."""
    return float(100 * (rates.max() - rates.min())) if len(rates) > 1 else 0.0


def group_rates(
    accepted: pd.Series, good: pd.Series, groups: pd.Series
) -> dict[str, Any]:
    """Rates and gaps of one grouping; rows with a missing group are left out.

    Args:
        accepted: True where the decision is ACCEPTE.
        good: True for a good payer (TARGET = 0).
        groups: Group of each row (NaN = left out), same index.

    Returns:
        raw_gap_pts, eo_gap_pts, recall_gap_pts, the per-group rates
        (a group without good payers or defaults has no rate of that kind)
        and excluded_rows.
    """
    keep = groups.notna()
    acc, ok, grp = accepted[keep], good[keep], groups[keep]
    raw = acc.groupby(grp, observed=True).mean()
    eo = acc[ok].groupby(grp[ok], observed=True).mean()
    recall = (~acc[~ok]).groupby(grp[~ok], observed=True).mean()
    return {
        "raw_gap_pts": gap_pts(raw),
        "eo_gap_pts": gap_pts(eo),
        "recall_gap_pts": gap_pts(recall),
        "acceptance_by_group": {str(k): float(v) for k, v in raw.items()},
        "acceptance_good_payers_by_group": {str(k): float(v) for k, v in eo.items()},
        "recall_by_group": {str(k): float(v) for k, v in recall.items()},
        "excluded_rows": int((~keep).sum()),
    }


def audit(df: pd.DataFrame, decisions: npt.ArrayLike, *, target: str) -> dict[str, Any]:
    """Gender and age-band rates and gaps (D5, D8).

    Args:
        df: Rows with CODE_GENDER, AGE_YEARS and the target.
        decisions: One D4 code per row of `df`.
        target: Label column (1 = default).

    Returns:
        {"gender": ..., "age": ...}, each a `group_rates` output.
    """
    idx = df.index
    accepted = pd.Series(np.asarray(decisions) == ACCEPT, index=idx)
    good = pd.Series(df[target].to_numpy() == 0, index=idx)
    gender = df[GENDER_COL].where(df[GENDER_COL] != GENDER_EXCLUDED)
    age = pd.cut(df[AGE_COL], AGE_BINS, right=False, labels=AGE_LABELS)
    return {
        "gender": group_rates(accepted, good, gender),
        "age": group_rates(accepted, good, age),
    }


def _check(value: float, limit: float) -> dict[str, Any]:
    return {
        "value": value,
        "limit": limit,
        "pass": bool(round(value, _DIGITS) <= limit),
    }


def d5_checks(
    fair: dict[str, Any],
    *,
    age_anchor_pts: float,
    max_gender_gap_pts: float,
    age_gap_margin_pts: float,
) -> dict[str, dict[str, Any]]:
    """The three D5 conditions, each {value, limit, pass} (bounds inclusive).

    Args:
        fair: An `audit` output (only the gap fields are read).
        age_anchor_pts: The baseline's age equal-opportunity gap.
        max_gender_gap_pts: Limit of both gender gaps.
        age_gap_margin_pts: Allowed age gap above the anchor.

    Returns:
        gender_raw_gap, gender_eo_gap, age_eo_gap.
    """
    g = fair["gender"]
    return {
        "gender_raw_gap": _check(g["raw_gap_pts"], max_gender_gap_pts),
        "gender_eo_gap": _check(g["eo_gap_pts"], max_gender_gap_pts),
        "age_eo_gap": _check(
            fair["age"]["eo_gap_pts"], age_anchor_pts + age_gap_margin_pts
        ),
    }
