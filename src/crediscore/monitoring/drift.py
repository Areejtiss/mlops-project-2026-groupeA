"""Drift detection: PSI, KS and chi2 tests, drift report (contract 4).

Compares a current batch with a reference batch, feature by feature, and
writes the drift report JSON consumed by Iyed (Prometheus gauges) and Areej
(monitoring flow: retrain on alert).

Usage:
    from crediscore.monitoring.drift import psi

    psi(reference["AMT_INCOME_TOTAL"], current["AMT_INCOME_TOTAL"], bins=10)
"""

import numpy as np
import pandas as pd

# Floor of a bin share in the PSI, so that an empty bin gives a finite term.
EPSILON = 1e-4
_NAN_BIN = -1


def _numeric_bins(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[pd.Series, pd.Series]:
    """Bin index of every value, from the deduplicated reference quantiles.

    Each quantile value is its own bin, and so is each open interval between
    two of them, plus the two open-ended outer intervals: a 0/1 column keeps
    a bin for 0 and one for 1, and a constant column separates "below" from
    "above". Missing values get their own bin.
    """
    ref = reference.dropna().to_numpy(dtype=float)
    cuts = (
        np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if ref.size
        else np.array([])
    )

    def index(s: pd.Series) -> pd.Series:
        x = s.to_numpy(dtype=float)
        idx = np.searchsorted(cuts, x, side="left") + np.searchsorted(
            cuts, x, side="right"
        )
        return pd.Series(np.where(np.isnan(x), _NAN_BIN, idx))

    return index(reference), index(current)


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index of `current` against `reference`.

    PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%), each share floored
    at EPSILON. Numeric columns are binned on the reference quantiles
    (see `_numeric_bins`); other columns use one bin per category, a category
    unseen in the reference included. Missing values are a bin of their own.
    The reference sets the bins, so psi(a, b) != psi(b, a) in general.

    Args:
        reference: Reference values (e.g. the training or test batch).
        current: Current values of the same feature.
        bins: Number of quantile intervals of the reference (numeric only).

    Returns:
        The PSI, >= 0 and finite. Rule of thumb: < 0.1 stable, >= 0.2 drift.

    Raises:
        ValueError: If bins < 2.
    """
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    if pd.api.types.is_numeric_dtype(reference):
        ref_bins, cur_bins = _numeric_bins(reference, current, bins)
    else:
        ref_bins, cur_bins = reference, current
    shares = pd.concat(
        [
            ref_bins.value_counts(normalize=True, dropna=False),
            cur_bins.value_counts(normalize=True, dropna=False),
        ],
        axis=1,
    )
    shares = shares.fillna(0.0).clip(lower=EPSILON)
    ref_share, cur_share = shares.iloc[:, 0], shares.iloc[:, 1]
    return float(((cur_share - ref_share) * np.log(cur_share / ref_share)).sum())
