"""Drift detection: PSI, KS and chi2 tests, drift report (contract 4).

Compares a current batch with a reference batch, feature by feature, and
writes the drift report JSON consumed by Iyed (Prometheus gauges) and Areej
(monitoring flow: retrain on alert).

Usage:
    from crediscore.monitoring.drift import psi

    psi(reference["AMT_INCOME_TOTAL"], current["AMT_INCOME_TOTAL"], bins=10)
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from crediscore.config import MonitoringConfig, Params
from crediscore.models.train import prepare_xy

REPORT_KEYS = ("timestamp", "drift_share", "psi", "ks", "chi2", "auc_drop", "alert")
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


def _counts(reference: pd.Series, current: pd.Series) -> pd.DataFrame:
    """Count per value (NaN included) in each batch, 0 where absent."""
    counts = pd.concat(
        [
            reference.value_counts(dropna=False),
            current.value_counts(dropna=False),
        ],
        axis=1,
    )
    return counts.fillna(0.0)


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
    counts = _counts(ref_bins, cur_bins)
    shares = (counts / counts.sum()).clip(lower=EPSILON)
    ref_share, cur_share = shares.iloc[:, 0], shares.iloc[:, 1]
    return float(((cur_share - ref_share) * np.log(cur_share / ref_share)).sum())


def ks_test(reference: pd.Series, current: pd.Series) -> float | None:
    """Two-sample Kolmogorov-Smirnov p-value, on the non-missing values.

    Args:
        reference: Reference values of a numeric feature.
        current: Current values of the same feature.

    Returns:
        The p-value, or None when one side has no value at all (the PSI, which
        counts missing values, still shows that drift).
    """
    ref, cur = reference.dropna(), current.dropna()
    if ref.empty or cur.empty:
        return None
    return float(stats.ks_2samp(ref, cur).pvalue)


def chi2_test(reference: pd.Series, current: pd.Series) -> float:
    """Chi-squared test of independence between batch and category.

    Missing values are a category of their own, and so is a category unseen
    in the reference.

    Args:
        reference: Reference values of a categorical feature.
        current: Current values of the same feature.

    Returns:
        The p-value; 1.0 when both batches hold a single category.
    """
    counts = _counts(reference, current)
    if len(counts) < 2:
        return 1.0
    return float(stats.chi2_contingency(counts.T.to_numpy()).pvalue)


def drift_share(p_values: Mapping[str, float | None], alpha: float) -> float:
    """Share of features whose p-value is strictly below alpha.

    Args:
        p_values: p-value per model feature (None = untested, not drifted).
        alpha: Significance level (`monitoring.alpha`).

    Returns:
        Drifted features / all features, in [0, 1].

    Raises:
        ValueError: If `p_values` is empty.
    """
    if not p_values:
        raise ValueError("no feature to compute drift_share on")
    drifted = sum(p is not None and p < alpha for p in p_values.values())
    return drifted / len(p_values)


def is_alert(
    share: float,
    psi_values: Mapping[str, float],
    drop: float | None,
    config: MonitoringConfig,
) -> bool:
    """Alert rule of contract 4.

    Alert when `share` >= `drift_share_alert`, or the PSI of one key feature
    >= `psi_alert`, or `drop` > `auc_drop_alert` (when the AUC is known).

    Args:
        share: The batch's drift_share.
        psi_values: PSI per feature.
        drop: Reference AUC minus current AUC, or None without labels.
        config: The `monitoring` parameters.

    Returns:
        True if the batch must raise the alert (retrain).
    """
    return (
        share >= config.drift_share_alert
        or any(psi_values[f] >= config.psi_alert for f in config.key_features)
        or (drop is not None and drop > config.auc_drop_alert)
    )


def auc_drop(
    model: Any, reference: pd.DataFrame, current: pd.DataFrame, params: Params
) -> float | None:
    """Concept drift: reference AUC minus current AUC of the same model.

    Args:
        model: Fitted classifier with `predict_proba` (contract 2 artifact).
        reference: Labelled reference batch.
        current: Current batch, labelled or not.
        params: Validated parameters (features and target name).

    Returns:
        The AUC drop (> 0 = the model got worse), or None when the current
        batch has no label.

    Raises:
        ValueError: If a labelled batch holds a single class.
    """
    if params.target not in current or current[params.target].isna().all():
        return None
    aucs = []
    for batch in (reference, current):
        X, y = prepare_xy(batch, params)
        aucs.append(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    return float(aucs[0] - aucs[1])


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: Params,
    *,
    auc_drop: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Drift report of `current` against `reference` (contract 4).

    KS for numeric features, chi2 for categorical ones, PSI for all of them;
    `drift_share` and the alert follow `params.monitoring`.

    Args:
        reference: Reference batch with `params.features.all`.
        current: Current batch with the same columns.
        params: Validated parameters.
        auc_drop: Result of `auc_drop`, None without labels.
        now: Report time; defaults to now. Written in UTC, ISO 8601.

    Returns:
        A JSON-ready dict with exactly the keys of `REPORT_KEYS`.
    """
    cfg = params.monitoring
    psi_values = {
        f: psi(reference[f], current[f], cfg.psi_bins) for f in params.features.all
    }
    ks = {f: ks_test(reference[f], current[f]) for f in params.features.numeric}
    chi2 = {f: chi2_test(reference[f], current[f]) for f in params.features.categorical}
    share = drift_share({**ks, **chi2}, cfg.alpha)
    stamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    return {
        "timestamp": stamp,
        "drift_share": share,
        "psi": psi_values,
        "ks": ks,
        "chi2": chi2,
        "auc_drop": auc_drop,
        "alert": is_alert(share, psi_values, auc_drop, cfg),
    }
