"""Drift detection: PSI, KS and chi2 tests, drift report (contract 4).

Compares a current batch with a reference batch, feature by feature, and
writes the drift report JSON consumed by Iyed (Prometheus gauges) and Areej
(monitoring flow: retrain on alert).

Usage:
    poetry run python -m crediscore.monitoring.drift
        --reference data/processed/test.parquet
        --current data/simulated/economic_crisis.parquet
        [--model models/lightgbm | --model models:/crediscore@champion]
        [--out-dir reports/drift] [--params params.yaml]
    -> writes <out-dir>/<YYYY-MM-DD_HHMMSS>.json and .html (Evidently, UTC
       time of the report) and <out-dir>/latest.json (same JSON). The exit
       code is 0 whatever the alert: the alert lives in the JSON.

In code:
    report = compute_drift_report(reference, current, params,
                                  auc_drop=auc_drop(model, reference, current,
                                                    params))
"""

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from scipy import stats
from sklearn.metrics import roc_auc_score

from crediscore.config import MonitoringConfig, Params, load_params
from crediscore.models.evaluate import load_model
from crediscore.models.train import prepare_xy

REPORT_KEYS = ("timestamp", "drift_share", "psi", "ks", "chi2", "auc_drop", "alert")
# Floor of a bin share in the PSI, so that an empty bin gives a finite term.
EPSILON = 1e-4
_NAN_BIN = -1
# Half-width of the single bin of a constant column, relative to its value.
_CONSTANT_PAD = 1e-6
# auc_drop needs at least this many defaults in each labelled batch.
MIN_POSITIVES = 50


def _numeric_edges(reference: pd.Series, bins: int) -> npt.NDArray[np.float64]:
    """Bin edges from the reference, none of them on a reference value.

    Inner edges sit halfway between each inner quantile ("lower" method, so a
    reference value) and the next distinct reference value; deduplicating
    the quantiles merges the deciles that fall on the same heaped value into
    one edge (a 0/1 column gets one edge, at 0.5). Outer edges sit half a gap
    beyond the min and the max, so values outside the reference range get
    their own open-ended bins. A constant column is padded by 1e-6 relative.
    """
    ref = reference.dropna().to_numpy(dtype=float)
    values = np.unique(ref)
    if values.size == 0:
        return np.array([], dtype=float)
    if values.size == 1:
        pad = _CONSTANT_PAD * max(abs(values[0]), 1.0)
        return np.array([values[0] - pad, values[0] + pad])
    inner = np.unique(
        np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1], method="lower")
    )
    inner = inner[inner < values[-1]]  # the max has no next value
    following = values[np.searchsorted(values, inner, side="right")]
    low = values[0] - (values[1] - values[0]) / 2
    high = values[-1] + (values[-1] - values[-2]) / 2
    return np.unique(np.r_[low, (inner + following) / 2, high])


def _numeric_bins(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[pd.Series, pd.Series]:
    """Bin index of every value (see `_numeric_edges`); NaN has its own bin."""
    edges = _numeric_edges(reference, bins)

    def index(s: pd.Series) -> pd.Series:
        x = s.to_numpy(dtype=float)
        idx = np.searchsorted(edges, x, side="right")
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
    (see `_numeric_edges`); other columns use one bin per category, a category
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
        The p-value; 0.0 (certain drift) when exactly one side has no value;
        None when neither side has a value.
    """
    ref, cur = reference.dropna(), current.dropna()
    if ref.empty and cur.empty:
        return None
    if ref.empty or cur.empty:
        return 0.0
    return float(stats.ks_2samp(ref, cur).pvalue)


def chi2_test(reference: pd.Series, current: pd.Series) -> float:
    """Chi-squared test of independence between batch and category.

    Missing values are a category of their own, and so is a category unseen
    in the reference.

    Args:
        reference: Reference values of a categorical feature.
        current: Current values of the same feature.

    Returns:
        The p-value; 1.0 when both batches hold a single category. With two
        categories (1 degree of freedom) scipy applies the Yates correction.
    """
    # A single category gives dof 0, for which scipy returns p = 1.
    counts = _counts(reference, current)
    return float(stats.chi2_contingency(counts.T.to_numpy()).pvalue)


def drift_share(p_values: Mapping[str, float | None], alpha: float) -> float:
    """Share of features whose p-value is strictly below alpha.

    Args:
        p_values: p-value per model feature (None = no value on either
            side, not drifted).
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
    *,
    n_rows: int,
) -> bool:
    """Alert rule of contract 4.

    Alert when `share` >= `drift_share_alert`, or the PSI of one key feature
    >= `psi_alert`, or `drop` > `auc_drop_alert` (when the AUC is known).
    The first two rules only apply when `n_rows` >= `min_rows`: on a small
    batch the tests and the PSI are too noisy to retrain on.

    Args:
        share: The batch's drift_share.
        psi_values: PSI per feature.
        drop: Reference AUC minus current AUC, or None when not computable.
        config: The `monitoring` parameters.
        n_rows: Rows of the smaller of the two batches.

    Returns:
        True if the batch must raise the alert (retrain).
    """
    data_drift = share >= config.drift_share_alert or any(
        psi_values[f] >= config.psi_alert for f in config.key_features
    )
    return (n_rows >= config.min_rows and data_drift) or (
        drop is not None and drop > config.auc_drop_alert
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
        The AUC drop (> 0 = the model got worse), or None when it is not
        computable: a batch without the label column, or whose labelled rows
        are fewer than `monitoring.min_labelled`, hold fewer than
        MIN_POSITIVES defaults or a single class. Rows without a label are
        left out.
    """
    aucs = []
    for batch in (reference, current):
        if params.target not in batch:
            return None
        labelled = batch.dropna(subset=[params.target])
        X, y = prepare_xy(labelled, params)
        positives = int(y.sum())
        if (
            len(y) < params.monitoring.min_labelled
            or positives < MIN_POSITIVES
            or positives == len(y)
        ):
            return None
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
        now: Report time, timezone-aware; defaults to now. Written in UTC,
            ISO 8601, to the second.

    Returns:
        A JSON-ready dict with exactly the keys of `REPORT_KEYS`.

    Raises:
        ValueError: If a batch has no row, `now` has no timezone, or
            `auc_drop` is NaN or infinite.
    """
    if reference.empty or current.empty:
        raise ValueError("empty batch: the reference and current need rows")
    if now is not None and now.utcoffset() is None:
        raise ValueError("now must carry a timezone (e.g. datetime.now(UTC))")
    if auc_drop is not None and not np.isfinite(auc_drop):
        raise ValueError(f"auc_drop must be finite or None, got {auc_drop}")
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
        "alert": is_alert(
            share,
            psi_values,
            auc_drop,
            cfg,
            n_rows=min(len(reference), len(current)),
        ),
    }


def write_html(
    reference: pd.DataFrame, current: pd.DataFrame, params: Params, path: Path
) -> None:
    """Evidently 0.7 `DataDriftPreset` HTML report on the model features.

    Same tests and alpha as our JSON (KS numeric, chi2 categorical), so the
    two agree on which feature drifted; Evidently's own drift_share threshold
    is set to `monitoring.drift_share_alert`.

    Args:
        reference: Reference batch.
        current: Current batch.
        params: Validated parameters.
        path: HTML file to write.
    """
    cfg = params.monitoring
    definition = DataDefinition(
        numerical_columns=params.features.numeric,
        categorical_columns=params.features.categorical,
    )
    preset = DataDriftPreset(
        columns=params.features.all,
        num_method="ks",
        cat_method="chisquare",
        threshold=cfg.alpha,
        drift_share=cfg.drift_share_alert,
    )
    features = params.features.all
    snapshot = Report([preset]).run(
        Dataset.from_pandas(current[features], data_definition=definition),
        Dataset.from_pandas(reference[features], data_definition=definition),
    )
    snapshot.save_html(str(path))


def main(argv: list[str] | None = None) -> None:
    """Compare two parquet batches, write the JSON report and the HTML."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--model", help="train output dir or models:/ URI")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/drift"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)

    params = load_params(args.params)
    reference = pd.read_parquet(args.reference)
    current = pd.read_parquet(args.current)
    drop = (
        auc_drop(load_model(args.model)[0], reference, current, params)
        if args.model
        else None
    )
    report = compute_drift_report(reference, current, params, auc_drop=drop)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(report["timestamp"]).strftime("%Y-%m-%d_%H%M%S")
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    for name in (f"{stamp}.json", "latest.json"):
        (args.out_dir / name).write_text(text, encoding="utf-8")
    write_html(reference, current, params, args.out_dir / f"{stamp}.html")

    top = sorted(report["psi"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"Wrote {args.out_dir / stamp}.json/.html and latest.json")
    print(f"  drift_share {report['drift_share']:.3f}  auc_drop {drop}")
    print("  top PSI " + ", ".join(f"{f} {v:.3f}" for f, v in top))
    print(f"  alert {report['alert']}")


if __name__ == "__main__":
    main()
