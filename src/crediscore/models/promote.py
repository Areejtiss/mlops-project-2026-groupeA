"""Challenger -> champion promotion gate (T7, D5, L4-1).

`models:/crediscore@challenger` and `models:/crediscore@champion` are scored
on THE SAME fresh labelled batch (`--data`, required): the `holdout` stage's
`data/processed/holdout.parquet` (raw rows outside the featurize sample) or a
real production batch. Never the test split (except a manual demo where it
is named explicitly), and never `data/simulated/economic_crisis.parquet`: it
reuses test rows, so it is a drift-detection demo only. Each model uses
its own thresholds.json (authoritative; its
version tags must match it). The challenger becomes champion iff all of these
hold (bounds inclusive):
- auc: challenger ROC-AUC >= champion ROC-AUC;
- cost: challenger cost per file <= champion's (D3 costs, own thresholds);
- gender_raw_gap, gender_eo_gap: gender acceptance gaps (raw and equal
  opportunity, XNA excluded) <= `promotion.max_gender_gap_pts`;
- age_eo_gap: age equal-opportunity gap <= the baseline's (the age anchor,
  read from `--metrics`, written by `evaluate`) + `promotion.age_gap_margin_pts`.
No champion yet: only the three fairness gates apply. The SYNOPSIS targets
(AUC 0.75, PR-AUC 0.25, cost -15 %) are reported KPIs, not gates.

Fail closed (not promoted, nothing changed), with the code in `failed`:
`no_challenger`, `already_champion` (challenger == champion version),
`age_anchor_missing`, `batch_too_small` (< `promotion.min_rows`),
`gender_groups_missing` (fewer than 2 gender groups with
`promotion.min_group_rows` rows), `thresholds_mismatch` (version tags !=
thresholds.json), or `["error", "<message>"]` on any exception.

On promotion, in this order: the version is tagged
`promotion_previous_champion`, `promotion_data_md5` and each check's value and
limit (`promotion_<check>`, `promotion_<check>_limit`); alias `challenger` is
removed; alias `champion` moves to the challenger's version; only then is
`promoted_at` written, so an aborted promotion leaves none. `--dry-run`
computes and writes the decision but changes nothing. The decision goes to
stdout and to `--output` (JSON, rewritten at every run).

Exit code: 0 if the decision is "promote" (also with --dry-run), else 1.

Usage:
    poetry run python -m crediscore.models.promote
        --data data/processed/holdout.parquet       (one line)
    poetry run python -m crediscore.models.promote --dry-run
        --data <batch.parquet> --output reports/promotion.json

Defaults: --metrics reports/metrics.json, --output reports/promotion.json,
--params params.yaml. The MLflow server comes from MLFLOW_TRACKING_URI.
"""

import argparse
import json
import math
import operator
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlflow.tracking import MlflowClient

from crediscore.config import Params, PromotionConfig, load_params
from crediscore.fairness.audit import (
    GENDER_COL,
    GENDER_EXCLUDED,
    audit,
    d5_checks,
)
from crediscore.models.evaluate import load_model
from crediscore.models.registry import MODEL_NAME
from crediscore.models.threshold import decide as decide_zones
from crediscore.models.threshold import zone_metrics
from crediscore.models.train import file_md5, prepare_xy, score_metrics


def score(
    df: pd.DataFrame, proba: npt.ArrayLike, th: dict[str, float], params: Params
) -> dict[str, Any]:
    """One model on the batch with its thresholds (no I/O).

    Args:
        df: Held-out rows with the target, CODE_GENDER and AGE_YEARS.
        proba: The model's calibrated probabilities on `df`.
        th: The model's thresholds (`low`, `high`).
        params: Validated parameters (costs, target).

    Returns:
        thresholds, roc_auc, pr_auc, brier, cost_per_file, recall,
        review_share and the fairness audit.
    """
    y = df[params.target].to_numpy().astype(int)
    p = np.asarray(proba, dtype=float)
    low, high = th["low"], th["high"]
    zones = zone_metrics(y, p, low, high, **params.cost.model_dump())
    return {
        "thresholds": {"low": low, "high": high},
        **score_metrics(y, p),
        "cost_per_file": zones["cost_per_file"],
        "recall": zones["recall"],
        "review_share": zones["review_share"],
        "fairness": audit(df, decide_zones(p, low, high), target=params.target),
    }


def _check(
    value: float, limit: float, op: Callable[[float, float], bool]
) -> dict[str, Any]:
    return {"value": value, "limit": limit, "pass": bool(op(value, limit))}


def decide(
    challenger: dict[str, Any],
    champion: dict[str, Any] | None,
    *,
    age_anchor_pts: float,
    promotion: PromotionConfig,
) -> dict[str, Any]:
    """Apply the gate to two `score` outputs (pure).

    Args:
        challenger: The challenger's scores.
        champion: The champion's scores on the same batch, None if none yet.
        age_anchor_pts: The baseline's age equal-opportunity gap.
        promotion: The gate tolerances (`params.promotion`).

    Returns:
        promote (bool), failed (check names) and checks ({value, limit, pass}).
    """
    checks: dict[str, dict[str, Any]] = {}
    if champion is not None:
        checks["auc"] = _check(challenger["roc_auc"], champion["roc_auc"], operator.ge)
        checks["cost"] = _check(
            challenger["cost_per_file"], champion["cost_per_file"], operator.le
        )
    checks.update(
        d5_checks(
            challenger["fairness"],
            age_anchor_pts=age_anchor_pts,
            max_gender_gap_pts=promotion.max_gender_gap_pts,
            age_gap_margin_pts=promotion.age_gap_margin_pts,
        )
    )
    failed = [name for name, c in checks.items() if not c["pass"]]
    return {"promote": not failed, "failed": failed, "checks": checks}


def read_age_anchor(path: Path) -> float | None:
    """The baseline's age equal-opportunity gap from evaluate's metrics.json.

    Returns:
        The anchor in points, or None if the file or the number is missing,
        not a number (a bool is not), or not finite.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        anchor = payload["baseline"]["fairness"]["age"]["eo_gap_pts"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if isinstance(anchor, bool) or not isinstance(anchor, int | float):
        return None
    return float(anchor) if math.isfinite(anchor) else None


def batch_check(df: pd.DataFrame, promotion: PromotionConfig) -> str | None:
    """Fail-closed guards on the batch before any score (gaps too noisy).

    Returns:
        `batch_too_small` below `min_rows` rows, `gender_groups_missing` if
        fewer than 2 gender groups (XNA excluded) have `min_group_rows` rows,
        else None (bounds inclusive).
    """
    if len(df) < promotion.min_rows:
        return "batch_too_small"
    if not gender_groups_ok(df, promotion.min_group_rows):
        return "gender_groups_missing"
    return None


def gender_groups_ok(df: pd.DataFrame, min_group_rows: int) -> bool:
    """True if at least 2 gender groups (XNA excluded) have min_group_rows rows.

    Below that a gender gap compares nothing (one group gives 0) or is too
    noisy. Also used by the drift report's D5 status (lot 6).
    """
    sizes = df[GENDER_COL][df[GENDER_COL] != GENDER_EXCLUDED].value_counts()
    return int((sizes >= min_group_rows).sum()) >= 2


def thresholds_match(client: MlflowClient, version: str, th: dict[str, Any]) -> bool:
    """True if the version tags carry exactly the thresholds.json values (D9)."""
    tags = client.get_model_version(MODEL_NAME, version).tags
    try:
        return bool(
            float(tags["threshold_low"]) == th["low"]
            and float(tags["threshold_high"]) == th["high"]
        )
    except (KeyError, ValueError):
        return False


def apply_promotion(
    client: MlflowClient,
    version: str,
    previous: str | None,
    checks: dict[str, dict[str, Any]],
    tags: dict[str, str],
    promoted_at: str,
) -> None:
    """Tag the version, drop `challenger`, move `champion`, then `promoted_at`.

    The champion alias moves after the other tags, so a failure part-way
    leaves the old champion serving and a re-run sees the true previous
    champion; `promoted_at` is written once the alias has moved, so an
    aborted promotion never carries it.
    """
    for name, c in checks.items():
        tags[f"promotion_{name}"] = repr(c["value"])
        tags[f"promotion_{name}_limit"] = repr(c["limit"])
    tags["promotion_previous_champion"] = previous or "none"
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)
    client.delete_registered_model_alias(MODEL_NAME, "challenger")
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    client.set_model_version_tag(MODEL_NAME, version, "promoted_at", promoted_at)


def _fail(decision: dict[str, Any], code: str) -> None:
    decision.update(promote=False, failed=[code], checks={})


def _run(args: argparse.Namespace, decision: dict[str, Any]) -> str:
    """Fill `decision` (and promote unless dry run); return the log message."""
    params = load_params(args.params)
    decision["data_md5"] = file_md5(args.data)
    client = MlflowClient()
    try:
        aliases = {
            k: str(v)
            for k, v in client.get_registered_model(MODEL_NAME).aliases.items()
        }
    except mlflow.exceptions.MlflowException:
        aliases = {}
    ch_version, champ_version = aliases.get("challenger"), aliases.get("champion")
    anchor = read_age_anchor(args.metrics)
    decision.update(
        challenger_version=ch_version,
        champion_version=champ_version,
        age_anchor_pts=anchor,
    )
    if ch_version is None:
        _fail(decision, "no_challenger")
        return f"no @challenger version of {MODEL_NAME}: nothing to promote"
    if ch_version == champ_version:
        _fail(decision, "already_champion")
        return f"@challenger v{ch_version} is already @champion: nothing to do"
    if anchor is None:
        _fail(decision, "age_anchor_missing")
        return (
            f"no age anchor (baseline age equal-opportunity gap) in {args.metrics}: "
            "run evaluate first; not promoted (fail closed)"
        )
    df = pd.read_parquet(args.data)
    decision["n_rows"] = len(df)
    problem = batch_check(df, params.promotion)
    if problem is not None:
        _fail(decision, problem)
        return f"{args.data} ({len(df)} rows): {problem}; not promoted (fail closed)"
    X, _ = prepare_xy(df, params)
    for side, version in [("challenger", ch_version), ("champion", champ_version)]:
        if version is None:
            continue
        model, th = load_model(f"models:/{MODEL_NAME}/{version}")
        if not thresholds_match(client, version, th):
            _fail(decision, "thresholds_mismatch")
            return (
                f"v{version}: threshold tags differ from its thresholds.json; "
                "not promoted (fail closed)"
            )
        decision["scores"][side] = score(df, model.predict_proba(X)[:, 1], th, params)
    decision.update(
        decide(
            decision["scores"]["challenger"],
            decision["scores"]["champion"],
            age_anchor_pts=anchor,
            promotion=params.promotion,
        )
    )
    if decision["promote"] and not args.dry_run:
        apply_promotion(
            client,
            ch_version,
            champ_version,
            decision["checks"],
            {"promotion_data_md5": decision["data_md5"]},
            promoted_at=decision["timestamp"],
        )
        decision["applied"] = True
    verdict = "promoted" if decision["applied"] else "would promote"
    if not decision["promote"]:
        verdict = f"not promoted, failed: {', '.join(decision['failed'])}"
    return (
        f"challenger v{ch_version} vs champion v{champ_version or '-'} "
        f"on {args.data}: {verdict}"
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point (see the module docstring).

    Returns:
        0 if the decision is "promote", else 1 (also on any error).
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="fresh labelled batch: data/processed/holdout.parquet or production"
        " (not test, not the crisis batch)",
    )
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/promotion.json"))
    parser.add_argument("--dry-run", action="store_true", help="change nothing")
    args = parser.parse_args(argv)
    decision: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "data": str(args.data),
        "data_md5": None,
        "n_rows": None,
        "model": MODEL_NAME,
        "challenger_version": None,
        "champion_version": None,
        "age_anchor_pts": None,
        "applied": False,
        "scores": {"challenger": None, "champion": None},
        "promote": False,
        "failed": [],
        "checks": {},
    }
    try:
        message = _run(args, decision)
    except Exception as err:  # fail closed: a stale decision must never survive
        detail = f"{type(err).__name__}: {err}"
        decision.update(promote=False, failed=["error", detail])
        message = f"error, not promoted: {detail}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(f"promotion: {message} ({args.output})")
    return 0 if decision["promote"] else 1


if __name__ == "__main__":
    sys.exit(main())
