"""Challenger -> champion promotion gate (T7, D5, L4-1).

`models:/crediscore@challenger` and `models:/crediscore@champion` are scored
on THE SAME held-out batch (`--data`), each with its own thresholds.json
(never on stored historical metrics). The challenger becomes champion iff all
of these hold (bounds inclusive):
- auc: challenger ROC-AUC >= champion ROC-AUC;
- cost: challenger cost per file <= champion's (D3 costs, own thresholds);
- gender_raw_gap, gender_eo_gap: gender acceptance gaps (raw and equal
  opportunity, XNA excluded) <= `promotion.max_gender_gap_pts`;
- age_eo_gap: age equal-opportunity gap <= the baseline's (the age anchor,
  read from `--metrics`, written by `evaluate`) + `promotion.age_gap_margin_pts`.
No champion yet: only the three fairness gates apply. A missing age anchor
fails closed. The SYNOPSIS targets (AUC 0.75, PR-AUC 0.25, cost -15 %) are
reported KPIs, not gates.

On promotion: alias `champion` moves to the challenger's version, alias
`challenger` is removed, and the version is tagged `promoted_at`,
`promotion_previous_champion`, `promotion_data_md5` and each check's value
and limit (`promotion_<check>`, `promotion_<check>_limit`). Otherwise nothing
changes. `--dry-run` computes and writes the decision but changes nothing.
The decision goes to stdout and to `--output` (JSON, `failed` lists the
conditions that failed: a check name, `no_challenger` or
`age_anchor_missing`).

Exit code: 0 if the decision is "promote" (also with --dry-run), else 1.

Usage:
    poetry run python -m crediscore.models.promote
    poetry run python -m crediscore.models.promote --dry-run
        --data data/simulated/batch.parquet --output reports/promotion.json

Defaults: --data data/processed/test.parquet, --metrics reports/metrics.json,
--output reports/promotion.json, --params params.yaml. The MLflow server
comes from MLFLOW_TRACKING_URI.
"""

import argparse
import json
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
from crediscore.fairness.audit import audit, d5_checks
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
        The anchor in points, or None if the file or the number is missing.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        anchor = payload["baseline"]["fairness"]["age"]["eo_gap_pts"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return float(anchor) if isinstance(anchor, int | float) else None


def apply_promotion(
    client: MlflowClient,
    version: str,
    previous: str | None,
    checks: dict[str, dict[str, Any]],
    tags: dict[str, str],
) -> None:
    """Move `champion` to `version`, drop `challenger`, tag the version."""
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    client.delete_registered_model_alias(MODEL_NAME, "challenger")
    for name, c in checks.items():
        tags[f"promotion_{name}"] = repr(c["value"])
        tags[f"promotion_{name}_limit"] = repr(c["limit"])
    tags["promotion_previous_champion"] = previous or "none"
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point (see the module docstring).

    Returns:
        0 if the decision is "promote", else 1.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/test.parquet")
    )
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/promotion.json"))
    parser.add_argument("--dry-run", action="store_true", help="change nothing")
    args = parser.parse_args(argv)
    params = load_params(args.params)
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
    decision: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "data": str(args.data),
        "data_md5": file_md5(args.data),
        "model": MODEL_NAME,
        "challenger_version": ch_version,
        "champion_version": champ_version,
        "age_anchor_pts": anchor,
        "applied": False,
        "scores": {"challenger": None, "champion": None},
    }
    if ch_version is None:
        message = f"no @challenger version of {MODEL_NAME}: nothing to promote"
        decision.update(promote=False, failed=["no_challenger"], checks={})
    elif anchor is None:
        message = (
            f"no age anchor (baseline age equal-opportunity gap) in {args.metrics}: "
            "run evaluate first; not promoted (fail closed)"
        )
        decision.update(promote=False, failed=["age_anchor_missing"], checks={})
    else:
        df = pd.read_parquet(args.data)
        X, _ = prepare_xy(df, params)
        for side, version in [("challenger", ch_version), ("champion", champ_version)]:
            if version is not None:
                model, th = load_model(f"models:/{MODEL_NAME}/{version}")
                decision["scores"][side] = score(
                    df, model.predict_proba(X)[:, 1], th, params
                )
        decision["n_rows"] = len(df)
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
                {
                    "promoted_at": decision["timestamp"],
                    "promotion_data_md5": decision["data_md5"],
                },
            )
            decision["applied"] = True
        verdict = "promoted" if decision["applied"] else "would promote"
        if not decision["promote"]:
            verdict = f"not promoted, failed: {', '.join(decision['failed'])}"
        message = (
            f"challenger v{ch_version} vs champion v{champ_version or '-'} "
            f"on {args.data}: {verdict}"
        )
    decision.setdefault("n_rows", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(f"promotion: {message} ({args.output})")
    return 0 if decision["promote"] else 1


if __name__ == "__main__":
    sys.exit(main())
