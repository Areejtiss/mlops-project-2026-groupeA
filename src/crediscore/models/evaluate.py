"""Evaluate a model on the test split, against the LogReg baseline (L3-4).

The ONLY reader of the test split for model selection. It loads the model
(a `train` output directory or a `models:/` URI) with its own thresholds.json,
retrains the LogReg baseline exactly like `train` does on the same train
parquet, applies the same `decide` to both on test and writes
`reports/metrics.json` (`dvc metrics show`):

- ROC-AUC, PR-AUC, Brier, recall (D8), review share, cost per file;
- KPI cost vs the baseline = (cost_model - cost_baseline) / cost_baseline,
  and cost vs "accept all";
- thin file (D10): AUC, acceptance rate, good payers accepted vs the classic
  scorecard (refuse when EXT_SOURCE_1 and EXT_SOURCE_3 are missing, else our
  decision);
- fairness (D5, only ACCEPTE counts as accepted, D8): gender (XNA excluded)
  and age band raw and equal-opportunity gaps, and the D5 status: gender
  both gaps <= 10 pts; age equal-opportunity gap <= baseline's + 2 pts;
- every SYNOPSIS target with its measured value, met or missed, and the gap.

If MLFLOW_TRACKING_URI is set, the numbers are also logged as an `evaluate`
run of experiment `crediscore`.

Usage:
    poetry run python -m crediscore.models.evaluate --model models
    poetry run python -m crediscore.models.evaluate
        --model models:/crediscore@champion --output reports/metrics.json

Defaults: --train data/processed/train.parquet, --test
data/processed/test.parquet, --params params.yaml, --output reports/metrics.json.
"""

import argparse
import json
import operator
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score

from crediscore.config import Params, load_params
from crediscore.models.cost import ACCEPT, REFUSE, business_cost
from crediscore.models.threshold import NEVER, decide, zone_metrics
from crediscore.models.train import EXPERIMENT, prepare_xy, score_metrics, train

# D5 (decided 2026-10-25). ponytail: constants until lot 4 adds params `promotion.*`.
GENDER_COL, GENDER_EXCLUDED = "CODE_GENDER", "XNA"
GENDER_MAX_GAP_PTS = 10.0
AGE_COL, AGE_TOLERANCE_PTS = "AGE_YEARS", 2.0
AGE_BINS, AGE_LABELS = [0.0, 30.0, 50.0, np.inf], ["<30", "30-50", "≥50"]
# SYNOPSIS targets (validated by the teacher); recall comes from params (D8).
TARGETS: dict[str, tuple[str, float]] = {
    "roc_auc": (">=", 0.75),
    "pr_auc": (">=", 0.25),
    "brier": ("<=", 0.07),
    "cost_vs_baseline": ("<=", -0.15),
}
_OPS = {">=": operator.ge, "<=": operator.le}


def _gap(rates: pd.Series) -> float:
    """Max - min of the group rates, in points (0 with fewer than 2 groups)."""
    return float(100 * (rates.max() - rates.min())) if len(rates) > 1 else 0.0


def _group_gaps(
    accepted: pd.Series, good: pd.Series, groups: pd.Series
) -> dict[str, Any]:
    keep = groups.notna()
    acc, ok, grp = accepted[keep], good[keep], groups[keep]
    raw = acc.groupby(grp, observed=True).mean()
    eo = acc[ok].groupby(grp[ok], observed=True).mean()
    return {
        "raw_gap_pts": _gap(raw),
        "eo_gap_pts": _gap(eo),
        "acceptance_by_group": {str(k): float(v) for k, v in raw.items()},
        "acceptance_good_payers_by_group": {str(k): float(v) for k, v in eo.items()},
        "excluded_rows": int((~keep).sum()),
    }


def fairness(
    df: pd.DataFrame, decisions: npt.ArrayLike, *, target: str
) -> dict[str, Any]:
    """Gender and age-band acceptance gaps, raw and equal opportunity (D5).

    Args:
        df: Test rows with CODE_GENDER, AGE_YEARS and the target.
        decisions: One D4 code per row; only ACCEPTE counts as accepted (D8).
        target: Label column (0 = repaid: the equal-opportunity population).

    Returns:
        {"gender": ..., "age": ...}, each with raw_gap_pts, eo_gap_pts, the
        per-group rates and the number of rows left out (XNA, missing age).
    """
    idx = df.index
    accepted = pd.Series(np.asarray(decisions) == ACCEPT, index=idx)
    good = pd.Series(df[target].to_numpy() == 0, index=idx)
    gender = df[GENDER_COL].where(df[GENDER_COL] != GENDER_EXCLUDED)
    age = pd.cut(df[AGE_COL], AGE_BINS, right=False, labels=AGE_LABELS)
    return {
        "gender": _group_gaps(accepted, good, gender),
        "age": _group_gaps(accepted, good, age),
    }


def thin_file(
    y: npt.ArrayLike,
    proba: npt.ArrayLike,
    decisions: npt.ArrayLike,
    thin: npt.ArrayLike,
    costs: dict[str, float],
) -> dict[str, Any]:
    """Thin-file KPI against the classic-scorecard policy (D10).

    The scorecard refuses every thin file and keeps our decision elsewhere.

    Args:
        y: Labels.
        proba: Calibrated probabilities.
        decisions: Our D4 codes.
        thin: 1 for a thin file (EXT_SOURCE_1 and EXT_SOURCE_3 missing).
        costs: cost_fn, cost_fp, cost_review.

    Returns:
        Thin share, AUC (None if one class only), acceptance rate, good payers
        accepted (thin and all) for us and the scorecard, their ratio, and the
        scorecard cost per file.
    """
    y, p, d = np.asarray(y), np.asarray(proba, dtype=float), np.asarray(decisions)
    t = np.asarray(thin).astype(bool)
    scorecard = np.where(t, REFUSE, d)
    good = y == 0
    ours = int(np.sum(good & (d == ACCEPT)))
    theirs = int(np.sum(good & (scorecard == ACCEPT)))
    return {
        "thin_share": float(t.mean()),
        "thin_roc_auc": (
            float(roc_auc_score(y[t], p[t])) if len(set(y[t].tolist())) == 2 else None
        ),
        "thin_acceptance_rate": float(np.mean(d[t] == ACCEPT)) if t.any() else None,
        "thin_good_payers_accepted": int(np.sum(good & t & (d == ACCEPT))),
        "scorecard_thin_good_payers_accepted": int(
            np.sum(good & t & (scorecard == ACCEPT))
        ),
        "good_payers_accepted": ours,
        "scorecard_good_payers_accepted": theirs,
        "good_payers_accepted_ratio": ours / theirs if theirs else None,
        "scorecard_cost_per_file": business_cost(y, scorecard, **costs) / len(y),
    }


def report(
    df: pd.DataFrame, proba: npt.ArrayLike, low: float, high: float, params: Params
) -> dict[str, Any]:
    """Every test metric of one model with its thresholds (no I/O)."""
    y = df[params.target].to_numpy().astype(int)
    p = np.asarray(proba, dtype=float)
    costs = {
        "cost_fn": params.cost.cost_fn,
        "cost_fp": params.cost.cost_fp,
        "cost_review": params.cost.cost_review,
    }
    d = decide(p, low, high)
    return {
        "thresholds": {"low": low, "high": high},
        **score_metrics(y, p),
        **zone_metrics(y, p, low, high, **costs),
        "accept_all_cost_per_file": zone_metrics(y, p, NEVER, NEVER, **costs)[
            "cost_per_file"
        ],
        "thin_file": thin_file(y, p, d, df["IS_THIN_FILE"].to_numpy(), costs),
        "fairness": fairness(df, d, target=params.target),
    }


def d5_status(
    model_fair: dict[str, Any], baseline_fair: dict[str, Any]
) -> dict[str, Any]:
    """D5: gender gaps <= 10 pts (both); age EO gap <= baseline's + 2 pts.

    Args:
        model_fair: `fairness` output of the evaluated model.
        baseline_fair: `fairness` output of the baseline on the same test rows.

    Returns:
        gender_pass, age_pass, pass and the age limit used (bounds inclusive).
    """
    g = model_fair["gender"]
    limit = baseline_fair["age"]["eo_gap_pts"] + AGE_TOLERANCE_PTS
    gender_ok = bool(
        g["raw_gap_pts"] <= GENDER_MAX_GAP_PTS and g["eo_gap_pts"] <= GENDER_MAX_GAP_PTS
    )
    age_ok = bool(model_fair["age"]["eo_gap_pts"] <= limit)
    return {
        "gender_pass": gender_ok,
        "age_pass": age_ok,
        "pass": gender_ok and age_ok,
        "gender_max_gap_pts": GENDER_MAX_GAP_PTS,
        "age_eo_limit_pts": limit,
    }


def kpis(
    model: dict[str, Any], baseline: dict[str, Any], params: Params
) -> dict[str, Any]:
    """Cost KPIs, D5 status and every target (value, target, met, gap).

    `gap` = value - target (signed); `met` uses inclusive bounds. Targets are
    never relaxed here (L3-6).
    """
    base_cost = baseline["cost_per_file"]
    values = {
        **{k: model[k] for k in ("roc_auc", "pr_auc", "brier", "recall")},
        "cost_vs_baseline": (model["cost_per_file"] - base_cost) / base_cost,
    }
    targets = {**TARGETS, "recall": (">=", params.threshold.min_recall)}
    return {
        "cost_vs_baseline": values["cost_vs_baseline"],
        "cost_vs_accept_all": (
            model["cost_per_file"] - model["accept_all_cost_per_file"]
        )
        / model["accept_all_cost_per_file"],
        "d5": d5_status(model["fairness"], baseline["fairness"]),
        "targets": {
            k: {
                "value": values[k],
                "target": t,
                # rounded: (0.255 - 0.30) / 0.30 must meet -0.15 exactly
                "met": bool(_OPS[op](round(values[k], 12), t)),
                "gap": values[k] - t,
            }
            for k, (op, t) in targets.items()
        },
    }


def load_model(ref: str) -> tuple[Any, dict[str, float]]:
    """Model and its thresholds.json, from a `train` directory or `models:/`."""
    if ref.startswith("models:/"):
        model = mlflow.sklearn.load_model(ref)
        name, _, alias = ref.removeprefix("models:/").partition("@")
        client = mlflow.MlflowClient()
        if alias:
            version = client.get_model_version_by_alias(name, alias)
        else:
            name, _, number = name.partition("/")
            version = client.get_model_version(name, number)
        th = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/thresholds.json")
        return model, th
    path = Path(ref)
    th = json.loads((path / "thresholds.json").read_text(encoding="utf-8"))
    return joblib.load(path / "model.joblib"), th


def flatten(d: dict[str, Any], prefix: str) -> dict[str, float]:
    """Numeric leaves of a nested dict as MLflow metric names (`a.b.c`)."""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        elif isinstance(v, bool | int | float):
            out[key.replace("≥", "ge").replace("<", "lt")] = float(v)  # age bands
    return out


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--model", required=True, help="train output dir or models:/")
    parser.add_argument(
        "--train", type=Path, default=Path("data/processed/train.parquet")
    )
    parser.add_argument(
        "--test", type=Path, default=Path("data/processed/test.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args(argv)
    params = load_params(args.params)

    model, th = load_model(args.model)
    base = train(pd.read_parquet(args.train), "logreg", params)
    base_th = base["thresholds"]
    test = pd.read_parquet(args.test)
    X, _ = prepare_xy(test, params)
    model_rep = report(
        test, model.predict_proba(X)[:, 1], th["low"], th["high"], params
    )
    base_rep = report(
        test,
        base["model"].predict_proba(X)[:, 1],
        base_th["low"],
        base_th["high"],
        params,
    )
    metrics = {
        "model_uri": args.model,
        "n_test": len(test),
        "thresholds": {"low": th["low"], "high": th["high"]},
        "model": model_rep,
        "baseline": base_rep,
        "kpi": kpis(model_rep, base_rep, params),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="evaluate"):
            mlflow.set_tags({"stage": "evaluate", "evaluated_model": args.model})
            mlflow.log_metrics(
                {
                    **flatten(model_rep, "test_"),
                    **flatten(base_rep, "baseline_test_"),
                    **flatten(metrics["kpi"], "kpi_"),
                }
            )
            mlflow.log_dict(metrics, "metrics.json")
    k = metrics["kpi"]
    print(
        f"test ROC-AUC {model_rep['roc_auc']:.4f} "
        f"(baseline {base_rep['roc_auc']:.4f}), "
        f"cost/file {model_rep['cost_per_file']:.4f} (baseline "
        f"{base_rep['cost_per_file']:.4f}, KPI {k['cost_vs_baseline']:+.1%}), "
        f"D5 pass {k['d5']['pass']}"
    )


if __name__ == "__main__":
    main()
