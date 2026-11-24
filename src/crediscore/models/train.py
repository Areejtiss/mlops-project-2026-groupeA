"""Train, calibrate, set the decision thresholds, save and (optionally) log.

Steps: load params; carve validation out of train (D1); fit the pipeline inside
`CalibratedClassifierCV(ensemble=False)` (T2); optimize the 3-zone thresholds
on validation (D3, D8); write `model.joblib`, `thresholds.json` and
`metrics.json` and `model_params.json` (T4). If MLFLOW_TRACKING_URI is set,
log one run with metrics, curves, the model (cloudpickle, all-optional
signature, contract 2) and tags. The model is NOT registered here
(`crediscore.models.registry`).

`--tune N` first runs N Optuna trials on the fit part (`crediscore.models.tune`,
each trial a nested MLflow run), then fits the best params as above.
`--model-params FILE` fixes the hyperparameters instead (e.g. the
`model_params.json` of a tuned run). `--exclude COL` keeps COL in the input
but out of the model and requires `--experiment NAME`, which tags the run
(D5: `no_age`).

The input parquet is featurized (Areej's output): it already has
`params.features.all`, built by `crediscore.data.featurize` (`fix_days_employed`
then `add_derived_features`).

Usage:
    poetry run python -m crediscore.models.train --model logreg
    poetry run python -m crediscore.models.train --model lightgbm --output-dir models
    poetry run python -m crediscore.models.train --model xgboost --tune 25
    poetry run python -m crediscore.models.train --model lightgbm --exclude AGE_YEARS
        --experiment no_age --model-params models/model_params.json   (one line)
    # CML mode: train on a small file (split internally), also write roc.png
    poetry run python -m crediscore.models.train --output-dir reports/cml
        --sample tests/fixtures/ci_sample.parquet       (one line)

Defaults: --train data/processed/train.parquet, --params params.yaml.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import yaml  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from mlflow.models import ModelSignature  # noqa: E402
from mlflow.types import ColSpec, DataType, Schema, TensorSpec  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from crediscore.config import Params, load_params  # noqa: E402
from crediscore.models.cost import DECISIONS  # noqa: E402
from crediscore.models.pipeline import (  # noqa: E402
    MODELS,
    build_pipeline,
    calibrate,
    imbalance_params,
)
from crediscore.models.split import split_validation  # noqa: E402
from crediscore.models.threshold import (  # noqa: E402
    NEVER,
    decide,
    optimize_thresholds,
    zone_metrics,
)
from crediscore.models.tune import SEARCH_SPACES, TrialCallback, tune  # noqa: E402

EXPERIMENT = "crediscore"
# Unit of the SHAP contributions of each model (contract 2 section 6).
SHAP_UNIT = {
    "logreg": "log_odds",
    "lightgbm": "log_odds",
    "xgboost": "log_odds",
    "random_forest": "probability",
}
DVC_FILE = Path("data/raw/application_train.csv.dvc")


def prepare_xy(df: pd.DataFrame, params: Params) -> tuple[pd.DataFrame, np.ndarray]:
    """Select the model input (numerics as float64, contract 2) and the label.

    Args:
        df: Featurized rows with `params.features.all` and the target.
        params: Validated parameters.

    Returns:
        (X with exactly `params.features.all`, y as an int array).
    """
    X = df[params.features.all].copy()
    X[params.features.numeric] = X[params.features.numeric].astype("float64")
    return X, df[params.target].to_numpy().astype(int)


def fit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    name: str,
    params: Params,
    *,
    numeric: list[str] | None = None,
    model_params: dict[str, Any] | None = None,
) -> CalibratedClassifierCV:
    """Fit the calibrated pipeline on the fit part (T2).

    `numeric` defaults to `params.features.numeric`; `model_params` override
    the classifier defaults, on top of `imbalance_params` (T5).
    """
    pipe = build_pipeline(
        name,
        numeric=params.features.numeric if numeric is None else numeric,
        categorical=params.features.categorical,
        seed=params.data.random_state,
        model_params={**imbalance_params(name, y), **(model_params or {})},
    )
    return calibrate(pipe).fit(X, y)


def score_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Threshold-free quality metrics (accuracy is never reported)."""
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


def thresholds_payload(
    low: float, high: float, params: Params, val: dict[str, float]
) -> dict[str, float]:
    """Content of thresholds.json (contract 2 section 5), full precision."""
    return {
        "low": low,
        "high": high,
        "cost_fn": params.cost.cost_fn,
        "cost_fp": params.cost.cost_fp,
        "cost_review": params.cost.cost_review,
        "min_recall": params.threshold.min_recall,
        "max_review_share": params.threshold.max_review_share,
        "val_recall": val["recall"],
        "val_review_share": val["review_share"],
        "val_cost_per_file": val["cost_per_file"],
    }


def build_signature(numeric: list[str], categorical: list[str]) -> ModelSignature:
    """Input signature of contract 2: double/string columns, all optional."""
    cols: list[ColSpec | TensorSpec] = [
        ColSpec(DataType.double, c, required=False) for c in numeric
    ]
    cols += [ColSpec(DataType.string, c, required=False) for c in categorical]
    return ModelSignature(inputs=Schema(cols))


def make_figures(
    y: np.ndarray, proba: np.ndarray, low: float, high: float
) -> dict[str, Figure]:
    """ROC, PR, calibration curves and the label x decision matrix."""
    figs: dict[str, Figure] = {}
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, label=f"AUC {roc_auc_score(y, proba):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="Recall", title="ROC (validation)")
    ax.legend(loc="lower right")
    figs["roc_curve"] = fig

    fig, ax = plt.subplots(figsize=(4.5, 4))
    prec, rec, _ = precision_recall_curve(y, proba)
    ax.plot(rec, prec, label=f"AP {average_precision_score(y, proba):.3f}")
    ax.axhline(y.mean(), ls="--", color="grey")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (validation)")
    ax.legend(loc="upper right")
    figs["pr_curve"] = fig

    fig, ax = plt.subplots(figsize=(4.5, 4))
    frac, mean_p = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, marker="o", label=f"Brier {brier_score_loss(y, proba):.4f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set(xlabel="Predicted", ylabel="Observed default rate", title="Calibration")
    ax.legend(loc="upper left")
    figs["calibration_curve"] = fig

    fig, ax = plt.subplots(figsize=(5, 3))
    counts = confusion_counts(y, proba, low, high)
    ax.imshow(counts, cmap="Blues")
    for (i, j), v in np.ndenumerate(counts):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks(range(3), DECISIONS)
    ax.set_yticks(range(2), ["TARGET=0", "TARGET=1"])
    ax.set_title("Decisions on validation")
    figs["confusion_matrix"] = fig
    for f in figs.values():
        f.tight_layout()
    return figs


def confusion_counts(
    y: np.ndarray, proba: np.ndarray, low: float, high: float
) -> npt.NDArray[np.int64]:
    """2 x 3 counts: label (0, 1) x decision (ACCEPTE, A_ETUDIER, REFUSE)."""
    d = decide(proba, low, high)
    return np.array(
        [[int(np.sum((y == t) & (d == c))) for c in DECISIONS] for t in (0, 1)]
    )


def read_dvc_md5(path: Path) -> str | None:
    """md5 of the first output of a .dvc pointer file, None if absent."""
    if not path.exists():
        return None
    meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(meta["outs"][0]["md5"])


def file_md5(path: Path) -> str:
    """md5 of a file's bytes (same algorithm as DVC)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def log_to_mlflow(
    *,
    model: CalibratedClassifierCV,
    name: str,
    params: Params,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    figures: dict[str, Figure],
    counts: npt.NDArray[np.int64],
    train_data_hash: str,
    extra_tags: dict[str, str],
) -> str:
    """Log into the active run (not registered). Returns its run id.

    Tags `raw_data_hash` (DVC md5 of the raw CSV, if the pointer file exists)
    and `train_data_hash` (md5 of the file actually trained on), plus
    `extra_tags` (experiment, excluded features). The hyperparameters are
    logged as `model__<name>` params.
    """
    threshold_tags = {
        "threshold_low": repr(thresholds["low"]),
        "threshold_high": repr(thresholds["high"]),
        "shap_unit": SHAP_UNIT[name],
    }
    tags = {
        "model_type": name,
        "sklearn_version": sklearn.__version__,
        "serialization_format": "cloudpickle",
        "train_data_hash": train_data_hash,
        **threshold_tags,
        **extra_tags,
    }
    optional = [("git_sha", _git_sha()), ("raw_data_hash", read_dvc_md5(DVC_FILE))]
    for key, value in optional:
        if value is not None:
            tags[key] = value
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_to_mlflow needs an active run")
    mlflow.set_tags(tags)
    mlflow.log_params(
        {
            "model": name,
            "seed": params.data.random_state,
            "val_size": params.data.val_size,
            "n_fit": metrics["n_fit"],
            "n_val": metrics["n_val"],
            "cost_fn": params.cost.cost_fn,
            "cost_fp": params.cost.cost_fp,
            "cost_review": params.cost.cost_review,
            "min_recall": params.threshold.min_recall,
            "max_review_share": params.threshold.max_review_share,
            "calibration": "isotonic_cv5_ensemble_false",
            **{f"model__{k}": v for k, v in metrics["model_params"].items()},
        }
    )
    if metrics["tune_trials"]:
        mlflow.log_params(
            {"tune_trials": metrics["tune_trials"], "tune_objective": "cv5_roc_auc"}
        )
        mlflow.log_metric("best_cv_roc_auc", metrics["best_cv_roc_auc"])
    mlflow.log_metrics({f"val_{k}": v for k, v in metrics["val"].items()})
    mlflow.log_metric(
        "val_accept_all_cost_per_file", metrics["accept_all_cost_per_file"]
    )
    mlflow.log_dict(thresholds, "thresholds.json")
    mlflow.log_dict(metrics, "metrics.json")
    mlflow.log_dict(metrics["model_params"], "model_params.json")
    labels = [f"TARGET={t}" for t in (0, 1)]
    mlflow.log_dict(
        {lab: dict(zip(DECISIONS, map(int, row))) for lab, row in zip(labels, counts)},
        "confusion_matrix.json",
    )
    for fig_name, fig in figures.items():
        mlflow.log_figure(fig, f"{fig_name}.png")
    mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format="cloudpickle",
        signature=build_signature(params.features.numeric, params.features.categorical),
        tags=threshold_tags,
    )
    return str(run.info.run_id)


def log_trial(number: int, hp: dict[str, Any], cv_auc: float) -> None:
    """Log one Optuna trial as a nested run of the active run."""
    with mlflow.start_run(run_name=f"trial-{number}", nested=True):
        mlflow.set_tag("trial", str(number))
        mlflow.log_params(hp)
        mlflow.log_metric("cv_roc_auc", cv_auc)


def train(
    df: pd.DataFrame,
    name: str,
    params: Params,
    *,
    model_params: dict[str, Any] | None = None,
    exclude: tuple[str, ...] = (),
    n_trials: int = 0,
    on_trial: TrialCallback | None = None,
) -> dict[str, Any]:
    """Split, (tune,) fit, calibrate and optimize the thresholds. No I/O.

    Args:
        df: Featurized training rows (features, target).
        name: Model name, a key of `MODELS`.
        params: Validated parameters.
        model_params: Fixed hyperparameters, over the `build_pipeline` defaults.
        exclude: Numeric features kept in the input but not given to the model.
        n_trials: If > 0, Optuna trials on the fit part; the best params are
            merged over `model_params`.
        on_trial: Passed to `tune` (e.g. `log_trial`).

    Returns:
        model, thresholds (dict), metrics (dict, with the `model_params`
        used), and the validation y/proba.
    """
    numeric = [c for c in params.features.numeric if c not in exclude]
    fit, val = split_validation(
        df,
        target=params.target,
        val_size=params.data.val_size,
        seed=params.data.random_state,
    )
    X_fit, y_fit = prepare_xy(fit, params)
    X_val, y_val = prepare_xy(val, params)
    hp = dict(model_params or {})
    best_cv = None
    if n_trials > 0:
        best, best_cv = tune(
            X_fit,
            y_fit,
            name,
            numeric=numeric,
            categorical=params.features.categorical,
            seed=params.data.random_state,
            n_trials=n_trials,
            on_trial=on_trial,
        )
        hp.update(best)
    model = fit_model(X_fit, y_fit, name, params, numeric=numeric, model_params=hp)
    proba = model.predict_proba(X_val)[:, 1]
    costs = {
        "cost_fn": params.cost.cost_fn,
        "cost_fp": params.cost.cost_fp,
        "cost_review": params.cost.cost_review,
    }
    low, high = optimize_thresholds(
        y_val,
        proba,
        **costs,
        min_recall=params.threshold.min_recall,
        max_review_share=params.threshold.max_review_share,
    )
    val_metrics = {
        **score_metrics(y_val, proba),
        **zone_metrics(y_val, proba, low, high, **costs),
        "distinct_proba": float(np.unique(proba).size),
    }
    accept_all = zone_metrics(y_val, proba, NEVER, NEVER, **costs)["cost_per_file"]
    metrics = {
        "model": name,
        "n_fit": len(fit),
        "n_val": len(val),
        "val": val_metrics,
        "accept_all_cost_per_file": accept_all,
        "model_params": hp,
        "excluded_features": list(exclude),
        "tune_trials": n_trials,
        "best_cv_roc_auc": best_cv,
    }
    return {
        "model": model,
        "thresholds": thresholds_payload(low, high, params, val_metrics),
        "metrics": metrics,
        "y_val": y_val,
        "proba_val": proba,
    }


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--train", type=Path, default=Path("data/processed/train.parquet")
    )
    parser.add_argument("--model", choices=sorted(MODELS), default="logreg")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--sample",
        type=Path,
        default=None,
        help="CML mode: train on this file instead of --train, also write roc.png",
    )
    parser.add_argument(
        "--tune", type=int, default=0, metavar="N", help="Optuna trials (L3-2)"
    )
    parser.add_argument(
        "--model-params", type=Path, default=None, help="JSON hyperparameters"
    )
    parser.add_argument(
        "--exclude", action="append", default=[], help="numeric feature to leave out"
    )
    parser.add_argument("--experiment", default=None, help="value of the tag")
    args = parser.parse_args(argv)
    params = load_params(args.params)
    unknown = set(args.exclude) - set(params.features.numeric)
    if unknown:
        parser.error(f"--exclude: not numeric features: {sorted(unknown)}")
    if args.exclude and not args.experiment:
        # A run without some features is a measurement, never a candidate.
        parser.error("--exclude needs --experiment (e.g. --experiment no_age)")
    if args.tune and args.model not in SEARCH_SPACES:
        parser.error(f"--tune: no search space for {args.model}")
    # MLflow prints an emoji: a redirected Windows console (cp1252) would crash.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")

    data_file: Path = args.sample or args.train
    df = pd.read_parquet(data_file)
    model_params = (
        json.loads(args.model_params.read_text(encoding="utf-8"))
        if args.model_params
        else None
    )
    tracking = bool(os.environ.get("MLFLOW_TRACKING_URI"))
    run_name = args.model + (f"-{args.experiment}" if args.experiment else "")
    if tracking:
        mlflow.set_experiment(EXPERIMENT)
    run_ctx = mlflow.start_run(run_name=run_name) if tracking else None
    with run_ctx or contextlib.nullcontext():
        result = train(
            df,
            args.model,
            params,
            model_params=model_params,
            exclude=tuple(args.exclude),
            n_trials=args.tune,
            on_trial=log_trial if tracking else None,
        )
        th, metrics = result["thresholds"], result["metrics"]
        y_val, proba = result["y_val"], result["proba_val"]

        out: Path = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(result["model"], out / "model.joblib")
        for file_name, payload in [
            ("thresholds.json", th),
            ("metrics.json", metrics),
            ("model_params.json", metrics["model_params"]),
        ]:
            (out / file_name).write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
        figures = make_figures(y_val, proba, th["low"], th["high"])
        if args.sample is not None:
            figures["roc_curve"].savefig(out / "roc.png", dpi=100)
        if tracking:
            extra_tags = {}
            if args.experiment:
                extra_tags["experiment"] = args.experiment
            if args.exclude:
                extra_tags["excluded_features"] = ",".join(args.exclude)
            run_id = log_to_mlflow(
                model=result["model"],
                name=args.model,
                params=params,
                metrics=metrics,
                thresholds=th,
                figures=figures,
                counts=confusion_counts(y_val, proba, th["low"], th["high"]),
                train_data_hash=file_md5(data_file),
                extra_tags=extra_tags,
            )
            print(f"MLflow run {run_id}")
    for fig in figures.values():
        plt.close(fig)
    val = metrics["val"]
    print(
        f"{run_name}: val ROC-AUC {val['roc_auc']:.4f}, PR-AUC {val['pr_auc']:.4f}, "
        f"Brier {val['brier']:.4f} | thresholds {th['low']!r} / {th['high']!r} | "
        f"recall {val['recall']:.3f}, review {val['review_share']:.3f}, "
        f"cost/file {val['cost_per_file']:.4f} (accept all "
        f"{metrics['accept_all_cost_per_file']:.4f})"
    )


if __name__ == "__main__":
    main()
