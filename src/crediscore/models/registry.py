"""Select the best training run and register it as `crediscore` (L3-1, L3-3).

Selection (validation only, the test split is never read here):
- candidates: finished top-level `train` runs of experiment `crediscore`
  trained on the same file (`train_data_hash` = md5 of `--train`), whose
  thresholds meet the D3/D8 constraints (recall >= min_recall, review share
  <= max_review_share);
- keep those with validation ROC-AUC >= 0.75 if any reaches it;
- the best is the lowest validation cost per file, ties broken by ROC-AUC.
A winner tagged `experiment` (e.g. the D5 `no_age` run) is a measurement:
it is flagged and only registered with `--allow-experiment`.

Registration: the run's logged model becomes a new version of `crediscore`.
Alias `champion` if none exists yet, otherwise `challenger` (promotion is
lot 4). The thresholds (full precision), `shap_unit`, `model_type` and data
hashes are copied from the run to model-version tags (contract 2 section 7).

Usage:
    poetry run python -m crediscore.models.registry --select
    poetry run python -m crediscore.models.registry --run-id <run id>
"""

import argparse
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from crediscore.models.train import EXPERIMENT, file_md5

MODEL_NAME = "crediscore"
# SYNOPSIS ROC-AUC target, used as the selection floor (L3-1).
MIN_AUC = 0.75
REQUIRED_TAGS = (
    "threshold_low",
    "threshold_high",
    "shap_unit",
    "model_type",
    "train_data_hash",
)
VERSION_TAGS = (*REQUIRED_TAGS, "raw_data_hash", "git_sha")


def select_run(
    candidates: list[dict[str, Any]], min_auc: float = MIN_AUC
) -> dict[str, Any]:
    """Pick the run to register (L3-1).

    Args:
        candidates: One dict per run: run_id, roc_auc, cost_per_file, recall,
            review_share, min_recall, max_review_share (validation values).
        min_auc: AUC floor, applied only if at least one feasible run reaches it.

    Returns:
        The chosen candidate.

    Raises:
        ValueError: If no candidate meets the recall and review constraints.
    """
    feasible = [
        c
        for c in candidates
        if c["recall"] >= c["min_recall"] and c["review_share"] <= c["max_review_share"]
    ]
    if not feasible:
        raise ValueError("no candidate run meets the recall and review constraints")
    pool = [c for c in feasible if c["roc_auc"] >= min_auc] or feasible
    return min(pool, key=lambda c: (c["cost_per_file"], -c["roc_auc"]))


def candidate(run: Run) -> dict[str, Any]:
    """Selection fields of a `train` run (validation metrics and constraints)."""
    m, p = run.data.metrics, run.data.params
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "roc_auc": m["val_roc_auc"],
        "cost_per_file": m["val_cost_per_file"],
        "recall": m["val_recall"],
        "review_share": m["val_review_share"],
        "min_recall": float(p["min_recall"]),
        "max_review_share": float(p["max_review_share"]),
        "experiment": run.data.tags.get("experiment"),
    }


def training_runs(client: MlflowClient, train_data_hash: str) -> list[Run]:
    """Finished top-level train runs of the experiment, on the given data."""
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return []
    runs = client.search_runs(
        [exp.experiment_id], filter_string="attributes.status = 'FINISHED'"
    )
    return [
        r
        for r in runs
        if "mlflow.parentRunId" not in r.data.tags
        and "model_type" in r.data.tags
        and r.data.tags.get("train_data_hash") == train_data_hash
        and "val_cost_per_file" in r.data.metrics
    ]


def register(
    client: MlflowClient, run_id: str, name: str = MODEL_NAME
) -> tuple[str, str]:
    """Register the run's logged model, copy its tags, set the alias.

    Args:
        client: MLflow client of the tracking server.
        run_id: A finished `train` run.
        name: Registered model name.

    Returns:
        (version, alias): alias `champion` if the model had none, else
        `challenger`.

    Raises:
        ValueError: If the run lacks a tag of `REQUIRED_TAGS`.
    """
    tags = client.get_run(run_id).data.tags
    missing = [k for k in REQUIRED_TAGS if k not in tags]
    if missing:
        raise ValueError(f"run {run_id} lacks the tags {missing}")
    try:
        aliases = client.get_registered_model(name).aliases
    except mlflow.exceptions.MlflowException:
        aliases = {}
    alias = "challenger" if "champion" in aliases else "champion"
    version = str(mlflow.register_model(f"runs:/{run_id}/model", name).version)
    for key in VERSION_TAGS:
        if key in tags:
            client.set_model_version_tag(name, version, key, tags[key])
    client.set_registered_model_alias(name, alias, version)
    return version, alias


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--select", action="store_true", help="apply L3-1")
    which.add_argument("--run-id", help="register this run as is")
    parser.add_argument(
        "--train", type=Path, default=Path("data/processed/train.parquet")
    )
    parser.add_argument(
        "--allow-experiment", action="store_true", help="register a tagged winner"
    )
    args = parser.parse_args(argv)
    client = MlflowClient()
    run_id = args.run_id
    if args.select:
        cands = [candidate(r) for r in training_runs(client, file_md5(args.train))]
        for c in sorted(cands, key=lambda c: c["cost_per_file"]):
            print(
                f"{c['run_name']:24s} {c['run_id']} AUC {c['roc_auc']:.4f} "
                f"cost/file {c['cost_per_file']:.4f} recall {c['recall']:.3f} "
                f"review {c['review_share']:.3f}"
            )
        try:
            best = select_run(cands)
        except ValueError as err:
            raise SystemExit(str(err)) from err
        print(f"selected {best['run_name']} {best['run_id']}")
        if best["experiment"] and not args.allow_experiment:
            raise SystemExit(
                f"the winner is tagged experiment={best['experiment']}: not "
                "registered (pass --allow-experiment to register it anyway)"
            )
        run_id = best["run_id"]
    version, alias = register(client, run_id)
    print(f"registered {MODEL_NAME} version {version} as @{alias}")


if __name__ == "__main__":
    main()
