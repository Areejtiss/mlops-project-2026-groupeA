"""End-to-end retraining loop on a sqlite MLflow (L6-3, hand-off to Areej).

The exact command sequence of docs/handoff/boucle-reentrainement.md, run
through each module's `main` on tiny synthetic data: bootstrap champion,
drift alert, retrain, registry, promote, aliases moved.
"""

import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml

from crediscore.config import load_params
from crediscore.models import evaluate, promote, registry, train
from crediscore.models.registry import MODEL_NAME
from crediscore.monitoring import drift
from conftest import PARAMS_FILE, make_synthetic_frame


def aliases() -> dict[str, str]:
    model = mlflow.MlflowClient().get_registered_model(MODEL_NAME)
    return {k: str(v) for k, v in model.aliases.items()}


def test_drift_alert_retrain_registry_promote_moves_the_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)

    # params.yaml with promotion minimums sized for a 1 500-row batch.
    raw = yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))
    raw["promotion"].update(min_rows=1000, min_group_rows=300)
    Path("params.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    p = ["--params", "params.yaml"]
    params = load_params(Path("params.yaml"))

    frame = make_synthetic_frame(params, n=7000)
    current = frame.iloc[3500:4500].copy()
    current["AMT_INCOME_TOTAL"] += 1.0  # the crisis: incomes move
    frame.iloc[:1000].to_parquet("train_v1.parquet")
    frame.iloc[1500:3500].to_parquet("test.parquet")
    current.to_parquet("current.parquet")
    pd.concat([frame.iloc[:1500], current]).to_parquet("train_v2.parquet")
    frame.iloc[5500:].to_parquet("holdout.parquet")

    # Bootstrap: first model, no champion yet -> it becomes champion.
    train.main([*p, "--train", "train_v1.parquet", "--output-dir", "m1"])
    registry.main([*p, "--select", "--train", "train_v1.parquet"])
    assert aliases() == {"champion": "1"}
    evaluate.main(
        [*p, "--model", "m1", "--train", "train_v1.parquet"]
        + ["--test", "test.parquet", "--output", "metrics.json"]
    )
    promo = [*p, "--data", "holdout.parquet", "--metrics", "metrics.json"]
    assert promote.main([*promo, "--output", "promotion0.json"]) == 1
    first = json.loads(Path("promotion0.json").read_text(encoding="utf-8"))
    assert first["failed"] == ["no_challenger"]  # keep the current model

    # 1. detect_drift: alert -> retrain.
    drift.main(
        [*p, "--reference", "test.parquet", "--current", "current.parquet"]
        + ["--model", f"models:/{MODEL_NAME}@champion", "--metrics", "metrics.json"]
        + ["--out-dir", "drift"]
    )
    report = json.loads(Path("drift/latest.json").read_text(encoding="utf-8"))
    assert report["alert"] is True
    assert report["psi"]["AMT_INCOME_TOTAL"] >= params.monitoring.psi_alert
    assert report["prediction"] is not None
    assert report["fairness"] is not None and report["fairness"]["d5"] is not None

    # 2. train on the refreshed data, 3. registry -> challenger.
    train.main([*p, "--train", "train_v2.parquet", "--output-dir", "m2"])
    registry.main([*p, "--select", "--train", "train_v2.parquet"])
    assert aliases() == {"champion": "1", "challenger": "2"}

    # 4a. promote on the fresh hold-out batch. The challenger is better on AUC
    # and cost, but learned the synthetic age signal more strongly: the D5
    # age gate fails, exit 1 = keep the current model, nothing changed.
    assert promote.main([*promo, "--output", "blocked.json"]) == 1
    blocked = json.loads(Path("blocked.json").read_text(encoding="utf-8"))
    assert blocked["failed"] == ["age_eo_gap"]
    assert aliases() == {"champion": "1", "challenger": "2"}

    # 4b. Same chain with a wider age margin (test-only params): exit 0 =
    # promoted, the aliases move (the gate itself is tested in test_promote).
    raw["promotion"]["age_gap_margin_pts"] = 10
    Path("params.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert promote.main([*promo, "--output", "promotion.json"]) == 0
    decision = json.loads(Path("promotion.json").read_text(encoding="utf-8"))
    assert decision["promote"] is True and decision["applied"] is True
    assert (decision["challenger_version"], decision["champion_version"]) == ("2", "1")
    assert decision["failed"] == []
    assert aliases() == {"champion": "2"}
    tags = mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    assert tags["promotion_previous_champion"] == "1"
    assert "promoted_at" in tags
