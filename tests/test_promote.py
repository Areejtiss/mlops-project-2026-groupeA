"""Tests for the challenger -> champion promotion gate (T7, D5, L4-1)."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from crediscore.config import PromotionConfig, load_params
from crediscore.models import promote as pr
from crediscore.models import train as train_module
from crediscore.models.evaluate import load_model
from crediscore.models.registry import MODEL_NAME, register
from crediscore.models.threshold import zone_metrics
from conftest import make_synthetic_frame

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"
PROMO = PromotionConfig(max_gender_gap_pts=10, age_gap_margin_pts=2)


# --- decide (pure) -------------------------------------------------------------


def scores(
    auc: float = 0.76,
    cost: float = 0.30,
    gender_raw: float = 5.0,
    gender_eo: float = 5.0,
    age_eo: float = 33.0,
) -> dict[str, Any]:
    return {
        "roc_auc": auc,
        "cost_per_file": cost,
        "fairness": {
            "gender": {"raw_gap_pts": gender_raw, "eo_gap_pts": gender_eo},
            "age": {"raw_gap_pts": 40.0, "eo_gap_pts": age_eo},
        },
    }


def decide(ch: dict, champ: dict | None, anchor: float = 38.7) -> dict[str, Any]:
    return pr.decide(ch, champ, age_anchor_pts=anchor, promotion=PROMO)


def test_all_conditions_pass() -> None:
    out = decide(scores(auc=0.77, cost=0.29), scores(auc=0.75, cost=0.32))
    assert out["promote"] is True
    assert out["failed"] == []
    assert list(out["checks"]) == [
        "auc",
        "cost",
        "gender_raw_gap",
        "gender_eo_gap",
        "age_eo_gap",
    ]
    assert out["checks"]["auc"] == {"value": 0.77, "limit": 0.75, "pass": True}
    assert out["checks"]["cost"] == {"value": 0.29, "limit": 0.32, "pass": True}
    assert out["checks"]["age_eo_gap"]["limit"] == pytest.approx(40.7)


@pytest.mark.parametrize(
    ("challenger", "failing"),
    [
        (scores(auc=0.7599), "auc"),
        (scores(cost=0.3001), "cost"),
        (scores(gender_raw=10.01), "gender_raw_gap"),
        (scores(gender_eo=10.01), "gender_eo_gap"),
        (scores(age_eo=40.71), "age_eo_gap"),
    ],
)
def test_each_condition_fails_alone(challenger: dict, failing: str) -> None:
    out = decide(challenger, scores(auc=0.76, cost=0.30))
    assert out["promote"] is False
    assert out["failed"] == [failing]


def test_equality_boundaries_promote() -> None:
    champ = scores(auc=0.76, cost=0.30)
    ch = scores(auc=0.76, cost=0.30, gender_raw=10.0, gender_eo=10.0, age_eo=40.7)
    out = decide(ch, champ, anchor=38.7)
    assert out["promote"] is True, out["failed"]


def test_a_better_champion_on_both_counts_blocks() -> None:
    out = decide(scores(auc=0.74, cost=0.33), scores(auc=0.75, cost=0.32))
    assert out["failed"] == ["auc", "cost"]


def test_no_champion_yet_only_the_fairness_gates_apply() -> None:
    out = decide(scores(auc=0.60, cost=0.90), None)
    assert out["promote"] is True
    assert list(out["checks"]) == ["gender_raw_gap", "gender_eo_gap", "age_eo_gap"]
    assert decide(scores(gender_eo=11.0), None)["failed"] == ["gender_eo_gap"]


def test_the_tolerances_come_from_the_promotion_config() -> None:
    strict = PromotionConfig(max_gender_gap_pts=4, age_gap_margin_pts=0)
    out = pr.decide(scores(), None, age_anchor_pts=33.0, promotion=strict)
    assert out["failed"] == ["gender_raw_gap", "gender_eo_gap"]
    assert out["checks"]["age_eo_gap"]["limit"] == 33.0


# --- age anchor ----------------------------------------------------------------


def test_age_anchor_is_the_baseline_eo_gap_of_metrics_json(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "model": {"fairness": {"age": {"eo_gap_pts": 1.0}}},
                "baseline": {"fairness": {"age": {"eo_gap_pts": 38.675}}},
            }
        ),
        "utf-8",
    )
    assert pr.read_age_anchor(path) == 38.675


@pytest.mark.parametrize(
    "content",
    [None, "not json", "{}", '{"baseline": {"fairness": {}}}', "[1]"]
    + ['{"baseline": {"fairness": {"age": {"eo_gap_pts": "38"}}}}'],
)
def test_missing_or_broken_age_anchor_is_none(
    tmp_path: Path, content: str | None
) -> None:
    path = tmp_path / "metrics.json"
    if content is not None:
        path.write_text(content, "utf-8")
    assert pr.read_age_anchor(path) is None


# --- main on a sqlite registry -------------------------------------------------


@pytest.fixture(scope="module")
def tracked(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A sqlite store with two finished train runs (logreg, lightgbm), a
    held-out batch and a metrics.json with the age anchor."""
    base = tmp_path_factory.mktemp("promote")
    params = load_params(PARAMS_FILE)
    train_file = base / "train.parquet"
    make_synthetic_frame(params).to_parquet(train_file)
    make_synthetic_frame(params, n=1500).to_parquet(base / "heldout.parquet")
    hp = base / "hp.json"
    hp.write_text(json.dumps({"n_estimators": 40}), encoding="utf-8")
    write_anchor(base / "metrics.json", 38.7)
    old_cwd, old_uri = Path.cwd(), os.environ.get("MLFLOW_TRACKING_URI")
    uri = f"sqlite:///{(base / 'mlflow.db').as_posix()}"
    os.chdir(base)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    try:
        common = ["--params", str(PARAMS_FILE), "--train", str(train_file)]
        train_module.main([*common, "--output-dir", "m1"])
        train_module.main(
            [*common, "--model", "lightgbm", "--model-params", str(hp)]
            + ["--output-dir", "m2"]
        )
    finally:
        os.chdir(old_cwd)
        if old_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = old_uri
    return base


def write_anchor(path: Path, anchor: float) -> Path:
    payload = {"baseline": {"fairness": {"age": {"eo_gap_pts": anchor}}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def store(tracked: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the tracked store (the registry is mutated)."""
    shutil.copy(tracked / "mlflow.db", tmp_path / "mlflow.db")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return tracked


def run_ids() -> dict[str, str]:
    runs = mlflow.search_runs(experiment_names=["crediscore"], output_format="list")
    return {r.data.tags["model_type"]: r.info.run_id for r in runs}


def aliases() -> dict[str, str]:
    model = mlflow.MlflowClient().get_registered_model(MODEL_NAME)
    return {k: str(v) for k, v in model.aliases.items()}


def run_main(
    store: Path, tmp_path: Path, *extra: str, metrics: Path | None = None
) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "promotion.json"
    code = pr.main(
        [
            "--params",
            str(PARAMS_FILE),
            "--data",
            str(store / "heldout.parquet"),
            "--metrics",
            str(metrics or store / "metrics.json"),
            "--output",
            str(out),
            *extra,
        ]
    )
    return code, json.loads(out.read_text(encoding="utf-8"))


def register_pair(champion: str, challenger: str) -> None:
    client = mlflow.MlflowClient()
    assert register(client, run_ids()[champion])[1] == "champion"
    assert register(client, run_ids()[challenger])[1] == "challenger"


def test_same_model_as_challenger_is_promoted_with_aliases_and_tags(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Same run twice: equal AUC and equal cost on the same batch -> promote.
    register_pair("lightgbm", "lightgbm")
    code, decision = run_main(store, tmp_path)
    assert code == 0, decision["failed"]
    assert aliases() == {"champion": "2"}
    assert decision["promote"] is True
    assert decision["applied"] is True
    assert decision["challenger_version"] == "2"
    assert decision["champion_version"] == "1"
    checks = decision["checks"]
    assert checks["auc"]["value"] == checks["auc"]["limit"]
    assert checks["cost"]["value"] == checks["cost"]["limit"]
    tags = mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    datetime.fromisoformat(tags["promoted_at"])
    assert tags["promoted_at"] == decision["timestamp"]
    assert tags["promotion_previous_champion"] == "1"
    assert tags["promotion_data_md5"] == decision["data_md5"]
    for name, check in checks.items():
        assert float(tags[f"promotion_{name}"]) == check["value"]
        assert float(tags[f"promotion_{name}_limit"]) == check["limit"]
    assert (
        "promoted_at"
        not in mlflow.MlflowClient().get_model_version(MODEL_NAME, "1").tags
    )
    assert "promoted" in capsys.readouterr().out


def test_a_failing_gate_changes_nothing_and_exits_1(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_pair("lightgbm", "lightgbm")
    anchor = write_anchor(tmp_path / "metrics.json", -10.0)
    code, decision = run_main(store, tmp_path, metrics=anchor)
    assert code == 1
    assert decision["promote"] is False
    assert decision["applied"] is False
    assert decision["failed"] == ["age_eo_gap"]
    assert decision["checks"]["age_eo_gap"]["limit"] == -8.0
    assert aliases() == {"champion": "1", "challenger": "2"}
    assert (
        "promoted_at"
        not in mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    )
    assert "age_eo_gap" in capsys.readouterr().out


def test_dry_run_decides_but_changes_nothing(store: Path, tmp_path: Path) -> None:
    register_pair("lightgbm", "lightgbm")
    code, decision = run_main(store, tmp_path, "--dry-run")
    assert code == 0
    assert decision["promote"] is True
    assert (decision["dry_run"], decision["applied"]) == (True, False)
    assert aliases() == {"champion": "1", "challenger": "2"}
    assert (
        "promoted_at"
        not in mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    )


def test_both_models_are_scored_on_the_batch_with_their_own_thresholds(
    store: Path, tmp_path: Path
) -> None:
    register_pair("logreg", "lightgbm")
    _, decision = run_main(store, tmp_path, "--dry-run")
    params = load_params(PARAMS_FILE)
    batch = pd.read_parquet(store / "heldout.parquet")
    X, y = train_module.prepare_xy(batch, params)
    costs = params.cost.model_dump()
    for side, version in [("champion", "1"), ("challenger", "2")]:
        model, th = load_model(f"models:/{MODEL_NAME}/{version}")
        p = model.predict_proba(X)[:, 1]
        got = decision["scores"][side]
        assert got["thresholds"] == {"low": th["low"], "high": th["high"]}
        assert got["roc_auc"] == pytest.approx(roc_auc_score(y, p))
        expected = zone_metrics(y, p, th["low"], th["high"], **costs)
        assert got["cost_per_file"] == pytest.approx(expected["cost_per_file"])
        assert got["recall"] == pytest.approx(expected["recall"])
    th1 = load_model(f"models:/{MODEL_NAME}/1")[1]
    th2 = load_model(f"models:/{MODEL_NAME}/2")[1]
    assert th1["low"] != th2["low"]  # really two threshold sets
    assert decision["n_rows"] == len(batch)
    checks = decision["checks"]
    assert checks["auc"]["value"] == decision["scores"]["challenger"]["roc_auc"]
    assert checks["auc"]["limit"] == decision["scores"]["champion"]["roc_auc"]
    ch_fair = decision["scores"]["challenger"]["fairness"]
    assert checks["gender_eo_gap"]["value"] == ch_fair["gender"]["eo_gap_pts"]
    assert checks["age_eo_gap"]["value"] == ch_fair["age"]["eo_gap_pts"]


def test_no_champion_yet_promotes_on_the_fairness_gates(
    store: Path, tmp_path: Path
) -> None:
    register_pair("logreg", "lightgbm")
    mlflow.MlflowClient().delete_registered_model_alias(MODEL_NAME, "champion")
    code, decision = run_main(store, tmp_path)
    assert code == 0, decision["failed"]
    assert decision["champion_version"] is None
    assert decision["scores"]["champion"] is None
    assert "auc" not in decision["checks"]
    assert aliases() == {"champion": "2"}
    tags = mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    assert tags["promotion_previous_champion"] == "none"


def test_missing_age_anchor_fails_closed(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_pair("lightgbm", "lightgbm")
    code, decision = run_main(store, tmp_path, metrics=tmp_path / "absent.json")
    assert code == 1
    assert decision["promote"] is False
    assert decision["failed"] == ["age_anchor_missing"]
    assert aliases() == {"champion": "1", "challenger": "2"}
    assert "age anchor" in capsys.readouterr().out


@pytest.mark.parametrize("registered", [False, True])
def test_no_challenger_exits_1(store: Path, tmp_path: Path, registered: bool) -> None:
    if registered:
        register(mlflow.MlflowClient(), run_ids()["logreg"])  # champion only
    code, decision = run_main(store, tmp_path)
    assert code == 1
    assert decision["failed"] == ["no_challenger"]
    assert decision["challenger_version"] is None
    if registered:
        assert aliases() == {"champion": "1"}


def test_decision_json_fields(store: Path, tmp_path: Path) -> None:
    register_pair("logreg", "lightgbm")
    _, decision = run_main(store, tmp_path, "--dry-run")
    assert set(decision) == {
        "timestamp",
        "dry_run",
        "data",
        "data_md5",
        "n_rows",
        "model",
        "challenger_version",
        "champion_version",
        "age_anchor_pts",
        "promote",
        "applied",
        "failed",
        "checks",
        "scores",
    }
    assert decision["model"] == MODEL_NAME
    assert decision["age_anchor_pts"] == 38.7
    assert decision["data"] == str(store / "heldout.parquet")
    assert decision["data_md5"] == train_module.file_md5(store / "heldout.parquet")
