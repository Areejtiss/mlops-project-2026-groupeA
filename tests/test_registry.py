"""Tests for model selection (L3-1) and registration with aliases (L3-3)."""

import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest

from crediscore.config import load_params
from crediscore.models import registry
from crediscore.models import train as train_module
from crediscore.models.registry import MODEL_NAME, VERSION_TAGS, register, select_run
from conftest import make_synthetic_frame

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"


def cand(run_id: str, auc: float, cost: float, **over: object) -> dict:
    c = {
        "run_id": run_id,
        "roc_auc": auc,
        "cost_per_file": cost,
        "recall": 0.66,
        "review_share": 0.14,
        "min_recall": 0.65,
        "max_review_share": 0.15,
        "experiment": None,
    }
    return {**c, **over}


# --- select_run (L3-1, pure) ---------------------------------------------------


def test_lowest_cost_among_runs_reaching_the_auc_floor() -> None:
    runs = [cand("a", 0.74, 0.20), cand("b", 0.76, 0.32), cand("c", 0.75, 0.31)]
    assert select_run(runs)["run_id"] == "c"  # a is cheaper but below 0.75


def test_every_run_is_a_candidate_when_none_reaches_the_floor() -> None:
    runs = [cand("a", 0.70, 0.33), cand("b", 0.74, 0.34)]
    assert select_run(runs)["run_id"] == "a"


def test_cost_ties_are_broken_by_auc() -> None:
    runs = [cand("a", 0.76, 0.30), cand("b", 0.77, 0.30), cand("c", 0.75, 0.30)]
    assert select_run(runs)["run_id"] == "b"


def test_runs_breaking_the_d3_d8_constraints_are_excluded() -> None:
    runs = [
        cand("low_recall", 0.80, 0.10, recall=0.64),
        cand("review_cap", 0.80, 0.10, review_share=0.151),
        cand("ok", 0.76, 0.30, recall=0.65, review_share=0.15),  # inclusive
    ]
    assert select_run(runs)["run_id"] == "ok"


def test_no_feasible_run_raises() -> None:
    with pytest.raises(ValueError, match="no candidate run"):
        select_run([cand("a", 0.8, 0.1, recall=0.1)])


def test_custom_floor() -> None:
    runs = [cand("a", 0.70, 0.20), cand("b", 0.80, 0.30)]
    assert select_run(runs, min_auc=0.65)["run_id"] == "a"


# --- registration on sqlite ------------------------------------------------------


@pytest.fixture(scope="module")
def tracked(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A sqlite store with two finished runs: logreg and a tagged no_age run."""
    base = tmp_path_factory.mktemp("store")
    params = load_params(PARAMS_FILE)
    data = base / "train.parquet"
    make_synthetic_frame(params).to_parquet(data)
    hp = base / "hp.json"
    hp.write_text(json.dumps({"n_estimators": 40}), encoding="utf-8")
    old_cwd, old_uri = Path.cwd(), os.environ.get("MLFLOW_TRACKING_URI")
    uri = f"sqlite:///{(base / 'mlflow.db').as_posix()}"
    os.chdir(base)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    try:
        common = ["--params", str(PARAMS_FILE), "--train", str(data)]
        train_module.main([*common, "--output-dir", "m1"])
        train_module.main(
            [
                *common,
                "--model",
                "lightgbm",
                "--model-params",
                str(hp),
                "--exclude",
                "AGE_YEARS",
                "--experiment",
                "no_age",
                "--output-dir",
                "m2",
            ]
        )
    finally:
        os.chdir(old_cwd)
        if old_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = old_uri
    return base


@pytest.fixture
def store(tracked: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the tracked store (the registry is mutated)."""
    shutil.copy(tracked / "mlflow.db", tmp_path / "mlflow.db")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return tracked


def runs_by_model() -> dict[str, mlflow.entities.Run]:
    runs = mlflow.search_runs(experiment_names=["crediscore"], output_format="list")
    return {r.data.tags["model_type"]: r for r in runs}


def test_first_registration_is_champion_then_challenger(store: Path) -> None:
    runs = runs_by_model()
    client = mlflow.MlflowClient()
    v1, alias1 = register(client, runs["logreg"].info.run_id)
    assert (v1, alias1) == ("1", "champion")
    v2, alias2 = register(client, runs["lightgbm"].info.run_id)
    assert (v2, alias2) == ("2", "challenger")
    model = client.get_registered_model(MODEL_NAME)
    assert model.aliases == {"champion": "1", "challenger": "2"}
    assert client.get_model_version_by_alias(MODEL_NAME, "champion").run_id == (
        runs["logreg"].info.run_id
    )


def test_version_tags_are_copied_exactly(store: Path) -> None:
    run = runs_by_model()["logreg"]
    client = mlflow.MlflowClient()
    version, _ = register(client, run.info.run_id)
    tags = client.get_model_version(MODEL_NAME, version).tags
    for key in VERSION_TAGS:
        assert tags[key] == run.data.tags[key], key
    assert set(VERSION_TAGS) >= {
        "threshold_low",
        "threshold_high",
        "shap_unit",
        "train_data_hash",
        "model_type",
    }
    th = json.loads((store / "m1" / "thresholds.json").read_text(encoding="utf-8"))
    assert float(tags["threshold_low"]) == th["low"]  # full precision
    assert float(tags["threshold_high"]) == th["high"]
    assert tags["shap_unit"] == "log_odds"


def test_registered_champion_loads_with_the_alias(store: Path) -> None:
    client = mlflow.MlflowClient()
    register(client, runs_by_model()["logreg"].info.run_id)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    params = load_params(PARAMS_FILE)
    X, _ = train_module.prepare_xy(make_synthetic_frame(params, n=50), params)
    assert model.predict_proba(X).shape == (50, 2)


def test_a_run_without_threshold_tags_is_refused(store: Path) -> None:
    client = mlflow.MlflowClient()
    run_id = runs_by_model()["logreg"].info.run_id
    client.delete_tag(run_id, "threshold_low")
    with pytest.raises(ValueError, match="threshold_low"):
        register(client, run_id)
    assert not client.search_registered_models()


def test_main_select_registers_the_selected_run(store: Path) -> None:
    runs = runs_by_model()
    client = mlflow.MlflowClient()
    expected = select_run([registry.candidate(r) for r in runs.values()])
    registry.main(
        ["--select", "--train", str(store / "train.parquet"), "--allow-experiment"]
    )
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert champion.run_id == expected["run_id"]


def test_main_select_ignores_runs_on_other_data(
    store: Path, tmp_path: Path, synthetic_frame: pd.DataFrame
) -> None:
    other = tmp_path / "other.parquet"
    synthetic_frame.head(10).to_parquet(other)
    with pytest.raises(SystemExit, match="no candidate run"):
        registry.main(["--select", "--train", str(other)])


def test_main_refuses_an_experiment_winner_without_the_flag(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_age = registry.candidate(runs_by_model()["lightgbm"])
    assert no_age["experiment"] == "no_age"
    monkeypatch.setattr(registry, "select_run", lambda c, **k: no_age)
    with pytest.raises(SystemExit, match="experiment=no_age"):
        registry.main(["--select", "--train", str(store / "train.parquet")])
    assert not mlflow.MlflowClient().search_registered_models()


def test_main_run_id_registers_that_run(store: Path) -> None:
    run_id = runs_by_model()["lightgbm"].info.run_id
    registry.main(["--run-id", run_id])
    client = mlflow.MlflowClient()
    assert client.get_model_version_by_alias(MODEL_NAME, "champion").run_id == run_id
