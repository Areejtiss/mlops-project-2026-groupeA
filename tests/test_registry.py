"""Tests for model selection (L3-1) and registration with aliases (L3-3)."""

import json
import os
import shutil
import subprocess
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
CONFIG = registry.run_config(load_params(PARAMS_FILE))


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
        "config": CONFIG,
    }
    return {**c, **over}


# --- select_run (L3-1, pure) ---------------------------------------------------


def test_lowest_cost_among_runs_reaching_the_auc_floor() -> None:
    runs = [cand("a", 0.74, 0.20), cand("b", 0.76, 0.32), cand("c", 0.75, 0.31)]
    assert select_run(runs, CONFIG)["run_id"] == "c"  # a is cheaper but below 0.75


def test_every_run_is_a_candidate_when_none_reaches_the_floor() -> None:
    runs = [cand("a", 0.70, 0.33), cand("b", 0.74, 0.34)]
    assert select_run(runs, CONFIG)["run_id"] == "a"


def test_cost_ties_are_broken_by_auc() -> None:
    runs = [cand("a", 0.76, 0.30), cand("b", 0.77, 0.30), cand("c", 0.75, 0.30)]
    assert select_run(runs, CONFIG)["run_id"] == "b"


def test_runs_breaking_the_d3_d8_constraints_are_excluded() -> None:
    runs = [
        cand("low_recall", 0.80, 0.10, recall=0.64),
        cand("review_cap", 0.80, 0.10, review_share=0.151),
        cand("ok", 0.76, 0.30, recall=0.65, review_share=0.15),  # inclusive
    ]
    assert select_run(runs, CONFIG)["run_id"] == "ok"


def test_no_feasible_run_raises() -> None:
    with pytest.raises(ValueError, match="no candidate run"):
        select_run([cand("a", 0.8, 0.1, recall=0.1)], CONFIG)


def test_runs_trained_with_other_params_are_excluded() -> None:
    other = {**CONFIG, "cost_review": 0.5}
    runs = [cand("old_costs", 0.80, 0.10, config=other), cand("ok", 0.76, 0.30)]
    assert select_run(runs, CONFIG)["run_id"] == "ok"
    with pytest.raises(ValueError, match="no candidate run"):
        select_run([cand("old_costs", 0.80, 0.10, config=other)], CONFIG)


def test_run_config_is_the_logged_cost_constraints_and_val_size() -> None:
    params = load_params(PARAMS_FILE)
    assert CONFIG == {
        "cost_fn": params.cost.cost_fn,
        "cost_fp": params.cost.cost_fp,
        "cost_review": params.cost.cost_review,
        "min_recall": params.threshold.min_recall,
        "max_review_share": params.threshold.max_review_share,
        "val_size": params.data.val_size,
    }


def test_run_config_takes_val_size_not_test_size() -> None:
    params = load_params(PARAMS_FILE)
    data = params.data.model_copy(update={"test_size": 0.3, "val_size": 0.1})
    other = params.model_copy(update={"data": data})
    assert registry.run_config(other)["val_size"] == 0.1


def test_search_is_filtered_on_the_server() -> None:
    seen: list[str] = []

    class FakeClient:
        def get_experiment_by_name(self, name: str) -> object:
            return type("E", (), {"experiment_id": "1"})()

        def search_runs(self, ids: list[str], filter_string: str) -> list:
            seen.append(filter_string)
            return []

    assert registry.training_runs(FakeClient(), "abc") == []  # type: ignore[arg-type]
    (f,) = seen
    assert "tags.train_data_hash = 'abc'" in f
    assert "tags.`mlflow.parentRunId` IS NULL" in f
    assert "attributes.status = 'FINISHED'" in f


def test_custom_floor() -> None:
    runs = [cand("a", 0.70, 0.20), cand("b", 0.80, 0.30)]
    assert select_run(runs, CONFIG, min_auc=0.65)["run_id"] == "a"


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
    dvc = base / "data" / "raw" / "application_train.csv.dvc"
    dvc.parent.mkdir(parents=True)
    dvc.write_text("outs:\n- md5: rawmd5\n  path: application_train.csv\n", "utf-8")
    # A throwaway repository, so that the runs get a git_sha tag even from a
    # git archive copy (no .git) of the project.
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], cwd=base, check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "x"], cwd=base, check=True
    )
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


def aliases() -> dict[str, str]:
    model = mlflow.MlflowClient().get_registered_model(MODEL_NAME)
    return {k: str(v) for k, v in model.aliases.items()}


def test_registration_is_always_challenger(store: Path) -> None:
    # Even the first model: promote gates it (fairness only), never registry.
    runs = runs_by_model()
    client = mlflow.MlflowClient()
    assert register(client, runs["logreg"].info.run_id) == ("1", "challenger")
    assert aliases() == {"challenger": "1"}
    client.set_registered_model_alias(MODEL_NAME, "champion", "1")
    assert register(client, runs["lightgbm"].info.run_id) == ("2", "challenger")
    assert aliases() == {"champion": "1", "challenger": "2"}
    assert client.get_model_version_by_alias(MODEL_NAME, "challenger").run_id == (
        runs["lightgbm"].info.run_id
    )


def test_main_select_exits_3_when_the_selected_run_is_already_champion(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--select", "--train", str(store / "train.parquet"), "--allow-experiment"]
    registry.main(args)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", "1")  # promoted
    client.delete_registered_model_alias(MODEL_NAME, "challenger")
    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_info:
        registry.main(args)
    assert exit_info.value.code == 3
    assert "nothing to register: selected run is already champion" in (
        capsys.readouterr().out
    )
    assert [str(v.version) for v in client.search_model_versions()] == ["1"]
    assert aliases() == {"champion": "1"}


def test_main_select_registers_when_the_champion_is_another_run(
    store: Path,
) -> None:
    runs = runs_by_model()
    client = mlflow.MlflowClient()
    expected = select_run([registry.candidate(r) for r in runs.values()], CONFIG)
    other = next(r for r in runs.values() if r.info.run_id != expected["run_id"])
    register(client, other.info.run_id)
    client.set_registered_model_alias(MODEL_NAME, "champion", "1")
    registry.main(
        ["--select", "--train", str(store / "train.parquet"), "--allow-experiment"]
    )
    assert aliases() == {"champion": "1", "challenger": "2"}


def test_version_tags_are_copied_exactly(store: Path) -> None:
    run = runs_by_model()["logreg"]
    client = mlflow.MlflowClient()
    version, _ = register(client, run.info.run_id)
    tags = client.get_model_version(MODEL_NAME, version).tags
    for key in VERSION_TAGS:
        assert tags[key] == run.data.tags[key], key
    assert len(tags["git_sha"]) == 40
    assert set(VERSION_TAGS) >= {
        "threshold_low",
        "threshold_high",
        "shap_unit",
        "train_data_hash",
        "raw_data_hash",
        "model_type",
    }
    assert tags["raw_data_hash"] == "rawmd5"
    th = json.loads((store / "m1" / "thresholds.json").read_text(encoding="utf-8"))
    assert float(tags["threshold_low"]) == th["low"]  # full precision
    assert float(tags["threshold_high"]) == th["high"]
    assert tags["shap_unit"] == "log_odds"


def test_registered_challenger_loads_with_the_alias(store: Path) -> None:
    client = mlflow.MlflowClient()
    register(client, runs_by_model()["logreg"].info.run_id)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@challenger")
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
    expected = select_run([registry.candidate(r) for r in runs.values()], CONFIG)
    registry.main(
        ["--select", "--train", str(store / "train.parquet"), "--allow-experiment"]
    )
    challenger = client.get_model_version_by_alias(MODEL_NAME, "challenger")
    assert challenger.run_id == expected["run_id"]


def test_main_select_ignores_runs_on_other_data(
    store: Path, tmp_path: Path, synthetic_frame: pd.DataFrame
) -> None:
    other = tmp_path / "other.parquet"
    synthetic_frame.head(10).to_parquet(other)
    with pytest.raises(SystemExit, match="no candidate run"):
        registry.main(["--select", "--train", str(other)])


def test_main_select_reads_the_params_file_it_is_given(
    store: Path, tmp_path: Path
) -> None:
    text = PARAMS_FILE.read_text(encoding="utf-8")
    assert "cost_review: 0.3 " in text
    other = tmp_path / "params.yaml"
    other.write_text(text.replace("cost_review: 0.3 ", "cost_review: 0.5 "), "utf-8")
    # The runs were logged with cost_review 0.3: none matches the other file.
    with pytest.raises(SystemExit, match="no candidate run"):
        registry.main(
            [
                "--select",
                "--params",
                str(other),
                "--train",
                str(store / "train.parquet"),
            ]
        )


def test_main_refuses_an_experiment_winner_without_the_flag(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_age = registry.candidate(runs_by_model()["lightgbm"])
    assert no_age["experiment"] == "no_age"
    monkeypatch.setattr(registry, "select_run", lambda *a, **k: no_age)
    with pytest.raises(SystemExit, match="experiment=no_age"):
        registry.main(["--select", "--train", str(store / "train.parquet")])
    assert not mlflow.MlflowClient().search_registered_models()


def test_main_run_id_registers_that_run(store: Path) -> None:
    run_id = runs_by_model()["lightgbm"].info.run_id
    registry.main(["--run-id", run_id])
    client = mlflow.MlflowClient()
    assert client.get_model_version_by_alias(MODEL_NAME, "challenger").run_id == run_id


def test_no_experiment_means_no_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    assert registry.training_runs(mlflow.MlflowClient(), "any") == []


# --- lot 3 mutation survivors ------------------------------------------------


def test_nested_trials_are_not_candidates_and_fields_map_to_validation_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("crediscore")
    tags = {"model_type": "lightgbm", "train_data_hash": "h"}
    metrics = {
        "val_roc_auc": 0.71,
        "val_pr_auc": 0.21,
        "val_cost_per_file": 0.31,
        "val_recall": 0.66,
        "val_review_share": 0.14,
    }
    with mlflow.start_run(run_name="parent") as parent:
        mlflow.set_tags(tags)
        mlflow.log_params({"min_recall": 0.65, "max_review_share": 0.15})
        mlflow.log_metrics(metrics)
        with mlflow.start_run(run_name="trial-0", nested=True):
            mlflow.set_tags(tags)
            mlflow.log_metrics({**metrics, "val_cost_per_file": 0.01})
    runs = registry.training_runs(mlflow.MlflowClient(), "h")
    assert [r.info.run_id for r in runs] == [parent.info.run_id]
    c = registry.candidate(runs[0])
    assert (c["roc_auc"], c["cost_per_file"], c["recall"], c["review_share"]) == (
        0.71,
        0.31,
        0.66,
        0.14,
    )
    assert (c["min_recall"], c["max_review_share"]) == (0.65, 0.15)


def test_a_run_with_excluded_features_is_an_experiment_even_untagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("crediscore")
    with mlflow.start_run() as run:
        mlflow.set_tags({"model_type": "lightgbm", "excluded_features": "AGE_YEARS"})
        mlflow.log_params({"min_recall": 0.65, "max_review_share": 0.15})
        mlflow.log_metrics(
            {
                "val_roc_auc": 0.8,
                "val_cost_per_file": 0.3,
                "val_recall": 0.7,
                "val_review_share": 0.1,
            }
        )
    c = registry.candidate(mlflow.MlflowClient().get_run(run.info.run_id))
    assert c["experiment"] == "excluded:AGE_YEARS"


def test_candidate_carries_the_logged_config(store: Path) -> None:
    assert registry.candidate(runs_by_model()["logreg"])["config"] == CONFIG
