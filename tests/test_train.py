"""End-to-end tests of the training entry point on the synthetic frame (T4)."""

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from crediscore.config import Params
from crediscore.models import train as train_module
from crediscore.models.threshold import decide
from crediscore.models.train import build_signature, main, prepare_xy, read_dvc_md5

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"


@pytest.fixture
def train_file(tmp_path: Path, synthetic_frame: pd.DataFrame) -> Path:
    path = tmp_path / "train.parquet"
    synthetic_frame.to_parquet(path)
    return path


def test_main_writes_artifacts_and_logs_an_mlflow_run(
    tmp_path: Path,
    train_file: Path,
    params: Params,
    synthetic_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)  # mlflow artifacts land under tmp_path
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)  # an earlier test set the process-wide URI
    dvc = tmp_path / "data" / "raw" / "application_train.csv.dvc"
    dvc.parent.mkdir(parents=True)
    dvc.write_text("outs:\n- md5: rawmd5\n  path: application_train.csv\n", "utf-8")
    monkeypatch.setattr(train_module, "_git_sha", lambda: "abc123sha")
    out = tmp_path / "models"
    main(
        [
            "--params",
            str(PARAMS_FILE),
            "--train",
            str(train_file),
            "--model",
            "logreg",
            "--output-dir",
            str(out),
        ]
    )

    model = joblib.load(out / "model.joblib")
    th = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert len(model.calibrated_classifiers_) == 1
    assert th["low"] <= th["high"]
    assert th["cost_review"] == params.cost.cost_review
    assert th["min_recall"] == params.threshold.min_recall
    assert th["val_recall"] >= params.threshold.min_recall
    assert th["val_review_share"] <= params.threshold.max_review_share
    assert 0.6 < metrics["val"]["roc_auc"] <= 1
    assert not (out / "roc.png").exists()  # only in --sample mode

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["crediscore"], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    for key in [
        "val_roc_auc",
        "val_pr_auc",
        "val_brier",
        "val_recall",
        "val_cost",
        "val_cost_per_file",
        "val_review_share",
    ]:
        assert key in run.data.metrics, key
    assert run.data.metrics["val_recall"] == pytest.approx(th["val_recall"])
    tags = run.data.tags
    assert float(tags["threshold_low"]) == th["low"]  # full precision
    assert float(tags["threshold_high"]) == th["high"]
    assert tags["shap_unit"] == "log_odds"
    assert tags["model_type"] == "logreg"
    assert tags["serialization_format"] == "cloudpickle"
    assert tags["raw_data_hash"] == "rawmd5"  # md5 of the raw CSV, from DVC
    assert tags["train_data_hash"] == hashlib.md5(train_file.read_bytes()).hexdigest()
    assert tags["git_sha"] == "abc123sha"
    assert "data_hash" not in tags
    assert run.data.params["model"] == "logreg"
    names = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    assert {
        "thresholds.json",
        "metrics.json",
        "roc_curve.png",
        "pr_curve.png",
        "calibration_curve.png",
        "confusion_matrix.png",
    } <= names

    logged = mlflow.search_logged_models(output_format="list")
    assert len(logged) == 1
    assert logged[0].tags["shap_unit"] == "log_odds"
    reloaded = mlflow.sklearn.load_model(logged[0].model_uri)
    X, _ = prepare_xy(synthetic_frame, params)
    np.testing.assert_array_equal(
        reloaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1]
    )
    inputs = mlflow.models.get_model_info(logged[0].model_uri).signature.inputs
    assert inputs.input_names() == params.features.all
    assert not any(col.required for col in inputs.inputs)


def test_sample_mode_lightgbm_without_mlflow(
    tmp_path: Path, train_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    out = tmp_path / "cml"
    main(
        [
            "--params",
            str(PARAMS_FILE),
            "--sample",
            str(train_file),
            "--model",
            "lightgbm",
            "--output-dir",
            str(out),
        ]
    )
    for name in ["model.joblib", "thresholds.json", "metrics.json", "roc.png"]:
        assert (out / name).exists(), name
    assert not (tmp_path / "mlruns").exists()
    model = joblib.load(out / "model.joblib")
    th = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(train_file)
    d = decide(model.predict_proba(df)[:, 1], th["low"], th["high"])
    assert set(d.tolist()) <= {"ACCEPTE", "A_ETUDIER", "REFUSE"}


def test_prepare_xy_selects_and_casts(
    params: Params, synthetic_frame: pd.DataFrame
) -> None:
    df = synthetic_frame.copy()
    df["CNT_CHILDREN"] = df["CNT_CHILDREN"].round().astype("int64")
    X, y = prepare_xy(df, params)
    assert list(X.columns) == params.features.all
    assert (X[params.features.numeric].dtypes == "float64").all()
    assert y.dtype.kind == "i" and set(np.unique(y)) <= {0, 1}


def test_signature_is_all_optional_double_and_string(params: Params) -> None:
    sig = build_signature(params.features.numeric, params.features.categorical)
    cols = sig.inputs.inputs
    assert [c.name for c in cols] == params.features.all
    assert {c.type.name for c in cols[: len(params.features.numeric)]} == {"double"}
    assert {c.type.name for c in cols[len(params.features.numeric) :]} == {"string"}
    assert not any(c.required for c in cols)


def test_tune_logs_nested_trials_and_fits_the_best_params(
    tmp_path: Path, train_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    out = tmp_path / "models"
    main(
        [
            "--params",
            str(PARAMS_FILE),
            "--train",
            str(train_file),
            "--model",
            "lightgbm",
            "--tune",
            "2",
            "--output-dir",
            str(out),
        ]
    )
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["crediscore"], output_format="list")
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    (parent,) = parents
    children = [r for r in runs if r not in parents]
    assert len(children) == 2
    assert {r.data.tags["mlflow.parentRunId"] for r in children} == {parent.info.run_id}
    assert all("cv_roc_auc" in r.data.metrics for r in children)
    best = max(children, key=lambda r: r.data.metrics["cv_roc_auc"])
    assert parent.data.metrics["best_cv_roc_auc"] == best.data.metrics["cv_roc_auc"]
    assert parent.data.params["tune_trials"] == "2"
    used = json.loads((out / "model_params.json").read_text(encoding="utf-8"))
    assert {k: str(v) for k, v in used.items()} == best.data.params
    for key, value in used.items():
        assert parent.data.params[f"model__{key}"] == str(value)
    model = joblib.load(out / "model.joblib")
    inner = model.calibrated_classifiers_[0].estimator[-1]
    assert inner.n_estimators == used["n_estimators"]


def test_model_params_file_exclude_and_experiment_tag(
    tmp_path: Path,
    train_file: Path,
    synthetic_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The D5 run: same params as a tuned run, without AGE_YEARS, tagged.
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    hp = tmp_path / "hp.json"
    hp.write_text(json.dumps({"n_estimators": 40, "num_leaves": 7}), "utf-8")
    out = tmp_path / "noage"
    main(
        [
            "--params",
            str(PARAMS_FILE),
            "--train",
            str(train_file),
            "--model",
            "lightgbm",
            "--model-params",
            str(hp),
            "--exclude",
            "AGE_YEARS",
            "--experiment",
            "no_age",
            "--output-dir",
            str(out),
        ]
    )
    model = joblib.load(out / "model.joblib")
    inner = model.calibrated_classifiers_[0].estimator
    assert (inner[-1].n_estimators, inner[-1].num_leaves) == (40, 7)
    assert not any("AGE_YEARS" in c for c in inner[0].get_feature_names_out())
    X = synthetic_frame.copy()
    changed = X.assign(AGE_YEARS=99.0)
    np.testing.assert_array_equal(
        model.predict_proba(changed[X.columns])[:, 1],
        model.predict_proba(X)[:, 1],
    )
    mlflow.set_tracking_uri(uri)
    (run,) = mlflow.search_runs(experiment_names=["crediscore"], output_format="list")
    assert run.data.tags["experiment"] == "no_age"
    assert run.data.tags["excluded_features"] == "AGE_YEARS"
    assert run.info.run_name == "lightgbm-no_age"
    assert run.data.params["model__n_estimators"] == "40"


def test_exclude_rejects_a_column_that_is_not_a_numeric_feature(
    train_file: Path,
) -> None:
    with pytest.raises(SystemExit):
        main(
            ["--params", str(PARAMS_FILE), "--train", str(train_file), "--exclude", "X"]
        )


def test_xgboost_gets_scale_pos_weight_from_the_fit_part(
    params: Params, synthetic_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    real = train_module.build_pipeline

    def spy(*a, **k):
        seen.update(k.get("model_params") or {})
        return real(*a, **k)

    monkeypatch.setattr(train_module, "build_pipeline", spy)
    result = train_module.train(synthetic_frame, "xgboost", params)
    n_fit = result["metrics"]["n_fit"]
    y_val = result["y_val"]
    n_pos_fit = synthetic_frame[params.target].sum() - y_val.sum()
    assert seen["scale_pos_weight"] == pytest.approx((n_fit - n_pos_fit) / n_pos_fit)
    assert train_module.SHAP_UNIT["xgboost"] == "log_odds"


def test_shap_unit_per_model() -> None:
    # Contract 2 section 6: probability for RandomForest, log-odds otherwise.
    assert train_module.SHAP_UNIT == {
        "logreg": "log_odds",
        "lightgbm": "log_odds",
        "xgboost": "log_odds",
        "random_forest": "probability",
    }


def test_read_dvc_md5(tmp_path: Path) -> None:
    dvc = tmp_path / "x.csv.dvc"
    dvc.write_text("outs:\n- md5: abc123\n  path: x.csv\n", encoding="utf-8")
    assert read_dvc_md5(dvc) == "abc123"
    assert read_dvc_md5(tmp_path / "missing.dvc") is None


def test_unknown_model_is_rejected(train_file: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--train", str(train_file), "--model", "svm"])


def test_main_survives_a_cp1252_console(
    tmp_path: Path, train_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MLflow prints an emoji; a redirected Windows stdout is cp1252 (b4c696a).
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    main(
        [
            "--params",
            str(PARAMS_FILE),
            "--sample",
            str(train_file),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    print("\U0001f3c3 View run")  # what mlflow writes after each run
    console.flush()
    written = console.buffer.getvalue()  # type: ignore[attr-defined]
    assert b"? View run" in written


def test_git_sha_inside_and_outside_a_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Throwaway repositories: the test must not depend on the checkout it runs
    # in (a git archive copy or a Docker build context has no .git).
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo, plain = tmp_path / "repo", tmp_path / "plain"
    repo.mkdir()
    plain.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "x"], cwd=repo, check=True
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(repo)
    assert train_module._git_sha() == head
    monkeypatch.chdir(plain)
    assert train_module._git_sha() is None


def test_sample_mode_tags_the_md5_of_the_sample_file(
    tmp_path: Path,
    train_file: Path,
    synthetic_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)  # an earlier test set the process-wide URI
    other = tmp_path / "other.parquet"  # a --train that must NOT be hashed
    synthetic_frame.head(500).to_parquet(other)
    main(
        [
            "--params",
            str(PARAMS_FILE),
            "--train",
            str(other),
            "--sample",
            str(train_file),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    mlflow.set_tracking_uri(uri)
    (run,) = mlflow.search_runs(experiment_names=["crediscore"], output_format="list")
    assert (
        run.data.tags["train_data_hash"]
        == hashlib.md5(train_file.read_bytes()).hexdigest()
    )


def test_accept_all_cost_counts_probabilities_of_exactly_one(
    params: Params, synthetic_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every validation proba is 1.0: "accept all" must still accept them (NEVER
    # cut), so its cost is cost_fn per default.
    class AllOnes:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.c_[np.zeros(len(X)), np.ones(len(X))]

    monkeypatch.setattr(train_module, "fit_model", lambda *a, **k: AllOnes())
    result = train_module.train(synthetic_frame, "logreg", params)
    y_val = result["y_val"]
    assert result["metrics"]["accept_all_cost_per_file"] == pytest.approx(
        params.cost.cost_fn * y_val.mean()
    )


def test_tuning_sees_the_excluded_feature_list(
    params: Params, synthetic_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_tune(X: object, y: object, name: str, **kw: object):
        seen.update(kw)
        return {"n_estimators": 20}, 0.7

    monkeypatch.setattr(train_module, "tune", fake_tune)
    out = train_module.train(
        synthetic_frame, "lightgbm", params, exclude=("AGE_YEARS",), n_trials=1
    )
    assert seen["numeric"] == [c for c in params.features.numeric if c != "AGE_YEARS"]
    assert out["metrics"]["model_params"]["n_estimators"] == 20


def test_exclude_requires_an_experiment_tag(
    train_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--params",
                str(PARAMS_FILE),
                "--train",
                str(train_file),
                "--exclude",
                "AGE_YEARS",
            ]
        )
    assert "--exclude needs --experiment" in capsys.readouterr().err
