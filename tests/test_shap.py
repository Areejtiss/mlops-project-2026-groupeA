"""Tests for the SHAP explainer of the served model (T3, contract 2 section 6)."""

import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from crediscore.config import Params
from crediscore.data.featurize import add_derived_features, fix_days_employed
from crediscore.explain import shap_explainer as se
from crediscore.explain.shap_explainer import Explainer, Factor
from crediscore.models.pipeline import build_pipeline, calibrate
from crediscore.models.train import SHAP_UNIT, prepare_xy

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"
SMALL = {
    "logreg": {},
    "lightgbm": {"n_estimators": 30},
    "xgboost": {"n_estimators": 30},
    "random_forest": {"n_estimators": 20},
}
DERIVED = [
    "ANNUITY_INCOME_RATIO",
    "CREDIT_INCOME_RATIO",
    "CREDIT_TERM",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "DAYS_EMPLOYED_ANOM",
    "EXT_SOURCE_MISSING_COUNT",
    "IS_THIN_FILE",
]


def fit(params: Params, frame: pd.DataFrame, name: str, numeric: list[str]) -> object:
    X, y = prepare_xy(frame, params)
    pipe = build_pipeline(
        name,
        numeric=numeric,
        categorical=params.features.categorical,
        seed=params.data.random_state,
        model_params=SMALL[name],
    )
    return calibrate(pipe).fit(X, y)


@pytest.fixture(scope="module")
def data() -> tuple[Params, pd.DataFrame]:
    from crediscore.config import load_params
    from conftest import make_synthetic_frame

    params = load_params(PARAMS_FILE)
    return params, make_synthetic_frame(params)


@pytest.fixture(scope="module")
def models(data: tuple[Params, pd.DataFrame]) -> dict[str, object]:
    params, frame = data
    return {name: fit(params, frame, name, params.features.numeric) for name in SMALL}


def rows(data: tuple[Params, pd.DataFrame], n: int = 20) -> pd.DataFrame:
    params, frame = data
    X, _ = prepare_xy(frame.iloc[:n], params)
    return X


def margin(model: object, X: pd.DataFrame, unit: str) -> np.ndarray:
    pipe = model.calibrated_classifiers_[0].estimator  # type: ignore[attr-defined]
    p = pipe.predict_proba(X)[:, 1]
    return logit(p) if unit == "log_odds" else p


@pytest.mark.parametrize("name", list(SMALL))
def test_three_factors_per_row_sorted_by_absolute_contribution(
    data: tuple[Params, pd.DataFrame], models: dict[str, object], name: str
) -> None:
    X = rows(data)
    out = Explainer(models[name]).explain(X)
    assert len(out) == len(X)
    for factors in out:
        assert len(factors) == 3
        assert all(isinstance(f, Factor) for f in factors)
        mags = [abs(f.contribution) for f in factors]
        assert mags == sorted(mags, reverse=True)


@pytest.mark.parametrize("name", list(SMALL))
def test_factors_are_the_largest_contributions_on_original_columns(
    data: tuple[Params, pd.DataFrame], models: dict[str, object], name: str
) -> None:
    params, _ = data
    X = rows(data)
    ex = Explainer(models[name])
    _, contrib = ex.contributions(X)
    assert list(contrib.columns) == params.features.all
    for i, factors in enumerate(ex.explain(X)):
        top = contrib.iloc[i].abs().sort_values(ascending=False, kind="stable")
        assert [f.feature for f in factors] == list(top.index[:3])
        for f in factors:
            assert f.contribution == pytest.approx(contrib.iloc[i][f.feature])


@pytest.mark.parametrize("name", list(SMALL))
def test_additivity_in_the_explainer_space(
    data: tuple[Params, pd.DataFrame], models: dict[str, object], name: str
) -> None:
    X = rows(data)
    ex = Explainer(models[name])
    assert ex.unit == SHAP_UNIT[name]
    base, contrib = ex.contributions(X)
    expected = margin(models[name], X, ex.unit)
    assert base + contrib.sum(axis=1).to_numpy() == pytest.approx(expected, abs=1e-5)


def test_one_hot_contributions_are_summed_back_to_their_column(
    data: tuple[Params, pd.DataFrame], models: dict[str, object]
) -> None:
    # Linear model, independent features: changing OCCUPATION_TYPE moves only
    # the OCCUPATION_TYPE contribution, by the coefficient difference.
    X = rows(data, 1)
    ex = Explainer(models["logreg"])
    changed = X.copy()
    changed["OCCUPATION_TYPE"] = "C" if X["OCCUPATION_TYPE"].iloc[0] != "C" else "A"
    _, before = ex.contributions(X)
    _, after = ex.contributions(changed)
    diff = (after - before).iloc[0]
    assert diff.drop("OCCUPATION_TYPE").abs().max() == pytest.approx(0.0, abs=1e-12)
    assert diff["OCCUPATION_TYPE"] != pytest.approx(0.0)


def test_value_is_the_raw_input_and_none_when_missing(
    data: tuple[Params, pd.DataFrame], models: dict[str, object]
) -> None:
    X = rows(data, 50)
    out = Explainer(models["lightgbm"]).explain(X)
    for i, factors in enumerate(out):
        for f in factors:
            raw = X.iloc[i][f.feature]
            if pd.isna(raw):
                assert f.value is None
            else:
                assert f.value == raw
                assert type(f.value) in (float, int, str)
            assert type(f.contribution) is float
            assert type(f.feature) is str


def test_explain_is_deterministic(
    data: tuple[Params, pd.DataFrame], models: dict[str, object]
) -> None:
    X = rows(data)
    assert Explainer(models["lightgbm"]).explain(X) == Explainer(
        models["lightgbm"]
    ).explain(X)


def test_excluded_feature_is_never_a_factor(data: tuple[Params, pd.DataFrame]) -> None:
    params, frame = data
    numeric = [c for c in params.features.numeric if c != "AGE_YEARS"]
    model = fit(params, frame, "lightgbm", numeric)
    ex = Explainer(model)
    _, contrib = ex.contributions(rows(data))
    assert "AGE_YEARS" not in contrib.columns
    assert all(f.feature != "AGE_YEARS" for fs in ex.explain(rows(data)) for f in fs)


def test_one_raw_request_with_nones_follows_the_serving_path(
    data: tuple[Params, pd.DataFrame], models: dict[str, object]
) -> None:
    params, frame = data
    raw = frame.iloc[[0]].drop(columns=DERIVED + ["TARGET"]).copy()
    raw["DAYS_BIRTH"] = -30 * 365.25
    raw["DAYS_EMPLOYED"] = None
    raw["EXT_SOURCE_1"] = None
    raw["OCCUPATION_TYPE"] = None
    raw = raw.astype(object)  # a JSON request: every field may be None
    X = add_derived_features(fix_days_employed(raw))[params.features.all]
    X[params.features.numeric] = X[params.features.numeric].astype("float64")
    model = models["lightgbm"]
    assert 0 <= model.predict_proba(X)[0, 1] <= 1  # type: ignore[attr-defined]
    (factors,) = Explainer(model).explain(X)
    assert len(factors) == 3
    for f in factors:
        if f.feature in ("EXT_SOURCE_1", "OCCUPATION_TYPE", "EMPLOYED_YEARS"):
            assert f.value is None


def test_logreg_reference_can_be_a_background_sample(
    data: tuple[Params, pd.DataFrame], models: dict[str, object]
) -> None:
    X = rows(data)
    ex = Explainer(models["logreg"], background=rows(data, 200))
    base, contrib = ex.contributions(X)
    expected = margin(models["logreg"], X, "log_odds")
    assert base + contrib.sum(axis=1).to_numpy() == pytest.approx(expected, abs=1e-6)
    assert base != pytest.approx(Explainer(models["logreg"]).contributions(X)[0])


def test_unknown_classifier_is_rejected(data: tuple[Params, pd.DataFrame]) -> None:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline

    params, frame = data
    X, y = prepare_xy(frame, params)
    pipe = build_pipeline(
        "logreg", params.features.numeric, params.features.categorical, 0
    )
    pipe = Pipeline([("pre", pipe.named_steps["pre"]), ("model", GaussianNB())])
    model = CalibratedClassifierCV(pipe, cv=2, ensemble=False).fit(X, y)
    with pytest.raises(ValueError, match="GaussianNB"):
        Explainer(model)


@pytest.mark.slow
def test_one_row_latency_has_a_generous_ci_bound(
    data: tuple[Params, pd.DataFrame], models: dict[str, object]
) -> None:
    X = rows(data, 50)
    ex = Explainer(models["lightgbm"])
    ex.explain(X.iloc[[0]])  # warm-up
    times = []
    for i in range(len(X)):
        start = time.perf_counter()
        ex.explain(X.iloc[[i]])
        times.append(time.perf_counter() - start)
    assert np.percentile(times, 95) < 0.5  # strict 50 ms measured locally (T9)


# --- summary plot ------------------------------------------------------------


def test_summary_figure_ranks_mean_absolute_contributions(
    data: tuple[Params, pd.DataFrame], models: dict[str, object]
) -> None:
    ex = Explainer(models["lightgbm"])
    X = rows(data, 100)
    importance = se.mean_abs_contributions(ex, X)
    _, contrib = ex.contributions(X)
    assert importance.equals(contrib.abs().mean().sort_values(ascending=False))
    fig = se.summary_figure(importance, unit=ex.unit, top=5)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == list(importance.index[:5])[::-1]  # largest on top
    assert "log_odds" in ax.get_xlabel()


def test_main_writes_the_png_and_logs_a_run_linked_to_the_model_run(
    tmp_path: Path,
    data: tuple[Params, pd.DataFrame],
    models: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frame = data
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("crediscore")
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            models["lightgbm"], name="model", serialization_format="cloudpickle"
        )
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", "crediscore")
    mlflow.MlflowClient().set_registered_model_alias(
        "crediscore", "champion", version.version
    )
    data_file = tmp_path / "train.parquet"
    frame.to_parquet(data_file)
    out = tmp_path / "fig" / "shap_summary.png"
    se.main(
        [
            "--params",
            str(PARAMS_FILE),
            "--model",
            "models:/crediscore@champion",
            "--data",
            str(data_file),
            "--rows",
            "300",
            "--output",
            str(out),
        ]
    )
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    runs = mlflow.search_runs(
        experiment_names=["crediscore"],
        filter_string="tags.kind = 'shap_summary'",
        output_format="list",
    )
    (shap_run,) = runs
    tags = shap_run.data.tags
    assert tags["model_run_id"] == run.info.run_id
    assert tags["model_uri"] == "models:/crediscore@champion"
    assert tags["model_version"] == str(version.version)
    assert tags["shap_unit"] == "log_odds"
    assert shap_run.data.params["rows"] == "300"
    files = [a.path for a in mlflow.MlflowClient().list_artifacts(shap_run.info.run_id)]
    assert files == ["shap_summary.png"]


def test_main_without_tracking_only_writes_the_png(
    tmp_path: Path,
    data: tuple[Params, pd.DataFrame],
    models: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import joblib

    params, frame = data
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    joblib.dump(models["random_forest"], model_dir / "model.joblib")
    data_file = tmp_path / "train.parquet"
    frame.to_parquet(data_file)
    out = tmp_path / "shap.png"
    se.main(
        [
            "--params",
            str(PARAMS_FILE),
            "--model",
            str(model_dir),
            "--data",
            str(data_file),
            "--rows",
            "5000",  # more than the file: every row is used
            "--output",
            str(out),
        ]
    )
    assert out.stat().st_size > 1000
