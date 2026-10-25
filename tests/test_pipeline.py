"""Tests for the preprocessing + model pipeline and its calibration (T2, T5)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from crediscore.config import Params
from crediscore.models.pipeline import (
    MODELS,
    N_JOBS,
    build_pipeline,
    calibrate,
    imbalance_params,
)

NAMES = sorted(MODELS)


def make(name: str, params: Params) -> Pipeline:
    return build_pipeline(
        name,
        numeric=params.features.numeric,
        categorical=params.features.categorical,
        seed=params.data.random_state,
    )


def test_all_models_are_available() -> None:
    assert NAMES == ["lightgbm", "logreg", "random_forest", "xgboost"]


@pytest.mark.parametrize("name", NAMES)
def test_fits_and_learns_the_signal(
    name: str, params: Params, synthetic_frame: pd.DataFrame
) -> None:
    X, y = synthetic_frame[params.features.all], synthetic_frame[params.target]
    proba = make(name, params).fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert roc_auc_score(y, proba) > 0.7


@pytest.mark.parametrize("name", NAMES)
def test_unknown_category_does_not_crash(
    name: str, params: Params, synthetic_frame: pd.DataFrame
) -> None:
    X, y = synthetic_frame[params.features.all], synthetic_frame[params.target]
    pipe = make(name, params).fit(X, y)
    new = X.head(3).copy()
    new["OCCUPATION_TYPE"] = "Astronaut"
    proba = pipe.predict_proba(new)[:, 1]
    assert np.isfinite(proba).all()


@pytest.mark.parametrize("name", NAMES)
def test_sensitive_and_id_columns_are_dropped(
    name: str, params: Params, synthetic_frame: pd.DataFrame
) -> None:
    df, y = synthetic_frame, synthetic_frame[params.target]
    extra = df[params.features.all + ["CODE_GENDER", params.id_col]]
    with_extra = make(name, params).fit(extra, y)
    without = make(name, params).fit(df[params.features.all], y)
    changed = extra.copy()
    changed["CODE_GENDER"] = changed["CODE_GENDER"].map({"F": "M", "M": "F"})
    changed[params.id_col] = changed[params.id_col][::-1].to_numpy()
    reference = without.predict_proba(df[params.features.all])[:, 1]
    np.testing.assert_array_equal(with_extra.predict_proba(extra)[:, 1], reference)
    np.testing.assert_array_equal(with_extra.predict_proba(changed)[:, 1], reference)
    used = set(with_extra[0].get_feature_names_out())
    assert not any("CODE_GENDER" in c or params.id_col in c for c in used)


@pytest.mark.parametrize("name", NAMES)
def test_none_and_nan_categories_are_both_imputed(
    name: str, params: Params, synthetic_frame: pd.DataFrame
) -> None:
    # Parquet and JSON null give None, CSV gives NaN: both mean "missing".
    X, y = synthetic_frame[params.features.all], synthetic_frame[params.target]
    pipe = make(name, params).fit(X, y)
    cat = params.features.categorical
    as_none, as_nan, as_mode = X.head(5).copy(), X.head(5).copy(), X.head(5).copy()
    as_none[cat], as_nan[cat] = None, np.nan
    for col in cat:
        as_mode[col] = X[col].mode()[0]
    expected = pipe.predict_proba(as_mode)[:, 1]
    np.testing.assert_allclose(pipe.predict_proba(as_none)[:, 1], expected)
    np.testing.assert_allclose(pipe.predict_proba(as_nan)[:, 1], expected)


def test_logreg_is_balanced(params: Params) -> None:
    assert make("logreg", params)[-1].class_weight == "balanced"


def test_lightgbm_handles_imbalance(params: Params) -> None:
    assert make("lightgbm", params)[-1].get_params()["is_unbalance"] is True


def test_random_forest_is_balanced_and_bounded(params: Params) -> None:
    rf = make("random_forest", params)[-1]
    assert rf.class_weight == "balanced"
    assert rf.n_jobs == N_JOBS and 0 < N_JOBS <= 8
    assert rf.random_state == params.data.random_state


def test_xgboost_is_bounded_and_seeded(params: Params) -> None:
    xgb = make("xgboost", params)[-1]
    assert xgb.n_jobs == N_JOBS
    assert xgb.random_state == params.data.random_state


def test_imbalance_params_weights_xgboost_positives_only() -> None:
    y = np.array([0] * 9 + [1] * 3)
    assert imbalance_params("xgboost", y) == {"scale_pos_weight": 3.0}
    for name in ["logreg", "lightgbm", "random_forest"]:
        assert imbalance_params(name, y) == {}


def test_model_params_override_the_defaults(params: Params) -> None:
    pipe = build_pipeline(
        "xgboost",
        numeric=params.features.numeric,
        categorical=params.features.categorical,
        seed=params.data.random_state,
        model_params={"max_depth": 2, "scale_pos_weight": 7.5},
    )
    assert pipe[-1].max_depth == 2
    assert pipe[-1].scale_pos_weight == 7.5


def test_numeric_subset_drops_a_feature(
    params: Params, synthetic_frame: pd.DataFrame
) -> None:
    # The D5 experiment: AGE_YEARS stays in the input but not in the model.
    numeric = [c for c in params.features.numeric if c != "AGE_YEARS"]
    pipe = build_pipeline(
        "lightgbm",
        numeric=numeric,
        categorical=params.features.categorical,
        seed=params.data.random_state,
    )
    X, y = synthetic_frame[params.features.all], synthetic_frame[params.target]
    pipe.fit(X, y)
    assert not any("AGE_YEARS" in c for c in pipe[0].get_feature_names_out())
    changed = X.copy()
    changed["AGE_YEARS"] = 99.0
    np.testing.assert_array_equal(
        pipe.predict_proba(changed)[:, 1], pipe.predict_proba(X)[:, 1]
    )


def test_unknown_model_raises(params: Params) -> None:
    with pytest.raises(ValueError, match="unknown model 'svm'"):
        make("svm", params)


@pytest.mark.parametrize("name", NAMES)
def test_calibrated_model_has_one_member_fit_on_all_train(
    name: str, params: Params, synthetic_frame: pd.DataFrame
) -> None:
    X, y = synthetic_frame[params.features.all], synthetic_frame[params.target]
    model = calibrate(make(name, params)).fit(X, y)
    assert len(model.calibrated_classifiers_) == 1
    inner = model.calibrated_classifiers_[0].estimator
    full = make(name, params).fit(X, y)
    np.testing.assert_allclose(
        inner.predict_proba(X)[:, 1], full.predict_proba(X)[:, 1]
    )
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_calibration_settings(params: Params) -> None:
    model = calibrate(make("logreg", params))
    assert (model.cv, model.method, model.ensemble) == (5, "isotonic", False)
