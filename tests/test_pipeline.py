"""Tests for the preprocessing + model pipeline and its calibration (T2, T5)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from crediscore.config import Params
from crediscore.models.pipeline import MODELS, build_pipeline, calibrate

NAMES = sorted(MODELS)


def make(name: str, params: Params) -> Pipeline:
    return build_pipeline(
        name,
        numeric=params.features.numeric,
        categorical=params.features.categorical,
        seed=params.data.random_state,
    )


def test_both_models_are_available() -> None:
    assert NAMES == ["lightgbm", "logreg"]


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
