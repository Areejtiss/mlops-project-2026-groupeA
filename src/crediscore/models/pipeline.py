"""Preprocessing + classifier pipeline, and its calibration (contract 2, T2, T5).

Numeric columns: median imputation + scaling. Categorical columns:
most-frequent imputation + one-hot (unknown categories ignored). Every other
column (CODE_GENDER, SK_ID_CURR, TARGET...) is dropped. Imbalance is handled by
class weights, never resampling (T5): `class_weight="balanced"` for LogReg and
the random forest, `is_unbalance` for LightGBM, `scale_pos_weight` (from the
labels, `imbalance_params`) for XGBoost. Calibration fixes the inflated scores.

Usage:
    from crediscore.models.pipeline import build_pipeline, calibrate
    from crediscore.models.pipeline import imbalance_params

    pipe = build_pipeline("logreg", numeric=num, categorical=cat, seed=42)
    extra = {"max_depth": 4, **imbalance_params("xgboost", y)}
    pipe = build_pipeline("xgboost", numeric=num, categorical=cat, seed=42,
                          model_params=extra)
    model = calibrate(pipe).fit(X, y)       # the served object
    model.calibrated_classifiers_[0].estimator   # pipeline SHAP explains
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Threads per random forest / XGBoost fit: leaves cores for the other jobs.
N_JOBS = 4


def none_to_nan(X: Any) -> pd.DataFrame:
    """Turn None (parquet, JSON null) into NaN, which SimpleImputer imputes.

    Module-level (not a lambda) so that the fitted pipeline pickles.
    """
    df = pd.DataFrame(X).astype(object)
    return df.where(df.notna(), np.nan)


def _logreg(seed: int) -> Any:
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)


def _lightgbm(seed: int) -> Any:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=seed,
        verbose=-1,
    )


def _random_forest(seed: int) -> Any:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=N_JOBS,
        random_state=seed,
    )


def _xgboost(seed: int) -> Any:
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=seed,
    )


MODELS: dict[str, Callable[[int], Any]] = {
    "logreg": _logreg,
    "lightgbm": _lightgbm,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
}


def imbalance_params(name: str, y: npt.ArrayLike) -> dict[str, float]:
    """Label-dependent imbalance setting: XGBoost's `scale_pos_weight` (T5).

    Args:
        name: Model name.
        y: Training labels (0/1).

    Returns:
        `{"scale_pos_weight": n_neg / n_pos}` for XGBoost, `{}` otherwise (the
        other models balance themselves from the labels at fit time).
    """
    if name != "xgboost":
        return {}
    y = np.asarray(y)
    return {"scale_pos_weight": float(np.sum(y == 0) / np.sum(y == 1))}


def build_pipeline(
    name: str,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build the unfitted `pre` + `model` pipeline of contract 2.

    Args:
        name: Model name, a key of `MODELS`.
        numeric: Numeric feature columns (`params.features.numeric`, or a
            subset: the other input columns are then dropped).
        categorical: Categorical feature columns (`params.features.categorical`).
        seed: Random seed of the classifier.
        model_params: Classifier hyperparameters overriding the defaults
            (Optuna's best trial, `imbalance_params`).

    Returns:
        An unfitted two-step pipeline.

    Raises:
        ValueError: If `name` is not a known model.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "cat",
                make_pipeline(
                    FunctionTransformer(none_to_nan, feature_names_out="one-to-one"),
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    model = MODELS[name](seed).set_params(**(model_params or {}))
    return Pipeline([("pre", pre), ("model", model)])


def calibrate(pipeline: Pipeline) -> CalibratedClassifierCV:
    """Wrap a pipeline in the isotonic calibration of contract 2 (unfitted).

    With `ensemble=False`, one isotonic map is learned on 5-fold out-of-fold
    predictions and the pipeline is refitted on all rows: the served model is a
    monotone function of `calibrated_classifiers_[0].estimator`.

    Args:
        pipeline: The unfitted pipeline from `build_pipeline`.

    Returns:
        An unfitted `CalibratedClassifierCV`.
    """
    return CalibratedClassifierCV(pipeline, cv=5, method="isotonic", ensemble=False)
