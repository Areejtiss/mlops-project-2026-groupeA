"""Preprocessing + classifier pipeline, and its calibration (contract 2, T2, T5).

Numeric columns: median imputation + scaling. Categorical columns:
most-frequent imputation + one-hot (unknown categories ignored). Every other
column (CODE_GENDER, SK_ID_CURR, TARGET...) is dropped. Imbalance is handled by
class weights, never resampling (T5); calibration fixes the inflated scores.

Usage:
    from crediscore.models.pipeline import build_pipeline, calibrate

    pipe = build_pipeline("logreg", numeric=num, categorical=cat, seed=42)
    model = calibrate(pipe).fit(X, y)       # the served object
    model.calibrated_classifiers_[0].estimator   # pipeline SHAP explains
"""

from collections.abc import Callable
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


MODELS: dict[str, Callable[[int], Any]] = {"logreg": _logreg, "lightgbm": _lightgbm}


def build_pipeline(
    name: str, numeric: list[str], categorical: list[str], seed: int
) -> Pipeline:
    """Build the unfitted `pre` + `model` pipeline of contract 2.

    Args:
        name: Model name, a key of `MODELS` ("logreg" or "lightgbm").
        numeric: Numeric feature columns (`params.features.numeric`).
        categorical: Categorical feature columns (`params.features.categorical`).
        seed: Random seed of the classifier.

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
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", MODELS[name](seed))])


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
