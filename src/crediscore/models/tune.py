"""Optuna search of the boosted models' hyperparameters on the fit part (L3-2).

Objective: mean ROC-AUC of a 5-fold stratified CV of the uncalibrated
pipeline, on the fit part only (80 % of train; validation and test are never
read here). Sampler: `TPESampler(seed=params.data.random_state)`. Calibration
is left out of the objective: isotonic calibration is monotone, so it barely
moves the ranking that ROC-AUC measures, and it would multiply the cost by 6.

Search spaces (small on purpose, about 25 trials each):

| LightGBM | range | XGBoost | range |
|---|---|---|---|
| n_estimators | 100-600 | n_estimators | 100-600 |
| learning_rate | 0.02-0.2 (log) | learning_rate | 0.02-0.2 (log) |
| num_leaves | 15-127 | max_depth | 3-8 |
| min_child_samples | 20-300 | min_child_weight | 1-50 (log) |
| colsample_bytree | 0.5-1.0 | colsample_bytree | 0.5-1.0 |
| reg_lambda | 1e-3-10 (log) | reg_lambda | 1e-3-10 (log) |

Entry point: `python -m crediscore.models.train --model lightgbm --tune 25`
(the best params then go through the normal fit + calibration + thresholds).

Usage:
    from crediscore.models.tune import tune

    best, cv_auc = tune(X_fit, y_fit, "lightgbm", numeric=num, categorical=cat,
                        seed=42, n_trials=25, on_trial=print)
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from crediscore.models.pipeline import build_pipeline, imbalance_params

TrialCallback = Callable[[int, dict[str, Any], float], None]


def _lightgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _xgboost_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 50.0, log=True
        ),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "lightgbm": _lightgbm_space,
    "xgboost": _xgboost_space,
}


def tune(
    X: pd.DataFrame,
    y: np.ndarray,
    name: str,
    *,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    n_trials: int,
    on_trial: TrialCallback | None = None,
) -> tuple[dict[str, Any], float]:
    """Seeded TPE search maximizing the 5-fold CV ROC-AUC on (X, y).

    Args:
        X: Fit-part features (never validation or test rows).
        y: Fit-part labels.
        name: "lightgbm" or "xgboost".
        numeric: Numeric columns the pipeline uses.
        categorical: Categorical columns the pipeline uses.
        seed: Seed of the sampler, the folds and the models.
        n_trials: Number of trials.
        on_trial: Called after each trial with (number, params, cv_auc), e.g.
            to log a nested MLflow run.

    Returns:
        (best params, best mean CV ROC-AUC).

    Raises:
        ValueError: If `name` has no search space.
    """
    if name not in SEARCH_SPACES:
        raise ValueError(
            f"no search space for {name!r}, expected {sorted(SEARCH_SPACES)}"
        )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    balance = imbalance_params(name, y)

    def objective(trial: optuna.Trial) -> float:
        hp = SEARCH_SPACES[name](trial)
        pipe = build_pipeline(name, numeric, categorical, seed, {**balance, **hp})
        auc = float(np.mean(cross_val_score(pipe, X, y, cv=folds, scoring="roc_auc")))
        if on_trial is not None:
            on_trial(trial.number, hp, auc)
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params), float(study.best_value)
