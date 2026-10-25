"""Tests for the Optuna search on the fit part (L3-2)."""

import pandas as pd
import pytest

from crediscore.config import Params
from crediscore.models.train import prepare_xy
from crediscore.models.tune import SEARCH_SPACES, tune


def run(name: str, params: Params, df: pd.DataFrame, n_trials: int, **kw: object):
    X, y = prepare_xy(df, params)
    return tune(
        X,
        y,
        name,
        numeric=params.features.numeric,
        categorical=params.features.categorical,
        seed=params.data.random_state,
        n_trials=n_trials,
        **kw,
    )


def test_only_the_boosted_models_are_tuned() -> None:
    assert sorted(SEARCH_SPACES) == ["lightgbm", "xgboost"]


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_best_params_are_deterministic_with_the_seed(
    name: str, params: Params, synthetic_frame: pd.DataFrame
) -> None:
    best, value = run(name, params, synthetic_frame, 2)
    again, value_again = run(name, params, synthetic_frame, 2)
    assert best == again
    assert value == value_again
    assert 0.6 < value <= 1  # mean CV ROC-AUC on a learnable frame


def test_every_trial_is_reported_with_its_params_and_cv_auc(
    params: Params, synthetic_frame: pd.DataFrame
) -> None:
    seen: list[tuple[int, dict[str, object], float]] = []
    best, value = run(
        "lightgbm",
        params,
        synthetic_frame,
        3,
        on_trial=lambda n, hp, auc: seen.append((n, hp, auc)),
    )
    assert [n for n, _, _ in seen] == [0, 1, 2]
    assert value == max(auc for _, _, auc in seen)
    assert best == next(hp for _, hp, auc in seen if auc == value)
    assert set(best) == {
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "colsample_bytree",
        "reg_lambda",
    }


def test_untunable_model_is_rejected(
    params: Params, synthetic_frame: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="no search space for 'logreg'"):
        run("logreg", params, synthetic_frame, 2)
