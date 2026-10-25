"""Behaviour tests of the served (calibrated) models on the synthetic frame.

Quality floor, output range, invariance to the ID and to the sensitive
column, and a directional check averaged over many rows (T9).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from crediscore.config import Params, load_params
from crediscore.models.pipeline import MODELS
from crediscore.models.train import fit_model, prepare_xy
from conftest import PARAMS_FILE, make_synthetic_frame

_FRAME = make_synthetic_frame(load_params(PARAMS_FILE))
FIT, HELD_OUT = _FRAME.iloc[:1500], _FRAME.iloc[1500:]
_CACHE: dict[str, object] = {}


@pytest.fixture(params=sorted(MODELS))
def model(request: pytest.FixtureRequest, params: Params):
    """One calibrated model per type, fitted once per session."""
    name = request.param
    if name not in _CACHE:
        X, y = prepare_xy(FIT, params)
        _CACHE[name] = fit_model(X, y, name, params)
    return _CACHE[name]


def proba(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df)[:, 1]


def test_auc_above_a_floor_on_held_out_rows(model, params: Params) -> None:
    y = HELD_OUT[params.target]
    assert roc_auc_score(y, proba(model, HELD_OUT)) > 0.7


def test_outputs_are_probabilities(model) -> None:
    p = proba(model, HELD_OUT)
    assert p.shape == (len(HELD_OUT),)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_id_and_gender_change_nothing(model, params: Params) -> None:
    changed = HELD_OUT.copy()
    changed[params.id_col] = changed[params.id_col].to_numpy()[::-1] + 7
    changed["CODE_GENDER"] = changed["CODE_GENDER"].map({"F": "M", "M": "F"})
    # atol: the random forest sums its trees in thread order (~1e-16).
    np.testing.assert_allclose(
        proba(model, changed), proba(model, HELD_OUT), rtol=0, atol=1e-12
    )


def test_higher_ext_source_2_does_not_raise_risk_on_average(model) -> None:
    better = HELD_OUT.copy()
    better["EXT_SOURCE_2"] = (better["EXT_SOURCE_2"] + 0.2).clip(upper=1.0)
    assert proba(model, better).mean() <= proba(model, HELD_OUT).mean()
