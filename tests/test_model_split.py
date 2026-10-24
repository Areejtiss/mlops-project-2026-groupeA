"""Tests for the validation carve-out from train (D1)."""

import pandas as pd
import pytest

from crediscore.config import Params
from crediscore.models.split import split_validation


def split(df: pd.DataFrame, params: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_validation(
        df,
        target=params.target,
        val_size=params.data.val_size,
        seed=params.data.random_state,
    )


def test_sizes(params: Params, synthetic_frame: pd.DataFrame) -> None:
    fit, val = split(synthetic_frame, params)
    assert len(val) == 400
    assert len(fit) == 1600


def test_stratified_within_half_a_point(
    params: Params, synthetic_frame: pd.DataFrame
) -> None:
    fit, val = split(synthetic_frame, params)
    rate = synthetic_frame[params.target].mean()
    assert abs(fit[params.target].mean() - rate) <= 0.005
    assert abs(val[params.target].mean() - rate) <= 0.005


def test_no_id_overlap_and_nothing_lost(
    params: Params, synthetic_frame: pd.DataFrame
) -> None:
    fit, val = split(synthetic_frame, params)
    ids_fit, ids_val = set(fit[params.id_col]), set(val[params.id_col])
    assert not ids_fit & ids_val
    assert ids_fit | ids_val == set(synthetic_frame[params.id_col])


def test_deterministic_with_the_seed(
    params: Params, synthetic_frame: pd.DataFrame
) -> None:
    a = split(synthetic_frame, params)
    b = split(synthetic_frame, params)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


def test_other_seed_gives_another_split(
    params: Params, synthetic_frame: pd.DataFrame
) -> None:
    _, val_a = split(synthetic_frame, params)
    _, val_b = split_validation(
        synthetic_frame, target=params.target, val_size=0.2, seed=7
    )
    assert set(val_a[params.id_col]) != set(val_b[params.id_col])


def test_val_size_is_respected(params: Params, synthetic_frame: pd.DataFrame) -> None:
    _, val = split_validation(
        synthetic_frame, target=params.target, val_size=0.1, seed=0
    )
    assert len(val) == 200


def test_input_not_mutated(params: Params, synthetic_frame: pd.DataFrame) -> None:
    before = synthetic_frame.copy()
    split(synthetic_frame, params)
    pd.testing.assert_frame_equal(synthetic_frame, before)


def test_missing_target_raises(params: Params, synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        split(synthetic_frame.drop(columns=params.target), params)


def test_original_index_is_kept(params: Params, synthetic_frame: pd.DataFrame) -> None:
    df = synthetic_frame.set_axis(synthetic_frame.index + 10_000)
    fit, val = split(df, params)
    for part in (fit, val):
        pd.testing.assert_frame_equal(part, df.loc[part.index])
