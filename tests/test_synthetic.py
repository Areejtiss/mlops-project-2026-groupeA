"""The shared synthetic frame has the shape the model tests rely on."""

import pandas as pd

from crediscore.config import Params
from conftest import make_synthetic_frame


def test_synthetic_frame_shape(params: Params, synthetic_frame: pd.DataFrame) -> None:
    df = synthetic_frame
    extra = [params.target, "CODE_GENDER", params.id_col]
    assert list(df.columns) == params.features.all + extra
    assert len(df) == 2000
    assert 0.07 <= df[params.target].mean() <= 0.10
    assert df[["EXT_SOURCE_1", "EXT_SOURCE_3"]].isna().any().all()
    assert df[params.id_col].is_unique


def test_synthetic_frame_is_deterministic(params: Params) -> None:
    pd.testing.assert_frame_equal(
        make_synthetic_frame(params), make_synthetic_frame(params)
    )
