"""Tests for the shared derived-feature function (D2, D10)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crediscore.config import load_params
from crediscore.data.features import DERIVED, add_derived_features

PARAMS = load_params(Path(__file__).resolve().parents[1] / "params.yaml")


def raw_row(**overrides: object) -> pd.DataFrame:
    """One applicant with every raw column the model features need."""
    row: dict[str, object] = {
        col: 1.0 for col in PARAMS.features.numeric if col not in DERIVED
    }
    row.update({col: "x" for col in PARAMS.features.categorical})
    row.update(
        AMT_INCOME_TOTAL=100_000.0,
        AMT_CREDIT=400_000.0,
        AMT_ANNUITY=20_000.0,
        DAYS_BIRTH=-365.25 * 40,
        DAYS_EMPLOYED=-365.25 * 5,
        EXT_SOURCE_1=0.5,
        EXT_SOURCE_2=0.6,
        EXT_SOURCE_3=0.7,
    )
    row.update(overrides)
    return pd.DataFrame([row])


def test_output_has_every_model_feature() -> None:
    out = add_derived_features(raw_row())
    assert set(PARAMS.features.all) <= set(out.columns)


def test_derived_list_matches_params() -> None:
    raw = set(raw_row().columns)
    assert set(DERIVED) == set(PARAMS.features.all) - raw


def test_values() -> None:
    out = add_derived_features(raw_row()).iloc[0]
    assert out["ANNUITY_INCOME_RATIO"] == pytest.approx(0.2)
    assert out["CREDIT_INCOME_RATIO"] == pytest.approx(4.0)
    assert out["CREDIT_TERM"] == pytest.approx(0.05)
    assert out["AGE_YEARS"] == pytest.approx(40.0)
    assert out["EMPLOYED_YEARS"] == pytest.approx(5.0)
    assert out["DAYS_EMPLOYED_ANOM"] == 0
    assert out["EXT_SOURCE_MISSING_COUNT"] == 0
    assert out["IS_THIN_FILE"] == 0


@pytest.mark.parametrize(
    ("zero_col", "nan_cols"),
    [
        ("AMT_INCOME_TOTAL", ["ANNUITY_INCOME_RATIO", "CREDIT_INCOME_RATIO"]),
        ("AMT_CREDIT", ["CREDIT_TERM"]),
    ],
)
def test_division_by_zero_gives_nan_not_inf(zero_col: str, nan_cols: list[str]) -> None:
    out = add_derived_features(raw_row(**{zero_col: 0.0}))
    for col in nan_cols:
        assert np.isnan(out[col].iloc[0]), col


def test_zero_over_zero_gives_nan() -> None:
    out = add_derived_features(raw_row(AMT_ANNUITY=0.0, AMT_CREDIT=0.0))
    assert np.isnan(out["CREDIT_TERM"].iloc[0])


def test_days_employed_sentinel() -> None:
    out = add_derived_features(raw_row(DAYS_EMPLOYED=365243)).iloc[0]
    assert np.isnan(out["DAYS_EMPLOYED"])
    assert np.isnan(out["EMPLOYED_YEARS"])
    assert out["DAYS_EMPLOYED_ANOM"] == 1


def test_missing_days_employed_is_not_flagged_as_sentinel() -> None:
    out = add_derived_features(raw_row(DAYS_EMPLOYED=np.nan)).iloc[0]
    assert out["DAYS_EMPLOYED_ANOM"] == 0
    assert np.isnan(out["EMPLOYED_YEARS"])


@pytest.mark.parametrize(
    ("ext1", "ext2", "ext3", "missing", "thin"),
    [
        (np.nan, np.nan, np.nan, 3, 1),
        (np.nan, 0.6, np.nan, 2, 1),  # D10: only EXT_SOURCE_1 and _3 matter
        (0.5, np.nan, np.nan, 2, 0),
        (np.nan, np.nan, 0.7, 2, 0),
        (0.5, 0.6, 0.7, 0, 0),
    ],
)
def test_thin_file_and_missing_count(
    ext1: float, ext2: float, ext3: float, missing: int, thin: int
) -> None:
    df = raw_row(EXT_SOURCE_1=ext1, EXT_SOURCE_2=ext2, EXT_SOURCE_3=ext3)
    out = add_derived_features(df).iloc[0]
    assert out["EXT_SOURCE_MISSING_COUNT"] == missing
    assert out["IS_THIN_FILE"] == thin


def test_input_is_not_mutated() -> None:
    df = raw_row(DAYS_EMPLOYED=365243)
    before = df.copy()
    add_derived_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_row_wise_and_index_preserved() -> None:
    a = raw_row()
    b = raw_row(AMT_INCOME_TOTAL=50_000.0, DAYS_EMPLOYED=365243)
    both = pd.concat([a, b], ignore_index=True).set_axis([10, 20])
    out = add_derived_features(both)
    assert list(out.index) == [10, 20]
    pd.testing.assert_frame_equal(
        out.iloc[[0]].reset_index(drop=True), add_derived_features(a)
    )
