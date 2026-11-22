"""Tests for the featurize stage, on small synthetic DataFrames."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crediscore.config import Params, load_params
from crediscore.data.featurize import (
    DAYS_EMPLOYED_SENTINEL,
    add_derived_features,
    drop_sparse_columns,
    featurize,
    fix_days_employed,
    safe_divide,
)
from crediscore.data.schema import required_raw_columns

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"


@pytest.fixture
def params() -> Params:
    return load_params(PARAMS_FILE)


def raw_row(**overrides: float) -> dict[str, float]:
    """One plausible raw applicant; override any column."""
    row = {
        "AMT_INCOME_TOTAL": 200_000.0,
        "AMT_CREDIT": 500_000.0,
        "AMT_ANNUITY": 25_000.0,
        "DAYS_BIRTH": -365.25 * 40,
        "DAYS_EMPLOYED": -365.25 * 5,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.7,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 2.0,
    }
    row.update(overrides)
    return row


def derive(**overrides: float) -> pd.Series:
    """Derived features of a single applicant."""
    df = fix_days_employed(pd.DataFrame([raw_row(**overrides)]))
    return add_derived_features(df).iloc[0]


# --- safe_divide -----------------------------------------------------------


def test_safe_divide_normal_case() -> None:
    result = safe_divide(pd.Series([10.0, 3.0]), pd.Series([2.0, 4.0]))
    assert result.tolist() == [5.0, 0.75]


def test_safe_divide_by_zero_gives_nan_not_inf() -> None:
    result = safe_divide(pd.Series([10.0, 0.0]), pd.Series([0.0, 0.0]))
    assert result.isna().all()


def test_safe_divide_propagates_nan() -> None:
    result = safe_divide(pd.Series([np.nan, 1.0]), pd.Series([2.0, np.nan]))
    assert result.isna().all()


# --- ratios ----------------------------------------------------------------


def test_ratios_values() -> None:
    row = derive()
    assert row["ANNUITY_INCOME_RATIO"] == pytest.approx(0.125)
    assert row["CREDIT_INCOME_RATIO"] == pytest.approx(2.5)
    assert row["CREDIT_TERM"] == pytest.approx(0.05)


def test_ratios_with_zero_income_are_nan() -> None:
    row = derive(AMT_INCOME_TOTAL=0.0)
    assert np.isnan(row["ANNUITY_INCOME_RATIO"])
    assert np.isnan(row["CREDIT_INCOME_RATIO"])
    assert row["CREDIT_TERM"] == pytest.approx(0.05)  # unaffected


def test_ratio_with_missing_annuity_is_nan() -> None:
    row = derive(AMT_ANNUITY=np.nan)
    assert np.isnan(row["ANNUITY_INCOME_RATIO"])
    assert np.isnan(row["CREDIT_TERM"])


def test_ages_in_years() -> None:
    row = derive()
    assert row["AGE_YEARS"] == pytest.approx(40.0)
    assert row["EMPLOYED_YEARS"] == pytest.approx(5.0)


# --- DAYS_EMPLOYED anomaly ---------------------------------------------------


def test_anomaly_becomes_nan_and_flagged() -> None:
    df = pd.DataFrame({"DAYS_EMPLOYED": [DAYS_EMPLOYED_SENTINEL, -1000.0]})
    fixed = fix_days_employed(df)
    assert np.isnan(fixed.loc[0, "DAYS_EMPLOYED"])
    assert fixed["DAYS_EMPLOYED_ANOM"].tolist() == [1, 0]
    assert fixed.loc[1, "DAYS_EMPLOYED"] == -1000.0  # normal row untouched


def test_anomaly_gives_missing_employed_years() -> None:
    row = derive(DAYS_EMPLOYED=DAYS_EMPLOYED_SENTINEL)
    assert np.isnan(row["EMPLOYED_YEARS"])
    assert row["DAYS_EMPLOYED_ANOM"] == 1


def test_fix_does_not_modify_input() -> None:
    df = pd.DataFrame({"DAYS_EMPLOYED": [DAYS_EMPLOYED_SENTINEL]})
    fix_days_employed(df)
    assert df.loc[0, "DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL


# --- thin file ---------------------------------------------------------------


def test_well_known_client_is_not_thin_file() -> None:
    row = derive()
    assert row["EXT_SOURCE_MISSING_COUNT"] == 0
    assert row["IS_THIN_FILE"] == 0


def test_one_missing_score_is_not_thin_file() -> None:
    row = derive(EXT_SOURCE_1=np.nan)
    assert row["EXT_SOURCE_MISSING_COUNT"] == 1
    assert row["IS_THIN_FILE"] == 0


def test_two_missing_scores_is_thin_file() -> None:
    row = derive(EXT_SOURCE_1=np.nan, EXT_SOURCE_3=np.nan)
    assert row["EXT_SOURCE_MISSING_COUNT"] == 2
    assert row["IS_THIN_FILE"] == 1


@pytest.mark.parametrize("requests", [0.0, np.nan])
def test_no_bureau_history_is_thin_file(requests: float) -> None:
    assert derive(AMT_REQ_CREDIT_BUREAU_YEAR=requests)["IS_THIN_FILE"] == 1


# --- sparse columns ------------------------------------------------------------


def test_drop_sparse_columns() -> None:
    df = pd.DataFrame({"full": [1, 2, 3, 4, 5], "sparse": [1, None, None, None, None]})
    out, dropped = drop_sparse_columns(df, threshold=0.6, keep=["full"])
    assert dropped == ["sparse"]
    assert list(out.columns) == ["full"]


def test_drop_sparse_refuses_to_drop_a_required_column() -> None:
    df = pd.DataFrame({"needed": [1, None, None, None, None]})
    with pytest.raises(ValueError, match="needed"):
        drop_sparse_columns(df, threshold=0.6, keep=["needed"])


# --- whole stage ---------------------------------------------------------------


def test_featurize_output(params: Params) -> None:
    rng = np.random.default_rng(0)
    n = 2000
    raw = pd.DataFrame(
        {col: rng.uniform(1, 10, size=n) for col in required_raw_columns(params)}
    )
    for col in params.features.categorical:
        raw[col] = rng.choice(["a", "b"], size=n)
    raw[params.id_col] = np.arange(n)
    raw[params.target] = (np.arange(n) % 10 == 0).astype(int)  # 10%
    raw["DAYS_BIRTH"] = -12_000.0
    raw["DAYS_EMPLOYED"] = -1_000.0
    raw["NOISE"] = np.nan  # 100% missing: must be dropped

    small = params.model_copy(
        update={"data": params.data.model_copy(update={"sample_size": 1000})}
    )
    out = featurize(raw, small)

    assert len(out) == 1000
    assert out[params.target].mean() == pytest.approx(0.10)
    assert list(out.columns) == [
        params.id_col,
        params.target,
        *params.sensitive_cols,
        *params.features.all,
    ]
    assert "CODE_GENDER" not in params.features.all
