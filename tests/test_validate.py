"""Tests for the validate stage, on a small synthetic DataFrame (no real data)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crediscore.config import Params, load_params
from crediscore.data.schema import DERIVED_FEATURES, required_raw_columns
from crediscore.data.validate import validate, write_report

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"
N_ROWS = 1000


@pytest.fixture
def params() -> Params:
    return load_params(PARAMS_FILE)


@pytest.fixture
def good_df(params: Params) -> pd.DataFrame:
    """A DataFrame with every required column and an 8% default rate."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {col: rng.normal(size=N_ROWS) for col in required_raw_columns(params)}
    )
    df[params.id_col] = np.arange(N_ROWS)
    df[params.target] = (np.arange(N_ROWS) < 80).astype(int)  # exactly 8%
    return df


def shape_of(df: pd.DataFrame) -> tuple[int, int]:
    return df.shape[0], df.shape[1]


def failed_checks(report: dict[str, object]) -> set[str]:
    checks = report["checks"]
    assert isinstance(checks, list)
    return {c["name"] for c in checks if not c["passed"]}


def test_required_columns_include_sources_not_derived(params: Params) -> None:
    required = required_raw_columns(params)
    assert "DAYS_BIRTH" in required  # source of AGE_YEARS
    assert "CODE_GENDER" in required  # kept for the fairness audit
    assert not set(DERIVED_FEATURES) & set(required)
    assert len(required) == len(set(required))


def test_good_data_passes(params: Params, good_df: pd.DataFrame) -> None:
    report = validate(good_df, params, expected_shape=shape_of(good_df))
    assert report["passed"] is True
    assert report["target_rate"] == pytest.approx(0.08)


def test_wrong_shape_fails(params: Params, good_df: pd.DataFrame) -> None:
    report = validate(good_df, params, expected_shape=(307_511, 122))
    assert failed_checks(report) == {"shape"}


def test_missing_column_fails(params: Params, good_df: pd.DataFrame) -> None:
    corrupted = good_df.drop(columns=["AMT_INCOME_TOTAL"])
    report = validate(corrupted, params, expected_shape=shape_of(corrupted))
    assert failed_checks(report) == {"required_columns"}


def test_non_binary_target_fails(params: Params, good_df: pd.DataFrame) -> None:
    good_df.loc[0, params.target] = 2
    report = validate(good_df, params, expected_shape=shape_of(good_df))
    assert "target_binary" in failed_checks(report)


def test_missing_target_value_fails(params: Params, good_df: pd.DataFrame) -> None:
    good_df[params.target] = good_df[params.target].astype(float)
    good_df.loc[0, params.target] = np.nan
    report = validate(good_df, params, expected_shape=shape_of(good_df))
    assert "target_binary" in failed_checks(report)


def test_unexpected_target_rate_fails(params: Params, good_df: pd.DataFrame) -> None:
    good_df[params.target] = (np.arange(N_ROWS) < 500).astype(int)  # 50%
    report = validate(good_df, params, expected_shape=shape_of(good_df))
    assert failed_checks(report) == {"target_rate"}


def test_duplicated_ids_fail(params: Params, good_df: pd.DataFrame) -> None:
    good_df.loc[1, params.id_col] = good_df.loc[0, params.id_col]
    report = validate(good_df, params, expected_shape=shape_of(good_df))
    assert failed_checks(report) == {"id_unique"}


def test_report_is_written_as_json(
    tmp_path: Path, params: Params, good_df: pd.DataFrame
) -> None:
    report = validate(good_df, params, expected_shape=shape_of(good_df))
    path = tmp_path / "reports" / "validation.json"
    write_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8"))["passed"] is True
