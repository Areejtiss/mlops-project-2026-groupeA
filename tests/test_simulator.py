"""Tests for the DriftSimulator (economic crisis scenario)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crediscore.config import DriftConfig, load_params
from crediscore.monitoring.simulator import (
    CREDIT_COLS,
    ECONOMIC_CRISIS_COLS,
    EMPLOYMENT_COLS,
    INCOME_COLS,
    DriftSimulator,
)

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ci_sample.parquet"


@pytest.fixture
def config() -> DriftConfig:
    return load_params(PARAMS_FILE).drift


@pytest.fixture
def batch() -> pd.DataFrame:
    """The committed 5k fixture: real featurized rows, available in CI."""
    return pd.read_parquet(FIXTURE)


@pytest.fixture
def drifted(config: DriftConfig, batch: pd.DataFrame) -> pd.DataFrame:
    return DriftSimulator(config).economic_crisis(batch)


def mean_ratio(after: pd.DataFrame, before: pd.DataFrame, col: str) -> float:
    return float(after[col].mean() / before[col].mean())


def test_income_drops(
    config: DriftConfig, batch: pd.DataFrame, drifted: pd.DataFrame
) -> None:
    factor = config.economic_crisis.income_factor
    for col in INCOME_COLS:
        assert mean_ratio(drifted, batch, col) == pytest.approx(factor, abs=0.01)


def test_credit_grows(
    config: DriftConfig, batch: pd.DataFrame, drifted: pd.DataFrame
) -> None:
    factor = config.economic_crisis.credit_factor
    for col in CREDIT_COLS:
        assert mean_ratio(drifted, batch, col) == pytest.approx(factor, abs=0.01)


def test_employment_shrinks(
    config: DriftConfig, batch: pd.DataFrame, drifted: pd.DataFrame
) -> None:
    factor = config.economic_crisis.employment_factor
    for col in EMPLOYMENT_COLS:
        assert mean_ratio(drifted, batch, col) == pytest.approx(factor, abs=0.01)


def test_debt_burden_increases(batch: pd.DataFrame, drifted: pd.DataFrame) -> None:
    # Lower income and higher loans: both ratios to income go up.
    for col in ["ANNUITY_INCOME_RATIO", "CREDIT_INCOME_RATIO"]:
        assert drifted[col].mean() > batch[col].mean() * 1.4


def test_ratios_are_consistent(drifted: pd.DataFrame) -> None:
    expected = drifted["AMT_CREDIT"] / drifted["AMT_INCOME_TOTAL"]
    pd.testing.assert_series_equal(
        drifted["CREDIT_INCOME_RATIO"], expected, check_names=False
    )


def test_other_columns_are_unchanged(
    batch: pd.DataFrame, drifted: pd.DataFrame
) -> None:
    others = [c for c in batch.columns if c not in ECONOMIC_CRISIS_COLS]
    assert "TARGET" in others and "EXT_SOURCE_2" in others
    pd.testing.assert_frame_equal(drifted[others], batch[others])


def test_missing_values_stay_missing(
    batch: pd.DataFrame, drifted: pd.DataFrame
) -> None:
    # EMPLOYED_YEARS is NaN for the DAYS_EMPLOYED anomaly: it must not be filled.
    assert batch["EMPLOYED_YEARS"].isna().any()
    pd.testing.assert_series_equal(
        drifted["EMPLOYED_YEARS"].isna(), batch["EMPLOYED_YEARS"].isna()
    )


def test_no_negative_amounts(drifted: pd.DataFrame) -> None:
    assert (drifted[INCOME_COLS + CREDIT_COLS].dropna() >= 0).all().all()


def test_same_seed_gives_same_batch(config: DriftConfig, batch: pd.DataFrame) -> None:
    sim = DriftSimulator(config)
    pd.testing.assert_frame_equal(
        sim.economic_crisis(batch), sim.economic_crisis(batch)
    )


def test_different_seed_gives_different_batch(
    config: DriftConfig, batch: pd.DataFrame
) -> None:
    sim = DriftSimulator(config)
    a = sim.economic_crisis(batch, seed=1)
    b = sim.economic_crisis(batch, seed=2)
    assert not a["AMT_INCOME_TOTAL"].equals(b["AMT_INCOME_TOTAL"])


def test_input_is_not_modified(config: DriftConfig, batch: pd.DataFrame) -> None:
    original = batch.copy()
    DriftSimulator(config).economic_crisis(batch)
    pd.testing.assert_frame_equal(batch, original)


def test_zero_noise_applies_exact_factors(
    config: DriftConfig, batch: pd.DataFrame
) -> None:
    crisis = config.economic_crisis.model_copy(update={"noise_std": 0.0})
    exact = DriftSimulator(config.model_copy(update={"economic_crisis": crisis}))
    drifted = exact.economic_crisis(batch)
    np.testing.assert_allclose(
        drifted["AMT_INCOME_TOTAL"], batch["AMT_INCOME_TOTAL"] * crisis.income_factor
    )


def test_missing_column_is_rejected(config: DriftConfig, batch: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="AMT_CREDIT"):
        DriftSimulator(config).economic_crisis(batch.drop(columns=["AMT_CREDIT"]))
