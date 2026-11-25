"""Shared fixtures: a deterministic synthetic applicant frame (no data needed).

The frame has exactly `params.features.all` + TARGET + CODE_GENDER +
SK_ID_CURR, NaNs in EXT_SOURCE_*, about 8 % positives and a learnable signal
(low EXT_SOURCE_2/3 means more defaults), so that models reach a real AUC.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crediscore.config import Params, load_params

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"


def make_synthetic_frame(params: Params, n: int = 2000) -> pd.DataFrame:
    """Build the synthetic frame, seeded with `params.data.random_state`."""
    rng = np.random.default_rng(params.data.random_state)
    df = pd.DataFrame(
        {col: rng.normal(size=n) for col in params.features.numeric},
    )
    for col, missing in [
        ("EXT_SOURCE_1", 0.55),
        ("EXT_SOURCE_2", 0.01),
        ("EXT_SOURCE_3", 0.2),
    ]:
        df[col] = rng.uniform(size=n)
        df.loc[rng.uniform(size=n) < missing, col] = np.nan
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MISSING_COUNT"] = ext.isna().sum(axis=1).astype(float)
    # Rule of crediscore.data.featurize (D10). AMT_REQ_CREDIT_BUREAU_YEAR is
    # never 0 or NaN in this frame, so only the EXT_SOURCE part applies.
    df["IS_THIN_FILE"] = (df["EXT_SOURCE_MISSING_COUNT"] >= 2).astype(float)
    df["DAYS_EMPLOYED_ANOM"] = (rng.uniform(size=n) < 0.18).astype(float)
    df["AGE_YEARS"] = rng.uniform(21, 69, size=n)
    for col in params.features.categorical:
        df[col] = rng.choice(["A", "B", "C", None], size=n, p=[0.5, 0.3, 0.15, 0.05])
    logit = (
        -3.25
        - 4.0 * (ext["EXT_SOURCE_2"].fillna(0.5) - 0.5)
        - 3.0 * (ext["EXT_SOURCE_3"].fillna(0.5) - 0.5)
        - 0.03 * (df["AGE_YEARS"] - 45)
        + 0.6 * (df["OCCUPATION_TYPE"] == "C")
    )
    df[params.target] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["CODE_GENDER"] = rng.choice(["F", "M"], size=n, p=[0.65, 0.35])
    df[params.id_col] = np.arange(100_000, 100_000 + n)
    return df


@pytest.fixture(scope="session")
def params() -> Params:
    """The real params.yaml."""
    return load_params(PARAMS_FILE)


@pytest.fixture
def synthetic_frame(params: Params) -> pd.DataFrame:
    """A fresh copy of the 2 000-row synthetic frame (safe to mutate)."""
    return make_synthetic_frame(params)
