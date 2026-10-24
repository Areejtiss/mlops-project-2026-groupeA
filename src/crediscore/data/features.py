"""Derived features, shared by the featurize stage and the API (D2, D10).

One pure, row-wise function, so that training and serving compute the ratios
exactly the same way (no train/serve skew). The model input is then exactly
`params.features.all`.

Usage:
    from crediscore.data.features import add_derived_features

    df = add_derived_features(raw_df)   # raw columns in, raw + derived out
"""

import numpy as np
import pandas as pd

DAYS_EMPLOYED_SENTINEL = 365243
DERIVED = (
    "ANNUITY_INCOME_RATIO",
    "CREDIT_INCOME_RATIO",
    "CREDIT_TERM",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "DAYS_EMPLOYED_ANOM",
    "EXT_SOURCE_MISSING_COUNT",
    "IS_THIN_FILE",
)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, with NaN instead of +-inf when den is 0."""
    return (num / den).replace([np.inf, -np.inf], np.nan)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 8 derived model features to a copy of the raw frame.

    `DAYS_EMPLOYED == 365243` (a sentinel, not 1000 years of employment) is
    replaced by NaN and flagged in `DAYS_EMPLOYED_ANOM`. A thin file has both
    `EXT_SOURCE_1` and `EXT_SOURCE_3` missing (D10).

    Args:
        df: Raw application rows. Needs AMT_INCOME_TOTAL, AMT_CREDIT,
            AMT_ANNUITY, DAYS_BIRTH, DAYS_EMPLOYED and EXT_SOURCE_1/2/3.

    Returns:
        A new frame, same index: every input column plus the `DERIVED` ones.
        The input is not modified.
    """
    out = df.copy()
    anom = out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].mask(anom)
    ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    out["ANNUITY_INCOME_RATIO"] = _ratio(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["CREDIT_INCOME_RATIO"] = _ratio(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["CREDIT_TERM"] = _ratio(out["AMT_ANNUITY"], out["AMT_CREDIT"])
    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    out["EMPLOYED_YEARS"] = -out["DAYS_EMPLOYED"] / 365.25
    out["DAYS_EMPLOYED_ANOM"] = anom.astype(int)
    out["EXT_SOURCE_MISSING_COUNT"] = ext.isna().sum(axis=1)
    thin = ext["EXT_SOURCE_1"].isna() & ext["EXT_SOURCE_3"].isna()
    out["IS_THIN_FILE"] = thin.astype(int)
    return out
