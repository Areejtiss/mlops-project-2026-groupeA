"""DVC stage 2 - sample, clean and build the model features.

Usage:
    poetry run python -m crediscore.data.featurize

Steps, in order:
    1. stratified sample of ``data.sample_size`` rows;
    2. drop columns with more than ``data.missing_threshold`` missing values;
    3. DAYS_EMPLOYED == 365243 (sentinel for "not employed") -> NaN + flag;
    4. derived features (ratios, ages, thin-file indicators);
    5. keep only id, target, sensitive columns and the params.yaml features.

Output: data/processed/features.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

from crediscore.config import Params, load_params
from crediscore.data.sampling import stratified_sample
from crediscore.data.schema import required_raw_columns

FEATURES_FILE = "features.parquet"

# Home Credit uses DAYS_EMPLOYED = 365243 (~1000 years) as a placeholder,
# mostly for pensioners and unemployed clients.
DAYS_EMPLOYED_SENTINEL = 365_243
DAYS_PER_YEAR = 365.25
EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
# A client is a "thin file" when at least this many external scores are missing.
THIN_FILE_MIN_MISSING_EXT = 2
# Raw numeric columns read by the derived features. They are cast to float64
# first: a one-row API request with a JSON null gives an object column of None.
RAW_INPUTS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    *EXT_SOURCES,
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that gives NaN (never inf) on a zero denominator."""
    result = numerator / denominator.where(denominator != 0)
    return result.replace([np.inf, -np.inf], np.nan)


def drop_sparse_columns(
    df: pd.DataFrame, threshold: float, keep: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose share of missing values is above ``threshold``.

    Args:
        df: Input data.
        threshold: Maximum allowed share of missing values (e.g. 0.6).
        keep: Columns the pipeline needs. If one of them is too sparse, the
            function raises instead of silently removing it.

    Returns:
        The reduced DataFrame and the sorted list of dropped columns.

    Raises:
        ValueError: If a column in ``keep`` is above the threshold.
    """
    missing_share = df.isna().mean()
    dropped = sorted(missing_share[missing_share > threshold].index)
    needed = sorted(set(dropped) & set(keep))
    if needed:
        raise ValueError(
            f"required columns are more than {threshold:.0%} missing: {needed}. "
            "Remove them from params.yaml or raise data.missing_threshold."
        )
    return df.drop(columns=dropped), dropped


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the 365243 sentinel by NaN and flag it in DAYS_EMPLOYED_ANOM."""
    out = df.copy()
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].astype("float64")  # None -> NaN
    is_anom = out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    out["DAYS_EMPLOYED_ANOM"] = is_anom.astype(int)
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].mask(is_anom)
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features listed in params.yaml.

    Expects DAYS_EMPLOYED to be already fixed (see ``fix_days_employed``).
    """
    out = df.copy()
    out[RAW_INPUTS] = out[RAW_INPUTS].astype("float64")  # None -> NaN
    # Debt burden and size of the loan relative to income
    out["ANNUITY_INCOME_RATIO"] = safe_divide(
        out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"]
    )
    out["CREDIT_INCOME_RATIO"] = safe_divide(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    # Share of the credit repaid each year (inverse of the loan duration)
    out["CREDIT_TERM"] = safe_divide(out["AMT_ANNUITY"], out["AMT_CREDIT"])
    # DAYS_* are negative counts of days before the application
    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / DAYS_PER_YEAR
    out["EMPLOYED_YEARS"] = -out["DAYS_EMPLOYED"] / DAYS_PER_YEAR

    ext_missing = out[EXT_SOURCES].isna().sum(axis=1)
    out["EXT_SOURCE_MISSING_COUNT"] = ext_missing.astype(int)
    # Thin file: external bureaus barely know the client (inclusion target).
    bureau_requests = out["AMT_REQ_CREDIT_BUREAU_YEAR"]
    is_thin = (
        (ext_missing >= THIN_FILE_MIN_MISSING_EXT)
        | bureau_requests.isna()
        | (bureau_requests == 0)
    )
    out["IS_THIN_FILE"] = is_thin.astype(int)
    return out


def output_columns(params: Params) -> list[str]:
    """Columns written by the stage: id, target, sensitive, then features."""
    return [params.id_col, params.target, *params.sensitive_cols, *params.features.all]


def featurize(raw: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Run the 5 featurize steps on the raw data and return the result."""
    sample = stratified_sample(
        raw, params.target, params.data.sample_size, params.data.random_state
    )
    sample, dropped = drop_sparse_columns(
        sample, params.data.missing_threshold, keep=required_raw_columns(params)
    )
    print(f"Dropped {len(dropped)} columns with too many missing values.")
    sample = fix_days_employed(sample)
    sample = add_derived_features(sample)
    return sample[output_columns(params)].reset_index(drop=True)


def main() -> None:
    """Command-line entry point (DVC stage `featurize`)."""
    params = load_params()
    raw = pd.read_csv(params.data.raw_path)
    features = featurize(raw, params)

    out_dir = Path(params.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / FEATURES_FILE, index=False)

    print(
        f"Wrote {out_dir / FEATURES_FILE}: {features.shape[0]:,} rows x "
        f"{features.shape[1]} columns ({len(params.features.all)} features), "
        f"default rate {features[params.target].mean():.4f}, "
        f"thin-file rate {features['IS_THIN_FILE'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
