"""Column contract shared by the validate and featurize stages.

params.yaml lists the model features. Some of them do not exist in the raw file:
they are derived by the featurize stage. This module says which ones, and which
raw columns they are built from, so that validate can check the raw file
contains everything featurize will need.
"""

from crediscore.config import Params

# Derived feature -> raw columns it is computed from.
DERIVED_FEATURES: dict[str, list[str]] = {
    "ANNUITY_INCOME_RATIO": ["AMT_ANNUITY", "AMT_INCOME_TOTAL"],
    "CREDIT_INCOME_RATIO": ["AMT_CREDIT", "AMT_INCOME_TOTAL"],
    "CREDIT_TERM": ["AMT_ANNUITY", "AMT_CREDIT"],
    "AGE_YEARS": ["DAYS_BIRTH"],
    "EMPLOYED_YEARS": ["DAYS_EMPLOYED"],
    "DAYS_EMPLOYED_ANOM": ["DAYS_EMPLOYED"],
    "EXT_SOURCE_MISSING_COUNT": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    "IS_THIN_FILE": [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ],
}


def raw_feature_columns(params: Params) -> list[str]:
    """Model features read directly from the raw file (not derived)."""
    return [col for col in params.features.all if col not in DERIVED_FEATURES]


def required_raw_columns(params: Params) -> list[str]:
    """Every raw column the pipeline needs, without duplicates, in a stable order.

    Identifier and target, sensitive columns (fairness audit), raw features and
    the sources of the derived features.
    """
    cols = [params.id_col, params.target, *params.sensitive_cols]
    cols += raw_feature_columns(params)
    for feature in params.features.all:
        cols += DERIVED_FEATURES.get(feature, [])
    return list(dict.fromkeys(cols))  # dedupe, keep first-seen order
