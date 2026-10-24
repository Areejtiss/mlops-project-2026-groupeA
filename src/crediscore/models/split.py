"""Carve a stratified validation set out of train (D1).

The validation part serves only threshold optimization and model selection;
calibration uses CV on the fit part (T2). The test split is never touched here.

Usage:
    from crediscore.models.split import split_validation

    fit, val = split_validation(train, target="TARGET", val_size=0.2, seed=42)
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_validation(
    df: pd.DataFrame, *, target: str, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into (fit, validation), stratified on `target`.

    Args:
        df: The training rows.
        target: Name of the binary label column used for stratification.
        val_size: Share of rows put in validation, in (0, 1).
        seed: Random seed (`params.data.random_state`).

    Returns:
        The fit and validation frames, original index kept.

    Raises:
        KeyError: If `target` is not a column of `df`.
    """
    fit, val = train_test_split(
        df, test_size=val_size, stratify=df[target], random_state=seed
    )
    return fit, val
