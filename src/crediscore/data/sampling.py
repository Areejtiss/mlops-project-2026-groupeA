"""Stratified sampling helpers (pure pandas, deterministic with a seed).

Stratifying on the target keeps the ~8% default rate identical in every sample
and split, which a plain random sample does not guarantee.
"""

import pandas as pd


def stratified_sample(
    df: pd.DataFrame, target: str, n: int, random_state: int
) -> pd.DataFrame:
    """Draw about ``n`` rows keeping the class proportions of ``target``.

    Each class is sampled with the same fraction, so the total may differ from
    ``n`` by one row per class because of rounding. If ``n`` is larger than
    ``df``, the whole DataFrame is returned.

    The result keeps the original index and order, so it is reproducible.
    """
    if n >= len(df):
        return df.copy()
    frac = n / len(df)
    sample = df.groupby(target, group_keys=False).sample(
        frac=frac, random_state=random_state
    )
    return sample.sort_index()


def stratified_split(
    df: pd.DataFrame, target: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, test), stratified on ``target``.

    The test set is drawn per class, the train set is everything else, so a row
    can never be in both.
    """
    test = df.groupby(target, group_keys=False).sample(
        frac=test_size, random_state=random_state
    )
    train = df.drop(index=test.index)
    return train.sort_index(), test.sort_index()
