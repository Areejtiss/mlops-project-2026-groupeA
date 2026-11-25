"""DVC stage `holdout` - a fresh labelled batch for promotion (for Areej's review).

Usage:
    poetry run python -m crediscore.data.holdout

`featurize` samples `data.sample_size` of the raw rows; the others are never
seen by train, evaluate or the drift simulator. This stage draws
`data.holdout_size` of them (stratified on the target, seeded with
`data.random_state`) and applies exactly the featurize processing (sparse
column guard, DAYS_EMPLOYED sentinel, derived features, same output columns),
so `promote --data data/processed/holdout.parquet` scores both models on rows
no model selection has read.

Output: data/processed/holdout.parquet
"""

from pathlib import Path

import pandas as pd

from crediscore.config import Params, load_params
from crediscore.data.featurize import (
    FEATURES_FILE,
    add_derived_features,
    drop_sparse_columns,
    fix_days_employed,
    output_columns,
)
from crediscore.data.sampling import stratified_sample
from crediscore.data.schema import required_raw_columns

HOLDOUT_FILE = "holdout.parquet"


def holdout(raw: pd.DataFrame, used_ids: pd.Series, params: Params) -> pd.DataFrame:
    """Draw and featurize the hold-out batch.

    Args:
        raw: The raw application file.
        used_ids: `id_col` of every row of features.parquet (left out).
        params: Validated parameters.

    Returns:
        About `data.holdout_size` rows with the features.parquet columns.

    Raises:
        ValueError: If fewer unused rows than `data.holdout_size` remain.
    """
    rest = raw[~raw[params.id_col].isin(used_ids)]
    size = params.data.holdout_size
    if len(rest) < size:
        raise ValueError(
            f"only {len(rest)} raw rows outside the featurize sample, "
            f"data.holdout_size is {size}"
        )
    batch = stratified_sample(rest, params.target, size, params.data.random_state)
    batch, _ = drop_sparse_columns(
        batch, params.data.missing_threshold, keep=required_raw_columns(params)
    )
    batch = add_derived_features(fix_days_employed(batch))
    return batch[output_columns(params)].reset_index(drop=True)


def main() -> None:
    """Command-line entry point (DVC stage `holdout`)."""
    params = load_params()
    out_dir = Path(params.data.processed_dir)
    used = pd.read_parquet(out_dir / FEATURES_FILE, columns=[params.id_col])
    batch = holdout(pd.read_csv(params.data.raw_path), used[params.id_col], params)
    batch.to_parquet(out_dir / HOLDOUT_FILE, index=False)
    print(
        f"Wrote {out_dir / HOLDOUT_FILE}: {len(batch):,} rows, "
        f"default rate {batch[params.target].mean():.4f}, "
        f"thin-file rate {batch['IS_THIN_FILE'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
