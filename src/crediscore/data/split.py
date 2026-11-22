"""DVC stage 3 - stratified train/test split and the CI fixture.

Usage:
    poetry run python -m crediscore.data.split

Outputs:
    data/processed/train.parquet, data/processed/test.parquet  (DVC, MinIO)
    tests/fixtures/ci_sample.parquet  (small, committed to Git: GitHub CI and
    CML cannot reach MinIO, so they train and test on this file only)
"""

from pathlib import Path

import pandas as pd

from crediscore.config import Params, load_params
from crediscore.data.featurize import FEATURES_FILE
from crediscore.data.sampling import stratified_sample, stratified_split

TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"
CI_SAMPLE_PATH = Path("tests/fixtures/ci_sample.parquet")
# check-added-large-files (pre-commit) rejects files above 1 MB.
CI_SAMPLE_MAX_BYTES = 1_000_000


def split(
    features: pd.DataFrame, params: Params
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, ci_sample), all stratified on the target."""
    train, test = stratified_split(
        features, params.target, params.data.test_size, params.data.random_state
    )
    ci_sample = stratified_sample(
        features,
        params.target,
        params.data.ci_sample_size,
        params.data.random_state,
    )
    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        ci_sample.reset_index(drop=True),
    )


def write_ci_sample(ci_sample: pd.DataFrame, path: Path = CI_SAMPLE_PATH) -> int:
    """Write the CI fixture and return its size in bytes.

    Raises:
        ValueError: If the file is larger than 1 MB (it could not be committed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ci_sample.to_parquet(path, index=False)
    size = path.stat().st_size
    if size > CI_SAMPLE_MAX_BYTES:
        raise ValueError(
            f"{path} is {size:,} bytes, above the 1 MB limit. "
            "Lower data.ci_sample_size in params.yaml."
        )
    return size


def main() -> None:
    """Command-line entry point (DVC stage `split`)."""
    params = load_params()
    out_dir = Path(params.data.processed_dir)
    features = pd.read_parquet(out_dir / FEATURES_FILE)
    train, test, ci_sample = split(features, params)

    train.to_parquet(out_dir / TRAIN_FILE, index=False)
    test.to_parquet(out_dir / TEST_FILE, index=False)
    size = write_ci_sample(ci_sample)

    target = params.target
    for name, df in [("train", train), ("test", test), ("ci_sample", ci_sample)]:
        print(f"{name:>9}: {len(df):>6,} rows, default rate {df[target].mean():.4f}")
    print(f"CI fixture: {CI_SAMPLE_PATH} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
