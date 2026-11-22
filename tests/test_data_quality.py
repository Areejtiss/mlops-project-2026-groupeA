"""Data quality checks on the REAL pipeline outputs (roadmap S4).

These tests read data/processed/*.parquet, produced by `dvc repro` or fetched by
`dvc pull`. GitHub CI cannot reach MinIO, so they are skipped there; run them
locally after `dvc pull`. The same rules are tested on synthetic data in
test_split.py, which always runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crediscore.config import Params, load_params
from crediscore.data.featurize import output_columns

ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(ROOT / "params.yaml")
PROCESSED = ROOT / PARAMS.data.processed_dir
FILES = {name: PROCESSED / f"{name}.parquet" for name in ("features", "train", "test")}

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in FILES.values()),
    reason="processed data not available (run `dvc pull` or `dvc repro`)",
)
# Roadmap S4: default rate identical within +/- 0.5 point in every split.
RATE_TOLERANCE = 0.005


@pytest.fixture(scope="module")
def data() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(path) for name, path in FILES.items()}


@pytest.fixture
def params() -> Params:
    return PARAMS


def test_default_rate_identical_in_each_split(
    data: dict[str, pd.DataFrame], params: Params
) -> None:
    reference = data["features"][params.target].mean()
    for name in ("train", "test"):
        rate = data[name][params.target].mean()
        assert rate == pytest.approx(reference, abs=RATE_TOLERANCE), name


def test_no_client_in_both_train_and_test(
    data: dict[str, pd.DataFrame], params: Params
) -> None:
    train_ids = set(data["train"][params.id_col])
    test_ids = set(data["test"][params.id_col])
    assert not train_ids & test_ids


def test_split_covers_the_sample_exactly(
    data: dict[str, pd.DataFrame], params: Params
) -> None:
    assert len(data["train"]) + len(data["test"]) == len(data["features"])
    assert len(data["features"]) == pytest.approx(params.data.sample_size, abs=2)
    assert len(data["test"]) / len(data["features"]) == pytest.approx(
        params.data.test_size, abs=0.001
    )


def test_code_gender_kept_but_not_a_feature(
    data: dict[str, pd.DataFrame], params: Params
) -> None:
    assert "CODE_GENDER" not in params.features.all
    for df in data.values():
        assert "CODE_GENDER" in df.columns  # needed by the fairness audit


def test_columns_follow_params(data: dict[str, pd.DataFrame], params: Params) -> None:
    for df in data.values():
        assert list(df.columns) == output_columns(params)


def test_no_infinite_values(data: dict[str, pd.DataFrame]) -> None:
    numeric = data["features"].select_dtypes("number")
    assert not np.isinf(numeric.to_numpy()).any()


def test_derived_features_in_plausible_ranges(data: dict[str, pd.DataFrame]) -> None:
    df = data["features"]
    assert df["AGE_YEARS"].between(18, 75).all()
    assert df["EMPLOYED_YEARS"].dropna().between(0, 60).all()
    assert set(df["EXT_SOURCE_MISSING_COUNT"].unique()) <= {0, 1, 2, 3}
    for flag in ("DAYS_EMPLOYED_ANOM", "IS_THIN_FILE"):
        assert set(df[flag].unique()) <= {0, 1}


def test_anomaly_flag_matches_missing_employment(
    data: dict[str, pd.DataFrame],
) -> None:
    df = data["features"]
    flagged = df["DAYS_EMPLOYED_ANOM"] == 1
    assert df.loc[flagged, "EMPLOYED_YEARS"].isna().all()
    assert df.loc[~flagged, "EMPLOYED_YEARS"].notna().all()
