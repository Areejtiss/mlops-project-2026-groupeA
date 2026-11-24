"""Tests for the promotion hold-out batch (R2, for Areej's review).

Synthetic tests always run; the checks on the real `holdout.parquet` need
`dvc repro -s holdout` (or `dvc pull`) and are skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crediscore.config import Params, load_params
from crediscore.data.featurize import featurize
from crediscore.data.holdout import holdout
from crediscore.data.schema import required_raw_columns

ROOT = Path(__file__).resolve().parents[1]
RATE_TOLERANCE = 0.005  # +/- 0.5 point, as for the splits (roadmap S4)


def raw_frame(params: Params, n: int = 3000) -> pd.DataFrame:
    """A raw-like frame: every required column, 1 in 12 defaults, sentinels."""
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        {col: rng.uniform(1, 10, size=n) for col in required_raw_columns(params)}
    )
    for col in params.features.categorical:
        raw[col] = rng.choice(["a", "b"], size=n)
    raw["CODE_GENDER"] = rng.choice(["F", "M"], size=n)
    raw[params.id_col] = np.arange(n)
    raw[params.target] = (np.arange(n) % 12 == 0).astype(int)
    raw["DAYS_BIRTH"] = -12_000.0
    raw["DAYS_EMPLOYED"] = np.where(np.arange(n) % 5 == 0, 365_243, -1_000.0)
    raw.loc[::7, "EXT_SOURCE_1"] = np.nan
    raw["NOISE"] = np.nan  # 100 % missing: dropped, never in the output
    return raw


def small(params: Params, **data: int) -> Params:
    return params.model_copy(update={"data": params.data.model_copy(update=data)})


@pytest.fixture
def synth(params: Params) -> tuple[pd.DataFrame, pd.DataFrame, Params]:
    p = small(params, sample_size=1000, holdout_size=600)
    raw = raw_frame(p)
    return raw, featurize(raw, p), p


def test_no_row_of_the_featurize_sample(
    synth: tuple[pd.DataFrame, pd.DataFrame, Params],
) -> None:
    raw, features, p = synth
    out = holdout(raw, features[p.id_col], p)
    assert not set(out[p.id_col]) & set(features[p.id_col])
    assert abs(len(out) - 600) <= 2  # one row per class of rounding


def test_stratified_default_rate(
    synth: tuple[pd.DataFrame, pd.DataFrame, Params],
) -> None:
    raw, features, p = synth
    out = holdout(raw, features[p.id_col], p)
    assert out[p.target].mean() == pytest.approx(
        raw[p.target].mean(), abs=RATE_TOLERANCE
    )


def test_same_columns_dtypes_and_processing_as_featurize(
    synth: tuple[pd.DataFrame, pd.DataFrame, Params],
) -> None:
    raw, features, p = synth
    out = holdout(raw, features[p.id_col], p)
    assert list(out.columns) == list(features.columns)
    assert out.dtypes.equals(features.dtypes)
    # featurize on exactly these raw rows (no sampling) gives the same frame.
    rows = raw[raw[p.id_col].isin(out[p.id_col])]
    expected = featurize(rows, small(p, sample_size=len(raw)))
    pd.testing.assert_frame_equal(out, expected)
    assert out["DAYS_EMPLOYED_ANOM"].sum() > 0  # the sentinel was handled


def test_deterministic_and_seeded(
    synth: tuple[pd.DataFrame, pd.DataFrame, Params],
) -> None:
    raw, features, p = synth
    first = holdout(raw, features[p.id_col], p)
    pd.testing.assert_frame_equal(first, holdout(raw, features[p.id_col], p))
    other = holdout(raw, features[p.id_col], small(p, random_state=1))
    assert set(other[p.id_col]) != set(first[p.id_col])


def test_too_few_unused_rows_is_refused(
    synth: tuple[pd.DataFrame, pd.DataFrame, Params],
) -> None:
    raw, features, p = synth
    with pytest.raises(ValueError, match="holdout_size"):
        holdout(raw, features[p.id_col], small(p, holdout_size=2500))


# --- the real file (dvc repro -s holdout) -------------------------------------

PARAMS = load_params(ROOT / "params.yaml")
PROCESSED = ROOT / PARAMS.data.processed_dir
REAL = {n: PROCESSED / f"{n}.parquet" for n in ("holdout", "features", "test")}
real = pytest.mark.skipif(
    not all(path.exists() for path in REAL.values()),
    reason="processed data not available (run `dvc repro -s holdout`)",
)


@pytest.fixture(scope="module")
def data() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(path) for name, path in REAL.items()}


@real
def test_real_holdout_is_disjoint_from_the_sample(
    data: dict[str, pd.DataFrame],
) -> None:
    ids = data["holdout"][PARAMS.id_col]
    assert ids.is_unique
    assert not set(ids) & set(data["features"][PARAMS.id_col])
    assert abs(len(ids) - PARAMS.data.holdout_size) <= 2


@real
def test_real_holdout_has_the_columns_and_dtypes_of_test(
    data: dict[str, pd.DataFrame],
) -> None:
    assert list(data["holdout"].columns) == list(data["test"].columns)
    assert data["holdout"].dtypes.equals(data["test"].dtypes)


@real
def test_real_holdout_default_rate(data: dict[str, pd.DataFrame]) -> None:
    rate = data["holdout"][PARAMS.target].mean()
    reference = data["features"][PARAMS.target].mean()
    assert rate == pytest.approx(reference, abs=RATE_TOLERANCE)


@real
@pytest.mark.slow
def test_real_holdout_is_reproduced_from_the_raw_file(
    data: dict[str, pd.DataFrame],
) -> None:
    raw = pd.read_csv(ROOT / PARAMS.data.raw_path)
    again = holdout(raw, data["features"][PARAMS.id_col], PARAMS)
    pd.testing.assert_frame_equal(again, data["holdout"])
