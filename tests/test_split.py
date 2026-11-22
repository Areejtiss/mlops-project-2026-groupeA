"""Tests for the split stage and the sampling helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crediscore.config import Params, load_params
from crediscore.data.sampling import stratified_sample, stratified_split
from crediscore.data.split import split, write_ci_sample

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"
# Roadmap S4: default rate identical within +/- 0.5 point in every split.
RATE_TOLERANCE = 0.005


@pytest.fixture
def params() -> Params:
    return load_params(PARAMS_FILE)


@pytest.fixture
def data(params: Params) -> pd.DataFrame:
    """10,000 rows with an 8% default rate, as in the real data."""
    n = 10_000
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            params.id_col: np.arange(100_000, 100_000 + n),
            params.target: (rng.random(n) < 0.08).astype(int),
            "x": rng.normal(size=n),
        }
    )


def small_params(params: Params, ci_sample_size: int = 500) -> Params:
    data = params.data.model_copy(update={"ci_sample_size": ci_sample_size})
    return params.model_copy(update={"data": data})


def test_split_sizes(params: Params, data: pd.DataFrame) -> None:
    train, test = stratified_split(data, params.target, 0.2, random_state=42)
    assert len(test) == pytest.approx(2000, abs=2)
    assert len(train) + len(test) == len(data)


def test_split_keeps_default_rate(params: Params, data: pd.DataFrame) -> None:
    train, test, ci = split(data, small_params(params))
    rate = data[params.target].mean()
    for part in (train, test, ci):
        assert part[params.target].mean() == pytest.approx(rate, abs=RATE_TOLERANCE)


def test_no_client_in_both_train_and_test(params: Params, data: pd.DataFrame) -> None:
    train, test, _ = split(data, small_params(params))
    assert not set(train[params.id_col]) & set(test[params.id_col])


def test_split_is_deterministic(params: Params, data: pd.DataFrame) -> None:
    first = split(data, small_params(params))
    second = split(data, small_params(params))
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_split(
    params: Params, data: pd.DataFrame
) -> None:
    _, test_a = stratified_split(data, params.target, 0.2, random_state=1)
    _, test_b = stratified_split(data, params.target, 0.2, random_state=2)
    assert set(test_a[params.id_col]) != set(test_b[params.id_col])


def test_sample_size(params: Params, data: pd.DataFrame) -> None:
    sample = stratified_sample(data, params.target, 1000, random_state=42)
    assert len(sample) == pytest.approx(1000, abs=2)


def test_sample_larger_than_data_returns_all(
    params: Params, data: pd.DataFrame
) -> None:
    sample = stratified_sample(data, params.target, 50_000, random_state=42)
    assert len(sample) == len(data)


def test_ci_sample_is_written(
    tmp_path: Path, params: Params, data: pd.DataFrame
) -> None:
    _, _, ci = split(data, small_params(params))
    path = tmp_path / "fixtures" / "ci_sample.parquet"
    size = write_ci_sample(ci, path)
    assert 0 < size < 1_000_000
    pd.testing.assert_frame_equal(pd.read_parquet(path), ci)


def test_ci_sample_too_large_is_refused(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    big = pd.DataFrame(rng.random((50_000, 5)), columns=list("abcde"))
    with pytest.raises(ValueError, match="1 MB"):
        write_ci_sample(big, tmp_path / "big.parquet")
