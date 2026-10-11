"""Tests for params.yaml loading and validation (contract 1)."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from crediscore.config import load_params

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"


def write_params(tmp_path: Path, changes: dict[str, Any]) -> Path:
    """Copy the real params.yaml with some top-level sections replaced."""
    params = yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))
    for section, value in changes.items():
        if isinstance(value, dict) and isinstance(params.get(section), dict):
            params[section].update(value)
        else:
            params[section] = value
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return path


def test_real_params_file_is_valid() -> None:
    params = load_params(PARAMS_FILE)
    assert params.target == "TARGET"
    assert params.data.sample_size == 100_000
    assert len(params.features.numeric) >= 25
    assert len(params.features.categorical) >= 8


def test_code_gender_is_not_a_feature() -> None:
    params = load_params(PARAMS_FILE)
    assert "CODE_GENDER" in params.sensitive_cols
    assert "CODE_GENDER" not in params.features.all


@pytest.mark.parametrize("test_size", [0, 1, 1.5, -0.2])
def test_invalid_test_size_is_rejected(tmp_path: Path, test_size: float) -> None:
    path = write_params(tmp_path, {"data": {"test_size": test_size}})
    with pytest.raises(ValidationError, match="test_size"):
        load_params(path)


def test_sensitive_column_as_feature_is_rejected(tmp_path: Path) -> None:
    params = yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))
    categorical = params["features"]["categorical"] + ["CODE_GENDER"]
    path = write_params(tmp_path, {"features": {"categorical": categorical}})
    with pytest.raises(ValidationError, match="sensitive columns"):
        load_params(path)


def test_target_as_feature_is_rejected(tmp_path: Path) -> None:
    params = yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))
    numeric = params["features"]["numeric"] + ["TARGET"]
    path = write_params(tmp_path, {"features": {"numeric": numeric}})
    with pytest.raises(ValidationError, match="TARGET must not be a feature"):
        load_params(path)


def test_overlap_numeric_categorical_is_rejected(tmp_path: Path) -> None:
    params = yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))
    categorical = params["features"]["categorical"] + ["AMT_CREDIT"]
    path = write_params(tmp_path, {"features": {"categorical": categorical}})
    with pytest.raises(ValidationError, match="both numeric and categorical"):
        load_params(path)


def test_duplicate_feature_is_rejected(tmp_path: Path) -> None:
    params = yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))
    numeric = params["features"]["numeric"] + ["AMT_CREDIT"]
    path = write_params(tmp_path, {"features": {"numeric": numeric}})
    with pytest.raises(ValidationError, match="duplicate column"):
        load_params(path)


def test_too_few_categorical_features_is_rejected(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"features": {"categorical": ["NAME_CONTRACT_TYPE"]}})
    with pytest.raises(ValidationError, match="categorical"):
        load_params(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"data": {"test_sise": 0.2}})
    with pytest.raises(ValidationError, match="test_sise"):
        load_params(path)


def test_thresholds_must_be_ordered(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"threshold": {"low": 0.6, "high": 0.3}})
    with pytest.raises(ValidationError, match="low must be smaller"):
        load_params(path)


def test_ci_sample_must_be_smaller_than_sample(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"data": {"ci_sample_size": 200_000}})
    with pytest.raises(ValidationError, match="ci_sample_size"):
        load_params(path)


def test_params_are_read_only() -> None:
    params = load_params(PARAMS_FILE)
    with pytest.raises(ValidationError):
        params.data.test_size = 0.5  # type: ignore[misc]
