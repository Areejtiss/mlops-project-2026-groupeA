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
    with pytest.raises(ValidationError, match="low must be <= "):
        load_params(path)


def test_equal_thresholds_are_allowed(tmp_path: Path) -> None:
    # low == high = 2 zones, no review (contract 2 section 5).
    path = write_params(tmp_path, {"threshold": {"low": 0.2, "high": 0.2}})
    params = load_params(path)
    assert params.threshold.low == params.threshold.high == 0.2


def test_ci_sample_must_be_smaller_than_sample(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"data": {"ci_sample_size": 200_000}})
    with pytest.raises(ValidationError, match="ci_sample_size"):
        load_params(path)


def test_holdout_size_is_loaded() -> None:
    assert load_params(PARAMS_FILE).data.holdout_size == 20_000


@pytest.mark.parametrize("value", [0, -1])
def test_holdout_size_must_be_positive(tmp_path: Path, value: int) -> None:
    path = write_params(tmp_path, {"data": {"holdout_size": value}})
    with pytest.raises(ValidationError, match=r"holdout_size\n.*greater than"):
        load_params(path)


def test_lot2_keys_are_loaded() -> None:
    params = load_params(PARAMS_FILE)
    assert params.data.val_size == 0.2
    assert params.cost.cost_review == 0.3
    assert params.threshold.max_review_share == 0.15
    assert params.threshold.min_recall == 0.65


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("data", "val_size", 0),
        ("data", "val_size", 1),
        ("cost", "cost_review", 0),
        ("cost", "cost_review", -0.3),
        ("threshold", "max_review_share", -0.1),
        ("threshold", "max_review_share", 1.5),
        ("threshold", "min_recall", -0.1),
        ("threshold", "min_recall", 1.5),
    ],
)
def test_invalid_lot2_key_is_rejected(
    tmp_path: Path, section: str, key: str, value: float
) -> None:
    path = write_params(tmp_path, {section: {key: value}})
    # Match the bound message, not just the key (an unknown key also names it).
    with pytest.raises(ValidationError, match=rf"{key}\n.*(greater|less) than"):
        load_params(path)


def test_params_are_read_only() -> None:
    params = load_params(PARAMS_FILE)
    with pytest.raises(ValidationError):
        params.data.test_size = 0.5  # type: ignore[misc]


def test_drift_params_are_loaded() -> None:
    crisis = load_params(PARAMS_FILE).drift.economic_crisis
    assert crisis.income_factor < 1 < crisis.credit_factor
    assert crisis.employment_factor < 1


def test_negative_drift_factor_is_rejected(tmp_path: Path) -> None:
    crisis = {
        "income_factor": -0.8,
        "credit_factor": 1.25,
        "employment_factor": 0.7,
        "noise_std": 0.05,
    }
    path = write_params(tmp_path, {"drift": {"economic_crisis": crisis}})
    with pytest.raises(ValidationError, match="income_factor"):
        load_params(path)


def test_promotion_keys_are_loaded() -> None:
    promotion = load_params(PARAMS_FILE).promotion
    assert promotion.max_gender_gap_pts == 10
    assert promotion.age_gap_margin_pts == 2


@pytest.mark.parametrize("key", ["max_gender_gap_pts", "age_gap_margin_pts"])
@pytest.mark.parametrize("value", [-0.1, 100.1])
def test_promotion_gap_outside_0_100_points_is_rejected(
    tmp_path: Path, key: str, value: float
) -> None:
    path = write_params(tmp_path, {"promotion": {key: value}})
    with pytest.raises(ValidationError, match=rf"{key}\n.*(greater|less) than"):
        load_params(path)


@pytest.mark.parametrize("value", [0, 100])
def test_promotion_gap_bounds_are_inclusive(tmp_path: Path, value: float) -> None:
    path = write_params(
        tmp_path,
        {"promotion": {"max_gender_gap_pts": value, "age_gap_margin_pts": value}},
    )
    promotion = load_params(path).promotion
    assert promotion.max_gender_gap_pts == promotion.age_gap_margin_pts == value


def test_promotion_batch_size_keys_are_loaded() -> None:
    promotion = load_params(PARAMS_FILE).promotion
    assert promotion.min_rows == 5000
    assert promotion.min_group_rows == 500


@pytest.mark.parametrize("key", ["min_rows", "min_group_rows"])
def test_promotion_row_minimum_must_be_positive(tmp_path: Path, key: str) -> None:
    path = write_params(tmp_path, {"promotion": {key: 0}})
    with pytest.raises(ValidationError, match=rf"{key}\n.*greater than"):
        load_params(path)


def test_two_gender_groups_must_fit_in_the_minimum_batch(tmp_path: Path) -> None:
    path = write_params(
        tmp_path, {"promotion": {"min_rows": 999, "min_group_rows": 500}}
    )
    with pytest.raises(ValidationError, match="at least twice min_group_rows"):
        load_params(path)
    ok = write_params(
        tmp_path, {"promotion": {"min_rows": 1000, "min_group_rows": 500}}
    )
    assert load_params(ok).promotion.min_rows == 1000


def test_monitoring_params_are_loaded() -> None:
    monitoring = load_params(PARAMS_FILE).monitoring
    assert monitoring.drift_share_alert == 0.30
    assert monitoring.psi_alert == 0.20
    assert monitoring.auc_drop_alert == 0.03
    assert monitoring.alpha == 0.05
    assert monitoring.psi_bins == 10
    assert monitoring.min_rows == 500
    assert monitoring.min_labelled == 1000
    assert monitoring.min_positives == 50
    assert monitoring.key_features == [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "ANNUITY_INCOME_RATIO",
        "EXT_SOURCE_2",
    ]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("drift_share_alert", 0),
        ("drift_share_alert", 1.1),
        ("psi_alert", 0),
        ("auc_drop_alert", -0.01),
        ("auc_drop_alert", 1),
        ("alpha", 0),
        ("alpha", 1),
        ("psi_bins", 1),
        ("min_rows", 1),
        ("min_labelled", 1),
        ("min_positives", 0),
    ],
)
def test_invalid_monitoring_key_is_rejected(
    tmp_path: Path, key: str, value: float
) -> None:
    path = write_params(tmp_path, {"monitoring": {key: value}})
    with pytest.raises(ValidationError, match=rf"{key}\n.*(greater|less) than"):
        load_params(path)


def test_monitoring_key_feature_must_be_a_feature(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"monitoring": {"key_features": ["CODE_GENDER"]}})
    with pytest.raises(ValidationError, match=r"not model features: \['CODE_GENDER'\]"):
        load_params(path)


def test_monitoring_needs_a_key_feature(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"monitoring": {"key_features": []}})
    with pytest.raises(ValidationError, match=r"key_features\n.*at least 1 item"):
        load_params(path)
