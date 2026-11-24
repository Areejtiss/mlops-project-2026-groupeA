"""Typed, validated access to params.yaml (contract 1).

Usage:
    from crediscore.config import load_params

    params = load_params()
    params.data.test_size          # 0.2
    params.features.numeric        # list of numeric feature names
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model: unknown keys are errors (catches typos like `test_sise`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    """Data locations, sampling and splitting parameters."""

    raw_path: Path
    processed_dir: Path
    sample_size: int = Field(gt=0)
    test_size: float = Field(gt=0, lt=1)
    val_size: float = Field(gt=0, lt=1)
    random_state: int = Field(ge=0)
    missing_threshold: float = Field(gt=0, lt=1)
    ci_sample_size: int = Field(gt=0)
    holdout_size: int = Field(gt=0)

    @model_validator(mode="after")
    def ci_sample_smaller_than_sample(self) -> Self:
        if self.ci_sample_size >= self.sample_size:
            raise ValueError("ci_sample_size must be smaller than sample_size")
        return self


class FeaturesConfig(StrictModel):
    """Model features: at least 25 numeric and 8 categorical, no overlap."""

    numeric: list[str] = Field(min_length=25)
    categorical: list[str] = Field(min_length=8)

    @model_validator(mode="after")
    def no_duplicates_or_overlap(self) -> Self:
        for name, cols in [
            ("numeric", self.numeric),
            ("categorical", self.categorical),
        ]:
            if len(cols) != len(set(cols)):
                raise ValueError(f"duplicate column in features.{name}")
        overlap = set(self.numeric) & set(self.categorical)
        if overlap:
            raise ValueError(f"columns both numeric and categorical: {sorted(overlap)}")
        return self

    @property
    def all(self) -> list[str]:
        """All model features, numeric first."""
        return self.numeric + self.categorical


class CostConfig(StrictModel):
    """Business cost of each error type."""

    cost_fn: float = Field(gt=0)
    cost_fp: float = Field(gt=0)
    cost_review: float = Field(gt=0)


class ThresholdConfig(StrictModel):
    """Constraints of the threshold search, plus an optional manual override.

    The thresholds are a training output that travels with the model version
    (D9). `low`/`high` stay None unless someone forces them by hand.
    """

    low: float | None = Field(default=None, ge=0, le=1)
    high: float | None = Field(default=None, ge=0, le=1)
    max_review_share: float = Field(ge=0, le=1)
    min_recall: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def low_not_above_high(self) -> Self:
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(
                "threshold.low must be <= threshold.high (equal = 2 zones)"
            )
        return self


class PromotionConfig(StrictModel):
    """Promotion gate (D5): gap tolerances in points, minimum batch sizes."""

    max_gender_gap_pts: float = Field(ge=0, le=100)
    age_gap_margin_pts: float = Field(ge=0, le=100)
    min_rows: int = Field(gt=0)
    min_group_rows: int = Field(gt=0)

    @model_validator(mode="after")
    def two_groups_fit_in_the_batch(self) -> Self:
        if self.min_rows < 2 * self.min_group_rows:
            raise ValueError("min_rows must be at least twice min_group_rows")
        return self


class EconomicCrisisConfig(StrictModel):
    """Intensity of the "economic crisis" drift scenario (multiplicative factors)."""

    income_factor: float = Field(gt=0)
    credit_factor: float = Field(gt=0)
    employment_factor: float = Field(gt=0)
    noise_std: float = Field(ge=0, lt=1)


class DriftConfig(StrictModel):
    """Drift simulation parameters."""

    seed: int = Field(ge=0)
    economic_crisis: EconomicCrisisConfig


class MonitoringConfig(StrictModel):
    """Drift detection: alert thresholds of the drift report (contract 4)."""

    alpha: float = Field(gt=0, lt=1)
    drift_share_alert: float = Field(gt=0, le=1)
    psi_alert: float = Field(gt=0)
    auc_drop_alert: float = Field(ge=0, lt=1)
    psi_bins: int = Field(ge=2)
    min_rows: int = Field(ge=2)
    min_labelled: int = Field(ge=2)
    key_features: list[str] = Field(min_length=1)


class Params(StrictModel):
    """The whole params.yaml file."""

    data: DataConfig
    id_col: str
    target: str
    sensitive_cols: list[str]
    features: FeaturesConfig
    cost: CostConfig
    threshold: ThresholdConfig
    promotion: PromotionConfig
    drift: DriftConfig
    monitoring: MonitoringConfig

    @model_validator(mode="after")
    def no_leak_into_features(self) -> Self:
        features = set(self.features.all)
        leaked = features & set(self.sensitive_cols)
        if leaked:
            raise ValueError(f"sensitive columns used as features: {sorted(leaked)}")
        for col in (self.target, self.id_col):
            if col in features:
                raise ValueError(f"{col} must not be a feature")
        unknown = [c for c in self.monitoring.key_features if c not in features]
        if unknown:
            raise ValueError(f"monitoring.key_features not model features: {unknown}")
        return self


def load_params(path: str | Path = "params.yaml") -> Params:
    """Read params.yaml and validate it.

    Args:
        path: Location of the YAML file.

    Returns:
        The validated, read-only parameters.

    Raises:
        pydantic.ValidationError: If a value is missing, has the wrong type or
            breaks a rule (e.g. test_size >= 1, CODE_GENDER used as a feature).
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Params.model_validate(raw)
