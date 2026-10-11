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
    random_state: int = Field(ge=0)
    missing_threshold: float = Field(gt=0, lt=1)
    ci_sample_size: int = Field(gt=0)

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


class ThresholdConfig(StrictModel):
    """Probability thresholds of the 3 decision zones (None until tuned)."""

    low: float | None = Field(default=None, ge=0, le=1)
    high: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def low_below_high(self) -> Self:
        if self.low is not None and self.high is not None and self.low >= self.high:
            raise ValueError("threshold.low must be smaller than threshold.high")
        return self


class Params(StrictModel):
    """The whole params.yaml file."""

    data: DataConfig
    id_col: str
    target: str
    sensitive_cols: list[str]
    features: FeaturesConfig
    cost: CostConfig
    threshold: ThresholdConfig

    @model_validator(mode="after")
    def no_leak_into_features(self) -> Self:
        features = set(self.features.all)
        leaked = features & set(self.sensitive_cols)
        if leaked:
            raise ValueError(f"sensitive columns used as features: {sorted(leaked)}")
        for col in (self.target, self.id_col):
            if col in features:
                raise ValueError(f"{col} must not be a feature")
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
