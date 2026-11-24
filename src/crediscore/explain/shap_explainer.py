"""SHAP explanations of the served model: top factors per row (T3, contract 2 §6).

`Explainer(model)` is built once (at API startup): it explains the
uncalibrated pipeline inside the served `CalibratedClassifierCV`
(`calibrated_classifiers_[0].estimator`): preprocessing, then the classifier.
- LogReg: `shap.LinearExplainer`. Reference: the training mean of each scaled
  numeric column and "no category" for one-hot columns, or the mean of a
  `background` frame if given.
- LightGBM, XGBoost: `shap.TreeExplainer`; RandomForest: the same with
  `feature_perturbation="tree_path_dependent"`.
Units (`unit`, the model-version tag `shap_unit`): log-odds of the uncalibrated
pipeline for LogReg, LightGBM and XGBoost; probability for RandomForest. The
contributions never add up to the served calibrated `proba`. A contribution
> 0 raises the default risk. One-hot contributions are summed back to their
original column.

The input is the model input: `params.features.all` after
`fix_days_employed` then `add_derived_features`, numerics as float64.

Usage:
    from crediscore.explain.shap_explainer import Explainer

    explainer = Explainer(model)              # once
    explainer.explain(X)[0]                   # 3 Factor(feature, value, contribution)

    # SHAP summary plot of the champion (bar chart of mean |contribution|),
    # logged as an MLflow run tagged kind=shap_summary if MLFLOW_TRACKING_URI is set
    poetry run python -m crediscore.explain.shap_explainer
        --model models:/crediscore@champion --output reports/figures/shap_summary.png

Defaults: --model models:/crediscore@champion, --data
data/processed/train.parquet, --rows 2000, --params params.yaml, --output
reports/figures/shap_summary.png.
"""

import argparse
import os
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from crediscore.config import load_params
from crediscore.models.evaluate import registry_version
from crediscore.models.train import EXPERIMENT, prepare_xy

TOP_K = 3


class Factor(NamedTuple):
    """One explained feature: original column, raw input value, contribution."""

    feature: str
    value: float | int | str | None
    contribution: float


def _raw(value: Any) -> float | int | str | None:
    """The input value as a plain Python value, None when missing."""
    if pd.isna(value):
        return None
    plain: float | int | str = value.item() if isinstance(value, np.generic) else value
    return plain


class Explainer:
    """SHAP explainer of a served model, built once.

    Attributes:
        unit: "log_odds" or "probability" (contract 2 section 6).
        columns: The original columns the model uses, in input order.
        base: The explainer's base value, in `unit`.
    """

    def __init__(self, model: Any, background: pd.DataFrame | None = None) -> None:
        """Build the SHAP explainer of `model`.

        Args:
            model: The served `CalibratedClassifierCV` (contract 2).
            background: LogReg only: rows whose mean is the reference.

        Raises:
            ValueError: If the classifier has no SHAP explainer here.
        """
        pipe = model.calibrated_classifiers_[0].estimator
        self._pre, est = pipe.named_steps["pre"], pipe.named_steps["model"]
        owners: list[str] = []
        for _, trans, cols in self._pre.transformers_:
            if trans == "drop":
                continue
            encoder = trans[-1]
            if hasattr(encoder, "categories_"):  # one-hot: one column per category
                owners += [
                    c for c, cats in zip(cols, encoder.categories_) for _ in cats
                ]
            else:
                owners += list(cols)
        self.columns = list(dict.fromkeys(owners))
        # (transformed column x original column) 0/1 matrix: sums one-hot back.
        self._group = (np.array(owners)[:, None] == np.array(self.columns)).astype(
            float
        )
        if isinstance(est, LogisticRegression):
            ref = (
                np.zeros((1, len(owners)))
                if background is None
                else self._transform(background)
            )
            masker = shap.maskers.Independent(ref, max_samples=len(ref))
            self._shap = shap.LinearExplainer(est, masker)
            self.unit = "log_odds"
        elif isinstance(est, LGBMClassifier | XGBClassifier):
            self._shap = shap.TreeExplainer(est)
            self.unit = "log_odds"
        elif isinstance(est, RandomForestClassifier):
            self._shap = shap.TreeExplainer(
                est, feature_perturbation="tree_path_dependent"
            )
            self.unit = "probability"
        else:
            raise ValueError(f"no SHAP explainer for {type(est).__name__}")
        # Base value from the model itself, on an all-zero row: shap 0.51
        # reports XGBoost 3's expected_value off by a constant (~0.04 log-odds).
        zero = np.zeros((1, len(owners)))
        p = float(est.predict_proba(zero)[0, 1])
        margin = p if self.unit == "probability" else float(np.log(p / (1 - p)))
        self.base = margin - float(self._values(zero).sum())

    def _transform(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        Xt = self._pre.transform(frame)
        return np.asarray(Xt.toarray() if hasattr(Xt, "toarray") else Xt, float)

    def contributions(self, frame: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Base value and the contribution of every original column, per row.

        Args:
            frame: Model input rows.

        Returns:
            (base, contributions): base + a row's sum is the pipeline's margin
            (log-odds) or probability, per `unit`.
        """
        values = self._values(self._transform(frame))
        return self.base, pd.DataFrame(
            values @ self._group, index=frame.index, columns=self.columns
        )

    def _values(self, Xt: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        values = np.asarray(self._shap.shap_values(Xt), dtype=float)
        return values[..., 1] if values.ndim == 3 else values  # RF: class 1

    def explain(self, frame: pd.DataFrame) -> list[list[Factor]]:
        """The `TOP_K` factors of each row, by decreasing |contribution|.

        Args:
            frame: Model input rows.

        Returns:
            One list of 3 `Factor` per row; `value` is the input value of the
            column (None if missing).
        """
        _, contrib = self.contributions(frame)
        raw = frame[self.columns].to_numpy(dtype=object)
        out = []
        for i, row in enumerate(contrib.to_numpy()):
            top = np.argsort(-np.abs(row), kind="stable")[:TOP_K]
            out.append(
                [Factor(self.columns[j], _raw(raw[i, j]), float(row[j])) for j in top]
            )
        return out


def mean_abs_contributions(explainer: Explainer, frame: pd.DataFrame) -> pd.Series:
    """Mean |contribution| per original column, largest first."""
    _, contrib = explainer.contributions(frame)
    return contrib.abs().mean().sort_values(ascending=False)


def summary_figure(importance: pd.Series, unit: str, top: int = 15) -> Figure:
    """Horizontal bar chart of the `top` columns, the largest on top."""
    shown = importance.iloc[:top][::-1]
    fig = Figure(figsize=(7, 0.35 * len(shown) + 1.2))
    ax = fig.subplots()
    ax.barh(list(shown.index), shown.to_numpy(), color="#4C72B0")
    ax.set_xlabel(f"mean |SHAP contribution| ({unit}, uncalibrated model)")
    ax.set_title("CrediScore: features driving the default risk")
    fig.tight_layout()
    return fig


def _load(ref: str) -> tuple[Any, dict[str, str]]:
    """Model and the tags linking it, from `models:/` or a `train` directory."""
    if ref.startswith("models:/"):
        version = registry_version(ref)
        tags = {"model_run_id": str(version.run_id), "model_version": version.version}
        return mlflow.sklearn.load_model(ref), {"model_uri": ref, **tags}
    return joblib.load(Path(ref) / "model.joblib"), {"model_uri": ref}


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--model", default="models:/crediscore@champion")
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/train.parquet")
    )
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/figures/shap_summary.png")
    )
    args = parser.parse_args(argv)
    params = load_params(args.params)
    model, tags = _load(args.model)
    df = pd.read_parquet(args.data)
    df = df.sample(min(args.rows, len(df)), random_state=params.data.random_state)
    X, _ = prepare_xy(df, params)
    explainer = Explainer(model)
    importance = mean_abs_contributions(explainer, X)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_figure(importance, explainer.unit).savefig(args.output, dpi=100)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="shap_summary"):
            mlflow.set_tags(
                {"kind": "shap_summary", "shap_unit": explainer.unit, **tags}
            )
            mlflow.log_params({"rows": len(X), "data": str(args.data)})
            mlflow.log_artifact(str(args.output))
    print(f"{args.output}: top features {list(importance.index[:5])}")


if __name__ == "__main__":
    main()
