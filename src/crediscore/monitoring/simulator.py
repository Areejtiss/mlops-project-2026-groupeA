"""Drift simulation: turn a clean batch into a "drifted" one.

Used to test the drift detection module (monitoring/drift.py) and, later, by the
monitoring flow to feed the API with realistic shifted traffic.

Usage:
    poetry run python -m crediscore.monitoring.simulator
    -> reads data/processed/test.parquet,
       writes data/simulated/economic_crisis.parquet

In code:
    sim = DriftSimulator(params.drift)
    drifted = sim.economic_crisis(batch)          # seed from params.yaml
    drifted = sim.economic_crisis(batch, seed=3)  # another reproducible draw

Scenarios work on featurized data (the output of the featurize stage). They
never modify the input DataFrame.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from crediscore.config import DriftConfig, load_params
from crediscore.data.featurize import safe_divide
from crediscore.data.split import TEST_FILE

OUTPUT_DIR = Path("data/simulated")
ECONOMIC_CRISIS_FILE = "economic_crisis.parquet"

INCOME_COLS = ["AMT_INCOME_TOTAL"]
# Bigger loans for the same duration: credit and annuity grow together.
CREDIT_COLS = ["AMT_CREDIT", "AMT_ANNUITY"]
EMPLOYMENT_COLS = ["EMPLOYED_YEARS"]
# Derived from the columns above: recomputed so the batch stays consistent.
RATIO_COLS = ["ANNUITY_INCOME_RATIO", "CREDIT_INCOME_RATIO", "CREDIT_TERM"]
ECONOMIC_CRISIS_COLS = INCOME_COLS + CREDIT_COLS + EMPLOYMENT_COLS + RATIO_COLS


class DriftSimulator:
    """Generate drifted copies of a batch, one method per scenario."""

    def __init__(self, config: DriftConfig) -> None:
        self.config = config

    def _rng(self, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(self.config.seed if seed is None else seed)

    @staticmethod
    def _shift(
        values: pd.Series, factor: float, noise_std: float, rng: np.random.Generator
    ) -> pd.Series:
        """values * factor * (1 + N(0, noise_std)), never negative, NaN kept."""
        noise = 1 + rng.normal(0.0, noise_std, size=len(values))
        return (values * factor * noise).clip(lower=0)

    def economic_crisis(
        self, df: pd.DataFrame, seed: int | None = None
    ) -> pd.DataFrame:
        """Economic crisis: lower incomes, bigger loans, shorter employment.

        Income is multiplied by ``income_factor``, credit and annuity by
        ``credit_factor``, employment seniority by ``employment_factor``, each
        with its own gaussian noise. The income/credit ratios are recomputed.
        Every other column is left exactly as it was.

        Args:
            df: Featurized batch (must contain ``ECONOMIC_CRISIS_COLS``).
            seed: Random seed; defaults to ``drift.seed`` from params.yaml.

        Returns:
            A drifted copy of ``df`` (same rows, columns and index).

        Raises:
            ValueError: If a required column is missing.
        """
        missing = [c for c in ECONOMIC_CRISIS_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"economic_crisis needs columns {missing}")

        cfg = self.config.economic_crisis
        rng = self._rng(seed)
        out = df.copy()
        for cols, factor in [
            (INCOME_COLS, cfg.income_factor),
            (CREDIT_COLS, cfg.credit_factor),
            (EMPLOYMENT_COLS, cfg.employment_factor),
        ]:
            for col in cols:
                out[col] = self._shift(out[col], factor, cfg.noise_std, rng)

        out["ANNUITY_INCOME_RATIO"] = safe_divide(
            out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"]
        )
        out["CREDIT_INCOME_RATIO"] = safe_divide(
            out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"]
        )
        out["CREDIT_TERM"] = safe_divide(out["AMT_ANNUITY"], out["AMT_CREDIT"])
        return out


def main() -> None:
    """Write the economic crisis version of the test set."""
    params = load_params()
    reference = pd.read_parquet(Path(params.data.processed_dir) / TEST_FILE)
    drifted = DriftSimulator(params.drift).economic_crisis(reference)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / ECONOMIC_CRISIS_FILE
    drifted.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}: {len(drifted):,} rows")
    for col in INCOME_COLS + CREDIT_COLS + EMPLOYMENT_COLS:
        before, after = reference[col].mean(), drifted[col].mean()
        print(f"  {col:<20} mean {before:>12,.2f} -> {after:>12,.2f}")


if __name__ == "__main__":
    main()
