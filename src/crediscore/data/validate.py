"""DVC stage 1 - validate the raw Home Credit file before anything uses it.

Usage:
    poetry run python -m crediscore.data.validate

Writes reports/validation.json (one entry per check) and exits with code 1 if
any check fails, so `dvc repro` and the Prefect flow stop right here instead of
training on bad data.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from crediscore.config import Params, load_params
from crediscore.data.download import EXPECTED_COLS, EXPECTED_ROWS
from crediscore.data.schema import required_raw_columns

REPORT_PATH = Path("reports/validation.json")

# The real default rate is ~8.07%. A value outside this range means the file
# is not the one we expect (wrong file, truncated, labels shuffled...).
TARGET_RATE_MIN = 0.07
TARGET_RATE_MAX = 0.09


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate(
    df: pd.DataFrame,
    params: Params,
    expected_shape: tuple[int, int] = (EXPECTED_ROWS, EXPECTED_COLS),
) -> dict[str, Any]:
    """Run every check on ``df`` and return the report (nothing is raised).

    Args:
        df: The raw application data.
        params: Validated params.yaml.
        expected_shape: Expected (rows, columns). Tests pass a small shape.

    Returns:
        A JSON-serialisable dict: ``passed`` (all checks ok), a few measured
        values, and the list of individual checks.
    """
    checks: list[dict[str, Any]] = []
    rows, cols = df.shape

    checks.append(
        _check(
            "shape",
            (rows, cols) == expected_shape,
            f"got {rows} x {cols}, expected {expected_shape[0]} x {expected_shape[1]}",
        )
    )

    missing = [c for c in required_raw_columns(params) if c not in df.columns]
    checks.append(
        _check(
            "required_columns",
            not missing,
            f"missing: {missing}" if missing else "all present",
        )
    )

    target_rate: float | None = None
    if params.target in df.columns:
        target = df[params.target]
        values = sorted(target.dropna().unique().tolist())
        is_binary = bool(target.notna().all()) and set(values) <= {0, 1}
        checks.append(_check("target_binary", is_binary, f"values: {values}"))
        if is_binary and rows > 0:
            target_rate = float(target.mean())
            checks.append(
                _check(
                    "target_rate",
                    TARGET_RATE_MIN <= target_rate <= TARGET_RATE_MAX,
                    f"{target_rate:.4f}, expected in "
                    f"[{TARGET_RATE_MIN}, {TARGET_RATE_MAX}]",
                )
            )

    if params.id_col in df.columns:
        n_dup = int(df[params.id_col].duplicated().sum())
        checks.append(_check("id_unique", n_dup == 0, f"{n_dup} duplicated ids"))

    return {
        "passed": all(c["passed"] for c in checks),
        "n_rows": rows,
        "n_cols": cols,
        "target_rate": target_rate,
        "checks": checks,
    }


def write_report(report: dict[str, Any], path: Path = REPORT_PATH) -> None:
    """Write the report as indented JSON (creates the folder if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Command-line entry point (DVC stage `validate`)."""
    params = load_params()
    df = pd.read_csv(params.data.raw_path)
    report = validate(df, params)
    write_report(report)

    for check in report["checks"]:
        status = "OK  " if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: {check['detail']}")
    if not report["passed"]:
        print(f"Validation failed, see {REPORT_PATH}", file=sys.stderr)
        sys.exit(1)
    print(f"Validation passed, report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
