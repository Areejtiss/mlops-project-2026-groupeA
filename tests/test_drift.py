"""Tests for the drift tests, drift_share, auc_drop and the report (contract 4)."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from crediscore.config import MonitoringConfig, Params, load_params
from crediscore.models.train import fit_model, prepare_xy
from crediscore.monitoring.drift import (
    REPORT_KEYS,
    auc_drop,
    chi2_test,
    compute_drift_report,
    drift_share,
    is_alert,
    ks_test,
    main,
)
from conftest import PARAMS_FILE, make_synthetic_frame

RNG_SEED = 42


@pytest.fixture
def monitoring() -> MonitoringConfig:
    return MonitoringConfig(
        alpha=0.05,
        drift_share_alert=0.30,
        psi_alert=0.20,
        auc_drop_alert=0.03,
        psi_bins=10,
        key_features=["AMT_INCOME_TOTAL"],
    )


# --- KS -------------------------------------------------------------------


def test_ks_detects_a_shifted_normal_and_not_the_same_one() -> None:
    rng = np.random.default_rng(RNG_SEED)
    ref = pd.Series(rng.normal(size=2000))
    assert ks_test(ref, pd.Series(rng.normal(loc=0.3, size=2000))) < 0.05
    assert ks_test(ref, pd.Series(rng.normal(size=2000))) > 0.05


def test_ks_ignores_missing_values() -> None:
    rng = np.random.default_rng(RNG_SEED)
    ref = pd.Series(rng.normal(size=500))
    cur = pd.Series(rng.normal(size=500))
    with_nan = pd.concat([cur, pd.Series([np.nan] * 300)], ignore_index=True)
    assert ks_test(ref, with_nan) == pytest.approx(ks_test(ref, cur))


@pytest.mark.parametrize("side", ["reference", "current"])
def test_ks_on_an_all_nan_column_is_none(side: str) -> None:
    values = pd.Series([1.0, 2.0, 3.0])
    nan = pd.Series([np.nan] * 3)
    ref, cur = (nan, values) if side == "reference" else (values, nan)
    assert ks_test(ref, cur) is None


# --- chi2 -----------------------------------------------------------------


def test_chi2_detects_shifted_proportions_and_not_the_same_ones() -> None:
    rng = np.random.default_rng(RNG_SEED)
    ref = pd.Series(rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2]))
    same = pd.Series(rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2]))
    moved = pd.Series(rng.choice(["a", "b", "c"], size=2000, p=[0.3, 0.3, 0.4]))
    assert chi2_test(ref, moved) < 0.05
    assert chi2_test(ref, same) > 0.05


def test_chi2_counts_missing_as_a_category() -> None:
    ref = pd.Series(["a", "b"] * 100)
    cur = pd.Series(["a", "b", None, None] * 50)  # same a/b split, half missing
    assert chi2_test(ref, cur) < 0.05
    assert chi2_test(ref, ref.copy()) == pytest.approx(1.0)


def test_chi2_detects_an_unseen_category() -> None:
    ref = pd.Series(["a", "b"] * 100)
    cur = pd.Series(["a", "b", "z", "z"] * 50)
    assert chi2_test(ref, cur) < 0.05


@pytest.mark.parametrize(
    "values", [["a"] * 10, [None] * 10], ids=["one-category", "all-missing"]
)
def test_chi2_with_a_single_category_is_one(values: list[str | None]) -> None:
    assert chi2_test(pd.Series(values), pd.Series(values)) == 1.0


# --- drift_share ----------------------------------------------------------


def test_drift_share_counts_p_values_strictly_below_alpha() -> None:
    p_values = {"a": 0.01, "b": 0.5, "c": None, "d": 0.05}
    # d sits on alpha (not below); c has no test (all NaN): not drifted.
    assert drift_share(p_values, alpha=0.05) == 0.25


def test_drift_share_needs_features() -> None:
    with pytest.raises(ValueError, match="no feature"):
        drift_share({}, alpha=0.05)


# --- alert rule -----------------------------------------------------------


@pytest.mark.parametrize(
    ("share", "psi_values", "drop", "expected"),
    [
        (0.0, {"AMT_INCOME_TOTAL": 0.0}, None, False),
        (0.30, {"AMT_INCOME_TOTAL": 0.0}, None, True),  # share on the threshold
        (0.2999, {"AMT_INCOME_TOTAL": 0.0}, None, False),
        (0.0, {"AMT_INCOME_TOTAL": 0.20}, None, True),  # key PSI on threshold
        (0.0, {"AMT_INCOME_TOTAL": 0.1999}, None, False),
        (0.0, {"AMT_INCOME_TOTAL": 0.0, "AMT_CREDIT": 5.0}, None, False),  # not key
        (0.0, {"AMT_INCOME_TOTAL": 0.0}, 0.03, False),  # drop must exceed .03
        (0.0, {"AMT_INCOME_TOTAL": 0.0}, 0.0301, True),
        (0.0, {"AMT_INCOME_TOTAL": 0.0}, -0.5, False),  # AUC got better
    ],
)
def test_alert_rule_at_each_boundary(
    monitoring: MonitoringConfig,
    share: float,
    psi_values: dict[str, float],
    drop: float | None,
    expected: bool,
) -> None:
    assert is_alert(share, psi_values, drop, monitoring) is expected


# --- report ---------------------------------------------------------------


def test_report_has_exactly_the_contract_keys_and_types(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    now = datetime(2026, 11, 24, 10, 30, tzinfo=UTC)
    report = compute_drift_report(synthetic_frame, synthetic_frame, params, now=now)
    assert REPORT_KEYS == (
        "timestamp",
        "drift_share",
        "psi",
        "ks",
        "chi2",
        "auc_drop",
        "alert",
    )
    assert list(report) == list(REPORT_KEYS)
    assert report["timestamp"] == "2026-11-24T10:30:00+00:00"
    assert isinstance(report["drift_share"], float)
    assert list(report["psi"]) == params.features.all
    assert list(report["ks"]) == params.features.numeric
    assert list(report["chi2"]) == params.features.categorical
    for section in ("psi", "ks", "chi2"):
        assert all(type(v) is float for v in report[section].values())
    assert report["auc_drop"] is None
    assert report["alert"] is False
    # Valid strict JSON: no NaN or numpy scalars.
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_report_on_identical_batches_shows_no_drift(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    report = compute_drift_report(synthetic_frame, synthetic_frame.copy(), params)
    assert report["drift_share"] == 0.0
    assert set(report["psi"].values()) == {0.0}
    assert report["alert"] is False


def test_report_timestamp_defaults_to_now_in_utc(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    before = datetime.now(UTC).replace(microsecond=0)
    report = compute_drift_report(synthetic_frame, synthetic_frame, params)
    stamp = datetime.fromisoformat(report["timestamp"])
    assert stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0
    assert before <= stamp <= datetime.now(UTC)


def test_report_flags_a_shifted_key_feature(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    current = synthetic_frame.copy()
    current["AMT_INCOME_TOTAL"] += 1.0  # one standard deviation
    report = compute_drift_report(synthetic_frame, current, params)
    assert report["ks"]["AMT_INCOME_TOTAL"] < params.monitoring.alpha
    assert report["psi"]["AMT_INCOME_TOTAL"] >= params.monitoring.psi_alert
    assert report["drift_share"] == pytest.approx(1 / len(params.features.all))
    assert report["alert"] is True


def test_report_drift_share_alone_raises_the_alert(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    current = synthetic_frame.copy()
    others = [
        c for c in params.features.numeric if c not in params.monitoring.key_features
    ]
    for col in others:
        current[col] = current[col] * 3 + 10
    report = compute_drift_report(synthetic_frame, current, params)
    assert all(report["psi"][c] == 0.0 for c in params.monitoring.key_features)
    assert report["drift_share"] >= params.monitoring.drift_share_alert
    assert report["alert"] is True


def test_report_tests_categorical_features_with_chi2(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    current = synthetic_frame.copy()
    current["NAME_INCOME_TYPE"] = "A"
    report = compute_drift_report(synthetic_frame, current, params)
    assert report["chi2"]["NAME_INCOME_TYPE"] < params.monitoring.alpha
    assert report["psi"]["NAME_INCOME_TYPE"] > params.monitoring.psi_alert
    assert report["drift_share"] == pytest.approx(1 / len(params.features.all))
    assert report["alert"] is False  # not a key feature, share below 0.30


def test_report_all_nan_numeric_column_is_null_in_ks(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    current = synthetic_frame.copy()
    current["EXT_SOURCE_1"] = np.nan
    report = compute_drift_report(synthetic_frame, current, params)
    assert report["ks"]["EXT_SOURCE_1"] is None
    assert report["psi"]["EXT_SOURCE_1"] > 1  # the NaN bin shows it
    assert report["drift_share"] == 0.0
    json.dumps(report, allow_nan=False)


def test_report_uses_the_given_auc_drop(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    report = compute_drift_report(
        synthetic_frame, synthetic_frame, params, auc_drop=0.05
    )
    assert report["auc_drop"] == 0.05
    assert report["alert"] is True


def test_report_uses_the_psi_bins_param(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    current = synthetic_frame.copy()
    current["AMT_CREDIT"] += 0.3
    two_bins = params.model_copy(
        update={"monitoring": params.monitoring.model_copy(update={"psi_bins": 2})}
    )
    ten = compute_drift_report(synthetic_frame, current, params)["psi"]
    two = compute_drift_report(synthetic_frame, current, two_bins)["psi"]
    assert two["AMT_CREDIT"] != pytest.approx(ten["AMT_CREDIT"])


# --- auc_drop (concept drift) ---------------------------------------------


class ExtSourceModel:
    """Synthetic model: the lower EXT_SOURCE_2, the riskier."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        risk = 1 - X["EXT_SOURCE_2"].fillna(0.5).to_numpy()
        return np.column_stack([1 - risk, risk])


def test_auc_drop_is_reference_minus_current_auc(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    ref, cur = synthetic_frame.iloc[:1000], synthetic_frame.iloc[1000:]
    risk = 1 - synthetic_frame["EXT_SOURCE_2"].fillna(0.5)
    expected = roc_auc_score(ref["TARGET"], risk[:1000]) - roc_auc_score(
        cur["TARGET"], risk[1000:]
    )
    assert auc_drop(ExtSourceModel(), ref, cur, params) == pytest.approx(expected)


def test_auc_drop_is_none_without_current_labels(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    unlabelled = synthetic_frame.drop(columns=params.target)
    assert auc_drop(ExtSourceModel(), synthetic_frame, unlabelled, params) is None
    missing = synthetic_frame.assign(**{params.target: np.nan})
    assert auc_drop(ExtSourceModel(), synthetic_frame, missing, params) is None


def test_concept_drift_scenario_c_raises_the_alert(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    """Scenario C: labels flipped for part of the "Commercial associate" group.

    The features do not move (no data drift); only the AUC can see it.
    """
    df = synthetic_frame.copy()
    rng = np.random.default_rng(params.data.random_state)
    df["NAME_INCOME_TYPE"] = rng.choice(
        ["Working", "Commercial associate"], size=len(df), p=[0.6, 0.4]
    )
    fit, ref = df.iloc[:1000], df.iloc[1000:]
    model = fit_model(*prepare_xy(fit, params), "logreg", params)

    cur = ref.copy()
    group = cur.index[cur["NAME_INCOME_TYPE"] == "Commercial associate"]
    flipped = rng.choice(group, size=len(group) // 2, replace=False)
    cur.loc[flipped, params.target] = 1 - cur.loc[flipped, params.target]

    drop = auc_drop(model, ref, cur, params)
    assert drop is not None and drop > params.monitoring.auc_drop_alert
    report = compute_drift_report(ref, cur, params, auc_drop=drop)
    assert report["drift_share"] == 0.0
    assert report["alert"] is True


# --- Evidently HTML and CLI -----------------------------------------------


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One CLI run on a shifted synthetic batch, with a saved model."""
    params = load_params(PARAMS_FILE)
    frame = make_synthetic_frame(params)
    tmp = tmp_path_factory.mktemp("drift_cli")
    ref, cur = frame.iloc[:1000], frame.iloc[1000:].copy()
    cur["AMT_INCOME_TOTAL"] += 1.0
    ref.to_parquet(tmp / "ref.parquet")
    cur.to_parquet(tmp / "cur.parquet")
    model_dir = tmp / "model"
    model_dir.mkdir()
    joblib.dump(
        fit_model(*prepare_xy(ref, params), "logreg", params),
        model_dir / "model.joblib",
    )
    (model_dir / "thresholds.json").write_text('{"low": 0.1, "high": 0.5}')
    out = tmp / "out"
    main(
        [
            "--reference",
            str(tmp / "ref.parquet"),
            "--current",
            str(tmp / "cur.parquet"),
            "--model",
            str(model_dir),
            "--out-dir",
            str(out),
            "--params",
            str(PARAMS_FILE),
        ]
    )
    return tmp, out


def test_cli_writes_latest_json_and_a_stamped_pair(cli_run: tuple[Path, Path]) -> None:
    _, out = cli_run
    latest = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    assert list(latest) == list(REPORT_KEYS)
    assert latest["alert"] is True  # alert lives in the JSON, the run still ends
    assert isinstance(latest["auc_drop"], float)
    stamp = datetime.fromisoformat(latest["timestamp"]).strftime("%Y-%m-%d_%H%M%S")
    assert json.loads((out / f"{stamp}.json").read_text(encoding="utf-8")) == latest
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["latest.json", f"{stamp}.json", f"{stamp}.html"]
    )


def test_cli_html_is_an_evidently_report_naming_the_drifted_features(
    cli_run: tuple[Path, Path],
) -> None:
    _, out = cli_run
    (html,) = out.glob("*.html")
    text = html.read_text(encoding="utf-8")
    assert "Drifted Columns" in text
    # One entry per model feature, with the tests we asked for.
    flags = dict(
        re.findall(
            r'"column_name": "(\w+)".*?"data_drift": "(Detected|Not Detected)"',
            text,
            re.S,
        )
    )
    params = load_params(PARAMS_FILE)
    assert sorted(flags) == sorted(params.features.all)  # no CODE_GENDER, no ID
    assert text.count('"K-S p_value"') >= len(params.features.numeric)
    assert text.count('"chi-square p_value"') >= len(params.features.categorical)
    # Same KS and alpha as the JSON: same drifted numeric features.
    report = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    ours = {f for f, p in report["ks"].items() if p is not None and p < 0.05}
    theirs = {f for f in params.features.numeric if flags[f] == "Detected"}
    assert "AMT_INCOME_TOTAL" in ours
    assert theirs == ours


def test_cli_without_model_has_null_auc_drop(
    cli_run: tuple[Path, Path], tmp_path: Path
) -> None:
    tmp, _ = cli_run
    main(
        [
            "--reference",
            str(tmp / "ref.parquet"),
            "--current",
            str(tmp / "ref.parquet"),
            "--out-dir",
            str(tmp_path),
            "--params",
            str(PARAMS_FILE),
        ]
    )
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["auc_drop"] is None
    assert latest["alert"] is False
    assert len(list(tmp_path.glob("*.html"))) == 1
