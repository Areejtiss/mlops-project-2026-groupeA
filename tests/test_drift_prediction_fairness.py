"""Tests for prediction drift and fairness over time in the drift report (lot 6).

L6-1: `prediction` (PSI of the probabilities, acceptance-rate shift) and L6-2:
`fairness` (audit of the current batch with the model's decisions, D5 status
against the anchor). Both are informative: they never raise the alert.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from crediscore.config import Params
from crediscore.fairness.audit import d5_checks
from crediscore.models.cost import ACCEPT
from crediscore.models.evaluate import fairness as evaluate_fairness
from crediscore.models.threshold import decide
from crediscore.models.train import fit_model, prepare_xy
from crediscore.monitoring.drift import (
    REPORT_KEYS,
    compute_drift_report,
    fairness_drift,
    main,
    prediction_drift,
    psi,
)
from conftest import PARAMS_FILE

TH = {"low": 0.3, "high": 0.6}


class Ext2Model:
    """Stub model: the probability of default is EXT_SOURCE_2 (NaN -> 0.5)."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        risk = X["EXT_SOURCE_2"].fillna(0.5).to_numpy()
        return np.column_stack([1 - risk, risk])


def with_monitoring(params: Params, **changes: int) -> Params:
    monitoring = params.monitoring.model_copy(update=changes)
    return params.model_copy(update={"monitoring": monitoring})


def proba(df: pd.DataFrame) -> np.ndarray:
    return df["EXT_SOURCE_2"].fillna(0.5).to_numpy()


# --- prediction drift (L6-1) ------------------------------------------------


def test_prediction_drift_on_identical_batches_is_zero(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    out = prediction_drift(Ext2Model(), TH, synthetic_frame, synthetic_frame, params)
    assert list(out) == [
        "psi_proba",
        "acceptance_rate_ref",
        "acceptance_rate_cur",
        "acceptance_rate_delta_pts",
    ]
    assert out["psi_proba"] == 0.0
    assert out["acceptance_rate_ref"] == out["acceptance_rate_cur"]
    assert out["acceptance_rate_delta_pts"] == 0.0


def test_prediction_drift_values_match_a_hand_computation(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    ref = synthetic_frame.iloc[:1000]
    cur = synthetic_frame.iloc[1000:].copy()
    cur["EXT_SOURCE_2"] = cur["EXT_SOURCE_2"] * 0.5  # lower risk in this stub
    out = prediction_drift(Ext2Model(), TH, ref, cur, params)
    acc_ref = float(np.mean(proba(ref) < TH["low"]))
    acc_cur = float(np.mean(proba(cur) < TH["low"]))
    assert acc_cur > acc_ref  # current is more often accepted
    assert out["acceptance_rate_ref"] == pytest.approx(acc_ref)
    assert out["acceptance_rate_cur"] == pytest.approx(acc_cur)
    assert out["acceptance_rate_delta_pts"] == pytest.approx(100 * (acc_cur - acc_ref))
    expected = psi(pd.Series(proba(ref)), pd.Series(proba(cur)), 10)
    assert out["psi_proba"] == pytest.approx(expected)
    assert out["psi_proba"] > 0.2
    # The reference sets the bins: swapping the batches gives another PSI.
    swapped = prediction_drift(Ext2Model(), TH, cur, ref, params)
    assert swapped["psi_proba"] != pytest.approx(out["psi_proba"])
    assert swapped["acceptance_rate_delta_pts"] == pytest.approx(
        -out["acceptance_rate_delta_pts"]
    )


def test_prediction_drift_counts_only_accepte_as_accepted(params: Params) -> None:
    # proba == low is A_ETUDIER, not accepted (decide's left-closed zones).
    df = pd.DataFrame({c: [np.nan] * 4 for c in params.features.all})
    df[params.features.numeric] = df[params.features.numeric].astype(float)
    df["EXT_SOURCE_2"] = [0.1, 0.3, 0.5, 0.9]
    out = prediction_drift(Ext2Model(), TH, df, df, params)
    assert out["acceptance_rate_ref"] == 0.25
    expected = np.mean(decide(df["EXT_SOURCE_2"].to_numpy(), 0.3, 0.6) == ACCEPT)
    assert out["acceptance_rate_cur"] == expected


def test_prediction_drift_uses_the_model_thresholds(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    low = prediction_drift(Ext2Model(), TH, synthetic_frame, synthetic_frame, params)
    high = prediction_drift(
        Ext2Model(), {"low": 0.7, "high": 0.8}, synthetic_frame, synthetic_frame, params
    )
    assert high["acceptance_rate_ref"] > low["acceptance_rate_ref"]
    assert high["acceptance_rate_ref"] == pytest.approx(
        float(np.mean(proba(synthetic_frame) < 0.7))
    )


def test_prediction_drift_uses_the_psi_bins_param(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    ref, cur = synthetic_frame.iloc[:1000], synthetic_frame.iloc[1000:].copy()
    cur["EXT_SOURCE_2"] = cur["EXT_SOURCE_2"] ** 2
    ten = prediction_drift(Ext2Model(), TH, ref, cur, params)["psi_proba"]
    two = prediction_drift(
        Ext2Model(), TH, ref, cur, with_monitoring(params, psi_bins=2)
    )["psi_proba"]
    assert two == pytest.approx(psi(pd.Series(proba(ref)), pd.Series(proba(cur)), 2))
    assert two != pytest.approx(ten)


def test_prediction_drift_needs_no_label(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    unlabelled = synthetic_frame.drop(columns=[params.target])
    out = prediction_drift(Ext2Model(), TH, synthetic_frame, unlabelled, params)
    assert out["psi_proba"] == 0.0


def test_prediction_drift_gives_the_model_float64_numerics(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    seen: list[pd.DataFrame] = []

    class Spy(Ext2Model):
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            seen.append(X)
            return super().predict_proba(X)

    ints = synthetic_frame.assign(CNT_CHILDREN=1)
    prediction_drift(Spy(), TH, ints, ints, params)
    assert [list(X.columns) for X in seen] == [params.features.all] * 2
    assert all(X["CNT_CHILDREN"].dtype == "float64" for X in seen)


# --- fairness over time (L6-2) ----------------------------------------------


def test_fairness_drift_is_the_audit_of_the_model_decisions(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    out = fairness_drift(Ext2Model(), TH, synthetic_frame, params, age_anchor_pts=None)
    assert out is not None
    assert list(out) == ["n_labelled", "gender", "age", "d5"]
    assert out["n_labelled"] == len(synthetic_frame)
    expected = evaluate_fairness(
        synthetic_frame,
        decide(proba(synthetic_frame), TH["low"], TH["high"]),
        target=params.target,
    )
    assert out["gender"] == expected["gender"]
    assert out["age"] == expected["age"]
    assert out["gender"]["raw_gap_pts"] > 0
    assert out["d5"] is None  # no anchor


def test_fairness_drift_uses_the_model_thresholds(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    other = {"low": 0.7, "high": 0.8}
    out = fairness_drift(
        Ext2Model(), other, synthetic_frame, params, age_anchor_pts=None
    )
    assert out is not None
    expected = evaluate_fairness(
        synthetic_frame, decide(proba(synthetic_frame), 0.7, 0.8), target=params.target
    )
    assert out["age"] == expected["age"]


def test_fairness_drift_d5_status_against_the_anchor(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    out = fairness_drift(Ext2Model(), TH, synthetic_frame, params, age_anchor_pts=38.7)
    assert out is not None and out["d5"] is not None
    checks = d5_checks(
        out,
        age_anchor_pts=38.7,
        max_gender_gap_pts=params.promotion.max_gender_gap_pts,
        age_gap_margin_pts=params.promotion.age_gap_margin_pts,
    )
    assert out["d5"] == {
        "age_anchor_pts": 38.7,
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
        "reason": None,
    }
    assert out["d5"]["checks"]["age_eo_gap"]["limit"] == pytest.approx(40.7)


def with_min_group_rows(params: Params, n: int) -> Params:
    promotion = params.promotion.model_copy(update={"min_group_rows": n})
    return params.model_copy(update={"promotion": promotion})


@pytest.mark.parametrize("gender", ["F", "XNA"])
def test_fairness_drift_d5_without_two_gender_groups_has_no_verdict(
    synthetic_frame: pd.DataFrame, params: Params, gender: str
) -> None:
    # All F: a gender gap of 0 would "pass" without any comparison. All XNA:
    # no group at all (XNA never counts, as in promote).
    one = synthetic_frame.assign(CODE_GENDER=gender)
    out = fairness_drift(Ext2Model(), TH, one, params, age_anchor_pts=38.7)
    assert out is not None and out["d5"] is not None
    assert out["d5"]["pass"] is None
    assert out["d5"]["reason"] == "gender_groups_missing"
    assert out["d5"]["age_anchor_pts"] == 38.7
    assert out["gender"]["raw_gap_pts"] == 0.0  # gaps still published
    assert "age_eo_gap" in out["d5"]["checks"]
    assert out["age"]["eo_gap_pts"] > 0


def test_fairness_drift_gender_groups_follow_min_group_rows(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    smaller = int(synthetic_frame["CODE_GENDER"].value_counts().min())
    at = with_min_group_rows(params, smaller)
    above = with_min_group_rows(params, smaller + 1)
    ok = fairness_drift(Ext2Model(), TH, synthetic_frame, at, age_anchor_pts=38.7)
    no = fairness_drift(Ext2Model(), TH, synthetic_frame, above, age_anchor_pts=38.7)
    assert ok is not None and ok["d5"]["reason"] is None  # bound inclusive
    assert isinstance(ok["d5"]["pass"], bool)
    assert no is not None and no["d5"]["reason"] == "gender_groups_missing"
    assert no["d5"]["pass"] is None


def test_fairness_drift_counts_gender_groups_on_labelled_rows_without_xna(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    n_m = int((synthetic_frame["CODE_GENDER"] == "M").sum())
    rules = with_min_group_rows(params, n_m)
    df = synthetic_frame.copy()
    m_rows = df.index[df["CODE_GENDER"] == "M"]
    df.loc[m_rows[:1], params.target] = np.nan  # one M row loses its label
    out = fairness_drift(Ext2Model(), TH, df, rules, age_anchor_pts=38.7)
    assert out is not None and out["d5"]["reason"] == "gender_groups_missing"
    extra = pd.concat(
        [synthetic_frame, synthetic_frame.iloc[:5].assign(CODE_GENDER="XNA")]
    )
    out = fairness_drift(
        Ext2Model(), TH, extra, with_min_group_rows(params, n_m), age_anchor_pts=38.7
    )
    assert out is not None and out["d5"]["reason"] is None


def test_fairness_drift_d5_fails_when_one_check_fails(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    out = fairness_drift(Ext2Model(), TH, synthetic_frame, params, age_anchor_pts=38.7)
    assert out is not None
    age_gap = out["age"]["eo_gap_pts"]
    margin = params.promotion.age_gap_margin_pts
    tight = fairness_drift(
        Ext2Model(), TH, synthetic_frame, params, age_anchor_pts=age_gap - margin - 1
    )
    loose = fairness_drift(
        Ext2Model(), TH, synthetic_frame, params, age_anchor_pts=age_gap - margin
    )
    assert tight is not None and tight["d5"]["pass"] is False
    assert tight["d5"]["checks"]["age_eo_gap"]["pass"] is False
    assert loose is not None
    assert loose["d5"]["checks"]["age_eo_gap"]["pass"] is True  # bound inclusive
    gender_ok = all(
        loose["d5"]["checks"][k]["pass"] for k in ("gender_raw_gap", "gender_eo_gap")
    )
    assert loose["d5"]["pass"] is gender_ok


def test_fairness_drift_follows_the_promotion_tolerances(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    strict = params.model_copy(
        update={
            "promotion": params.promotion.model_copy(
                update={"max_gender_gap_pts": 0.0, "age_gap_margin_pts": 0.0}
            )
        }
    )
    out = fairness_drift(Ext2Model(), TH, synthetic_frame, strict, age_anchor_pts=1.0)
    assert out is not None
    assert out["d5"]["checks"]["gender_raw_gap"]["limit"] == 0.0
    assert out["d5"]["checks"]["age_eo_gap"]["limit"] == 1.0
    assert out["d5"]["pass"] is False


def test_fairness_drift_leaves_unlabelled_rows_out(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    df = pd.concat([synthetic_frame, synthetic_frame.iloc[:300]], ignore_index=True)
    df.loc[len(synthetic_frame) :, params.target] = np.nan  # feedback not back yet
    out = fairness_drift(Ext2Model(), TH, df, params, age_anchor_pts=None)
    full = fairness_drift(Ext2Model(), TH, synthetic_frame, params, age_anchor_pts=None)
    assert out is not None and full is not None
    assert out["n_labelled"] == len(synthetic_frame)
    assert out["gender"] == full["gender"]
    assert out["age"] == full["age"]


@pytest.mark.parametrize(("n", "computed"), [(999, False), (1000, True)])
def test_fairness_drift_needs_min_labelled_rows(
    synthetic_frame: pd.DataFrame, params: Params, n: int, computed: bool
) -> None:
    cur = synthetic_frame.iloc[:n]
    assert cur[params.target].sum() >= params.monitoring.min_positives
    out = fairness_drift(Ext2Model(), TH, cur, params, age_anchor_pts=38.7)
    assert (out is not None) is computed


def test_fairness_drift_follows_the_min_labelled_param(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    cur = synthetic_frame.iloc[:800]
    assert fairness_drift(Ext2Model(), TH, cur, params, age_anchor_pts=None) is None
    lenient = with_monitoring(params, min_labelled=800)
    assert fairness_drift(Ext2Model(), TH, cur, lenient, age_anchor_pts=None)


def test_fairness_drift_needs_min_positives(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    y = synthetic_frame[params.target]
    goods = synthetic_frame[y == 0]
    bads = synthetic_frame[y == 1]
    k = params.monitoring.min_positives
    few = pd.concat([goods, bads.iloc[: k - 1]])
    enough = pd.concat([goods, bads.iloc[:k]])
    assert fairness_drift(Ext2Model(), TH, few, params, age_anchor_pts=None) is None
    assert fairness_drift(Ext2Model(), TH, enough, params, age_anchor_pts=None)
    lenient = with_monitoring(params, min_positives=k - 1)
    assert fairness_drift(Ext2Model(), TH, few, lenient, age_anchor_pts=None)


def test_fairness_drift_with_a_single_class_is_none(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    lenient = with_monitoring(params, min_positives=1)
    ones = synthetic_frame.assign(**{params.target: 1})
    assert fairness_drift(Ext2Model(), TH, ones, lenient, age_anchor_pts=None) is None


@pytest.mark.parametrize("column", ["TARGET", "CODE_GENDER"])
def test_fairness_drift_without_label_or_gender_is_none(
    synthetic_frame: pd.DataFrame, params: Params, column: str
) -> None:
    cur = synthetic_frame.drop(columns=[column])
    assert fairness_drift(Ext2Model(), TH, cur, params, age_anchor_pts=38.7) is None


# --- report and alert ---------------------------------------------------------


def test_report_keys_append_prediction_and_fairness_after_alert() -> None:
    assert REPORT_KEYS == (
        "timestamp",
        "drift_share",
        "psi",
        "ks",
        "chi2",
        "auc_drop",
        "alert",
        "prediction",
        "fairness",
    )


def test_report_without_model_has_null_prediction_and_fairness(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    report = compute_drift_report(synthetic_frame, synthetic_frame, params)
    assert list(report) == list(REPORT_KEYS)
    assert report["prediction"] is None
    assert report["fairness"] is None


def test_report_carries_prediction_and_fairness_without_alerting(
    synthetic_frame: pd.DataFrame, params: Params
) -> None:
    pred = {
        "psi_proba": 9.0,
        "acceptance_rate_ref": 0.9,
        "acceptance_rate_cur": 0.1,
        "acceptance_rate_delta_pts": -80.0,
    }
    fair = fairness_drift(Ext2Model(), TH, synthetic_frame, params, age_anchor_pts=0.0)
    assert fair is not None and fair["d5"]["pass"] is False
    report = compute_drift_report(
        synthetic_frame, synthetic_frame, params, prediction=pred, fairness=fair
    )
    assert report["prediction"] == pred
    assert report["fairness"] == fair
    assert report["alert"] is False  # informative only
    assert json.loads(json.dumps(report, allow_nan=False)) == report


# --- CLI ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def batches(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Reference and current parquet files and a saved logreg model."""
    from crediscore.config import load_params
    from conftest import make_synthetic_frame

    params = load_params(PARAMS_FILE)
    frame = make_synthetic_frame(params, n=3000)
    tmp = tmp_path_factory.mktemp("drift_l6")
    fit, ref, cur = frame.iloc[:1000], frame.iloc[1000:2000], frame.iloc[2000:]
    ref.to_parquet(tmp / "ref.parquet")
    cur.to_parquet(tmp / "cur.parquet")
    model_dir = tmp / "model"
    model_dir.mkdir()
    joblib.dump(
        fit_model(*prepare_xy(fit, params), "logreg", params),
        model_dir / "model.joblib",
    )
    (model_dir / "thresholds.json").write_text('{"low": 0.06, "high": 0.12}')
    (tmp / "metrics.json").write_text(
        json.dumps({"baseline": {"fairness": {"age": {"eo_gap_pts": 12.5}}}})
    )
    return tmp


def run_cli(batches: Path, out: Path, *extra: str) -> dict:
    main(
        [
            "--reference",
            str(batches / "ref.parquet"),
            "--current",
            str(batches / "cur.parquet"),
            "--out-dir",
            str(out),
            "--params",
            str(PARAMS_FILE),
            *extra,
        ]
    )
    return json.loads((out / "latest.json").read_text(encoding="utf-8"))


def test_cli_with_model_writes_prediction_and_fairness(
    batches: Path, tmp_path: Path, params: Params
) -> None:
    latest = run_cli(
        batches,
        tmp_path,
        "--model",
        str(batches / "model"),
        "--metrics",
        str(batches / "metrics.json"),
    )
    model = joblib.load(batches / "model" / "model.joblib")
    th = {"low": 0.06, "high": 0.12}
    ref = pd.read_parquet(batches / "ref.parquet")
    cur = pd.read_parquet(batches / "cur.parquet")
    assert latest["prediction"] == pytest.approx(
        prediction_drift(model, th, ref, cur, params)
    )
    expected = fairness_drift(model, th, cur, params, age_anchor_pts=12.5)
    assert expected is not None
    assert latest["fairness"]["n_labelled"] == expected["n_labelled"]
    assert latest["fairness"]["d5"]["age_anchor_pts"] == 12.5
    assert latest["fairness"]["d5"]["pass"] is expected["d5"]["pass"]
    assert latest["fairness"]["age"]["eo_gap_pts"] == pytest.approx(
        expected["age"]["eo_gap_pts"]
    )


def test_cli_with_model_and_no_metrics_file_has_no_d5_status(
    batches: Path, tmp_path: Path
) -> None:
    latest = run_cli(
        batches,
        tmp_path,
        "--model",
        str(batches / "model"),
        "--metrics",
        str(tmp_path / "absent.json"),
    )
    assert latest["fairness"] is not None
    assert latest["fairness"]["d5"] is None


def test_cli_without_model_has_null_prediction_and_fairness(
    batches: Path, tmp_path: Path
) -> None:
    latest = run_cli(batches, tmp_path, "--metrics", str(batches / "metrics.json"))
    assert latest["prediction"] is None
    assert latest["fairness"] is None


def test_cli_metrics_defaults_to_reports_metrics_json_of_the_cwd(
    batches: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "metrics.json").write_text(
        json.dumps({"baseline": {"fairness": {"age": {"eo_gap_pts": 21.5}}}})
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"baseline": {"fairness": {"age": {"eo_gap_pts": 99.0}}}})
    )
    monkeypatch.chdir(tmp_path)
    latest = run_cli(batches, tmp_path / "out", "--model", str(batches / "model"))
    assert latest["fairness"]["d5"]["age_anchor_pts"] == 21.5
