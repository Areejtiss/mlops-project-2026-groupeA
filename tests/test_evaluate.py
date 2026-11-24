"""Tests for the test-split evaluation and the SYNOPSIS KPIs (L3-4, D5, D10)."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from crediscore.config import Params
from crediscore.models import evaluate as ev
from crediscore.models.cost import ACCEPT, REFUSE, REVIEW
from crediscore.models.evaluate import d5_status, fairness, kpis, thin_file

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"
COSTS = {"cost_fn": 5.0, "cost_fp": 1.0, "cost_review": 0.3}


# --- fairness (D5, D8: only ACCEPTE counts as accepted) ----------------------


def toy_fairness_frame() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.DataFrame(
        {
            "CODE_GENDER": ["F", "F", "F", "M", "M", "XNA"],
            "AGE_YEARS": [25.0, 30.0, 50.0, 29.9, 60.0, np.nan],
            "TARGET": [0, 1, 0, 0, 0, 0],
        }
    )
    d = np.array([ACCEPT, ACCEPT, REFUSE, ACCEPT, REVIEW, ACCEPT])
    return df, d


def test_gender_gaps_exclude_xna_and_count_review_as_not_accepted() -> None:
    df, d = toy_fairness_frame()
    g = fairness(df, d, target="TARGET")["gender"]
    # F accepted 2/3, M 1/2 (review is not accepted); XNA left out.
    assert g["acceptance_by_group"] == pytest.approx({"F": 2 / 3, "M": 0.5})
    assert g["raw_gap_pts"] == pytest.approx(100 * (2 / 3 - 0.5))
    # Good payers: F rows 0 (accepted) and 2 (refused), M rows 3 and 4.
    assert g["acceptance_good_payers_by_group"] == pytest.approx({"F": 0.5, "M": 0.5})
    assert g["eo_gap_pts"] == pytest.approx(0.0)
    assert g["excluded_rows"] == 1


def test_age_bands_boundaries_and_missing_age() -> None:
    df, d = toy_fairness_frame()
    a = fairness(df, d, target="TARGET")["age"]
    # 25 and 29.9 -> <30; 30 -> 30-50; 50 and 60 -> >=50; NaN left out.
    assert a["acceptance_by_group"] == pytest.approx(
        {"<30": 1.0, "30-50": 1.0, "≥50": 0.0}
    )
    assert a["raw_gap_pts"] == pytest.approx(100.0)
    # 30-50 has no good payer: not a group of the equal-opportunity gap.
    assert a["acceptance_good_payers_by_group"] == pytest.approx(
        {"<30": 1.0, "≥50": 0.0}
    )
    assert a["eo_gap_pts"] == pytest.approx(100.0)
    assert a["excluded_rows"] == 1


def test_single_group_has_no_gap() -> None:
    df = pd.DataFrame(
        {"CODE_GENDER": ["F", "F"], "AGE_YEARS": [40.0, 41.0], "TARGET": [0, 0]}
    )
    out = fairness(df, np.array([ACCEPT, REFUSE]), target="TARGET")
    assert out["gender"]["raw_gap_pts"] == 0.0
    assert out["age"]["eo_gap_pts"] == 0.0


def fair(gender_raw: float, gender_eo: float, age_eo: float) -> dict:
    return {
        "gender": {"raw_gap_pts": gender_raw, "eo_gap_pts": gender_eo},
        "age": {"raw_gap_pts": 50.0, "eo_gap_pts": age_eo},
    }


def test_d5_status_boundaries_are_inclusive() -> None:
    base = fair(7.8, 6.6, 32.9)
    ok = d5_status(fair(10.0, 10.0, 34.9), base)
    assert ok["gender_pass"] and ok["age_pass"] and ok["pass"]
    assert ok["age_eo_limit_pts"] == pytest.approx(34.9)


@pytest.mark.parametrize(
    "model, failing",
    [
        (fair(10.01, 5.0, 30.0), "gender_pass"),
        (fair(5.0, 10.01, 30.0), "gender_pass"),
        (fair(5.0, 5.0, 34.91), "age_pass"),
    ],
)
def test_d5_status_each_condition_fails_alone(model: dict, failing: str) -> None:
    out = d5_status(model, fair(7.8, 6.6, 32.9))
    assert out[failing] is False
    assert out["pass"] is False
    other = {"gender_pass", "age_pass"} - {failing}
    assert all(out[k] is True for k in other)


def test_d5_age_limit_is_anchored_on_the_baseline_not_a_constant() -> None:
    out = d5_status(fair(5.0, 5.0, 40.0), fair(7.8, 6.6, 38.5))
    assert out["age_eo_limit_pts"] == pytest.approx(40.5)
    assert out["age_pass"] is True


# --- thin file (D10) ---------------------------------------------------------


def test_thin_file_vs_classic_scorecard_hand_computed() -> None:
    y = np.array([0, 0, 1, 0])
    proba = np.array([0.1, 0.3, 0.2, 0.05])
    thin = np.array([1, 1, 0, 0])
    d = np.array([ACCEPT, REFUSE, ACCEPT, ACCEPT])
    out = thin_file(y, proba, d, thin, COSTS)
    assert out["thin_share"] == 0.5
    assert out["thin_roc_auc"] is None  # only good payers among thin files
    assert out["thin_acceptance_rate"] == 0.5
    assert out["thin_good_payers_accepted"] == 1
    # The scorecard refuses every thin file, applies our cut elsewhere.
    assert out["scorecard_thin_good_payers_accepted"] == 0
    assert out["good_payers_accepted"] == 2
    assert out["scorecard_good_payers_accepted"] == 1
    assert out["good_payers_accepted_ratio"] == 2.0
    assert out["scorecard_cost_per_file"] == pytest.approx((1 + 1 + 5) / 4)


def test_thin_file_auc_when_both_classes_are_present() -> None:
    y = np.array([0, 1, 1, 0])
    proba = np.array([0.1, 0.9, 0.2, 0.3])
    thin = np.array([1, 1, 1, 0])
    out = thin_file(y, proba, np.array([ACCEPT] * 4), thin, COSTS)
    assert out["thin_roc_auc"] == pytest.approx(1.0)
    assert out["thin_share"] == 0.75


def test_thin_file_companions_acceptance_of_good_payers_and_losses() -> None:
    y = np.array([0, 0, 0, 1, 1, 0])
    thin = np.array([1, 1, 1, 1, 1, 0])
    d = np.array([ACCEPT, ACCEPT, REFUSE, ACCEPT, REVIEW, ACCEPT])
    out = thin_file(y, np.full(6, 0.1), d, thin, COSTS)
    assert out["thin_good_payer_acceptance_rate"] == pytest.approx(2 / 3)
    assert out["thin_accepted_default_rate"] == pytest.approx(1 / 3)
    none = thin_file(y, np.full(6, 0.1), np.array([REFUSE] * 6), thin, COSTS)
    assert none["thin_accepted_default_rate"] is None  # nothing accepted


def thin(accepted: int, rate: float, default_rate: float) -> dict:
    return {
        "thin_good_payers_accepted": accepted,
        "scorecard_thin_good_payers_accepted": 0,
        "thin_good_payer_acceptance_rate": rate,
        "thin_accepted_default_rate": default_rate,
        "good_payers_accepted_ratio": 1.62,
    }


# --- KPIs (L3-4, L3-6) -------------------------------------------------------


def rep(**over: float) -> dict:
    base = {
        "roc_auc": 0.76,
        "pr_auc": 0.26,
        "brier": 0.068,
        "recall": 0.66,
        "cost_per_file": 0.27,
        "accept_all_cost_per_file": 0.40,
        "fairness": fair(5.0, 5.0, 30.0),
        "thin_file": thin(100, 0.6, 0.05),
    }
    return {**base, **over}


def test_thin_file_target_is_the_literal_kpi_with_companions(params: Params) -> None:
    model = rep(thin_file=thin(4946, 0.654, 0.051))
    base = rep(thin_file=thin(4467, 0.589, 0.060))
    t = kpis(model, base, params)["targets"]["thin_file"]
    # SYNOPSIS: 2x the thin good payers of "refuse without history" (0).
    assert (t["value"], t["target"], t["met"], t["gap"]) == (4946, 0, True, 4946)
    assert t["companion"] == {
        "thin_good_payer_acceptance_rate": 0.654,
        "baseline_thin_good_payer_acceptance_rate": 0.589,
        "thin_accepted_default_rate": 0.051,
        "baseline_thin_accepted_default_rate": 0.060,
        "portfolio_good_payers_ratio_informative": 1.62,
    }
    zero = kpis(rep(thin_file=thin(0, 0.0, 0.0)), base, params)["targets"]
    assert zero["thin_file"]["met"] is False  # accepting none is not "2x"


def test_kpi_cost_vs_baseline_formula_and_status(params: Params) -> None:
    out = kpis(rep(cost_per_file=0.27), rep(cost_per_file=0.30), params)
    assert out["cost_vs_baseline"] == pytest.approx((0.27 - 0.30) / 0.30)
    t = out["targets"]["cost_vs_baseline"]
    assert t["target"] == -0.15 and t["met"] is False
    assert t["gap"] == pytest.approx(-0.1 + 0.15)
    assert out["cost_vs_accept_all"] == pytest.approx((0.27 - 0.40) / 0.40)
    met = kpis(rep(cost_per_file=0.255), rep(cost_per_file=0.30), params)
    assert met["targets"]["cost_vs_baseline"]["met"] is True  # exactly -15 %


def test_kpi_targets_directions(params: Params) -> None:
    out = kpis(rep(roc_auc=0.74, pr_auc=0.25, brier=0.071, recall=0.65), rep(), params)[
        "targets"
    ]
    assert out["roc_auc"]["met"] is False  # >= 0.75
    assert out["roc_auc"]["gap"] == pytest.approx(-0.01)
    assert out["pr_auc"]["met"] is True  # >= 0.25, inclusive
    below = kpis(rep(pr_auc=0.249), rep(), params)["targets"]["pr_auc"]
    assert below["met"] is False
    assert out["brier"]["met"] is False  # <= 0.07
    assert out["recall"]["met"] is True  # >= min_recall, inclusive
    assert out["recall"]["target"] == params.threshold.min_recall


def test_kpis_carry_the_d5_status(params: Params) -> None:
    out = kpis(rep(fairness=fair(5, 5, 33.0)), rep(fairness=fair(5, 5, 30.0)), params)
    assert out["d5"]["age_pass"] is False  # 33 > 30 + 2


# --- main --------------------------------------------------------------------


@pytest.fixture
def split_files(tmp_path: Path, synthetic_frame: pd.DataFrame) -> tuple[Path, Path]:
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    synthetic_frame.iloc[:1500].to_parquet(train)
    synthetic_frame.iloc[1500:].to_parquet(test)
    return train, test


def test_main_on_a_model_directory_writes_metrics_and_logs(
    tmp_path: Path,
    split_files: tuple[Path, Path],
    params: Params,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crediscore.models import train as train_module

    train, test = split_files
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    model_dir = tmp_path / "lgbm"
    train_module.main(
        [
            "--params",
            str(PARAMS_FILE),
            "--train",
            str(train),
            "--model",
            "lightgbm",
            "--output-dir",
            str(model_dir),
        ]
    )
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    out = tmp_path / "reports" / "metrics.json"
    ev.main(
        [
            "--params",
            str(PARAMS_FILE),
            "--model",
            str(model_dir),
            "--train",
            str(train),
            "--test",
            str(test),
            "--output",
            str(out),
        ]
    )
    m = json.loads(out.read_text(encoding="utf-8"))
    th = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
    assert m["thresholds"] == {"low": th["low"], "high": th["high"]}
    # The model is scored with its own thresholds.json, not the baseline's.
    assert m["model"]["thresholds"] == {"low": th["low"], "high": th["high"]}
    assert m["n_test"] == 500
    for side in ("model", "baseline"):
        r = m[side]
        for key in ["roc_auc", "pr_auc", "brier", "recall", "review_share"]:
            assert 0 <= r[key] <= 1, (side, key)
        assert r["cost_per_file"] > 0
        assert set(r["fairness"]) == {"gender", "age"}
        assert "thin_acceptance_rate" in r["thin_file"]
    kpi = m["kpi"]
    assert kpi["cost_vs_baseline"] == pytest.approx(
        (m["model"]["cost_per_file"] - m["baseline"]["cost_per_file"])
        / m["baseline"]["cost_per_file"]
    )
    assert kpi["d5"]["age_eo_limit_pts"] == pytest.approx(
        m["baseline"]["fairness"]["age"]["eo_gap_pts"] + 2
    )
    assert m["baseline"]["thresholds"]["low"] <= m["baseline"]["thresholds"]["high"]

    mlflow.set_tracking_uri(uri)
    (run,) = mlflow.search_runs(experiment_names=["crediscore"], output_format="list")
    assert run.data.tags["stage"] == "evaluate"
    assert run.data.tags["evaluated_model"] == str(model_dir)
    assert run.data.metrics["test_roc_auc"] == pytest.approx(m["model"]["roc_auc"])
    assert run.data.metrics["baseline_test_cost_per_file"] == pytest.approx(
        m["baseline"]["cost_per_file"]
    )
    assert run.data.metrics["kpi_cost_vs_baseline"] == pytest.approx(
        kpi["cost_vs_baseline"]
    )


def test_main_on_a_registry_uri_uses_the_version_thresholds(
    tmp_path: Path,
    split_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crediscore.models import train as train_module

    train, test = split_files
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    train_module.main(
        [
            "--params",
            str(PARAMS_FILE),
            "--train",
            str(train),
            "--output-dir",
            str(tmp_path / "m"),
        ]
    )
    mlflow.set_tracking_uri(uri)
    (run,) = mlflow.search_runs(experiment_names=["crediscore"], output_format="list")
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", "crediscore")
    client = mlflow.MlflowClient()
    client.set_registered_model_alias("crediscore", "champion", version.version)
    out = tmp_path / "metrics.json"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")  # no evaluate run this time
    ev.main(
        [
            "--params",
            str(PARAMS_FILE),
            "--model",
            "models:/crediscore@champion",
            "--train",
            str(train),
            "--test",
            str(test),
            "--output",
            str(out),
        ]
    )
    m = json.loads(out.read_text(encoding="utf-8"))
    th = json.loads((tmp_path / "m" / "thresholds.json").read_text(encoding="utf-8"))
    assert m["thresholds"] == {"low": th["low"], "high": th["high"]}
    assert m["model_uri"] == "models:/crediscore@champion"
    _, by_number = ev.load_model(f"models:/crediscore/{version.version}")
    assert by_number == th


# --- lot 3 mutation survivors ------------------------------------------------


def test_report_accept_all_accepts_probabilities_of_exactly_one(
    params: Params,
) -> None:
    df = pd.DataFrame(
        {
            "TARGET": [1, 0, 0, 1],
            "IS_THIN_FILE": [0, 1, 0, 1],
            "CODE_GENDER": ["F", "M", "F", "M"],
            "AGE_YEARS": [25.0, 40.0, 60.0, 35.0],
        }
    )
    out = ev.report(df, [1.0, 0.2, 0.1, 0.9], 0.5, 0.95, params)
    # "accept all" accepts even proba 1.0: both defaults are missed.
    assert out["accept_all_cost_per_file"] == pytest.approx(2 * 5 / 4)


def test_main_scores_the_model_and_a_logreg_baseline_with_its_own_thresholds(
    tmp_path: Path,
    split_files: tuple[Path, Path],
    params: Params,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.metrics import roc_auc_score

    from crediscore.models import train as train_module

    train, test = split_files
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    model_dir = tmp_path / "lgbm"
    common = ["--params", str(PARAMS_FILE), "--train", str(train)]
    train_module.main([*common, "--model", "lightgbm", "--output-dir", str(model_dir)])
    calls: list[tuple[str, dict]] = []

    def spy(df: pd.DataFrame, name: str, p: Params, **kw: object) -> dict:
        result = train_module.train(df, name, p, **kw)
        calls.append((name, result))
        return result

    monkeypatch.setattr(ev, "train", spy)
    out = tmp_path / "metrics.json"
    ev.main(
        [*common, "--model", str(model_dir), "--test", str(test), "--output", str(out)]
    )
    m = json.loads(out.read_text(encoding="utf-8"))
    ((name, base),) = calls
    assert name == "logreg"
    assert m["baseline"]["thresholds"] == {
        "low": base["thresholds"]["low"],
        "high": base["thresholds"]["high"],
    }
    test_df = pd.read_parquet(test)
    X, y = train_module.prepare_xy(test_df, params)
    model, _ = ev.load_model(str(model_dir))
    assert m["model"]["roc_auc"] == pytest.approx(
        roc_auc_score(y, model.predict_proba(X)[:, 1])
    )
    assert m["baseline"]["roc_auc"] == pytest.approx(
        roc_auc_score(y, base["model"].predict_proba(X)[:, 1])
    )


def test_flatten_keeps_numeric_leaves_including_ints_and_bools() -> None:
    d = {"a": 1, "b": True, "c": {"d": 2.5, "e": None, "f": "x"}, "<30": 3}
    assert ev.flatten(d, "p_") == {"p_a": 1.0, "p_b": 1.0, "p_c.d": 2.5, "p_lt30": 3.0}
