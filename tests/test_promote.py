"""Tests for the challenger -> champion promotion gate (T7, D5, L4-1)."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from crediscore.config import Params, PromotionConfig, load_params
from crediscore.models import promote as pr
from crediscore.models import train as train_module
from crediscore.models.evaluate import load_model
from crediscore.models.registry import MODEL_NAME, register
from crediscore.models.threshold import zone_metrics
from conftest import make_synthetic_frame

PARAMS_FILE = Path(__file__).resolve().parents[1] / "params.yaml"
PROMO = PromotionConfig(
    max_gender_gap_pts=10, age_gap_margin_pts=2, min_rows=10, min_group_rows=5
)


# --- decide (pure) -------------------------------------------------------------


def scores(
    auc: float = 0.76,
    cost: float = 0.30,
    gender_raw: float = 5.0,
    gender_eo: float = 5.0,
    age_eo: float = 33.0,
) -> dict[str, Any]:
    return {
        "roc_auc": auc,
        "cost_per_file": cost,
        "fairness": {
            "gender": {"raw_gap_pts": gender_raw, "eo_gap_pts": gender_eo},
            "age": {"raw_gap_pts": 40.0, "eo_gap_pts": age_eo},
        },
    }


def decide(ch: dict, champ: dict | None, anchor: float = 38.7) -> dict[str, Any]:
    return pr.decide(ch, champ, age_anchor_pts=anchor, promotion=PROMO)


def test_all_conditions_pass() -> None:
    out = decide(scores(auc=0.77, cost=0.29), scores(auc=0.75, cost=0.32))
    assert out["promote"] is True
    assert out["failed"] == []
    assert list(out["checks"]) == [
        "auc",
        "cost",
        "gender_raw_gap",
        "gender_eo_gap",
        "age_eo_gap",
    ]
    assert out["checks"]["auc"] == {"value": 0.77, "limit": 0.75, "pass": True}
    assert out["checks"]["cost"] == {"value": 0.29, "limit": 0.32, "pass": True}
    assert out["checks"]["age_eo_gap"]["limit"] == pytest.approx(40.7)


@pytest.mark.parametrize(
    ("challenger", "failing"),
    [
        (scores(auc=0.7599), "auc"),
        (scores(cost=0.3001), "cost"),
        (scores(gender_raw=10.01), "gender_raw_gap"),
        (scores(gender_eo=10.01), "gender_eo_gap"),
        (scores(age_eo=40.71), "age_eo_gap"),
    ],
)
def test_each_condition_fails_alone(challenger: dict, failing: str) -> None:
    out = decide(challenger, scores(auc=0.76, cost=0.30))
    assert out["promote"] is False
    assert out["failed"] == [failing]


def test_equality_boundaries_promote() -> None:
    champ = scores(auc=0.76, cost=0.30)
    ch = scores(auc=0.76, cost=0.30, gender_raw=10.0, gender_eo=10.0, age_eo=40.7)
    out = decide(ch, champ, anchor=38.7)
    assert out["promote"] is True, out["failed"]


def test_a_better_champion_on_both_counts_blocks() -> None:
    out = decide(scores(auc=0.74, cost=0.33), scores(auc=0.75, cost=0.32))
    assert out["failed"] == ["auc", "cost"]


def test_no_champion_yet_only_the_fairness_gates_apply() -> None:
    out = decide(scores(auc=0.60, cost=0.90), None)
    assert out["promote"] is True
    assert list(out["checks"]) == ["gender_raw_gap", "gender_eo_gap", "age_eo_gap"]
    assert decide(scores(gender_eo=11.0), None)["failed"] == ["gender_eo_gap"]


def test_the_tolerances_come_from_the_promotion_config() -> None:
    strict = PromotionConfig(
        max_gender_gap_pts=4, age_gap_margin_pts=0, min_rows=10, min_group_rows=5
    )
    out = pr.decide(scores(), None, age_anchor_pts=33.0, promotion=strict)
    assert out["failed"] == ["gender_raw_gap", "gender_eo_gap"]
    assert out["checks"]["age_eo_gap"]["limit"] == 33.0


# --- age anchor ----------------------------------------------------------------


def test_age_anchor_is_the_baseline_eo_gap_of_metrics_json(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "model": {"fairness": {"age": {"eo_gap_pts": 1.0}}},
                "baseline": {"fairness": {"age": {"eo_gap_pts": 38.675}}},
            }
        ),
        "utf-8",
    )
    assert pr.read_age_anchor(path) == 38.675


@pytest.mark.parametrize(
    "content",
    [None, "not json", "{}", '{"baseline": {"fairness": {}}}', "[1]"]
    + ['{"baseline": {"fairness": {"age": {"eo_gap_pts": "38"}}}}']
    + [
        '{"baseline": {"fairness": {"age": {"eo_gap_pts": %s}}}}' % v
        for v in ("true", "NaN", "Infinity", "-Infinity")
    ],
)
def test_missing_or_broken_age_anchor_is_none(
    tmp_path: Path, content: str | None
) -> None:
    path = tmp_path / "metrics.json"
    if content is not None:
        path.write_text(content, "utf-8")
    assert pr.read_age_anchor(path) is None


# --- main on a sqlite registry -------------------------------------------------


@pytest.fixture(scope="module")
def tracked(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A sqlite store with two finished train runs (logreg, lightgbm), a
    held-out batch and a metrics.json with the age anchor."""
    base = tmp_path_factory.mktemp("promote")
    params = load_params(PARAMS_FILE)
    train_file = base / "train.parquet"
    make_synthetic_frame(params).to_parquet(train_file)
    make_synthetic_frame(params, n=1500).to_parquet(base / "heldout.parquet")
    hp = base / "hp.json"
    hp.write_text(json.dumps({"n_estimators": 40}), encoding="utf-8")
    write_anchor(base / "metrics.json", 38.7)
    # The synthetic batch has 1 500 rows (about 975 F / 525 M): smaller minimums.
    small_params(base / "params.yaml", min_rows=1000, min_group_rows=300)
    old_cwd, old_uri = Path.cwd(), os.environ.get("MLFLOW_TRACKING_URI")
    uri = f"sqlite:///{(base / 'mlflow.db').as_posix()}"
    os.chdir(base)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    try:
        common = ["--params", str(PARAMS_FILE), "--train", str(train_file)]
        train_module.main([*common, "--output-dir", "m1"])
        train_module.main(
            [*common, "--model", "lightgbm", "--model-params", str(hp)]
            + ["--output-dir", "m2"]
        )
    finally:
        os.chdir(old_cwd)
        if old_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = old_uri
    return base


def small_params(path: Path, **promotion: int) -> Path:
    raw = yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))
    raw["promotion"].update(promotion)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def write_anchor(path: Path, anchor: float) -> Path:
    payload = {"baseline": {"fairness": {"age": {"eo_gap_pts": anchor}}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def store(tracked: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the tracked store (the registry is mutated)."""
    shutil.copy(tracked / "mlflow.db", tmp_path / "mlflow.db")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return tracked


def run_ids() -> dict[str, str]:
    runs = mlflow.search_runs(experiment_names=["crediscore"], output_format="list")
    return {r.data.tags["model_type"]: r.info.run_id for r in runs}


def aliases() -> dict[str, str]:
    model = mlflow.MlflowClient().get_registered_model(MODEL_NAME)
    return {k: str(v) for k, v in model.aliases.items()}


def run_main(
    store: Path,
    tmp_path: Path,
    *extra: str,
    metrics: Path | None = None,
    params: Path | None = None,
    data: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "promotion.json"
    code = pr.main(
        [
            "--params",
            str(params or store / "params.yaml"),
            "--data",
            str(data or store / "heldout.parquet"),
            "--metrics",
            str(metrics or store / "metrics.json"),
            "--output",
            str(out),
            *extra,
        ]
    )
    return code, json.loads(out.read_text(encoding="utf-8"))


def register_pair(champion: str, challenger: str) -> None:
    client = mlflow.MlflowClient()
    assert register(client, run_ids()[champion])[1] == "champion"
    assert register(client, run_ids()[challenger])[1] == "challenger"


def test_same_model_as_challenger_is_promoted_with_aliases_and_tags(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Same run twice: equal AUC and equal cost on the same batch -> promote.
    register_pair("lightgbm", "lightgbm")
    code, decision = run_main(store, tmp_path)
    assert code == 0, decision["failed"]
    assert aliases() == {"champion": "2"}
    assert decision["promote"] is True
    assert decision["applied"] is True
    assert decision["challenger_version"] == "2"
    assert decision["champion_version"] == "1"
    checks = decision["checks"]
    assert checks["auc"]["value"] == checks["auc"]["limit"]
    assert checks["cost"]["value"] == checks["cost"]["limit"]
    tags = mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    datetime.fromisoformat(tags["promoted_at"])
    assert tags["promoted_at"] == decision["timestamp"]
    assert tags["promotion_previous_champion"] == "1"
    assert tags["promotion_data_md5"] == decision["data_md5"]
    for name, check in checks.items():
        assert float(tags[f"promotion_{name}"]) == check["value"]
        assert float(tags[f"promotion_{name}_limit"]) == check["limit"]
    assert (
        "promoted_at"
        not in mlflow.MlflowClient().get_model_version(MODEL_NAME, "1").tags
    )
    assert "promoted" in capsys.readouterr().out


def test_a_failing_gate_changes_nothing_and_exits_1(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_pair("lightgbm", "lightgbm")
    anchor = write_anchor(tmp_path / "metrics.json", -10.0)
    code, decision = run_main(store, tmp_path, metrics=anchor)
    assert code == 1
    assert decision["promote"] is False
    assert decision["applied"] is False
    assert decision["failed"] == ["age_eo_gap"]
    assert decision["checks"]["age_eo_gap"]["limit"] == -8.0
    assert aliases() == {"champion": "1", "challenger": "2"}
    assert (
        "promoted_at"
        not in mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    )
    assert "age_eo_gap" in capsys.readouterr().out


def test_dry_run_decides_but_changes_nothing(store: Path, tmp_path: Path) -> None:
    register_pair("lightgbm", "lightgbm")
    code, decision = run_main(store, tmp_path, "--dry-run")
    assert code == 0
    assert decision["promote"] is True
    assert (decision["dry_run"], decision["applied"]) == (True, False)
    assert aliases() == {"champion": "1", "challenger": "2"}
    assert (
        "promoted_at"
        not in mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    )


def test_both_models_are_scored_on_the_batch_with_their_own_thresholds(
    store: Path, tmp_path: Path
) -> None:
    register_pair("logreg", "lightgbm")
    _, decision = run_main(store, tmp_path, "--dry-run")
    params = load_params(PARAMS_FILE)
    batch = pd.read_parquet(store / "heldout.parquet")
    X, y = train_module.prepare_xy(batch, params)
    costs = params.cost.model_dump()
    for side, version in [("champion", "1"), ("challenger", "2")]:
        model, th = load_model(f"models:/{MODEL_NAME}/{version}")
        p = model.predict_proba(X)[:, 1]
        got = decision["scores"][side]
        assert got["thresholds"] == {"low": th["low"], "high": th["high"]}
        assert got["roc_auc"] == pytest.approx(roc_auc_score(y, p))
        expected = zone_metrics(y, p, th["low"], th["high"], **costs)
        assert got["cost_per_file"] == pytest.approx(expected["cost_per_file"])
        assert got["recall"] == pytest.approx(expected["recall"])
    th1 = load_model(f"models:/{MODEL_NAME}/1")[1]
    th2 = load_model(f"models:/{MODEL_NAME}/2")[1]
    assert th1["low"] != th2["low"]  # really two threshold sets
    assert decision["n_rows"] == len(batch)
    checks = decision["checks"]
    assert checks["auc"]["value"] == decision["scores"]["challenger"]["roc_auc"]
    assert checks["auc"]["limit"] == decision["scores"]["champion"]["roc_auc"]
    ch_fair = decision["scores"]["challenger"]["fairness"]
    assert checks["gender_eo_gap"]["value"] == ch_fair["gender"]["eo_gap_pts"]
    assert checks["age_eo_gap"]["value"] == ch_fair["age"]["eo_gap_pts"]


def test_no_champion_yet_promotes_on_the_fairness_gates(
    store: Path, tmp_path: Path
) -> None:
    register_pair("logreg", "lightgbm")
    mlflow.MlflowClient().delete_registered_model_alias(MODEL_NAME, "champion")
    code, decision = run_main(store, tmp_path)
    assert code == 0, decision["failed"]
    assert decision["champion_version"] is None
    assert decision["scores"]["champion"] is None
    assert "auc" not in decision["checks"]
    assert aliases() == {"champion": "2"}
    tags = mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    assert tags["promotion_previous_champion"] == "none"


def test_missing_age_anchor_fails_closed(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_pair("lightgbm", "lightgbm")
    code, decision = run_main(store, tmp_path, metrics=tmp_path / "absent.json")
    assert code == 1
    assert decision["promote"] is False
    assert decision["failed"] == ["age_anchor_missing"]
    assert aliases() == {"champion": "1", "challenger": "2"}
    assert "age anchor" in capsys.readouterr().out


@pytest.mark.parametrize("registered", [False, True])
def test_no_challenger_exits_1(store: Path, tmp_path: Path, registered: bool) -> None:
    if registered:
        register(mlflow.MlflowClient(), run_ids()["logreg"])  # champion only
    code, decision = run_main(store, tmp_path)
    assert code == 1
    assert decision["failed"] == ["no_challenger"]
    assert decision["challenger_version"] is None
    if registered:
        assert aliases() == {"champion": "1"}


def test_decision_json_fields(store: Path, tmp_path: Path) -> None:
    register_pair("logreg", "lightgbm")
    _, decision = run_main(store, tmp_path, "--dry-run")
    assert set(decision) == {
        "timestamp",
        "dry_run",
        "data",
        "data_md5",
        "n_rows",
        "model",
        "challenger_version",
        "champion_version",
        "age_anchor_pts",
        "promote",
        "applied",
        "failed",
        "checks",
        "scores",
    }
    assert decision["model"] == MODEL_NAME
    assert decision["age_anchor_pts"] == 38.7
    assert decision["data"] == str(store / "heldout.parquet")
    assert decision["data_md5"] == train_module.file_md5(store / "heldout.parquet")


def test_score_audits_the_decisions_taken_with_the_model_thresholds(
    params: Params,
) -> None:
    df = pd.DataFrame(
        {
            "TARGET": [0, 0, 0, 0],
            "CODE_GENDER": ["F", "F", "M", "M"],
            "AGE_YEARS": [25.0, 40.0, 25.0, 60.0],
        }
    )
    # low 0.1, high 0.2: ACCEPTE, A_ETUDIER, ACCEPTE, REFUSE.
    out = pr.score(df, [0.05, 0.15, 0.05, 0.25], {"low": 0.1, "high": 0.2}, params)
    gender = out["fairness"]["gender"]["acceptance_by_group"]
    assert gender == pytest.approx({"F": 0.5, "M": 0.5})
    assert out["review_share"] == 0.25


# --- correction pass: batch guards, atomicity, stale output, thresholds -------


def test_batch_check_counts_gender_groups_without_xna() -> None:
    promo = PromotionConfig(
        max_gender_gap_pts=10, age_gap_margin_pts=2, min_rows=4, min_group_rows=2
    )
    df = pd.DataFrame({"CODE_GENDER": ["F", "F", "M", "XNA", "XNA"]})
    assert pr.batch_check(df, promo) == "gender_groups_missing"
    df = pd.DataFrame({"CODE_GENDER": ["F", "F", "M", "M"]})
    assert pr.batch_check(df, promo) is None  # boundaries inclusive
    assert pr.batch_check(df.iloc[:3], promo) == "batch_too_small"


def test_data_is_required() -> None:
    with pytest.raises(SystemExit) as err:
        pr.main(["--metrics", "m.json", "--output", "o.json"])
    assert err.value.code == 2


def test_small_batch_fails_closed(store: Path, tmp_path: Path) -> None:
    register_pair("lightgbm", "lightgbm")
    params = small_params(tmp_path / "p.yaml", min_rows=1501, min_group_rows=300)
    code, decision = run_main(store, tmp_path, params=params)
    assert (code, decision["failed"]) == (1, ["batch_too_small"])
    assert decision["n_rows"] == 1500
    assert aliases() == {"champion": "1", "challenger": "2"}


def test_single_gender_batch_fails_closed(store: Path, tmp_path: Path) -> None:
    register_pair("lightgbm", "lightgbm")
    params = small_params(tmp_path / "p.yaml", min_rows=1000, min_group_rows=480)
    batch = pd.read_parquet(store / "heldout.parquet")
    assert (batch["CODE_GENDER"] == "M").sum() >= 480  # both groups pass here
    females = tmp_path / "f.parquet"
    pd.concat([batch[batch["CODE_GENDER"] == "F"]] * 2).to_parquet(females)
    code, decision = run_main(store, tmp_path, params=params, data=females)
    assert (code, decision["failed"]) == (1, ["gender_groups_missing"])
    assert aliases() == {"champion": "1", "challenger": "2"}


def test_challenger_that_is_already_champion_is_a_no_op(
    store: Path, tmp_path: Path
) -> None:
    register(mlflow.MlflowClient(), run_ids()["lightgbm"])
    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "challenger", "1")
    code, decision = run_main(store, tmp_path)
    assert (code, decision["failed"]) == (1, ["already_champion"])
    assert aliases() == {"champion": "1", "challenger": "1"}
    tags = mlflow.MlflowClient().get_model_version(MODEL_NAME, "1").tags
    assert "promoted_at" not in tags


def test_tags_first_challenger_alias_next_champion_alias_last(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_pair("lightgbm", "lightgbm")
    calls: list[str] = []
    client_cls = mlflow.MlflowClient
    for name in (
        "set_model_version_tag",
        "delete_registered_model_alias",
        "set_registered_model_alias",
    ):
        original = getattr(client_cls, name)

        def spy(self: Any, *a: Any, _o: Any = original, _n: str = name) -> Any:
            calls.append(_n)
            return _o(self, *a)

        monkeypatch.setattr(client_cls, name, spy)
    code, _ = run_main(store, tmp_path)
    assert code == 0
    first_alias = calls.index("delete_registered_model_alias")
    assert set(calls[:first_alias]) == {"set_model_version_tag"}
    assert calls[first_alias:] == [
        "delete_registered_model_alias",
        "set_registered_model_alias",
    ]


@pytest.mark.parametrize(
    "failing", ["delete_registered_model_alias", "set_registered_model_alias"]
)
def test_failure_between_calls_never_records_a_false_previous_champion(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    register_pair("lightgbm", "lightgbm")

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("registry down")

    with monkeypatch.context() as m:
        m.setattr(mlflow.MlflowClient, failing, boom)
        code, decision = run_main(store, tmp_path)
    assert code == 1
    assert decision["failed"][0] == "error"
    assert "registry down" in decision["failed"][1]
    assert aliases()["champion"] == "1"  # the champion alias moves last
    code, decision = run_main(store, tmp_path)  # re-run, registry back
    tags = mlflow.MlflowClient().get_model_version(MODEL_NAME, "2").tags
    assert tags["promotion_previous_champion"] == "1"
    if failing == "delete_registered_model_alias":
        assert code == 0
        assert aliases() == {"champion": "2"}
    else:  # challenger alias already gone: nothing left to promote
        assert (code, decision["failed"]) == (1, ["no_challenger"])
        assert aliases() == {"champion": "1"}


def test_a_fatal_error_overwrites_a_stale_decision_file(
    store: Path, tmp_path: Path
) -> None:
    register_pair("lightgbm", "lightgbm")
    out = tmp_path / "promotion.json"
    out.write_text('{"promote": true, "stale": true}', encoding="utf-8")
    code, decision = run_main(store, tmp_path, data=tmp_path / "absent.parquet")
    assert code == 1
    assert decision["promote"] is False
    assert decision["failed"][0] == "error"
    assert "absent.parquet" in decision["failed"][1]
    assert "stale" not in decision


def test_version_tags_differing_from_thresholds_json_fail_closed(
    store: Path, tmp_path: Path
) -> None:
    register_pair("lightgbm", "lightgbm")
    client = mlflow.MlflowClient()
    client.set_model_version_tag(MODEL_NAME, "2", "threshold_low", "0.5")
    code, decision = run_main(store, tmp_path)
    assert (code, decision["failed"]) == (1, ["thresholds_mismatch"])
    assert aliases() == {"champion": "1", "challenger": "2"}
