"""The `train` and `evaluate` DVC stages (lot 3): wiring, params, no test leak."""

import json
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

from crediscore.config import Params
from crediscore.models.pipeline import MODELS, build_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stages() -> dict[str, Any]:
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    return dict(dvc["stages"])


def _option(cmd: str, flag: str) -> str:
    args = shlex.split(cmd)
    return args[args.index(flag) + 1]


@pytest.mark.parametrize("name", ["train", "evaluate"])
def test_every_listed_param_exists_in_params_yaml(
    stages: dict[str, Any], name: str
) -> None:
    raw = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    for key in stages[name]["params"]:
        node = raw
        for part in key.split("."):
            assert part in node, f"{name}: unknown param {key}"
            node = node[part]


def test_train_reads_train_only_and_outputs_the_model(stages: dict[str, Any]) -> None:
    train = stages["train"]
    assert "data/processed/train.parquet" in train["deps"]
    assert not any("test" in str(d) for d in train["deps"])  # ML rule: no leak
    assert {"models/model.joblib", "models/thresholds.json"} <= set(train["outs"])
    assert _option(train["cmd"], "--output-dir") == "models"


def test_train_config_is_a_valid_committed_champion_config(
    stages: dict[str, Any], params: Params
) -> None:
    cmd = stages["train"]["cmd"]
    model = _option(cmd, "--model")
    hp_file = _option(cmd, "--model-params")
    assert model in MODELS
    assert hp_file in stages["train"]["deps"]
    hp = json.loads((ROOT / hp_file).read_text(encoding="utf-8"))
    # set_params raises on a hyperparameter the classifier does not know.
    build_pipeline(model, params.features.numeric, params.features.categorical, 0, hp)


def test_evaluate_reads_the_model_both_splits_and_writes_metrics(
    stages: dict[str, Any],
) -> None:
    ev = stages["evaluate"]
    assert _option(ev["cmd"], "--model") == "models"
    assert {
        "models/model.joblib",
        "models/thresholds.json",
        "data/processed/train.parquet",
        "data/processed/test.parquet",
    } <= set(ev["deps"])
    assert {"reports/metrics.json": {"cache": False}} in ev["metrics"]


@pytest.mark.parametrize("name", ["train", "evaluate"])
def test_stages_depend_on_every_module_they_import(
    stages: dict[str, Any], name: str
) -> None:
    import sys

    __import__(f"crediscore.models.{name}")
    imported = {
        Path(m.__file__).resolve().relative_to(ROOT).as_posix()
        for key, m in sys.modules.items()
        if key.startswith("crediscore.")
        and getattr(m, "__file__", None)
        and not m.__file__.endswith("__init__.py")
    }
    # Each stage runs in a fresh process: only its own import graph counts.
    needed = {
        "train": {"train", "pipeline", "split", "threshold", "cost", "tune"},
        "evaluate": {"evaluate", "train", "pipeline", "split", "threshold"}
        | {"cost", "tune"},
    }[name]
    expected = {f"src/crediscore/models/{m}.py" for m in needed}
    expected.add("src/crediscore/config.py")
    assert expected <= imported
    assert expected <= set(stages[name]["deps"])


def test_evaluate_tracks_the_fairness_audit_and_the_promotion_tolerances(
    stages: dict[str, Any],
) -> None:
    ev = stages["evaluate"]
    assert "src/crediscore/fairness/audit.py" in ev["deps"]
    assert {
        "promotion.max_gender_gap_pts",
        "promotion.age_gap_margin_pts",
    } <= set(ev["params"])
    assert "promotion" not in ev["params"]  # min_rows is not read by evaluate
