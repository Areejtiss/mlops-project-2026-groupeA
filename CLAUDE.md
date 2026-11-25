# CLAUDE.md — CrediScore Inclusion

Explainable credit-scoring MLOps platform (Home Credit Default Risk, binary
`TARGET`, ~8 % positives). Team of 3, graded per person on `git log` and on
individual questions. Sources of truth, in this order: **the code on disk** >
`params.yaml` > `SYNOPSIS.md` (validated by the teacher: KPIs and targets) >
`Roadmap_CrediScore_Inclusion.md` (who does what, when) >
`PROJET_CREDIT_SCORING_INCLUSION.md` (early generic guide: its names
`credit_scoring`, 70/15/15 split, LGD cost, `bureau.csv` thin-file rule are
**outdated**, do not copy them).

Per-person context (who is using this copy, their plan) lives in
`CLAUDE.local.md`, which is never committed.

## Ownership (never edit another owner's file directly: propose a PR they review)

| Area | Owner | Files |
|---|---|---|
| Data, DVC, Prefect, drift simulator | Areej | `params.yaml`, `config.py`, `data/*`, `flows/*`, `prefect.yaml`, `monitoring/simulator.py`, `dvc.yaml` data stages |
| Models, MLflow, SHAP, fairness, drift calc | Achraf | `models/*`, `explain/*`, `fairness/*`, `monitoring/drift.py`, `dvc.yaml` train/evaluate stages, `configs/`, `notebooks/` |
| API, Docker, CI/CD, Prometheus | Iyed | `api/*`, `Dockerfile`, `docker-compose.yml`, `.github/*`, `deploy/*`, `scripts/deploy_blue_green.py`, `monitoring/metrics.py` |

Shared files (`tests/conftest.py`, `pyproject.toml`/`poetry.lock`, `README.md`):
coordinate before changing. Changing a **contract** (1 `params.yaml`, 2 model
artifact, 3 `/predict` response, 4 drift JSON) = PR reviewed by both other members.

## Commands (Windows, PowerShell; everything through `poetry run`)

```
poetry install                      # env from poetry.lock
poetry run pytest                   # tests
poetry run pytest --cov=crediscore  # coverage (target 75 %, enforced in CI from S4)
poetry run black --check src tests
poetry run flake8 src tests
poetry run mypy                     # strict, on src/ only
poetry run pre-commit run --all-files
docker compose up -d                # MinIO (+ MLflow once added)
poetry run dvc pull                 # empty local bucket? poetry run python -m crediscore.data.download; poetry run dvc push
poetry run dvc repro
```

## Code conventions (match the existing code)

- Code, identifiers, comments, docstrings: **English**. Docs for the team: French.
- Python 3.11, `src/crediscore/` layout, run modules with `python -m crediscore.<pkg>.<mod>`.
- `black` 88 cols, `flake8` (E203/W503 ignored), **`mypy --strict`** on `src/`:
  every function fully typed. Untyped libs (scikit-learn, mlflow, lightgbm,
  xgboost, shap, evidently…) go in the `[[tool.mypy.overrides]]` list, not
  scattered `# type: ignore`.
- Module docstring with a `Usage:` block; Google-style docstrings
  (`Args/Returns/Raises`) on public functions. Short comments only where a reason is not obvious.
- Config: `crediscore.config.load_params()` is called in `main()` and values are
  passed down; core functions stay pure (no file or MLflow access). No new
  hard-coded feature lists, seeds, thresholds or costs. New parameter = PR on
  `params.yaml` + `config.py` (Areej).
- Pydantic models inherit the `StrictModel` pattern (`extra="forbid", frozen=True`).
- Secrets only in `.env` / `.dvc/config.local`. MLflow URI from `MLFLOW_TRACKING_URI`, never hard-coded.

## ML rules (each one should be a test)

- `CODE_GENDER` (and every `sensitive_cols`) never enters the model; `SK_ID_CURR` and `TARGET` neither.
- Preprocessing and model are versioned and served as **one artifact** logged to MLflow with a signature (exact form: contract 2).
- For **model selection**, the test split is read once, by `evaluate`. Thresholds, calibration and Optuna use splits carved from train. (The drift simulator may reuse test rows as "production traffic"; that is not model selection.)
- Every random thing seeded with `params.data.random_state`.
- Accuracy is never reported as a quality metric (92 % by predicting 0).
- SYNOPSIS KPIs: business cost −15 % vs the LogReg baseline, recall (class 1) ≥ 65 %, thin-file good payers accepted ×2 vs a "refuse without history" policy, acceptance gap between groups ≤ 10 pts. Targets: ROC-AUC ≥ 0.75, PR-AUC ≥ 0.25, Brier ≤ 0.07, SHAP < 50 ms/row, API p95 < 150 ms.
- Registry: model name `crediscore`, aliases `champion` / `challenger` (stages are deprecated; the report justifies it).
- CI never needs MinIO or MLflow: tests use `tests/fixtures/ci_sample.parquet` or synthetic frames.

## Working method (méthode d'exigence, adapted)

- **No code without a written plan** in `docs/plans/`; a plan is reviewed by an independent agent before coding.
- **TDD**: the test fails for the right reason, then minimal code, then green, then commit.
- Each lot is reviewed adversarially (fresh agent, isolated copy via `git archive`, own probes, mutation: ≥ 30 planted bugs on new code) until "ready".
- Never claim green without the command output. Never predict an agent's result.
- Product/business decisions (costs, KPI definitions, fairness rule, decision labels, anything another member consumes) go to the human, with a recommendation. Technical choices: decide, write them in the plan, say it in one line.
- The human must be able to explain every module (individual questions): end each lot with a short walkthrough.

## Git

- Never commit on `main`/`dev`. One branch per topic: `feature/<topic>`.
- Commits: small, one subject each, `type(scope): message` as in the log
  (`feat(config): …`, `chore(setup): …`, `docs: …`). ≥ 3 commits/week per person.
- Add files one by one; never `git add -A`; never bare `git stash`.
- **Never push, open a PR, merge, or message teammates without explicit approval for that action.**
- Never commit data, models, `mlruns/`, `.env`, `CLAUDE.local.md`.
