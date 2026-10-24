# Achraf's roadmap — Modeling, MLflow, SHAP, fairness, drift

> **For agents:** execute lot by lot, task by task, in TDD. Read `CLAUDE.md` first.
> Do not start a lot whose "depends on" is not met, nor before the decisions it
> lists are answered (section 2).

- **Base read:** branch `dev` = `main` = `0de3860` (2026-10-11); `origin/feat/iyed-serving-cicd` = `91e5e1b` (2026-10-24, CI only, not merged).
- **Owner:** Achraf Gharbi. **Written:** 2026-10-24.
- **Status:** v2. Plan review on 2026-10-24 gave "ready with corrections"; all 4 blocking and 10 important corrections are applied below. Decisions taken 2026-10-24 (section 2). Lot 0 done 2026-10-24 (Python 3.11.9, Poetry 2.5.1, 22 tests green, MinIO up, raw data md5-verified and pushed). Lot 1 implemented 2026-10-24 (see "Lot 1 progress"); review corrections applied 2026-10-25; awaiting team approval of contracts 2/3. D5 decided 2026-10-25 (section 2).

---

## 0. Where we really are (verified 2026-10-24)

| Fact | Evidence |
|---|---|
| Areej's S2 is merged (Poetry, DVC + MinIO, download, `params.yaml`, `config.py`) | `git log 0de3860` |
| Areej's S3 (`validate`, `featurize`, `split`, `dvc.yaml`, `ci_sample.parquet`) **does not exist** | `src/crediscore/data/` has only `download.py`; `tests/fixtures/` has only `.gitkeep` |
| Iyed has an unmerged CI workflow (black, flake8, mypy, `pytest --cov` without `--cov-fail-under`). It triggers on push to `main` and `feat/**`, and on PRs to `main` | `ci.yml:4-7, 29-39` |
| Iyed's CI will likely fail: `cache: poetry` runs before Poetry is installed | `ci.yml:21-24` |
| **None of Achraf's S2 deliverables exist** (EDA, `cost.py`, baseline, contract 2) | `models/__init__.py` is a docstring only; `notebooks/.gitkeep` only |
| No ML dependency is installed (no scikit-learn, mlflow, lightgbm, xgboost, shap, evidently, optuna, scipy, matplotlib, jupyter) | `pyproject.toml:12-34` |
| The split is **80/20 only**, with no validation split | `params.yaml:10` |
| The cost is a flat 5:1 ratio | `params.yaml:80-82` |
| Thresholds are `null`. The comment uses a third label spelling, `A ETUDIER` | `params.yaml:84-88` |
| Only `CODE_GENDER` is sensitive. Age bands come from `AGE_YEARS`, which is also a model feature | `params.yaml:19-20, 61` |
| Only `application_train` is downloaded (no `bureau.csv`), so thin-file is defined from `EXT_SOURCE` only | `download.py:22` |
| **This machine:** no Python, no Poetry, no `.venv`, Docker daemon stopped, no `.env`, no data | checked 2026-10-24 |
| `git config user.email` = `achraf.gharbi@mafrah.com`; SYNOPSIS lists `gharbiachraf6@gmail.com` | `SYNOPSIS.md:11` |

**Bottom line:**
- Achraf is **one to two weeks behind**: S2 has not started, Areej's S2 merged on 2026-10-11, and Jalon 2 may be close (D7).
- He is also waiting on Areej's S3 data outputs.
- Plan to get unblocked: work on raw data and synthetic test data now, and plug into Areej's parquet files when they land. Deliver a walking-skeleton model early so that the team can meet the S3 checkpoint ("dvc repro produces a model, the API serves it in Docker and returns a decision").

---

## 1. Contracts Achraf produces or consumes

| # | Contract | Achraf's role | Status |
|---|---|---|---|
| 1 | `params.yaml` | Consumer. Requests new keys by PR to Areej | exists |
| 2 | Model artifact | **Owner** | to write (lot 1) |
| 3 | `/predict` response `{proba, decision, top_factors[3], model_version}` | Co-owner with Iyed (decision and SHAP parts). Changes go to the whole team | to align (lot 1) |
| 4 | Drift JSON `{timestamp, drift_share, psi, ks, chi2, auc_drop, alert}` | **Owner**, with Areej | to write (lot 5) |

Contract 2 must state:
- **Name and aliases:** registered as `crediscore`, with aliases `champion` and `challenger`.
- **Flavor:** `mlflow.sklearn`, because the API needs `predict_proba` and SHAP needs the fitted estimator.
- **Exact object:** a `CalibratedClassifierCV(..., ensemble=False)` that wraps the pipeline. Its `calibrated_classifiers_[0].estimator` is the full-train uncalibrated pipeline that SHAP explains, so there is no separate uncalibrated artifact. The scikit-learn version is pinned (≥ 1.6) for the API image.
- **Input:** the exact input columns and their dtypes (the signature), which depend on D2.
- **Output:** probability of class 1.
- **Thresholds (D9):** they travel with the model version, as model-version tags plus `thresholds.json`.
- **SHAP:** how it is computed, and the unit of `contribution`: log-odds (raw margin) for LogReg, LightGBM and XGBoost, probability for RandomForest. In both cases it does **not** add up to `proba`.
- **MLflow 3.17:** `serialization_format="cloudpickle"` (skops refuses `CalibratedClassifierCV`); always `models:/crediscore@<alias>`, never `runs:/…`; the sklearn flavor does not enforce the signature, so the API's Pydantic model validates inputs; numerics cast to float64 and every column optional in the signature.
- **Derived features (if D2a):** the function is pinned with mlflow `code_paths`, so later edits to Areej's function cannot silently change old champions.

---

## 2. Decisions — recommendation first

> **Decided 2026-10-24 (Achraf):** the recommended option is adopted for D1, D2, D3, D4, D6, D8, D9, D10. D3 values: `cost_review = 0.3`, `max_review_share = 0.15`; D8 adds `min_recall = 0.65`. Team-facing summary and actions per member: `docs/decisions/2026-10-24-decisions-modelisation.md` (to paste in the PR description). Decisions marked "Team" stay open to objection from Areej and Iyed until the PR is approved.
> **D5 decided 2026-10-25 (human), replacing "a then b":**
> - **Gender: absolute gate.** The acceptance-rate gap (`ACCEPTE` only) between F and M, `XNA` excluded, must be ≤ 10 pts, and the equal-opportunity gap must also be ≤ 10 pts. Baseline: 7.8 / 6.6 pts.
> - **Age bands (<30, 30–50, ≥50): no absolute gate.** Non-regression rule: the challenger's age equal-opportunity gap ≤ the champion's + 2 pts. Both age gaps are published at every evaluation and tracked over time (lot 6).
> - **Why (for the report):** age is a real risk driver in this data (the <30 band defaults about 2× the ≥50 band: 11.4 % vs 5.7 %) and a model feature. The baseline age gap is 34.5 pts raw / 32.9 pts equal opportunity, so an absolute 10-pt rule would block every promotion.
> - Lot 3 adds one MLflow experiment, LightGBM without `AGE_YEARS`, to measure the AUC/cost vs age-gap trade-off.
> - The teacher is told that the SYNOPSIS KPI "gap ≤ 10 pts" now applies to gender, and that age is "measured, explained, without degradation".
>
> **Closed 2026-10-24:** D7: no fixed dates; lots run back to back, as fast as quality allows. D11: the commit e-mail is Achraf's and is linked to his GitHub account.

Wording for non-developers in each line. Technical decisions are at the end.

| # | Question | Options | Recommendation | Who | Blocks |
|---|---|---|---|---|---|
| D1 | The data only has "train" and "test". Where do we tune the decision cut-offs and probability calibration without cheating on the test set? | a) set aside part of training data (new param `data.val_size`, e.g. 0.2) · b) ask Areej for a 70/15/15 split · c) cross-validation | **a)**: one new key in params (PR to Areej), no wait | Achraf + Areej | lot 2 |
| D2 | Ratios like "annuity / income" are computed by Areej's data step. At prediction time someone must compute them **exactly the same way**, or the model sees different numbers than in training ("train/serve skew") | a) the model artifact starts with Areej's function, so the API sends raw fields (`DAYS_BIRTH`…); this changes contracts 1, 2 and 3 · b) the API calls the same shared function before the model; the model takes `params.features.all` · c) the client sends already-computed ratios | **b)**: keeps contract 1 and the model input as they are; one shared function, imported by both Areej's step and Iyed's API, plus a test that both paths give identical columns | **Whole team** | lot 2 (contract 2) |
| D3 | With 3 zones (accept / review / refuse), what does one manual review cost, compared to a missed default (5) and a wrongly refused good client (1)? | a) add `cost.cost_review` and optimize both cut-offs · b) a fixed grey band around one cut-off · c) review band sized by analyst capacity (e.g. ≤ 10 % of files) | **a) + a capacity cap (c)**. With review = 0.3 the optimum is "accept if risk < 6 %, refuse if > 70 %": about **half** the applicants would go to review, which is unrealistic. A cap like "≤ 15 % of files to review" keeps it usable. Share of files in review shown with the numbers in lot 1 | Achraf + Areej (params) | lot 2 |
| D4 | Exact decision strings in the API answer | a) `ACCEPTE` / `A_ETUDIER` / `REFUSE` · b) `ACCEPTÉ` / `À ÉTUDIER` / `REFUSÉ` (roadmap and SYNOPSIS) | **a)**: no accents or spaces in a machine code; accents in the report and UI only. Also fix `params.yaml:85` | **Whole team** (contract 3) | lot 1 |
| D5 | Fairness gate for promotion. SYNOPSIS: "acceptance-rate gap ≤ 10 pts". Under-30s default about 2× more than over-50s, and age is itself a model feature, so the age gap may exceed 10 pts at any useful cut-off and **no model would ever be promoted** | a) measure on the baseline first (lot 1), then decide · b) gate on "equal opportunity": among clients who **did** repay, the acceptance gap between groups ≤ 10 pts; report the raw gap too · c) keep the raw gap | **Decided 2026-10-25 (human):** gender, absolute gate, raw ≤ 10 pts **and** equal opportunity ≤ 10 pts (XNA excluded); age, non-regression, challenger's equal-opportunity gap ≤ champion's + 2 pts, both age gaps published and tracked. Tell the teacher | Team | lot 4 |
| D6 | PR target branch | a) `main` · b) `dev`, then release to `main` | **a)**: CI only runs on PRs to `main`. Ask Iyed to add `feature/**` to the push trigger and to fix `cache: poetry` | Team | first PR |
| D7 | Real dates of S3…S8, Jalon 2, and the defense | — | needed to date the lots and re-plan | Human | planning |
| D8 | With 3 zones, how do the SYNOPSIS KPIs count "to review"? Recall ≥ 65 %: is a default sent to review "detected"? Acceptance rate and fairness: is review "accepted"? | a) review = detected for recall, **not** accepted for acceptance/fairness (prudent reading) · b) review = accepted everywhere · c) KPIs computed on a 2-zone cut (review merged into refuse) | **a)**. The single cost-optimal cut-off (risk 1/6) gives a recall of only ~50 % at AUC 0.76, so the 65 % target needs either a) or a recall constraint in the optimizer (applied anyway) | Team | lot 2 |
| D9 | Where do the cut-offs live? They are an **output** of training; writing them into `params.yaml` (an input of training) creates a DVC loop and two copies that can disagree | a) with the model version (tags + `thresholds.json`); `params.yaml` `threshold` becomes an optional manual override · b) in `params.yaml`, copied by hand | **a)**: PR to Areej to document the override | Achraf + Areej | lot 2 |
| D10 | Thin-file KPI: "accept 2× more good payers than refusing everyone without history" is always met (2 × 0). What is the comparison policy? | a) a "classic scorecard": refuse when `EXT_SOURCE_1` and `EXT_SOURCE_3` are missing, otherwise apply our cut-off · b) keep the literal KPI | **a)**: a meaningful number for the report | Team | lot 3 |
| D11 | Git identity: commits must be credited to Achraf on GitHub (graded on `git log`) | check that `achraf.gharbi@mafrah.com` is linked to his GitHub account, or switch to the linked address | **check before the first commit** | Achraf | lot 1 |

Technical decisions already taken (Achraf):

**T1. Dependencies.** All ML dependencies go in **one PR at the start of lot 1** (scikit-learn, mlflow, lightgbm, xgboost, shap, optuna, scipy, matplotlib, boto3, evidently pinned to an exact version, jupyter in dev) to avoid `poetry.lock` conflicts. Evidently's API changed a lot across 0.4 → 0.7. The mypy overrides cover scikit-learn, mlflow, lightgbm, xgboost, shap and evidently. Optuna ships type hints.

**T2. Splits and calibration.** The validation set is never reused for two fitted things:
- calibration is `CalibratedClassifierCV(pipeline, cv=5, method="isotonic", ensemble=False)` on the training part: one isotonic map learned on out-of-fold predictions, and the pipeline refitted on the whole training part. The served model is then exactly monotone in that one pipeline;
- the validation part only serves threshold optimization and model selection;
- Optuna uses CV on the training part.

**T3. SHAP.**
- SHAP explains the **uncalibrated** pipeline inside the served model (`calibrated_classifiers_[0].estimator`, T2). Factor order is an approximation of the calibrated decision; the report states this.
- Units: log-odds (raw margin) for LogReg, LightGBM and XGBoost; probability for RandomForest (which stays a candidate). Written as the `shap_unit` model-version tag.
- One-hot contributions are summed back to the original column.
- `TreeExplainer` is used for tree models and `LinearExplainer` for LogReg.
- For RandomForest, the latency test uses `feature_perturbation="tree_path_dependent"`, and the additivity check is done in probability space.

**T4. Artifacts and MLflow.** `train` writes `models/model.joblib` + `models/thresholds.json` (DVC outputs) and logs to MLflow only if `MLFLOW_TRACKING_URI` is set. A `--sample <parquet>` flag trains on the CI fixture and writes `metrics.json` + `roc.png` for Iyed's CML.

**T5. Class imbalance.** Use `class_weight="balanced"` / `scale_pos_weight`, never SMOTE, then calibrate.

**T6. Brier target.** A constant predictor at 8.07 % already scores Brier ≈ 0.074, so the 0.07 target is tight. Report it honestly.

**T7. Promotion.** Champion and challenger are scored **on the same, most recent held-out set**, at the time of promotion. They are never compared on their stored historical metrics.

**T8. MLflow server.** Run it with `--artifacts-destination s3://mlflow` (proxied artifacts), so that clients need no S3 credentials. Build a small image with boto3, since the official image lacks it.

**T9. Flaky tests.**
- The directional test averages over many rows (or uses LightGBM monotone constraints on `EXT_SOURCE_*`).
- Latency tests use a generous bound in CI, behind a `slow` marker; the strict 50 ms is measured locally.

**T10. Drift.**
- PSI quantile bins are deduplicated, for binary or discrete features.
- `drift_share` = share of features with p-value < α = 0.05 (KS or χ²), documented in contract 4.

---

## 3. File map (new files, all owned by Achraf unless noted)

```
notebooks/01_eda.ipynb, 02_baseline.ipynb     outputs cleared; figures saved to reports/figures/
docs/contracts/02-model-artifact.md, 04-drift-report.md
src/crediscore/models/cost.py          business_cost (pure, costs passed in)
src/crediscore/models/split.py         validation carve-out (D1)
src/crediscore/models/pipeline.py      build_pipeline(model_name, features, seed)
src/crediscore/models/threshold.py     optimize_thresholds (with recall + review-cap constraints), decide
src/crediscore/models/train.py         main(): fit, calibrate, thresholds, save, log
src/crediscore/models/evaluate.py      main(): reports/metrics.json incl. SYNOPSIS KPIs
src/crediscore/models/registry.py      register + set alias
src/crediscore/models/promote.py       challenger -> champion rule (T7)
src/crediscore/explain/shap_explainer.py
src/crediscore/fairness/audit.py
src/crediscore/monitoring/drift.py
tests/conftest.py (shared: coordinate with Iyed)   synthetic frame from params.yaml
tests/test_cost.py, test_split.py, test_threshold.py, test_pipeline.py, test_train.py,
tests/test_model_behaviour.py, test_promote.py, test_shap.py,
tests/test_fairness.py, test_psi.py, test_drift.py
```

Changes to other owners' files go by PR:

| File | Owner | What Achraf asks for |
|---|---|---|
| `pyproject.toml` / `poetry.lock` | shared | ML dependencies (T1) |
| `dvc.yaml` | Areej (data stages) | the train and evaluate stages |
| `docker-compose.yml` | Iyed | MLflow service and `mlflow` bucket |
| `params.yaml` | Areej | `data.val_size`, `cost.cost_review`, `threshold.max_review_share`, `threshold.min_recall`, `promotion.*`, `drift.*`, fix of the label spelling |

---

## 4. Lots

**Every lot ends with:**
- black, flake8, mypy and pytest green, with the outputs pasted;
- the plan boxes ticked, and the lot's own files committed;
- an **adversarial review** until "ready": a fresh agent, an isolated `git archive` copy, at least 30 mutants on the new code;
- a 10-minute walkthrough, so that Achraf can explain every function (individual questions).

The PR is opened **only after the human says so**.

### Lot 0 — Workstation ready (human, ~1 h) · depends on: nothing

- [ ] Install Python 3.11 (python.org, tick "Add to PATH"; the current `python` is the Microsoft Store stub) and Poetry 2.x. Restart VS Code.
- [ ] Run `poetry install`, then `poetry run pre-commit install`.
- [ ] Start Docker Desktop and `Copy-Item .env.example .env`. Fill in the MinIO and Kaggle secrets, then `docker compose up -d`.
- [ ] Run `poetry run dvc remote modify --local …` (README step 5).
- [ ] Get the data: `poetry run python -m crediscore.data.download`, then `poetry run dvc push` (the local bucket is empty).
- [ ] `poetry run pytest` → all green.
- [ ] D11: check the git e-mail on GitHub.

**Done when:** `data/raw/application_train.csv` has 307 511 × 122 rows and columns, and the tests pass.

### Lot 1 — S2 catch-up: dependencies, understanding, cost, baseline, contracts · branch `feature/eda-baseline` · ~8 h incl. review

**Depends on:** lot 0. **Decisions needed:** D4 and D11. D3, D5 and D8 get measured here.

**0. Dependencies PR (T1).**
- Stop point: a short review before anything else, because everything depends on it.

**1. `notebooks/01_eda.ipynb`.** Cover:
- class balance;
- missing rates;
- `DAYS_EMPLOYED = 365243`;
- the top correlations with `TARGET`;
- `EXT_SOURCE_*` missingness;
- the thin-file segment (EXT_SOURCE-based definition, agreed with Areej): its size and default rate;
- the default rate by `CODE_GENDER` (including `XNA`) and by age band (<30, 30–50, ≥50).

Figures go to `reports/figures/` for the report. Outputs are cleared before commit (1 MB pre-commit limit).

**2. `models/cost.py`, in TDD.** A pure function; the costs are passed in.

Tests first:
- all-correct → 0;
- one FN → `cost_fn`; one FP → `cost_fp`;
- mixed counts;
- review cost counted per reviewed file;
- length mismatch → `ValueError`; non-binary labels → `ValueError`.

Mutants to kill: swapped fn/fp, `>=` vs `>`, missing validation.

Also write in the notebook why the cost ratio is 5:1 (roadmap requirement).

**3. `notebooks/02_baseline.ipynb`.** LogReg (impute + scale + one-hot) on a local stratified split. Report:
- ROC-AUC, PR-AUC and Brier;
- for the single cost-optimal cut-off: cost and recall;
- for 3 zones under D3 options a and c: share of files in review and recall, with numbers for D3 and D8;
- the acceptance gap by gender and age band, raw and equal-opportunity → **answers D5 with numbers**.

**4. Contracts.**
- `docs/contracts/02-model-artifact.md` (section 1 list).
- The decision and SHAP parts of contract 3.
- Both are sent to Iyed and Areej for review.

**Done when:**
- the reference AUC and baseline cost are written in the notebook (the cost −15 % KPI is measured against this baseline);
- `cost.py` is 100 % covered;
- contracts 2 and 3 are approved by the team (their approval, not ours).

**Lot 1 progress (2026-10-24):**
- [x] 0. Dependencies (T1), one commit: scikit-learn 1.9.1, mlflow 3.17.2, lightgbm 4.7.0, xgboost 3.2.0, shap 0.51.0, optuna 5.0.0, scipy 1.17.1, matplotlib 3.11.2, boto3 1.43.117, **evidently pinned 0.7.23** (latest stable); jupyter, nbconvert, ipykernel in dev.
- [x] 1. `notebooks/01_eda.ipynb` + 5 figures in `reports/figures/eda_*.png`.
- [x] 2. `models/cost.py` in TDD, 27 tests, 100 % covered, mypy strict clean. Reviewer: round 1, 42 mutants, 33 killed / 9 survived; round 2, 45 mutants, 38 killed / 7 survived (2 equivalent, 5 now covered by tests). The 5:1 rationale is written in `02_baseline.ipynb`.
- [x] Review corrections (2026-10-25): asymmetric binary tests, 1-D error, float/bool decisions, BOM; `ensemble=False`; exact threshold search; contract 2 per coordinator D1–D3; `≥50` label. From now on the failing test is committed before the code (`test(...)` then `feat/fix(...)`).
- [x] 3. `notebooks/02_baseline.ipynb` + 3 figures + `reports/baseline_metrics.json`.
- [x] 4. `docs/contracts/02-model-artifact.md`, with the contract 3 proposal (decision + `top_factors`).
- [ ] Contracts 2 and 3 approved by Areej and Iyed (sent for review by the human).
- [ ] Adversarial review (≥ 30 mutants) and 10-minute walkthrough.

**Measured (baseline LogReg, stratified 100k sample, split 64k/16k/20k, seed 42, isotonic `ensemble=False`, exact threshold search over the 69 distinct validation probabilities):**

| | Validation | Test |
|---|---|---|
| ROC-AUC (calibrated) | 0.738 | **0.746** |
| PR-AUC (calibrated) | 0.213 | 0.220 |
| Brier uncalibrated / calibrated | 0.206 / 0.069 | 0.203 / **0.069** |
| Single cost-optimal cut-off 0.1707: recall, cost per file | 37.5 %, 0.348 | 38.6 %, 0.340 |
| 3 zones D3 (0.0926 / 0.1378): recall (margin), review share, cost per file | 66.2 % (+1.2 pt), 14.2 %, 0.333 | 66.5 % (+1.5 pt), 14.2 %, **0.325** (total 6 495.6) |
| 3 zones unconstrained (0.0571 / 0.7500): review share | 49.4 % | 48.7 % |

- Baseline cost on this test split: **0.325 per file** (≤ 0.276 for −15 %). Valid only on the same test split: lot 3 `evaluate` recomputes the baseline on Areej's test split.
- D3 confirmed: without a cap, half of the files go to review. D8: a single cut-off gives only 38 % recall (not ~50 % as estimated in D8); with recall ≥ 65 % it costs more than accepting everyone (0.409 vs 0.404). 3 zones fix this. The review cap is not hit exactly (14.2 %) because isotonic probabilities come in steps.
- **D5 measurement** (test, D3 thresholds, "accepted" = `ACCEPTE` only, D8): age gap **34.5 pts raw, 32.9 pts equal opportunity**; gender 7.8 / 6.6 pts. With the single cut-off: age 22.2 / 19.0, gender 5.9 / 4.5. Equal opportunity does not bring the age gap under 10 pts. D5 was then decided on 2026-10-25 (section 2): the baseline passes the gender gate; the age rule is non-regression.
- No `XNA` row in the test split (4 in the whole dataset).

**Deviations and why:**
- T1: mypy overrides cover scikit-learn, shap, scipy, boto3 and evidently (checked with mypy). mlflow, lightgbm, xgboost, optuna and matplotlib ship type hints, so they get no override. Version ranges allow Python 3.11 (scipy ≥ 1.18 and the newest shap need 3.12).
- The baseline uses the `params.data.sample_size` (100k) stratified sample, like the future featurize stage, not the full 307k rows.
- `VAL_SIZE`, `COST_REVIEW`, `MAX_REVIEW_SHARE`, `MIN_RECALL` are notebook constants, **pending Areej's params PR**. The derived features use a local stand-in for Areej's `add_derived_features` (D2).
- `business_cost` takes all three costs as required keyword arguments (no default `cost_review`), so a 3-zone call can never silently count review as free. Negative or NaN costs are not checked there: `config.py` already validates `gt=0`.
- `ensemble=False` (review, coordinator D1): the isotonic map has only 69 steps, so PR-AUC drops slightly (test 0.228 → 0.220) and thresholds are step values. `thresholds.json` must store them in full precision.

### Lot 2 — S3: walking skeleton, then traced training · branch `feature/mlflow-training` · ~12 h incl. 2 review rounds

**Depends on:** lot 1. **Decisions needed:** D1, D2, D3, D8, D9. For the DVC stage, Areej's `split` outputs (until then, train on a local split; wire the stage when her files exist).

**1. Walking skeleton first.** This serves the team's S3 checkpoint.
- Deliver a minimal LogReg `model.joblib` + `thresholds.json` + `decide()` to Iyed in the first days, so that the API can serve a decision in Docker.
- Stop point: a short review.

**2. MLflow service PR to Iyed (T8).**
- `docker-compose.yml` gets a service `mlflow` on port 5000, with a SQLite backend on a volume and artifacts proxied to `s3://mlflow` on MinIO.
- `minio-init` also creates the `mlflow` bucket.
- `.env.example` gets `MLFLOW_TRACKING_URI`.

**3. `tests/conftest.py`.** A deterministic synthetic DataFrame with:
- the model input columns from D2 (by default exactly `params.features.all`), plus `TARGET`, `CODE_GENDER` and `SK_ID_CURR`;
- NaNs in `EXT_SOURCE_*`;
- about 8 % positives and 2 000 rows.

**4. `models/split.py` (D1).** Tests:
- stratification within ±0.5 pt;
- deterministic with the seed;
- no ID overlap.

**5. `models/pipeline.py`.** `build_pipeline("logreg" | "lightgbm", …)` builds:
- a `ColumnTransformer`: median imputation + scaling for numeric columns; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` for categorical columns (LightGBM may skip imputation);
- then the model.

Tests:
- it fits on the synthetic frame;
- `predict_proba` is in [0,1];
- an unknown category does not crash;
- extra `CODE_GENDER` / `SK_ID_CURR` columns in the input are dropped and have no effect.

**6. `models/threshold.py`.**
- `optimize_thresholds(y, proba, costs, min_recall, max_review_share) -> (low, high)`.
- `decide(proba, low, high)`, returning the D4 codes.

Tests:
- a hand-computed toy case;
- `low < high` always;
- the constraints are respected;
- boundaries: `proba == low` → A_ETUDIER, `== high` → REFUSE;
- the cost at the optimum is ≤ the cost of "accept all" and of "refuse all".

**7. `models/train.py` `main()`.**
- Load params, split, fit, calibrate (T2), optimize the thresholds on validation.
- Save the calibrated model (`ensemble=False`, which contains the pipeline SHAP explains) + `thresholds.json` with full-precision thresholds.
- Log to MLflow if the URI is set: params, ROC-AUC, PR-AUC, Brier, recall@threshold, cost, review share, confusion matrix, ROC, PR and calibration curves, signature, DVC data hash as a tag.
- `--sample` flag (T4).

Test: `main()` on synthetic data, in a `tmp_path`, with the MLflow URI `sqlite:///<tmp_path>/mlflow.db` (not a file store), produces the run and the files.

**8. `dvc.yaml` stage `train`.** Deps: the train parquet and `src/crediscore/models`. Params: `data`, `features`, `cost`, `threshold`.

**Done when:**
- the API serves the skeleton model (with Iyed);
- two runs (LogReg, LightGBM) are side by side in the MLflow UI, with all metrics and curves;
- `dvc repro train` works.

### Lot 3 — S4: comparison, evaluate, registry (Jalon 2) · branch `feature/model-registry` · ~10 h incl. review

**Depends on:** lot 2 and D10.

**1. More models.** Add `random_forest` and `xgboost` to `build_pipeline`.

**2. Optuna.** 20–30 trials on LightGBM and XGBoost, each trial a nested MLflow run. Objective: CV ROC-AUC on the training part. Seeded sampler.

**2b. Age trade-off experiment (D5).** One more MLflow experiment: LightGBM **without `AGE_YEARS`**, to measure the AUC / cost vs age-gap trade-off for the report. It is a measurement, not a promotion candidate.

**3. `models/evaluate.py`.** This is the `evaluate` stage and the **only** reader of the test split for model selection. It writes `reports/metrics.json`, tracked with `dvc metrics`:
- ROC-AUC, PR-AUC, Brier, recall (D8), cost;
- **cost vs the LogReg baseline (KPI −15 %)**. `evaluate` **recomputes the LogReg baseline on the same test split**, with the same constraints. The −15 % reference is only valid that way, not against the lot 1 notebook number;
- thin-file AUC, acceptance rate, and good payers accepted vs the D10 policy;
- cost vs "accept all";
- fairness (D5): gender raw and equal-opportunity gaps, and **both age gaps, published at every evaluation** and tracked over time (lot 6).

**4. `models/registry.py`.** Register the best run as `crediscore`, and set the alias `champion` (first time) or `challenger`.

**5. Model behaviour tests** (`test_model_behaviour.py`), on the fixture or the synthetic frame:
- the AUC is above a floor;
- the output is in [0,1];
- **invariance**: changing `SK_ID_CURR` or `CODE_GENDER` changes nothing;
- **directional**: raising `EXT_SOURCE_2` does not raise risk on average (T9).

The `--sample` run on the 5k fixture finishes in under a minute, for Iyed's CML.

**Done when:**
- 4 models are compared in MLflow and the champion is in the Registry;
- ROC-AUC ≥ 0.75, PR-AUC ≥ 0.25 and recall ≥ 65 % on test, and cost −15 % vs the baseline;
- any target missed is **reported as missed to the team and teacher**, not silently relaxed;
- the Brier score is reported.

### Lot 4 — S5: SHAP, fairness, promotion (Jalon 3) · branch `feature/explainability-fairness` · ~10 h incl. 2 review rounds (promotion is critical)

**Depends on:** lot 3 and D5.

**1. `explain/shap_explainer.py`.** `Explainer(model)` is built once (T3); `explain(row)` returns 3 tuples `(feature, value, contribution)`.

Tests:
- 3 items, sorted by |contribution|;
- names are the original columns (not `onehot__x_y`);
- additivity holds in the explainer's own space;
- the latency bound (T9).

The SHAP summary plot is logged to MLflow.

**2. `fairness/audit.py`.** Acceptance rate (D8), recall and equal-opportunity rate, per `CODE_GENDER` and per age band; max gap in points. `XNA` is excluded explicitly and counted in the report.

Tests:
- a hand-made frame with known gaps;
- a single-group input;
- an empty group.

**3. `models/promote.py`.** The challenger becomes champion iff **all** of these hold on the same fresh held-out set (T7):
- `auc_ch ≥ auc_champ`;
- `auc_ch ≥ 0.75`;
- **gender (D5)**: the acceptance-rate gap (`ACCEPTE` only) between F and M, `XNA` excluded, is ≤ 10 pts, **and** the equal-opportunity gap is ≤ 10 pts;
- **age bands (D5)** (<30, 30–50, ≥50): non-regression only, `eo_gap_age_ch ≤ eo_gap_age_champ + 2` pts. No absolute age gate. Both age gaps are logged at every evaluation.

Otherwise nothing changes, and the exit code and log say why. The thresholds come from `params.yaml` (`promotion` key: 10 pts gender, +2 pts age tolerance).

Tests:
- the 2 roadmap cases;
- each condition failing alone;
- no champion yet;
- equality boundaries.

Stop point: a short review before it is wired into Prefect.

**4. Handover to Iyed** (pairing session): SHAP for `/predict`.

**Done when:** Iyed's API shows the 3 SHAP factors, and `promote.py` decides correctly on all test cases.

### Lot 5 — S6: drift computation (Jalon 4) · branch `feature/drift-detection` · ~10 h incl. review

**Depends on:** lot 3, plus Areej's `DriftSimulator` for the end-to-end check (unit tests do not need it). This lot can run in parallel with lot 4, since they touch different files.

**1. `docs/contracts/04-drift-report.md`,** with Areej and Iyed. It fixes:
- the field types;
- α and the `drift_share` definition (T10);
- the "key" features for the PSI alert: income, credit, the annuity/income ratio, `EXT_SOURCE_2`.

**2. `monitoring/drift.py`.**
- `psi(ref, cur, bins=10)`, written by hand: quantile bins on the reference, deduplicated, with an epsilon for empty bins.
- `ks_test`, `chi2_test`, `drift_share`.
- `compute_drift_report(ref, cur, labels=None) -> dict` (contract 4).
- The alert fires when `drift_share ≥ 0.30`, or PSI ≥ 0.20 on a key feature, or `auc_drop > 0.03`.

**3. Tests.**
- `tests/test_psi.py`: PSI = 0 on identical data; a hand-computed known value; empty bins stay finite; constant and binary columns; NaN handling.
- `test_drift.py`: a shifted normal is detected and the same distribution is not (seeded); the JSON validates against contract 4.

**4. Evidently report.** A `DataDriftPreset` HTML report (pinned version), written to `reports/drift/<date>.html`.

**5. Concept drift.** AUC on a labelled current batch vs the reference AUC.

**Done when:** on Areej's "crise économique" batch, the module flags drift and writes the JSON and the HTML.

### Lot 6 — S7: closing the loop (with Areej) · ~6 h

1. Prediction drift (PSI on the probabilities, shift in acceptance rate) added to the report.
2. Fairness over time: the audit runs on each monitoring batch.
3. The `training_flow` produces a coherent challenger after drift.
4. MLflow screenshots (runs comparison, Registry) for the slides.

**Done when:** automatic promotion is verified on a real case, with the history visible in the Registry.

### Lot 7 — S8: report and slides · ~4 h

**Report sections:**
- MLflow comparison (table + curves);
- the threshold and business cost (3 zones, D3/D8);
- SHAP, including the calibration caveat (T3);
- the fairness audit;
- the drift analysis;
- why aliases rather than Staging/Production (SYNOPSIS L48 still says Staging → Production).

**Slides and rehearsal:** results/KPI slides; rehearse the 1:00 and 2:30 demo slots.

---

## 5. Estimate and order

These are Claude's work hours, reviews included. Not included: Achraf's decision time, team reviews, waiting on Areej and Iyed, and Achraf's own learning time (count about +50 %).

| Lot | Estimate | Order |
|---|---|---|
| 0 | 1 h (human) | now |
| 1 | 8 h | now (catch-up) |
| 2 | 12 h | after D1–D3, D8, D9; skeleton first |
| 3 | 10 h | |
| 4 | 10 h | |
| 5 | 10 h | in parallel with lot 4 |
| 6–7 | 10 h | |
| **Total** | **~61 h** | re-plan with real dates (D7); recalibrate after lot 2 |

## 6. Risks

| Risk | Mitigation |
|---|---|
| Areej's S3 outputs are late | Synthetic frame + local split; the DVC stages are wired when her outputs land |
| The team misses the S3 checkpoint because no model exists | Walking skeleton at the start of lot 2 |
| The fairness gate never passes (age) | Measured in lot 1 (34.5 / 32.9 pts); D5 decided 2026-10-25: absolute gate on gender only, non-regression on age |
| Recall ≥ 65 % is not reached | Recall constraint in the optimizer; D8 |
| Brier ≤ 0.07 is not reached | CV calibration; reported honestly (T6) |
| `poetry.lock` conflicts | One early dependencies PR (T1) |
| `poetry.lock` is 896 KiB, close to the pre-commit 1 000 KB limit (`check-added-large-files`) | Ask Areej to exclude `poetry.lock` from `check-added-large-files` before the next dependency |
| Evidently API churn | Exact version pin; code against that version's docs only |
| SHAP latency | Explainer built once; measured locally |
| Iyed's CI is broken (`cache: poetry`) or skips `feature/*` pushes | Raised with Iyed (D6) |
| Commits not credited to Achraf | D11 check in lot 0 |
