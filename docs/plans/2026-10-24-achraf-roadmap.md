# Achraf's roadmap — Modeling, MLflow, SHAP, fairness, drift

> **For agents:** execute lot by lot, task by task, in TDD. Read `CLAUDE.md` first.
> Do not start a lot whose "depends on" is not met, nor before the decisions it
> lists are answered (section 2).

- **Base read:** branch `dev` = `main` = `0de3860` (2026-10-11); `origin/feat/iyed-serving-cicd` = `91e5e1b` (2026-10-24, CI only, not merged).
- **Owner:** Achraf Gharbi. **Written:** 2026-10-24.
- **Status:** v2. Plan review on 2026-10-24 gave "ready with corrections"; all 4 blocking and 10 important corrections are applied below. Decisions taken 2026-10-24 (section 2). Lot 0 done 2026-10-24 (Python 3.11.9, Poetry 2.5.1, 22 tests green, MinIO up, raw data md5-verified and pushed). Lot 1 implemented 2026-10-24 (see "Lot 1 progress"); review corrections applied 2026-10-25; awaiting team approval of contracts 2/3. D5 decided 2026-10-25 (section 2), age rule anchored to the baseline the same day. Lot 2 implemented 2026-10-25 (see "Lot 2 progress"); `dvc.yaml` train stage deferred until Areej's pipeline exists. Lot 3 implemented 2026-11-24 (see "Lot 3 progress"): champion LightGBM v1 registered; test ROC-AUC, PR-AUC and the −15 % cost KPI missed, reported with their gaps.  Lot 4 implemented 2026-11-24 (see "Lot 4 progress"), with the promotion gate made relative (L4-1).
- **Integration with `dev` (2026-11-24).** The three branches are rebased onto `origin/dev` = `3ed70dc` (Areej's PR #2, DVC pipeline, and PR #3, drift simulator). Consequences:
  - **D2:** Areej's `crediscore.data.featurize` (`fix_days_employed` then `add_derived_features`) is THE shared function; our `data/features.py` is removed. Its float64 cast of the raw inputs (one-row requests with `None`) is ported into her file, in a commit for her review.
  - **D10:** her merged thin-file definition is adopted (at least 2 of 3 `EXT_SOURCE` missing, or `AMT_REQ_CREDIT_BUREAU_YEAR` NaN or 0): 39.7 % of the real train split, against 11.9 % for our old stand-in.
  - **D6:** PRs target `dev` (team practice).
  - Real data: `dvc repro split` produces `data/processed/train.parquet` / `test.parquet` (80 000 / 20 000 rows, every `params.features.all` column); our throwaway prep is gone. Our model-split tests are now `tests/test_model_split.py` (Areej's `tests/test_split.py` tests her split stage).
  - Commit SHAs quoted below predate the rebase (e.g. `af69614` is now `8e4041c`).

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
| Only `application_train` is downloaded (no `bureau.csv`), so thin-file is defined from `EXT_SOURCE` only (2026-11-24: Areej's definition also uses `AMT_REQ_CREDIT_BUREAU_YEAR`, D10) | `download.py:22` |
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
> - **Age bands (<30, 30–50, ≥50): no absolute gate.** Non-regression rule **anchored to the baseline** (refined 2026-10-25, human-approved): The anchor is the LogReg baseline's age equal-opportunity gap, recomputed by `evaluate` on the evaluation set it is given; the challenger's gap must be ≤ that + 2 pts. "Fixed" means tied to the baseline, never to the champion, so it cannot creep. Current value 38.7 pts, so the limit is 40.7 pts (history: 34.9 was the provisional lot 1 limit, on another split). It applies to the first promotion too (no champion needed), and cannot creep 2 pts at every promotion as "≤ champion + 2" would. Both age gaps are published at every evaluation and tracked over time (lot 6).
> - **Why (for the report):** age is a real risk driver in this data (the <30 band defaults about 2× the ≥50 band: 11.4 % vs 5.7 %) and a model feature. The baseline age gap is 34.5 pts raw / 32.9 pts equal opportunity, so an absolute 10-pt rule would block every promotion.
> - Lot 3 adds one MLflow experiment, LightGBM without `AGE_YEARS`, to measure the AUC/cost vs age-gap trade-off.
> - The teacher is told that the SYNOPSIS KPI "gap ≤ 10 pts" now applies to gender, and that age is "measured, explained, without degradation".
>
> **Promotion gate revised 2026-11-24 (L4-1, human-approved "adopt the recommended option"):** the gate is **relative**. The current champion scores test AUC 0.7469, so an absolute "AUC ≥ 0.75" on a fresh batch would block every promotion. Challenger and champion are scored on the same held-out batch, each with its own thresholds; promote iff AUC_ch ≥ AUC_champ, cost_ch ≤ cost_champ, both gender gaps ≤ 10 pts and age EO gap ≤ anchor + 2 pts (anchor = baseline age EO gap read from `reports/metrics.json`; missing = fail closed). No champion: fairness gates only. 0.75 / 0.25 / −15 % stay reported KPIs. Tolerances in `params.yaml` `promotion:` (`max_gender_gap_pts: 10`, `age_gap_margin_pts: 2`).
>
> **Promotion completed 2026-11-24 (lot 4 review, human policy "take the recommended option"):** `--data` is **required**: a fresh labelled batch that is **not the test split** (production or Areej's simulated batch; `test.parquet` only for a manual demo, named explicitly). Fail closed below `promotion.min_rows: 5000` rows (`batch_too_small`) or with fewer than 2 gender groups of `promotion.min_group_rows: 500` rows (`gender_groups_missing`). `thresholds.json` is authoritative (version tags must equal it, else `thresholds_mismatch`). Write order: tags, then delete `challenger`, then move `champion` last; challenger == champion → `already_champion`. Any error → `failed: ["error", "<message>"]`, exit 1.
>
> **Closed 2026-10-24:** D7: no fixed dates; lots run back to back, as fast as quality allows. D11: the commit e-mail is Achraf's and is linked to his GitHub account.

Wording for non-developers in each line. Technical decisions are at the end.

| # | Question | Options | Recommendation | Who | Blocks |
|---|---|---|---|---|---|
| D1 | The data only has "train" and "test". Where do we tune the decision cut-offs and probability calibration without cheating on the test set? | a) set aside part of training data (new param `data.val_size`, e.g. 0.2) · b) ask Areej for a 70/15/15 split · c) cross-validation | **a)**: one new key in params (PR to Areej), no wait | Achraf + Areej | lot 2 |
| D2 | Ratios like "annuity / income" are computed by Areej's data step. At prediction time someone must compute them **exactly the same way**, or the model sees different numbers than in training ("train/serve skew") | a) the model artifact starts with Areej's function, so the API sends raw fields (`DAYS_BIRTH`…); this changes contracts 1, 2 and 3 · b) the API calls the same shared function before the model; the model takes `params.features.all` · c) the client sends already-computed ratios | **b)**: keeps contract 1 and the model input as they are; one shared function, imported by both Areej's step and Iyed's API, plus a test that both paths give identical columns | **Whole team** | lot 2 (contract 2) |
| D3 | With 3 zones (accept / review / refuse), what does one manual review cost, compared to a missed default (5) and a wrongly refused good client (1)? | a) add `cost.cost_review` and optimize both cut-offs · b) a fixed grey band around one cut-off · c) review band sized by analyst capacity (e.g. ≤ 10 % of files) | **a) + a capacity cap (c)**. With review = 0.3 the optimum is "accept if risk < 6 %, refuse if > 70 %": about **half** the applicants would go to review, which is unrealistic. A cap like "≤ 15 % of files to review" keeps it usable. Share of files in review shown with the numbers in lot 1 | Achraf + Areej (params) | lot 2 |
| D4 | Exact decision strings in the API answer | a) `ACCEPTE` / `A_ETUDIER` / `REFUSE` · b) `ACCEPTÉ` / `À ÉTUDIER` / `REFUSÉ` (roadmap and SYNOPSIS) | **a)**: no accents or spaces in a machine code; accents in the report and UI only. Also fix `params.yaml:85` | **Whole team** (contract 3) | lot 1 |
| D5 | Fairness gate for promotion. SYNOPSIS: "acceptance-rate gap ≤ 10 pts". Under-30s default about 2× more than over-50s, and age is itself a model feature, so the age gap may exceed 10 pts at any useful cut-off and **no model would ever be promoted** | a) measure on the baseline first (lot 1), then decide · b) gate on "equal opportunity": among clients who **did** repay, the acceptance gap between groups ≤ 10 pts; report the raw gap too · c) keep the raw gap | **Decided 2026-10-25 (human):** gender, absolute gate, raw ≤ 10 pts **and** equal opportunity ≤ 10 pts (XNA excluded); age, non-regression anchored to the baseline: challenger's equal-opportunity gap ≤ baseline's + 2 pts; the anchor is the LogReg baseline's age equal-opportunity gap, recomputed by `evaluate` on the evaluation set it is given; the challenger's gap must be ≤ that + 2 pts. "Fixed" means tied to the baseline, never to the champion, so it cannot creep. Current value 38.7 pts, so the limit is 40.7 pts (history: 34.9 was the provisional lot 1 limit, on another split); also for the first promotion; both age gaps published and tracked. Tell the teacher | Team | lot 4 |
| D6 | PR target branch | a) `main` · b) `dev`, then release to `main` | ~~a)~~ **b), revised 2026-11-24:** the team merges PRs into `dev` (PR #2, #3); our PRs target `dev`. Ask Iyed to run CI on PRs to `dev`, add `feature/**` to the push trigger and fix `cache: poetry` | Team | first PR |
| D7 | Real dates of S3…S8, Jalon 2, and the defense | — | needed to date the lots and re-plan | Human | planning |
| D8 | With 3 zones, how do the SYNOPSIS KPIs count "to review"? Recall ≥ 65 %: is a default sent to review "detected"? Acceptance rate and fairness: is review "accepted"? | a) review = detected for recall, **not** accepted for acceptance/fairness (prudent reading) · b) review = accepted everywhere · c) KPIs computed on a 2-zone cut (review merged into refuse) | **a)**. The single cost-optimal cut-off (risk 1/6) gives a recall of only ~50 % at AUC 0.76, so the 65 % target needs either a) or a recall constraint in the optimizer (applied anyway) | Team | lot 2 |
| D9 | Where do the cut-offs live? They are an **output** of training; writing them into `params.yaml` (an input of training) creates a DVC loop and two copies that can disagree | a) with the model version (tags + `thresholds.json`); `params.yaml` `threshold` becomes an optional manual override · b) in `params.yaml`, copied by hand | **a)**: PR to Areej to document the override | Achraf + Areej | lot 2 |
| D10 | Thin-file KPI: "accept 2× more good payers than refusing everyone without history" is always met (2 × 0). What is the comparison policy? | a) a "classic scorecard": refuse when `EXT_SOURCE_1` and `EXT_SOURCE_3` are missing, otherwise apply our cut-off · b) keep the literal KPI | **a)**: a meaningful number for the report. **Definition revised 2026-11-24:** thin file = Areej's merged `IS_THIN_FILE` (at least 2 of 3 `EXT_SOURCE` missing, or `AMT_REQ_CREDIT_BUREAU_YEAR` NaN or 0); the scorecard refuses every thin file | Team | lot 3 |
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
| `params.yaml` | Areej | `data.val_size`, `cost.cost_review`, `threshold.max_review_share`, `threshold.min_recall`, `promotion.*`, `monitoring.*` (our drift-detection alert thresholds, lot 5; never under Areej's `drift:` key, which configures her simulator), fix of the label spelling |

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

**Lot 2 progress (2026-10-25, branch `feature/mlflow-training`):**
- [x] Params PR for Areej (separate `(for Areej's review)` commits): `data.val_size`, `cost.cost_review`, `threshold.max_review_share`, `threshold.min_recall` with bounds and tests; `threshold.low/high` documented as an optional manual override (D9); `A_ETUDIER` spelling (D4).
- [x] `data/features.py` for Areej (D2, D10): `add_derived_features`, pure, row-wise, NaN instead of ±inf, `DAYS_EMPLOYED` 365243 → NaN, thin file = `EXT_SOURCE_1` and `_3` missing; 100 % covered. **Superseded 2026-11-24:** removed in favour of Areej's `data/featurize.py` (D2, D10).
- [x] `/models/` git-ignored (root-anchored).
- [x] 3. `tests/conftest.py`: deterministic 2 000-row synthetic frame (8.7 % positives, NaNs in `EXT_SOURCE_*`, learnable signal).
- [x] 4. `models/split.py` (D1), 100 %.
- [x] 5. `models/pipeline.py`: `build_pipeline("logreg" | "lightgbm")` + `calibrate()` (`ensemble=False`), 100 %.
- [x] 6. `models/threshold.py`: `decide`, `zone_metrics`, `optimize_thresholds` (exact search, midpoint cuts, rule A), 100 %. Quick mutation check on the boundaries: 7/7 killed (review cap `<`, recall `<=`, cost-tie `<=`, last argmin, `NEVER` → 1.0, no midpoint, strict `low < high`).
- [x] 7. `models/train.py`: `main()`, artifacts, `--sample` (CML: 4–5 s on 5 000 rows), MLflow logging; 98 % at 10ce7ae (uncovered: the `_git_sha` failure path, the optional-tag assignment and `__main__`); 99 % after the correction pass (only `__main__`).
- [x] 1. Walking skeleton: hand-off for Iyed in contract 2 §9 (snippet checked on a raw row). The API side is Iyed's.
- [x] 2. MLflow service for Iyed (T8): `deploy/mlflow/Dockerfile` (mlflow 3.17.2 + boto3 1.43.117), compose service `mlflow` (SQLite volume, `--artifacts-destination s3://mlflow`, healthcheck), `mlflow` bucket in `minio-init`, `.env.example`.
- [ ] 8. `dvc.yaml` stage `train`: **deferred**, Areej's `dvc.yaml` and `split` outputs do not exist yet. Until then a throwaway prep script (not committed) builds `data/processed/{train,test}.parquet` like the notebook.
- [ ] The API serves the skeleton (Iyed). Adversarial review (≥ 30 mutants) and walkthrough.
- Checks: black, flake8, mypy strict green; **176 tests** at 10ce7ae, **225** after the correction pass; total coverage 94 % → 95 % (`download.py` 64 % is Areej's, unchanged).

**Measured (real data, 100k stratified sample, 80k train → 64k fit / 16k val, seed 42; validation only, test untouched):**

| | LogReg (baseline) | LightGBM |
|---|---|---|
| ROC-AUC / PR-AUC / Brier | 0.7385 / 0.2133 / 0.0689 | **0.7535 / 0.2400 / 0.0678** |
| Distinct calibrated probabilities | 69 | 63 |
| Thresholds low / high (midpoints) | 0.08981 / 0.12321 | 0.08559 / 0.13261 |
| Recall (D8) / review share | 66.2 % / 14.2 % | 67.0 % / 14.4 % |
| Cost per file (accept all: 0.4037) | 0.3329 | **0.3179** (−4.5 % vs LogReg on val) |
| MLflow run id | `269f4acf43a646f5b09e74f6dcc3de7f` | `a6cea7a26be8445884bdaf3e2086c288` |

- LogReg reproduces the notebook validation numbers exactly (same split, same cost 0.3329).
- LightGBM (default-ish parameters, no tuning) is −4.5 % on cost: far from −15 %. Lot 3 (Optuna, XGBoost) has to close the gap, on the test split.
- Proof: REST API lists both runs FINISHED with the metrics, tags (`threshold_low/high` in full precision, `shap_unit`, raw md5, then named `data_hash`, renamed `raw_data_hash` in the correction pass), curves and JSON artifacts; objects are in MinIO bucket `mlflow`; both logged models reload through MLflow with identical probabilities to the local `model.joblib`.

**Deviations and why:**
- `optimize_thresholds` returns `low ≤ high`, not strictly `<`: the notebook, contract 2 (`low = high` = 2 zones) and the "accept all" candidate required by the round-2 review all need it. A strict search would also make `max_review_share = 0` infeasible. Infeasibility can then only come from a constraint outside [0, 1].
- Rule A (coordinator): cuts at the midpoint below the chosen step. `NEVER` (just above 1.0) is not a step and stays as is: halfway between 1.0 and the next float rounds back to 1.0 (found by a test with a proba of exactly 1.0).
- Found on the real run: parquet and JSON give `None` for missing categoricals, which `SimpleImputer` does not impute (it became a one-hot category). Fixed inside the pipeline (`none_to_nan` step), so the API gets it too; test added.
- Found on the real run: MLflow prints an emoji, which crashed on a redirected cp1252 Windows stdout (as under `dvc repro`). `main()` now replaces unencodable characters.
- Host-side `mlflow.sklearn.load_model` hung: the server advertises multipart downloads with presigned URLs for `http://minio:9000`. Host clients set `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` (`.env.example`, for Iyed's review).
- `joblib` added to the mypy untyped list (`pyproject.toml`, shared).
- LightGBM hyperparameters are fixed in `pipeline.py` until Optuna (lot 3). Both preprocessing branches are identical for both models (scaling is harmless for trees).
- The thresholds are run tags and logged-model tags (MLflow 3 `LoggedModel`); they become model-version tags at registration (lot 3).
- `train` does not apply the `params.threshold.low/high` override: it is a serving-time override (contract 2 §5).
- `docker compose -p mlops-project-2026-groupea up -d --build mlflow` from this worktree recreated the MinIO container (its config hash changed); data is on the named volume and was checked intact (158 MiB in `dvc-store`).

- **CML numbers on the 5k fixture look poor by design.** With 1 000 validation rows, the isotonic map has very few, coarse steps, and `min_recall = 0.65` must still be met: LogReg then costs **0.546 per file vs 0.405 for "accept all"** (LightGBM 0.425). This is not a bug. CML on the fixture checks that training runs and gives a plausible AUC, not the business cost; the cost KPI is read on the full split.

**Lot 2 correction pass (2026-10-25, review of 10ce7ae: "ready with corrections", no blocking item):**
- [x] Important 1, compose (Iyed): `--allowed-hosts "mlflow:5000,localhost:${MLFLOW_PORT:-5000},127.0.0.1:${MLFLOW_PORT:-5000}"`. Container recreated with `-p mlops-project-2026-groupea`. Proven: a container on the compose network loads both logged models (`http://mlflow:5000`, same probabilities as `model.joblib`, no extra env var); the host gets 200; an unlisted Host still gets 403. `dvc status -c` from the main copy: "Cache and remote 'minio' are in sync". Contract 2 §9 corrected (containers: allowed-hosts; host: `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`).
- [x] Important 2, `features.py` (Areej): the 8 raw inputs are cast to float64 first; a one-row request with `DAYS_BIRTH`/`DAYS_EMPLOYED` None no longer raises, and the pandas FutureWarning is gone. The reviewer's serving probe now serves 200/200 raw rows with None.
- [x] Minor 3–6, `threshold.py`: lowest-step cut = `cut / 2`; a midpoint that rounds onto the lower step (neighbouring doubles) keeps the cut; proba and thresholds must be finite and in [0, 1] (`NEVER` allowed) in `decide` and `optimize_thresholds`; one message per input check; both internal guards reached by monkeypatch tests (no `pragma`). Split index and `-inf` tests added. The baseline notebook uses the same midpoint rule (re-executed, outputs byte-identical).
- [x] Minor 7: run tags `raw_data_hash` (DVC md5 of the raw CSV) and `train_data_hash` (md5 of the file read); `git_sha` tested.
- [x] Minor 8 (Areej): the `params.yaml` override accepts `low == high`.
- [x] Minor 9: decisions doc (no separate uncalibrated artifact), contract 2 §2 (fit part = 80 % of train), §3 (parity test pending), §7 (tags), this coverage note.
- [x] Minor 10 (Iyed): `${MLFLOW_PORT:-5000}`, `MLFLOW_PORT` in `.env.example`, compose comment on the MinIO root account (local use only).
- [x] Minor 11: test of the cp1252 fix (fails without it). Commit `af69614` has a wrong subject (it adds the hash/git_sha tag test, not a split test); recorded in the next commit, history not rewritten.
- [x] Minor 12: CML note above.
- ~~Mutation replay: 78 run, 78 killed~~ **Withdrawn, did not reproduce.** In the `git archive` copy the old `git_sha` test failed on its own (no `.git`), so several "kills" were that failure, not the mutant. Rule since then: run the unmutated suite in the copy first, and quote mutation numbers only from a log.
- **Probes replayed:** `probe_threshold` (its brute force uses a 2.0 cut, now rejected by design: replayed with `NEVER`): 0 mismatches in 300 random cases, adjacent-float steps now resolved, float32 and JSON round-trips exact. `probe_lowest_step`: cut 0.005, float32 gives the same decisions. `probe_serving`: 200/200; a column absent from the request (not None) still raises `KeyError`. The API must send every column, with None if missing (contract 3, Iyed). `probe_mlflow`: contract-2 run, new tags present, reload identical.

**Lot 2 status: reviewed "ready" (2026-10-25) after two review rounds and a final minor pass.**
- Reviewer mutation counts: round 1, 63 mutants, 56 killed / 7 survived; round 2, 87 run, 79 killed / 8 survived (4 equivalent, 4 fixed in the final pass).
- Final minor pass: the `git_sha` test uses throwaway repositories (it failed outside a checkout); survivors #66 (accept-all keeps `NEVER`), #88 (every raw input cast to float64, warnings as errors), #90 (`--sample` tags the md5 of the sample file) and #93 (accept-all with probabilities of exactly 1.0) are killed. Contract 2 §3/§9 and the contract-3 proposal: every raw column must be present, the API fills an absent field with `None`. §5: float32 caveat for neighbouring doubles. Compose header shows `${MLFLOW_PORT:-5000}`.
- Replay of `mutate2.py` on a `git archive` copy of the final pass (unmutated suite first: 227 passed with no `.git`): **87 run, 83 killed, 4 survived, 9 skipped** (old mutants whose line changed). The 4 survivors are equivalent: #19 and #70 (the same checks with the same messages run again in `business_cost` / `decide`), #82 (`size` vs `shape` once both are 1-D), #61 (EXT_SOURCE column order). Log: `scratchpad/lot2-fix2/mutation3.log` of the session.
- Checks: 227 tests, coverage 95 % (new modules 100 %, `train.py` 99 %: `__main__`).

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

**Lot 3 coordinator decisions (final):**
- **L3-1 selection:** candidates need validation ROC-AUC ≥ 0.75 if any reaches it; the best has the lowest validation cost per file under the D3/D8 constraints; ties are broken by validation AUC. The test split is never used for selection. By design, an experiment-tagged winner aborts registration (`--allow-experiment` forces it).
- **L3-2 Optuna:** 20–30 trials each for LightGBM and XGBoost, 5-fold stratified CV ROC-AUC on the fit part, TPE seeded, nested runs.
- **L3-3 registry:** `crediscore`; `champion` if none exists, else `challenger`; tags as in contract 2 §7. No promotion logic (lot 4).
- **L3-4 baseline:** `evaluate` compares the model with a LogReg baseline trained identically on the same train.parquet, on test. KPI = (cost_model − cost_base) / cost_base.
- **L3-5 no_age:** an extra LightGBM run without `AGE_YEARS`, tag `experiment=no_age`.
- **L3-6 honesty:** missed targets are reported with their gap; a target is never relaxed and nothing is tuned on test.
- **Added 2026-11-24 (review of `a3e3db0`):**
  - a run trained with `--exclude` is an experiment: `train --exclude` requires `--experiment`, and `registry` treats any run tagged `excluded_features` as an experiment;
  - a candidate must have been logged with the costs, constraints and `val_size` of the current `params.yaml`; the data-hash and top-level filters run in the MLflow query;
  - D5 age anchor wording (product clarification): the anchor is the LogReg baseline's age equal-opportunity gap, recomputed by `evaluate` on the evaluation set it is given; "fixed" means tied to the baseline, never to the champion, so it cannot creep; current value 38.7 pts, limit 40.7 pts;
  - thin-file KPI (product decision, recommended option): the literal SYNOPSIS KPI is the pass flag in `kpi.targets.thin_file`; next to it, the thin good-payer acceptance rate and the default rate among accepted thin files, champion vs baseline; the whole-portfolio ratio is informative only.

**Lot 3 progress (2026-11-24, branch `feature/model-registry`, on `origin/dev` with Areej's pipeline):**
- [x] 1. `random_forest` and `xgboost` in `build_pipeline` (test `26dedfb`, code `51075eb`).
- [x] 2. Optuna (`models/tune.py`, `train --tune N`): seeded TPE, 5-fold stratified CV ROC-AUC on the fit part, one nested MLflow run per trial (`df0c5d1`, `df3b280`).
- [x] 2b. D5 experiment: `train --exclude AGE_YEARS --experiment no_age --model-params` (same commits); real run below.
- [x] 3. `models/evaluate.py` (`5cb81df`, `b107283`); `reports/metrics.json` committed as a DVC metric.
- [x] 4. `models/registry.py`: L3-1 selection + registration, `champion` / `challenger` (`bddbe66`, `f6680ab`).
- [x] 5. Behaviour tests (`b49f36e`). CML `--sample` on the 5k fixture: under a minute (4–5 s, measured in lot 2).
- [x] 6. `dvc.yaml` stages `train` and `evaluate` (this pass, test committed first). `train` = LightGBM with the tuned params of the selected run, frozen in `configs/champion_model_params.json`; each stage lists only the modules and params keys it reads; `train` never depends on `test.parquet` (tested). `dvc repro evaluate` ran end to end: same validation numbers as the MLflow run, and `models/model.joblib` gives **exactly** the probabilities of `models:/crediscore@champion` (max diff 0.0 on 20 000 rows).
- [x] 7. Real runs on the MLflow server, registration, evaluate on test, REST checks (below).
- [ ] Adversarial review by a fresh agent, and walkthrough.
- Checks: black, flake8, mypy strict green; **336 tests** (326 before this pass), total coverage 92 % (models: `evaluate` 99 %, `registry` 99 %, `train` 99 %, `tune` 100 %; the misses are the `__main__` lines, plus two guards in `train.py` that no test reaches: `log_to_mlflow` without an active run and `--tune` on a model without a search space).

**Measured, validation (16 000 rows carved from Areej's `train.parquet`, md5 `d0178dc…`; test not read for selection):**

| Run (experiment `crediscore`) | ROC-AUC | PR-AUC | Brier | Recall (D8) | Review | Cost/file | Run id |
|---|---|---|---|---|---|---|---|
| logreg | 0.7558 | 0.2382 | 0.0680 | 71.9 % | 14.8 % | 0.3395 | `1a16047475464bd4a5881b7ddb4df3a8` |
| random_forest (300 trees, leaf 20, fixed) | 0.7480 | 0.2337 | 0.0682 | 69.3 % | 15.0 % | 0.3326 | `bc029f926b51446fa2293f16f0edb61c` |
| **lightgbm, Optuna 25 trials (CV 0.7519)** | **0.7670** | **0.2574** | **0.0670** | 69.7 % | 14.9 % | **0.3070** | `1c3f3b8712aa43d491ec3882bf64e70e` |
| xgboost, Optuna 25 trials (CV 0.7512) | 0.7648 | 0.2543 | 0.0671 | 71.0 % | 11.4 % | 0.3428 | `2f5b60577b3c4081b6df59b1f43f4b9e` |
| lightgbm-no_age (D5, tuned LightGBM params) | 0.7649 | 0.2507 | 0.0672 | 69.4 % | 14.0 % | 0.3136 | `be2f0c6472d441d19e93443e9bf5b2a8` |

Accept all: 0.4037 per file. Each tuned run has its 25 nested trial runs (checked through the REST API).

- **Selection (L3-1):** every run meets the D3/D8 constraints; the 0.75 AUC floor drops random_forest; the lowest cost is LightGBM. **Registered `crediscore` version 1, alias `champion`**, from run `1c3f3b87…`, with the contract 2 §7 tags. REST: `GET registered-models/alias?name=crediscore&alias=champion` → v1, READY; `models:/crediscore@champion` reloads on the host with the same probabilities as the local artifacts.
- The 7 runs of 2026-10-25 (lot 2 and an earlier lot 3 attempt) were trained on the throwaway prep (other `train_data_hash`): they stay on the server and the selection ignores them.

**Measured, test (20 000 rows, `evaluate`; baseline LogReg retrained identically on the same train.parquet; `reports/metrics.json`):**

| | Baseline LogReg | Champion LightGBM v1 | Target | Status |
|---|---|---|---|---|
| ROC-AUC | 0.7346 | 0.7469 | ≥ 0.75 | **missed by 0.003** |
| PR-AUC | 0.2079 | 0.2267 | ≥ 0.25 | **missed by 0.023** |
| Brier | 0.0692 | 0.0685 | ≤ 0.07 | met (0.0015 margin) |
| Recall (D8) | 69.4 % | 66.9 % | ≥ 65 % | met (+1.9 pt) |
| Review share | 14.4 % | 14.5 % | (≤ 15 % on validation) | |
| Cost per file | 0.3519 | 0.3203 | −15 % vs baseline | **missed: −9.0 %** (6.0 pts short; −15 % needs ≤ 0.2991) |
| Cost vs accept all (0.4037) | −12.8 % | −20.7 % | | |
| Gender gap raw / EO (D5) | 8.4 / 7.0 | 7.7 / 6.3 | ≤ 10 / ≤ 10 | met |
| Age gap raw / EO (D5) | 39.7 / 38.7 | 34.6 / 33.1 | EO ≤ baseline + 2 = 40.7 | met |

- **KPI status:** recall and Brier met; ROC-AUC, PR-AUC and the −15 % cost KPI missed, with the gaps above. Nothing was relaxed and nothing was tuned on test (L3-6). The three models evaluated on test all lose about 0.02 AUC from validation (LogReg 0.756 → 0.735, LightGBM 0.767 → 0.747, no_age 0.765 → 0.746): the drop is shared, not specific to the selected model. To be reported to the team and the teacher.
- **D5 anchor (clarified 2026-11-24):** the anchor is the LogReg baseline's age equal-opportunity gap, recomputed by `evaluate` on the evaluation set it is given; the challenger's gap must be ≤ that + 2 pts. "Fixed" means tied to the baseline, never to the champion, so it cannot creep. Current value 38.7 pts, so the limit is 40.7 pts (history: 34.9 was the provisional lot 1 limit, on another split). Contract 2 §1 and the decisions doc carry the same wording.

**D5 experiment (LightGBM without `AGE_YEARS`, same tuned params, measured on test after the selection):**

| | With `AGE_YEARS` (v1) | Without | Baseline |
|---|---|---|---|
| ROC-AUC / PR-AUC | 0.7469 / 0.2267 | 0.7463 / 0.2251 | 0.7346 / 0.2079 |
| Cost per file (vs baseline) | 0.3203 (−9.0 %) | 0.3267 (−7.1 %) | 0.3519 |
| Age gap raw / EO (pts) | 34.6 / 33.1 | **37.1 / 35.6** | 39.7 / 38.7 |
| Gender gap raw / EO (pts) | 7.7 / 6.3 | 8.3 / 6.8 | 8.4 / 7.0 |
| Acceptance <30 / 30–50 / ≥50 | 46.9 / 63.9 / 81.5 % | 42.9 / 65.9 / 80.0 % | 37.3 / 61.0 / 76.9 % |

- Removing age does **not** reduce the age gap: it grows by 2.5 pts (EO), under-30 acceptance drops by 4 pts, and the cost rises. A likely reading (not proven here): age stays proxied by other features (employment years, `DAYS_*`, family), while the model loses the direct signal that separates good young applicants. Keeping `AGE_YEARS` is both cheaper and fairer on age; the report should say so.

**Thin file (D10, Areej's definition):**
- 39.4 % of the test split is thin file (39.7 % of train; 11.9 % under our old stand-in). The segment is no longer a small "no external score" niche but two applicants in five: at least 2 missing `EXT_SOURCE` **or** no bureau enquiry recorded for the last year (`AMT_REQ_CREDIT_BUREAU_YEAR` 0 or NaN).
- **SYNOPSIS KPI, literal (the pass flag, `kpi.targets.thin_file`): met.** The champion accepts **4 946 thin good payers vs 0** for "refuse without history" (target 2 × 0, and more than none).
- Companions (`kpi.targets.thin_file.companion`), champion vs baseline:
  - thin good payers accepted: **68.4 % vs 61.8 %** of them (+6.6 pts);
  - default rate among accepted thin files: **3.96 % vs 3.79 %** (+0.17 pt). The champion accepts more thin files (65.4 % vs 58.9 % of the segment), and their loss rate rises very slightly: "sans augmenter les pertes" holds within 0.2 pt, not strictly. To be said in the report.
- Champion thin-file ROC-AUC 0.755 (higher than on the whole test).
- Informative only: whole-portfolio ratio of good payers accepted vs the scorecard (refuse every thin file, our decision elsewhere) 12 928 / 7 982 = **1.62** (baseline 1.59); the scorecard costs 0.553 per file vs 0.320. With 39 % of files thin, ×2 on the whole portfolio is out of reach by construction, so it is not a target.

**Lot 3 mutation check (own, before the adversarial review):** 49 mutants on `evaluate.py`, `registry.py`, `tune.py` and the `train --tune/--exclude` path, on a `git archive` copy of `9ec520c` (no `.git`; unmutated suite first: 323 passed, 8 skipped for absent processed data). **49 run, 40 killed, 9 survived.** The 9 survivors were real test gaps, now covered: accept-all with a proba of 1.0 (#14); baseline model name, baseline thresholds and the class-1 column in `evaluate.main` (#24–26); nested trial runs and the candidate field mapping in `registry` (#34, #40); CV folds and `scale_pos_weight` in `tune` (#41, #43); the excluded feature list passed to `tune` (#48). Replay of the 9 with the new tests (unmutated selected suite first: 76 passed): **9 run, 9 killed**. Logs: `scratchpad/lot3/mutation.log` and `scratchpad/lot3/mutation_replay.log` of the session.

**Deviations and why:**
- The `train` stage trains a fixed champion config (model + frozen Optuna params in `configs/`) instead of tuning at every `dvc repro`: tuning is a one-off MLflow experiment, and the stage must stay reproducible and fast (about 1 min). Changing the champion config is a reviewed commit on that file.
- `dvc.lock` is committed with the `train` and `evaluate` entries only. Our `train` deps record `train.parquet` md5 `d0178dc…` (what Areej's `split` writes today), while the committed `split` entry still says `d75701c…`: `featurize.py` changed in the float64-cast commit for Areej's review, and her lock entries are hers to refresh. `dvc repro` also rewrote `tests/fixtures/ci_sample.parquet` and `reports/validation.json` (pyarrow version / line endings): restored with `git checkout`.
- `evaluate` was also run on the `no_age` model, for the D5 measurement only (output kept out of the repo). The champion had already been chosen on validation.

**Lot 3 correction pass (2026-11-24, review of `a3e3db0`: "ready with corrections", no blocking item):**
- [x] Important 1: `train --exclude` requires `--experiment`; `registry` treats a run tagged `excluded_features` as an experiment (`excluded:<cols>`), so it never registers silently.
- [x] Important 2: one D5 anchor wording in the decisions doc, this plan and contract 2 §1; 34.9 kept only as a history note.
- [x] Important 3: "Lot 3 coordinator decisions" block above.
- [x] Important 4: thin-file KPI in `kpi.targets.thin_file` (literal flag + companions); `reports/metrics.json` regenerated with `dvc repro -s evaluate`; numbers in "Thin file" above and in contract 2 §8.
- [x] Important 5 and minors: tests kill the reviewer's survivors #7, #17, #28, #29, #43 (the registry test store is now a throwaway git repo) and #54/#55 (search spaces pinned).
- [x] Minor: the MLflow query filters on `train_data_hash` and on top-level runs (`mlflow.parentRunId IS NULL`); candidates must carry the `params.yaml` costs, constraints and `val_size` (`run_config`). Read-only check on the real server: same champion, `1c3f3b87…`.
- [x] Minor: `train` and `evaluate` depend on `config.py`, `evaluate` also on `tune.py` (tested against each stage's import graph); relocked, data-stage entries untouched.
- [x] Minor: `configs/` added to Achraf's row in `CLAUDE.md`; corrupted coverage line fixed.
- Kept as is: taking `GENDER_COL` from `params.sensitive_cols` in `evaluate.py` is not a one-liner (`fairness` would need a new argument at every call), so the ponytail note stays until lot 4's `promotion.*` params.
- Found during the pass: `evaluate` wrote `reports/metrics.json` with CRLF on Windows, so the md5 in `dvc.lock` never matched the LF file Git checks out (`eol=lf`), and `dvc status` showed `evaluate` as modified after a commit. Fixed with `newline="\n"` (test committed first) and relocked.
- Also found: with the new `--exclude` guard, the old unknown-column test exited on that guard instead (reviewer mutant #62 survived). The test now passes `--experiment`.
- **Reviewer's `mutate.py` replayed** on a `git archive` copy of `3a06add`. Unmutated suites first: full suite 340 passed, 8 skipped; reviewer test list 118 passed. Result: **65 mutants, 61 killed, 1 survived (#62, fixed above), 3 skipped** (#33, #40, #41: their lines changed). Follow-up run with the fixed test: #62 killed; the 3 skipped mutants, adapted to the new lines, killed; 6 new mutants on the new code killed (10 of 10). Final checks: black, flake8, mypy strict green; 348 tests; coverage 93 %. Logs: `scratchpad/lot3/reviewer_replay.log` and `scratchpad/lot3/reviewer_replay_extra.log` of the session.

**Review round 2: ready; 86 mutants, 81 killed, 5 survivors (2 equivalent, 3 covered by these tests)**: `run_config` with `test_size != val_size` (#69), `registry --select` with another `--params` file (#72), the thin-file portfolio ratio companion taken from the model, not the baseline (#84).

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

**3. `models/promote.py`.** *(Revised 2026-11-24, L4-1: relative gate. `auc_ch ≥ 0.75` is dropped and `cost_ch ≤ cost_champ` is added; the age anchor is read from `reports/metrics.json` (38.7 pts, limit 40.7) and a missing anchor fails closed; exit 0 = promoted, 1 = not; `reports/promotion.json`; `--dry-run`. The text below is the original version.)* The challenger becomes champion iff **all** of these hold on the same fresh held-out set (T7):
- `auc_ch ≥ auc_champ`;
- ~~`auc_ch ≥ 0.75`~~ (removed 2026-11-24);
- **gender (D5)**: the acceptance-rate gap (`ACCEPTE` only) between F and M, `XNA` excluded, is ≤ 10 pts, **and** the equal-opportunity gap is ≤ 10 pts;
- **age bands (D5)** (<30, 30–50, ≥50): non-regression anchored to the baseline, `eo_gap_age_ch ≤ eo_gap_age_baseline + 2` pts: the anchor is the LogReg baseline's age equal-opportunity gap, recomputed by `evaluate` on the evaluation set it is given; the challenger's gap must be ≤ that + 2 pts. "Fixed" means tied to the baseline, never to the champion, so it cannot creep. Current value 38.7 pts, so the limit is 40.7 pts (history: 34.9 was the provisional lot 1 limit, on another split). It applies to the first promotion too. No absolute age gate. Both age gaps are logged at every evaluation.

Otherwise nothing changes, and the exit code and log say why. The thresholds come from `params.yaml` (`promotion` key: 10 pts gender, +2 pts age tolerance).

Tests:
- the 2 roadmap cases;
- each condition failing alone;
- no champion yet;
- equality boundaries.

Stop point: a short review before it is wired into Prefect.

**4. Handover to Iyed** (pairing session): SHAP for `/predict`.

**Done when:** Iyed's API shows the 3 SHAP factors, and `promote.py` decides correctly on all test cases.

**Lot 4 progress (2026-11-24, branch `feature/explainability-fairness`, on `a3e3db0`):**
- [x] Params (L4-1, `(for Areej's review)`): `promotion.max_gender_gap_pts: 10`, `promotion.age_gap_margin_pts: 2`, `PromotionConfig` with bounds [0, 100], tests first.
- [x] 2. `fairness/audit.py` (L4-2): acceptance (ACCEPTE only), equal-opportunity and recall (D8) rates per `CODE_GENDER` (XNA and missing excluded, counted) and per age band; max gaps in points; `d5_checks` (value, limit, pass; gaps compared at 1e-9 pts so `100 * (0.4 - 0.3)` is still 10). `evaluate` now takes its fairness numbers from the audit and its D5 tolerances from `params.promotion`; `reports/metrics.json` is **byte-identical**: `dvc repro -s evaluate` writes md5 `6382b250…`, the lot 3 lock value (the git blob differs only by CRLF). The `evaluate` stage now depends on `fairness/audit.py` and `promotion`; `dvc.lock` relocked (evaluate entry only).
- [x] 1. `explain/shap_explainer.py` (L4-3): `Explainer(model)` built once; `explain(frame)` → 3 `Factor(feature, value, contribution)` per row on original columns; `contributions()` for additivity; units per contract 2 §6. Summary plot (bar of mean |contribution|, 2 000 train rows) in `reports/figures/shap_summary.png` and on the user's MLflow: run `a146f2b6842d42adac5da1979fa5d0cd`, tag `kind=shap_summary`, `model_run_id` = champion run `1c3f3b87…`. Top features: EXT_SOURCE_3, EXT_SOURCE_2, EXT_SOURCE_1, AMT_GOODS_PRICE, CREDIT_TERM.
- [x] SHAP latency, local, champion v1, 200 real test rows one at a time: `explain` p50 13.1 ms, **p95 14.9 ms** (target < 50 ms); `predict_proba` + `explain` p95 33.2 ms; build 0.35 s. CI: `slow`-marked test, p95 < 500 ms.
- [x] 3. `models/promote.py` (L4-1, relative gate, see the decisions block). Real checks: `--dry-run` on the user's server → `no_challenger`, registry identical before/after (REST). Throwaway server (sqlite, port 5321, stopped): v1 = champion artifact, v2 = no_age as challenger → **not promoted, failed `auc` (0.7463 < 0.7469) and `cost` (0.3267 > 0.3203)**; gender 8.3 / 6.8 pts and age EO 35.6 ≤ 40.7 pass; exit 1; aliases unchanged. The champion's batch scores equal `reports/metrics.json`. Then the champion run re-registered as v3 challenger → promoted (equal AUC and cost), `champion` → v3, `challenger` removed, `promoted_at` and `promotion_*` tags set, exit 0.
- [x] 4. Hand-off to Iyed: contract 2 §10 (French); snippet run on raw test row `SK_ID_CURR 100295` with `OCCUPATION_TYPE = None`: `A_ETUDIER`, proba 0.0978, top factors EXT_SOURCE_2 (+0.40), EXT_SOURCE_1 (+0.21), REGION_POPULATION_RELATIVE (−0.16).
- [ ] Adversarial review by a fresh agent (promotion is critical: two rounds) and walkthrough. Stop point before the Prefect wiring (Areej).
- Checks: black, flake8, mypy strict green; **412 tests** (336 before); coverage 94 %; `audit` 100 %, `shap_explainer` 99 %, `promote` 99 %, `evaluate` 99 % (the misses are `__main__` lines).

**Lot 4 mutation check (own):** 54 mutants on `audit.py`, `shap_explainer.py` and `promote.py`, on a `git archive` copy of `73cfdca` (`PYTHONPATH` checked to import the copy; unmutated suite first: exit 0). **54 run, 51 killed, 3 survived.** #1 (`len(rates) > 0` in `gap_pts`) is equivalent (one group gives 0 either way). #7 (recall gap computed from the raw rates) and #49 (promote's audit with `high` as `low`) were real gaps, now covered; replay with the new tests (unmutated selected suite first: exit 0): **2 run, 2 killed**. Logs: `scratchpad/lot4/mutation.log` and `scratchpad/lot4/mutation_replay.log` of the session.

**Deviations and why:**
- The gate is relative (L4-1, human-approved): AUC 0.75 and −15 % cost are KPIs, not gates; cost ≤ champion's is added.
- SHAP base value is computed from the model, for robustness: the classifier's raw margin on an all-zero row (LightGBM `booster_.predict(raw_score=True)`, XGBoost `get_booster().predict(output_margin=True)`, LogReg `decision_function`; RF in probability) minus the SHAP sum there. *Corrected by the review:* shap 0.51's XGBoost 3 `expected_value` offset is not a fixed 0.04: it varies with the number of trees (+0.009 / −0.023 log-odds measured by the reviewer) and is 0 with `scale_pos_weight`, the production path. The first version took the logit of `predict_proba` (float32 for XGBoost); the review replaced it by the raw margins.
- LogReg reference without background: training mean for scaled numerics, "no category" for one-hot (no training data needed at API startup); a `background` frame can be passed.
- The summary plot is a matplotlib bar chart of mean |contribution| on original columns (a beeswarm cannot colour categorical columns); computed on train rows, not test.
- `evaluate.registry_version` extracted (resolves a `models:/` URI) and reused by the explainer CLI.
- The audit also publishes per-group recall; `evaluate` keeps only its previous keys, so `metrics.json` is unchanged.
- ~~A batch with a single gender group gives a gap of 0 and passes~~: fixed in the correction pass (`gender_groups_missing`).

**Lot 4 coordinator decisions (all final; product ones follow the human's standing policy "take the recommended option"):**

| # | Decision | Date |
|---|---|---|
| L4-1 | Relative promotion gate on the same batch, own thresholds: AUC ≥ champion's, cost ≤ champion's, gender raw and EO gaps ≤ 10 pts, age EO gap ≤ anchor + 2 pts (anchor from `reports/metrics.json`, missing = fail closed); no champion = fairness only; exit 0/1, `reports/promotion.json`; tolerances under `params.promotion` | 2026-11-24 |
| L4-2 | `fairness/audit.py` is the single home of the fairness numbers; `evaluate` refactored with `metrics.json` byte-identical | 2026-11-24 |
| L4-3 | `explain/shap_explainer.py`: `Explainer` built once, top 3 factors on original columns, units per contract 2 §6, summary plot logged to MLflow (`kind=shap_summary`) | 2026-11-24 |
| L4-4 | Hand-off to Iyed in contract 2 §10 (French), snippet run on a raw test row | 2026-11-24 |
| L4-5 | Out of scope: Prefect wiring (Areej), API code (Iyed) | 2026-11-24 |
| I1 | Near-atomic promotion: tags (incl. `promotion_previous_champion`), then delete `challenger`, then move `champion` last; challenger == champion → `already_champion`, exit 1 | 2026-11-24 (review) |
| I2 | `promotion.min_rows: 5000` (`batch_too_small`) and `promotion.min_group_rows: 500` with at least 2 gender groups (`gender_groups_missing`), both fail closed | 2026-11-24 (review) |
| I3 | `--data` required; promotion runs on a fresh batch, never the test split (except a named manual demo) | 2026-11-24 (review) |
| I4 | `evaluate` stage relocked at HEAD; it lists only `promotion.max_gender_gap_pts` and `promotion.age_gap_margin_pts` | 2026-11-24 (review) |
| m6 | `thresholds.json` is authoritative; version tags must match (`thresholds_mismatch`) | 2026-11-24 (review) |
| m5 | Any error rewrites `reports/promotion.json` with `failed: ["error", "<message>"]`; the file is git-ignored | 2026-11-24 (review) |

**Lot 4 correction pass (2026-11-25, review of `c92feca`: "ready with corrections", nothing blocking):**
- [x] I1: `apply_promotion` writes the tags (incl. `promotion_previous_champion`), deletes `challenger`, moves `champion` last; `already_champion` no-op. Tests: call order spied; a client raising on `delete_registered_model_alias` or on `set_registered_model_alias` leaves `champion` on v1, and the re-run records previous champion `1` (promoted after the first, `no_challenger` after the second).
- [x] I2 (for Areej's review): `promotion.min_rows: 5000`, `promotion.min_group_rows: 500`, both > 0 and `min_rows ≥ 2 × min_group_rows`; `batch_too_small`, `gender_groups_missing` (XNA not counted), tested.
- [x] I3: `--data` required (argparse exit 2 without it); docstring, contract §1 and the decisions block say: fresh batch, not the test split.
- [x] I4: `dvc repro -s evaluate` at HEAD: `metrics.json` md5 `6382b250…` unchanged; the evaluate stage lists only `promotion.max_gender_gap_pts` and `promotion.age_gap_margin_pts`; only the evaluate lock entry changed.
- [x] Minor 1–10: `registry_version` alias test (v2 under `@champion`); LogReg contributions = `coef × x_transformed`, base = intercept; base from the raw margin (LightGBM `booster_`, XGBoost `output_margin`, LogReg `decision_function`, RF probability), additivity against raw margins for the 4 models and a test that `predict_proba` is never used for a log-odds base; XGBoost offset wording fixed (plan, contract §6, code comment); fatal errors rewrite the decision file (stale-file test); `thresholds_mismatch` (tags differ, absent or unreadable); age anchor rejects bool, NaN, ±inf; `/reports/promotion.json` git-ignored (shared file); contract §10 sign wording (non-decreasing `proba`) and the exact raw request columns (23 numeric + `DAYS_BIRTH`, `DAYS_EMPLOYED` + 10 categorical).
- Found while replaying: the new `registry_version` test failed when run after other MLflow tests (stale active experiment id); fixed by setting its own experiment (`5fba2ff`). The reviewer's survivor #46 (`value.item()` branch) was dead code: `to_numpy(dtype=object)` already yields plain Python values; branch removed (`aa14056`).
- Serving snippet re-run after the base change: same output (`A_ETUDIER`, 0.0978, same 3 factors).
- **Reviewer probes replayed** on `git archive` copies (originals copied, not modified): `probe_shap.py`: additivity vs raw margin 1.8e-15 (LogReg), 1.7e-14 (LightGBM), 3.9e-6 (XGBoost, float32), 2.7e-15 (RF); p95 5.6 / 11.0 / 10.4 / 15.7 ms. `probe_shap2.py`: XGBoost offset without `scale_pos_weight` +0.0091 / −0.0232, additivity vs `output_margin` 3.4e-6 / 1.2e-6; contract row unchanged. `probe_promote.py` as written stops at B with `thresholds_mismatch`: it registers versions with `mlflow.register_model` without copying the threshold tags, which the gate now rejects by design. A v2 copy that tags versions like `registry.register` runs to the end: A `no_challenger`; B promote (equal), tags set; C own thresholds kept, `cost` fails; D anchor boundary exact (pass at `age − 2`, fail 1e-6 below), NaN / inf / missing / bool anchor → `age_anchor_missing`; E no champion → fairness only; F no thresholds.json → `error`; G `thresholds_mismatch`; H `delete` raising → aliases unchanged, re-run promotes with previous champion `2` (true); I, J (3 000 / 2 000 rows) → `batch_too_small`. Throwaway server port 5331 stopped.
- **Reviewer `mutate.py` replayed** on a `git archive` copy of `5fba2ff` (unmutated selected suite first: 146 passed): **53 mutants, 48 killed, 1 survived, 4 skipped** (lines rewritten by the fixes: #24, #29, #39, #40). Survivor #46 = the dead branch, removed afterwards. Logs: `scratchpad/lot4/review_mutation2.log`, `review_probe_promote.log` (original probe), `review_probe_promote_v2.log`, `review_probe_shap.log`, `review_probe_shap2.log` of the session. A first replay (`review_mutation.log`) is void: its unmutated suite had the failing test above.
- Checks: black, flake8, mypy strict green; **441 tests**; coverage 94 %; `audit` 100 %, `promote` 99 %, `shap_explainer` 99 %, `evaluate` 99 % (the misses are `__main__` lines).

**Lot 4 review round 2 (2026-11-25): "ready with corrections"; decisions final, product ones by the standing policy "adopt the recommended option":**
- [x] R1 (survivor #65): the CHAMPION version's `threshold_high` tag differing from its thresholds.json gives `thresholds_mismatch`, exit 1 (the check was only tested on the challenger). Kills "challenger only" and "high tag ignored".
- [x] R3: `promoted_at` is written after the champion alias move (one extra tag call), so an aborted promotion never carries it; the other `promotion_*` tags, `promotion_previous_champion` included, stay first. Failure-injection test (`set_registered_model_alias` raising), call-order test updated, contract 2 §1 and the decisions doc. Known ceiling: if only that last write fails, the alias has moved and the version has no `promoted_at` (exit 1, `error`).
- [x] R2 (product decision): **fresh hold-out batch for promotion.** `featurize` samples 100 000 of the 307 511 raw rows; the new DVC stage `holdout` (after `featurize`, `src/crediscore/data/holdout.py`, `data.holdout_size: 20000`, all "for Areej's review") draws 20 000 labelled rows, stratified and seeded, from the raw rows NOT in `features.parquet`, and processes them with Areej's own functions (`drop_sparse_columns` guard, `fix_days_employed`, `add_derived_features`, `output_columns`). `dvc repro -s holdout`: **20 000 rows, 1 615 defaults = 8.075 %** (sample 8.073 %), 13 250 F / 6 750 M, thin files 39.4 %; md5 `8bc410e8…`. Only the `holdout` lock entry was added. Tests: no `SK_ID_CURR` overlap with `features.parquet`, columns and dtypes of `test.parquet`, default rate within ±0.5 pt, deterministic and seeded, identical to `featurize` on the same raw rows, refused when too few rows are left, rebuilt from the raw CSV (slow). Own mutants on `holdout.py`: 6 run, 6 killed.
- Promotion batch (contract 2 §1, `promote.py` docstring and `--help`, decisions doc): `data/processed/holdout.parquet`, or a real labelled production batch. `data/simulated/economic_crisis.parquet` reuses test rows: drift-detection demo only, never a promotion batch.
- `dvc push` of `holdout.parquet` is left to the coordinator (MinIO is read-only for this session).

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
- The alert fires when `drift_share ≥ 0.30`, or PSI ≥ 0.20 on a key feature, or `auc_drop > 0.03`. These thresholds go in `params.yaml` under a **new `monitoring:` key** (PR on `params.yaml` + `config.py` for Areej), never under `drift:`, which is Areej's simulator configuration.

**3. Tests.**
- `tests/test_psi.py`: PSI = 0 on identical data; a hand-computed known value; empty bins stay finite; constant and binary columns; NaN handling.
- `test_drift.py`: a shifted normal is detected and the same distribution is not (seeded); the JSON validates against contract 4.

**4. Evidently report.** A `DataDriftPreset` HTML report (pinned version), written to `reports/drift/<date>.html`.

**5. Concept drift.** AUC on a labelled current batch vs the reference AUC.

**Done when:** on Areej's "crise économique" batch, the module flags drift and writes the JSON and the HTML.

**Lot 5 status (2026-11-24): implemented, TDD, before adversarial review.** Branch `feature/drift-detection`, built on lot 3 in progress (rebase on lot 3's final tip pending).
- [x] 1. Contract 4 written (`docs/contracts/04-drift-report.md`, French), **proposal for Areej and Iyed**.
- [x] 2. `monitoring/drift.py`: `psi`, `ks_test`, `chi2_test`, `drift_share`, `is_alert`, `auc_drop`, `compute_drift_report`, `write_html`, CLI `main`. `monitoring:` in `params.yaml` + `MonitoringConfig` (alpha 0.05, drift_share_alert 0.30, psi_alert 0.20, auc_drop_alert 0.03, psi_bins 10, key features `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `ANNUITY_INCOME_RATIO`, `EXT_SOURCE_2`; each key feature must be a model feature), "(for Areej's review)".
- [x] 3. Tests: `tests/test_psi.py` (18), `tests/test_drift.py` (37), 11 new in `test_config.py`.
- [x] 4. Evidently 0.7.23 `DataDriftPreset` HTML: `reports/drift/<YYYY-MM-DD_HHMMSS>.html` (UTC) + the JSON next to it + `latest.json`. DVC stage `detect_drift` after `simulate_crisis` (`latest.json`, `cache: false`, HTML outside DVC). `/reports/drift/` git-ignored.
- [x] 5. Concept drift: `auc_drop(model, reference, current, params)` = reference AUC − current AUC; `null` without current labels. CLI `--model <dir | models:/ URI>` (reuses `evaluate.load_model`).
- Real runs (a local LightGBM copied to `models/lightgbm`, see the caveat below; test = 20 000 rows):

  | Current vs reference | drift_share | Drifted (KS / χ² p < 0.05) | Key PSI (income / credit / annuity ratio / EXT_SOURCE_2) | auc_drop | alert |
  |---|---|---|---|---|---|
  | Economic crisis vs test | 0.171 (7/41) | income, credit, annuity, the 3 ratios, `EMPLOYED_YEARS` | 2.971 / 0.579 / 0.677 / 0.000 | 0.033 (0.816 → 0.784) | **true** |
  | Test half B vs half A (seed 42) | 0.024 (1/41, `EMPLOYED_YEARS`) | — | 0.006 / 0.004 / 0.003 / 0.003 (max PSI 0.011) | 0.003 | false |
  | Scenario C: 30 % of "Commercial associate" labels flipped vs test | 0.000 | none | all 0 | 0.162 (0.816 → 0.654) | **true** |

  Top PSI on the crisis: `AMT_INCOME_TOTAL` 2.97, `CREDIT_TERM` 0.70, `ANNUITY_INCOME_RATIO` 0.68, `AMT_CREDIT` 0.58, `CREDIT_INCOME_RATIO` 0.41. The Evidently HTML (about 6 MB) lists the 41 features and flags the same 7 on the crisis.
- Deviations and choices:
  - `compute_drift_report(reference, current, params, *, auc_drop=None, now=None)` instead of `labels=None`: the AUC needs the model, so the caller computes `auc_drop` first (core stays pure).
  - PSI bins: each deduplicated reference quantile is its own bin, and so is each open interval between them, plus the two open-ended outer bins (≤ 23 bins for 10 deciles). Keeps 0/1 columns at 2 bins and separates "below" and "above" a constant. Epsilon 1e-4 floors each share. NaN always has its own bin (zero contribution when absent on both sides).
  - `ks` and `chi2` hold **p-values**. KS on an all-missing side gives `null`, counted as not drifted (the PSI NaN bin shows it).
  - Evidently's "chi-square" is a goodness-of-fit test against the reference proportions; ours is the two-sample independence test. Test vs test, Evidently flags 3 categoricals (`NAME_INCOME_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`), we flag none. The JSON is the source of truth; on numerics both agree (tested).
  - `dvc.lock` not committed for `detect_drift`: the repro also re-ran featurize/split (featurize.py changed since the lock), and those md5s are pyarrow/line-ending artifacts; restored with `git checkout`.
  - The model is a local artifact of 2026-10-25 (`crediscore-lot2/models/lightgbm`), trained before Areej's current split: AUC 0.816 on today's test ≈ 0.820 on today's train, vs 0.754 on its own validation, so its training rows likely overlap today's test. The AUC **levels** are not meaningful; the **drops** are (same model, same rows, only the batch changes). Re-run with lot 3's final champion once rebased.
- Mutation (own, before the review): 41 hand-written mutants on `drift.py`, on a `git archive` copy (unmutated suite first: 53 passed). Round 1: **38 killed, 3 survived**. #17 (`len(counts) < 2` → `< 1`) was equivalent: scipy already returns p = 1 for a single category, so the guard was removed. #29 (`isna().all()` → `.any()`) showed that a partly labelled batch crashed: `auc_drop` now scores only the labelled current rows (test + fix). #32 (alpha hard-coded) is killed by a new "report follows alpha" test. Round 2 on the fixed HEAD (55 passed unmutated): **39 valid, 39 killed, 0 survived** (#17/#18 no longer apply). Logs: `scratchpad/lot5/mutation_drift.log` and `mutation_drift_round2.log` of the session; script `scratchpad/lot5/mutate.py`.
- Checks: black, flake8 and mypy clean; 392 tests pass (326 before the lot), coverage 93 %, `drift.py` 99 % (the `__main__` line).

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
| The fairness gate never passes (age) | Measured in lot 1 (34.5 / 32.9 pts); D5 decided 2026-10-25: absolute gate on gender only; age ≤ baseline + 2 pts (fixed anchor) |
| Recall ≥ 65 % is not reached | Recall constraint in the optimizer; D8 |
| Brier ≤ 0.07 is not reached | CV calibration; reported honestly (T6) |
| `poetry.lock` conflicts | One early dependencies PR (T1) |
| `poetry.lock` is 896 KiB, close to the pre-commit 1 000 KB limit (`check-added-large-files`) | Ask Areej to exclude `poetry.lock` from `check-added-large-files` before the next dependency |
| Evidently API churn | Exact version pin; code against that version's docs only |
| SHAP latency | Explainer built once; measured locally |
| Iyed's CI is broken (`cache: poetry`) or skips `feature/*` pushes | Raised with Iyed (D6) |
| Commits not credited to Achraf | D11 check in lot 0 |
