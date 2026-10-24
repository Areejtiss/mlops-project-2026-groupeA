# Contrat 2 — Artefact modèle

- **Propriétaire :** Achraf. **Consommateurs :** Iyed (API `/predict`), Areej (flows Prefect, promotion).
- **Statut :** **proposition du 24/10/2026, en attente de relecture par Areej et Iyed.** Tout changement passe par une PR relue par les deux autres membres.
- **Décisions appliquées :** D2 (option b), D4, D8, D9 et T2, T3 (voir `docs/decisions/2026-10-24-decisions-modelisation.md` et `docs/plans/2026-10-24-achraf-roadmap.md`).

## 1. Nom et alias

- Modèle enregistré dans le MLflow Model Registry sous le nom **`crediscore`**.
- Deux alias : **`champion`** (servi par l'API) et **`challenger`** (candidat à la promotion). Les stages `Staging`/`Production` sont dépréciés dans MLflow et ne sont pas utilisés.
- L'API charge `models:/crediscore@champion`. La « version du modèle » (`model_version`) est le numéro de version du Registry, en chaîne (ex. `"3"`).

## 2. Flavor et objet exact

- Flavor **`mlflow.sklearn`**, chargé par `mlflow.sklearn.load_model(...)`. L'API a besoin de `predict_proba`, et SHAP du pipeline ajusté : le flavor `pyfunc` ne suffit pas.
- Objet servi (artefact `model/` du run, celui qui est enregistré) : un **`CalibratedClassifierCV(pipeline, cv=5, method="isotonic")`** ajusté sur la partie train (T2).
- `pipeline` est un `sklearn.pipeline.Pipeline` en deux étapes :
  1. `pre` : un `ColumnTransformer`. Colonnes numériques : imputation médiane + `StandardScaler` (étape optionnelle pour les modèles à arbres). Colonnes catégorielles : imputation par la modalité la plus fréquente + `OneHotEncoder(handle_unknown="ignore")`. `remainder="drop"`.
  2. `model` : le classifieur (LogReg, LightGBM, XGBoost ou RandomForest).
- Le **pipeline non calibré** ajusté sur tout le train est stocké dans le même run, artefact **`uncalibrated_model/`** (même flavor). Il sert **uniquement à SHAP**. Il n'est jamais utilisé pour la décision.
- **Version de scikit-learn épinglée.** Un pickle scikit-learn n'est fiable qu'avec la même version : ≥ 1.6, actuellement **1.9.1** (`poetry.lock`). L'image Docker de l'API doit installer exactement la version de `poetry.lock`. La version est aussi écrite en tag (`sklearn_version`).

## 3. Entrée

- **Exactement les colonnes `params.features.all`** (D2 option b), dans cet ordre : les 31 numériques puis les 10 catégorielles de `params.yaml`.
- Les 8 features dérivées (`ANNUITY_INCOME_RATIO`, `CREDIT_INCOME_RATIO`, `CREDIT_TERM`, `AGE_YEARS`, `EMPLOYED_YEARS`, `DAYS_EMPLOYED_ANOM`, `EXT_SOURCE_MISSING_COUNT`, `IS_THIN_FILE`) **ne sont pas calculées par l'artefact**. Elles sont produites par la fonction partagée `add_derived_features(df)` d'Areej. Cette fonction est appelée par l'étape `featurize` **et** par l'API avant le modèle. Un test vérifie que les deux chemins donnent des colonnes identiques.
- Types (signature MLflow, `infer_signature` sur un échantillon du train) :

| Colonnes | Type | Manquant autorisé |
|---|---|---|
| 31 numériques (montants, `EXT_SOURCE_*`, `DAYS_*`, `CNT_*`, flags 0/1, dérivées) | `double` (float64). Les entiers et flags sont convertis en float64 par l'appelant | oui (`NaN`/`null`), imputé par le pipeline |
| 10 catégorielles (`NAME_*`, `OCCUPATION_TYPE`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `ORGANIZATION_TYPE`) | `string` | oui, imputé ; une modalité inconnue est ignorée (vecteur one-hot nul) |

- **Jamais en entrée :** `CODE_GENDER` (et toute colonne de `sensitive_cols`), `SK_ID_CURR`, `TARGET`. Si elles sont présentes, le `ColumnTransformer` les ignore (`remainder="drop"`). Un test le vérifie (lot 2).
- `EMPLOYED_YEARS` vaut `NaN` quand `DAYS_EMPLOYED = 365243` (sentinelle) ; `DAYS_EMPLOYED_ANOM` vaut alors 1.

## 4. Sortie

- `model.predict_proba(X)[:, 1]` : **probabilité calibrée de défaut** (classe 1, `TARGET = 1`), un float dans [0, 1], une par ligne.
- `predict()` n'est pas utilisé : la décision vient des seuils (section 5), pas du seuil 0,5.

## 5. Seuils de décision (D9)

- Les seuils sont un **résultat de l'entraînement** : ils sont optimisés sur le jeu de validation (D1), sur la probabilité calibrée. Ils **voyagent avec la version du modèle** :
  - tags de la version du modèle : `threshold_low`, `threshold_high` ;
  - artefact `thresholds.json` du run (et `models/thresholds.json` en sortie DVC) :

```json
{
  "low": 0.087,
  "high": 0.129,
  "cost_fn": 5.0,
  "cost_fp": 1.0,
  "cost_review": 0.3,
  "min_recall": 0.65,
  "max_review_share": 0.15,
  "val_recall": 0.6525,
  "val_review_share": 0.1486,
  "val_cost_per_file": 0.3293
}
```

  (valeurs de la baseline LogReg, `reports/baseline_metrics.json`).
- Règle (D4), bornes incluses comme indiqué :
  - `proba < low` → `ACCEPTE` ;
  - `low ≤ proba < high` → `A_ETUDIER` ;
  - `proba ≥ high` → `REFUSE`.
  - `low = high` est autorisé (2 zones, aucun dossier en étude).
- `threshold.low/high` de `params.yaml` devient un **forçage manuel optionnel** : s'il est renseigné, il remplace les seuils du modèle et l'API le journalise.
- Coût métier : `crediscore.models.cost.business_cost`. FN = 5, FP = 1, étude = 0,3 par dossier, un dossier étudié est supposé bien décidé (D3). Recall au sens D8 : un défaut envoyé en étude compte comme détecté.

## 6. SHAP (T3)

- SHAP explique le **pipeline non calibré** (`uncalibrated_model/`). L'explainer est construit une fois au chargement :
  - `LinearExplainer` pour la LogReg ;
  - `TreeExplainer` pour les modèles à arbres (RandomForest : `feature_perturbation="tree_path_dependent"`).
- Unité de `contribution` : **log-odds du modèle non calibré**. Les contributions s'additionnent (avec la valeur de base) au log-odds du modèle non calibré, **pas** à `proba` : on ne peut pas les sommer pour retrouver `proba`. L'ordre des facteurs est une approximation de la décision calibrée (la calibration isotonique est monotone, mais non linéaire). Le rapport le précise.
- Les contributions des colonnes one-hot sont **ré-additionnées par colonne d'origine** (`OCCUPATION_TYPE`, pas `cat__OCCUPATION_TYPE_Laborers`).
- `value` : la valeur reçue en entrée pour cette colonne, avant imputation et mise à l'échelle ; `null` si elle manquait.
- Signe : contribution > 0 = augmente le risque de défaut.
- Cible de latence : < 50 ms par ligne (mesurée en local, T9).

## 7. Tags et métadonnées du run

`model_type`, `sklearn_version`, `git_sha`, `data_hash` (hash DVC des données d'entraînement), `threshold_low`, `threshold_high`, et les métriques de validation (ROC-AUC, PR-AUC, Brier, recall, part en étude, coût). Les métriques de **test** sont écrites par `evaluate` (seul lecteur du test pour la sélection).

## 8. Référence baseline (notebook `02_baseline.ipynb`)

LogReg calibrée sur le test : ROC-AUC 0,746 ; PR-AUC 0,228 ; Brier 0,069 ; seuils 3 zones 0,087 / 0,129 ; recall 66,0 % ; 14,9 % en étude ; coût **0,321 par dossier**. Cette valeur est la référence du KPI −15 %.

---

## Proposition pour le contrat 3 (à valider par Iyed)

> **Proposition, en attente de relecture par Areej et Iyed.** Seules les parties décision et SHAP relèvent d'Achraf ; le reste du schéma `/predict` appartient à Iyed.

Réponse de `/predict` :

```json
{
  "proba": 0.1123,
  "decision": "A_ETUDIER",
  "top_factors": [
    {"feature": "EXT_SOURCE_3", "value": 0.21, "contribution": 0.48},
    {"feature": "AGE_YEARS", "value": 24.3, "contribution": 0.22},
    {"feature": "OCCUPATION_TYPE", "value": "Laborers", "contribution": -0.07}
  ],
  "model_version": "3"
}
```

| Champ | Type | Sens et unité |
|---|---|---|
| `proba` | `float` dans [0, 1] | probabilité **calibrée** de défaut (section 4) |
| `decision` | `Literal["ACCEPTE", "A_ETUDIER", "REFUSE"]` | codes D4, sans accents ni espaces. Libellés accentués (« Accepté », « À étudier », « Refusé ») dans l'interface et le rapport seulement |
| `top_factors` | liste d'**exactement 3** objets `{feature, value, contribution}`, triés par \|contribution\| décroissante | voir section 6 |
| `top_factors[].feature` | `str` | nom d'une colonne de `params.features.all` (colonne d'origine, jamais un nom one-hot) |
| `top_factors[].value` | `float \| str \| null` | valeur d'entrée brute de cette colonne |
| `top_factors[].contribution` | `float` | log-odds du modèle non calibré ; > 0 augmente le risque ; ne s'additionne pas à `proba` |
| `model_version` | `str` | version du Registry du modèle `champion` qui a répondu ; les seuils utilisés sont ceux de cette version (D9) |

Côté Pydantic (Iyed) : modèles `StrictModel`, `decision: Literal[...]`, `top_factors: list[Factor] = Field(min_length=3, max_length=3)`.
