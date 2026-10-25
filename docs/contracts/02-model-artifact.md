# Contrat 2 — Artefact modèle

- **Propriétaire :** Achraf. **Consommateurs :** Iyed (API `/predict`), Areej (flows Prefect, promotion).
- **Statut :** **proposition du 24/10/2026, en attente de relecture par Areej et Iyed.** Tout changement passe par une PR relue par les deux autres membres.
- **Décisions appliquées :** D2 (option b), D4, D8, D9 et T2, T3 (voir `docs/decisions/2026-10-24-decisions-modelisation.md` et `docs/plans/2026-10-24-achraf-roadmap.md`).

## 1. Nom et alias

- Modèle enregistré dans le MLflow Model Registry sous le nom **`crediscore`**.
- Deux alias : **`champion`** (servi par l'API) et **`challenger`** (candidat à la promotion). Les stages `Staging`/`Production` sont dépréciés dans MLflow et ne sont pas utilisés.
- **Promotion** (`models/promote.py`, T7, D5). Le challenger devient champion si **toutes** ces conditions sont vraies, sur le même jeu récent pour les deux modèles :
  - AUC du challenger ≥ AUC du champion, et ≥ 0,75 ;
  - **sexe** : écart de taux d'acceptation (`ACCEPTE` seulement) entre F et M, `XNA` exclu, ≤ 10 pts, **et** écart d'égalité des chances ≤ 10 pts ;
  - **âge** (<30, 30-50, ≥50) : écart d'égalité des chances du challenger ≤ celui de la **baseline** LogReg + 2 pts, soit ≤ 34,9 pts (référence fixe : la baseline recalculée par `evaluate` au lot 3 sur le même jeu de test, 32,9 pts d'ici là ; s'applique dès la première promotion, ne dérive pas). Pas de seuil absolu ; les deux écarts d'âge sont publiés à chaque évaluation.
- L'API charge `models:/crediscore@champion`. La « version du modèle » (`model_version`) est le numéro de version du Registry, en chaîne (ex. `"3"`).

## 2. Flavor et objet exact

- Flavor **`mlflow.sklearn`**, chargé par `mlflow.sklearn.load_model(...)`. L'API a besoin de `predict_proba`, et SHAP du pipeline ajusté : le flavor `pyfunc` ne suffit pas.
- Objet servi (le modèle enregistré) : un **`CalibratedClassifierCV(pipeline, cv=5, method="isotonic", ensemble=False)`** ajusté sur la partie train (T2).
  - Avec `ensemble=False`, scikit-learn apprend **une seule** fonction isotonique sur les prédictions hors-pli des 5 plis, puis réajuste le pipeline sur **toute la partie fit** (80 % du train ; les 20 % restants sont la validation, D1).
  - Le modèle servi est donc une fonction **monotone** d'un seul pipeline : `calibrated_classifiers_[0].estimator`. C'est ce pipeline que SHAP explique (section 6).
  - Il n'y a **pas d'artefact séparé** pour le modèle non calibré.
- `pipeline` est un `sklearn.pipeline.Pipeline` en deux étapes :
  1. `pre` : un `ColumnTransformer`. Colonnes numériques : imputation médiane + `StandardScaler` (étape optionnelle pour les modèles à arbres). Colonnes catégorielles : `None` converti en `NaN` (un `null` JSON ou parquet est donc imputé comme un `NaN`, et ne devient pas une modalité), imputation par la modalité la plus fréquente + `OneHotEncoder(handle_unknown="ignore")`. `remainder="drop"`.
  2. `model` : le classifieur (LogReg, LightGBM, XGBoost ou RandomForest).
- Conséquence de l'isotonique : la probabilité servie est **en paliers**. La baseline n'a que 69 valeurs distinctes sur la validation, et beaucoup de dossiers ont exactement la même probabilité.
- **MLflow 3.17, journalisation et chargement :**
  - `mlflow.sklearn.log_model(..., serialization_format="cloudpickle")`. Le format `skops` refuse `CalibratedClassifierCV` sans `skops_trusted_types`.
  - Le modèle est toujours adressé par **nom enregistré + alias** (`models:/crediscore@champion`), jamais par `runs:/<run_id>/<artefact>`.
- **Version de scikit-learn épinglée.** Un pickle scikit-learn n'est fiable qu'avec la même version : ≥ 1.6, actuellement **1.9.1** (`poetry.lock`). L'image Docker de l'API doit installer exactement la version de `poetry.lock`. La version est aussi écrite en tag (`sklearn_version`).

## 3. Entrée

- **Exactement les colonnes `params.features.all`** (D2 option b), dans cet ordre : les 31 numériques puis les 10 catégorielles de `params.yaml`.
- Les 8 features dérivées (`ANNUITY_INCOME_RATIO`, `CREDIT_INCOME_RATIO`, `CREDIT_TERM`, `AGE_YEARS`, `EMPLOYED_YEARS`, `DAYS_EMPLOYED_ANOM`, `EXT_SOURCE_MISSING_COUNT`, `IS_THIN_FILE`) **ne sont pas calculées par l'artefact**. Elles sont produites par la fonction partagée `add_derived_features(df)` d'Areej. Cette fonction est appelée par l'étape `featurize` **et** par l'API avant le modèle. Un test de parité (les deux chemins donnent des colonnes identiques) est **à écrire** quand l'étape `featurize` d'Areej et l'API d'Iyed existeront ; il n'existe pas encore.
- Types (signature MLflow, construite explicitement par `crediscore.models.train.build_signature`, pas inférée) :

| Colonnes | Type | Manquant autorisé |
|---|---|---|
| 31 numériques (montants, `EXT_SOURCE_*`, `DAYS_*`, `CNT_*`, flags 0/1, dérivées) | `double` (float64). `train` convertit toutes les numériques en float64 avant l'ajustement et la signature ; l'API fait de même | oui (`NaN`/`null`), imputé par le pipeline |
| 10 catégorielles (`NAME_*`, `OCCUPATION_TYPE`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `ORGANIZATION_TYPE`) | `string` | oui, imputé ; une modalité inconnue est ignorée (vecteur one-hot nul) |

- Dans la signature, **toutes les colonnes sont marquées optionnelles** (`required=False`) : n'importe quelle feature peut manquer en production. Une valeur manquante veut dire `None`, pas une colonne absente : **Toutes les colonnes brutes doivent être présentes** dans le DataFrame de la requête : l'API remplit un champ absent avec `None` ; sinon `add_derived_features` ou le modèle lèvent `KeyError`.
- **Le flavor `mlflow.sklearn` n'applique pas la signature.** Une colonne absente ou de mauvais type n'est pas rejetée par MLflow. Selon le cas, scikit-learn lève une erreur ou produit un résultat faux. **La validation des entrées est donc faite par le modèle Pydantic de l'API (Iyed)** : noms, types, valeurs manquantes autorisées. La signature sert de documentation et de référence pour ce schéma.

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
  "low": 0.0898,
  "high": 0.1232,
  "cost_fn": 5.0,
  "cost_fp": 1.0,
  "cost_review": 0.3,
  "min_recall": 0.65,
  "max_review_share": 0.15,
  "val_recall": 0.6618,
  "val_review_share": 0.1421,
  "val_cost_per_file": 0.3329
}
```

  Les valeurs ci-dessus sont celles de la baseline LogReg (`reports/baseline_metrics.json`), arrondies pour la lecture.
- **Dans le vrai fichier, `low` et `high` sont écrits en pleine précision (float64, jamais arrondis).**
- Les seuils sont cherchés **exactement** sur toutes les probabilités distinctes de la validation, pas sur une grille (`crediscore.models.threshold.optimize_thresholds`, qui rend toujours `low ≤ high`).
- **Règle du milieu.** Avec l'isotonique, une coupure posée exactement sur un palier ferait dépendre la décision d'une égalité de flottants : sur le test de la baseline, 1 350 dossiers ont `proba == low`, un arrondi à 4 décimales fait passer l'étude à 17,4 % et une proba en float32 fait tomber le recall à 58,9 %. Chaque seuil retenu est donc placé **au milieu entre son palier et le palier distinct immédiatement inférieur**. S'il n'y a pas de palier inférieur, la coupure est la moitié du palier. Si les deux paliers sont des doubles voisins (le milieu retombe sur le palier inférieur), la coupure reste sur son palier. Les probabilités et les seuils doivent être finis et dans [0, 1] (`NEVER`, juste au-dessus de 1, est admis pour « jamais refuser »). Les décisions sur la validation sont identiques (vérifié dans le code), et un aller-retour JSON ou une proba en float32 ne les change plus (testé), sauf le cas des doubles voisins où la coupe reste sur sa marche.
- **La décision est calculée sur la proba float64 non arrondie**, avant tout arrondi de la réponse de l'API.
- Règle (D4), bornes incluses comme indiqué :
  - `proba < low` → `ACCEPTE` ;
  - `low ≤ proba < high` → `A_ETUDIER` ;
  - `proba ≥ high` → `REFUSE`.
  - `low = high` est autorisé (2 zones, aucun dossier en étude).
- `threshold.low/high` de `params.yaml` devient un **forçage manuel optionnel** : s'il est renseigné, il remplace les seuils du modèle et l'API le journalise.
- Coût métier : `crediscore.models.cost.business_cost`. FN = 5, FP = 1, étude = 0,3 par dossier, un dossier étudié est supposé bien décidé (D3). Recall au sens D8 : un défaut envoyé en étude compte comme détecté.

## 6. SHAP (T3)

- SHAP explique le **pipeline non calibré contenu dans le modèle servi** : `model.calibrated_classifiers_[0].estimator` (section 2), sans artefact séparé. L'explainer est construit une fois au chargement :
  - `LinearExplainer` pour la LogReg ;
  - `TreeExplainer` pour les modèles à arbres (RandomForest : `feature_perturbation="tree_path_dependent"`).
- **Unité de `contribution`, selon le type de modèle :**

| Modèle | Unité de `contribution` | Les contributions + la valeur de base donnent |
|---|---|---|
| LogReg, LightGBM, XGBoost | **log-odds** (marge brute) du pipeline non calibré | le log-odds non calibré |
| RandomForest | **probabilité** (classe 1) du pipeline non calibré | la probabilité non calibrée |

- Dans les deux cas, les contributions **ne s'additionnent pas à `proba`** (la probabilité calibrée servie). Comme le modèle servi est une fonction monotone du pipeline expliqué (`ensemble=False`), un facteur qui pousse le score non calibré vers le haut pousse aussi `proba` vers le haut ou la laisse sur le même palier. Le rapport précise cette approximation.
- L'unité est écrite en tag de la version du modèle (`shap_unit` = `log_odds` ou `probability`), pour que l'API et le rapport l'affichent correctement.
- Les contributions des colonnes one-hot sont **ré-additionnées par colonne d'origine** (`OCCUPATION_TYPE`, pas `cat__OCCUPATION_TYPE_Laborers`).
- `value` : la valeur reçue en entrée pour cette colonne, avant imputation et mise à l'échelle ; `null` si elle manquait.
- Signe : contribution > 0 = augmente le risque de défaut.
- Cible de latence : < 50 ms par ligne (mesurée en local, T9).

## 7. Tags et métadonnées du run

- Tags de la **version du modèle** (lus par l'API avec le modèle) : `threshold_low`, `threshold_high` (pleine précision), `shap_unit`.
- Tags du **run** : `model_type`, `sklearn_version`, `git_sha`, `raw_data_hash` (md5 DVC du CSV brut, lu dans `data/raw/application_train.csv.dvc`), `train_data_hash` (md5 du fichier réellement lu par `train` : le parquet `--train` ou le fichier `--sample`), `serialization_format` (`cloudpickle`).
- Métriques de validation du run : ROC-AUC, PR-AUC, Brier, recall, part en étude, coût.
- Les métriques de **test** sont écrites par `evaluate`, seul lecteur du test pour la sélection.

## 8. Référence baseline (notebook `02_baseline.ipynb`)

LogReg calibrée (`ensemble=False`), sur le test :
- ROC-AUC 0,746 ; PR-AUC 0,220 ; Brier 0,069 ;
- seuils 3 zones 0,0898 / 0,1232 (milieux entre paliers ; paliers 0,0926 / 0,1378) ; recall 66,5 % ; 14,2 % en étude ;
- coût **0,325 par dossier**.

Ce chiffre n'est la référence du KPI −15 % que sur ce jeu de test : au lot 3, `evaluate` recalcule la baseline sur le split de test définitif et compare les modèles à ce recalcul.

## 9. Passation à Iyed : squelette « marchant » (lot 2)

En attendant le Registry (lot 3), l'API peut servir le modèle produit par `train` à partir de deux fichiers locaux. Ce sont les sorties DVC de `train`, non versionnées dans git :

```
poetry run python -m crediscore.models.train --model logreg     # -> models/model.joblib, models/thresholds.json, models/metrics.json
```

**Chemin de service (D2), dans cet ordre :** données brutes de la requête → `add_derived_features` → modèle → `decide`.

```python
import json

import joblib
import pandas as pd

from crediscore.config import load_params
from crediscore.data.features import add_derived_features
from crediscore.models.threshold import decide

params = load_params()
model = joblib.load("models/model.joblib")              # CalibratedClassifierCV
th = json.loads(open("models/thresholds.json", encoding="utf-8").read())

raw = pd.DataFrame([request_fields])                     # champs bruts validés par Pydantic, None si absent
X = add_derived_features(raw)[params.features.all]
X[params.features.numeric] = X[params.features.numeric].astype("float64")
proba = model.predict_proba(X)[:, 1]                     # float64, NON arrondie
decision = decide(proba, th["low"], th["high"])[0]       # "ACCEPTE" / "A_ETUDIER" / "REFUSE"
response = {"proba": round(float(proba[0]), 4), "decision": str(decision)}   # arrondi APRÈS la décision
```

- **Toutes les colonnes brutes doivent être présentes** dans le DataFrame de la requête : l'API remplit un champ absent avec `None` ; sinon `add_derived_features` ou le modèle lèvent `KeyError`.
- **Entrée brute attendue par `add_derived_features`** : les 23 colonnes brutes de `params.features.numeric`, plus `DAYS_BIRTH` et `DAYS_EMPLOYED` (à la place des 8 dérivées), plus les 10 catégorielles. Le modèle ignore toute autre colonne (`CODE_GENDER`, `SK_ID_CURR`…).
- `low` et `high` sont lus tels quels (float64), jamais arrondis. Si `params.threshold.low/high` est renseigné (forçage manuel, D9), l'API l'utilise à la place et le journalise.
- L'image de l'API doit installer `crediscore` (pour `add_derived_features`, `decide`, et la fonction `none_to_nan` référencée par le pickle) et exactement la version de scikit-learn de `poetry.lock`.
- Au lot 3, `joblib.load` sera remplacé par `mlflow.sklearn.load_model("models:/crediscore@champion")`, et les seuils seront lus dans les tags de la version (`threshold_low`, `threshold_high`, `shap_unit`). Accès au serveur MLflow :
  - **depuis un conteneur du réseau compose** (l'API) : `MLFLOW_TRACKING_URI=http://mlflow:5000`. Cela fonctionne parce que le serveur liste `mlflow:5000` dans `--allowed-hosts` (sans cela, MLflow 3.17 répond 403 « Invalid Host header ») ; les URL présignées vers `http://minio:9000` sont résolues dans le réseau. Vérifié : un conteneur du réseau charge les deux modèles, mêmes probabilités que `model.joblib` ;
  - **depuis la machine hôte** : `MLFLOW_TRACKING_URI=http://localhost:5000` **et** `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` (voir `.env.example`), sinon `load_model` reste bloqué sur `http://minio:9000`, que l'hôte ne résout pas.

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
| `proba` | `float` dans [0, 1] | probabilité **calibrée** de défaut (section 4). `decision` est calculée sur la proba float64 **avant** tout arrondi de ce champ |
| `decision` | `Literal["ACCEPTE", "A_ETUDIER", "REFUSE"]` | codes D4, sans accents ni espaces. Libellés accentués (« Accepté », « À étudier », « Refusé ») dans l'interface et le rapport seulement |
| `top_factors` | liste d'**exactement 3** objets `{feature, value, contribution}`, triés par \|contribution\| décroissante | voir section 6 |
| `top_factors[].feature` | `str` | nom d'une colonne de `params.features.all` (colonne d'origine, jamais un nom one-hot) |
| `top_factors[].value` | `float \| str \| null` | valeur d'entrée brute de cette colonne |
| `top_factors[].contribution` | `float` | unité selon le modèle (section 6) : log-odds pour LogReg, LightGBM et XGBoost ; probabilité pour RandomForest. > 0 augmente le risque ; ne s'additionne pas à `proba` |
| `model_version` | `str` | version du Registry du modèle `champion` qui a répondu ; les seuils utilisés sont ceux de cette version (D9) |

Côté Pydantic (Iyed) :
- réponse : modèles `StrictModel`, `decision: Literal[...]`, `top_factors: list[Factor] = Field(min_length=3, max_length=3)` ;
- **requête : un modèle `StrictModel` qui valide les entrées** (colonnes de la section 3, types, `None` autorisé). **Toutes les colonnes brutes doivent être présentes** dans le DataFrame de la requête : l'API remplit un champ absent avec `None` ; sinon `add_derived_features` ou le modèle lèvent `KeyError`. Le flavor sklearn n'applique pas la signature MLflow : c'est la seule barrière.
- L'unité de `contribution` est lue dans le tag `shap_unit` de la version servie. Le schéma peut l'exposer à l'interface sans changer la forme de la réponse.
