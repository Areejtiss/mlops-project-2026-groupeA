# Contrat 2 — Artefact modèle

- **Propriétaire :** Achraf. **Consommateurs :** Iyed (API `/predict`), Areej (flows Prefect, promotion).
- **Statut :** **proposition du 24/10/2026, en attente de relecture par Areej et Iyed.** Tout changement passe par une PR relue par les deux autres membres.
- **Décisions appliquées :** D2 (option b), D4, D8, D9 et T2, T3 (voir `docs/decisions/2026-10-24-decisions-modelisation.md` et `docs/plans/2026-10-24-achraf-roadmap.md`).

## 1. Nom et alias

- Modèle enregistré dans le MLflow Model Registry sous le nom **`crediscore`**.
- Deux alias : **`champion`** (servi par l'API) et **`challenger`** (candidat à la promotion). Les stages `Staging`/`Production` sont dépréciés dans MLflow et ne sont pas utilisés.
- **Promotion** (`crediscore.models.promote`, T7, D5, **règle relative révisée le 24/11/2026**, validée par l'humain). Le challenger (`models:/crediscore@challenger`) et le champion (`models:/crediscore@champion`) sont évalués sur **le même lot**, passé en entrée par `--data` (**obligatoire, sans valeur par défaut**), chacun avec **ses propres seuils** (`thresholds.json` de sa version, **qui fait foi** ; les tags `threshold_low/high` de la version doivent lui être exactement égaux, sinon échec fermé `thresholds_mismatch`).
  - **Lot à utiliser :** un lot récent étiqueté qui **n'est pas le split de test** (lot de production ou lot simulé). Pour la démo : le lot simulé d'Areej, ou un lot étiqueté mis de côté qu'elle fournit. `data/processed/test.parquet` n'est admis que pour une démo manuelle, en le nommant explicitement.
  - **Garde-fous sur le lot (échec fermé, décision du 24/11/2026) :** moins de `params.promotion.min_rows` = **5 000** lignes → `batch_too_small` ; moins de 2 groupes de sexe (`XNA` exclu) ayant chacun au moins `params.promotion.min_group_rows` = **500** lignes → `gender_groups_missing`. En dessous, les écarts sont trop bruités pour décider. Le challenger devient champion si **toutes** ces conditions sont vraies (bornes incluses) :
  - `auc` : AUC du challenger ≥ AUC du champion ;
  - `cost` : coût par dossier du challenger ≤ celui du champion (coûts D3, seuils de chaque version) ;
  - `gender_raw_gap`, `gender_eo_gap` : **sexe**, écart de taux d'acceptation (`ACCEPTE` seulement) entre F et M, `XNA` exclu, ≤ 10 pts, **et** écart d'égalité des chances ≤ 10 pts (`params.promotion.max_gender_gap_pts`) ;
  - `age_eo_gap` : **âge** (<30, 30-50, ≥50), écart d'égalité des chances du challenger ≤ celui de la **baseline** LogReg + 2 pts (`params.promotion.age_gap_margin_pts`). Référence fixe, lue dans `reports/metrics.json` écrit par `evaluate` (`baseline.fairness.age.eo_gap_pts`) : **38,7 pts** sur le split de test d'Areej, donc limite **40,7 pts** (34,9 pts était la valeur provisoire du lot 1). Si elle manque, la promotion est **refusée** (échec fermé, `age_anchor_missing`). Les deux écarts d'âge sont publiés à chaque évaluation.
  - **Pourquoi relative :** le champion actuel a une AUC test de 0,7469 ; un seuil absolu « AUC ≥ 0,75 » sur un lot récent bloquerait toute promotion et la boucle de réentraînement n'aboutirait jamais. Les cibles 0,75 / 0,25 / −15 % restent des **KPI publiés**, pas des conditions.
  - **Pas encore de champion :** seules les trois conditions d'équité s'appliquent.
  - **Challenger = champion** (même version sous les deux alias) : rien à faire, `already_champion`.
  - **Si promu, dans cet ordre :** (1) la version reçoit les tags `promoted_at`, `promotion_previous_champion`, `promotion_data_md5` et, pour chaque condition, `promotion_<condition>` (valeur) et `promotion_<condition>_limit` ; (2) l'alias `challenger` est supprimé ; (3) l'alias `champion` passe à la version du challenger, **en dernier**. Une panne en cours de route laisse donc l'ancien champion en service, et une relance ne peut pas enregistrer un faux champion précédent. Sinon rien ne change.
  - **Sortie :** code 0 si promu, 1 sinon (le flow Prefect d'Areej s'en sert) ; une ligne `promotion: ...` sur la sortie standard ; un JSON `reports/promotion.json` (réécrit à chaque exécution, ignoré par git) dont la clé `failed` liste les conditions en échec : un nom de condition, ou `no_challenger`, `already_champion`, `age_anchor_missing`, `batch_too_small`, `gender_groups_missing`, `thresholds_mismatch`, ou `["error", "<message>"]` pour toute erreur (échec fermé). `--dry-run` calcule et écrit la décision sans rien changer (même code de sortie).
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
- Les 8 features dérivées (`ANNUITY_INCOME_RATIO`, `CREDIT_INCOME_RATIO`, `CREDIT_TERM`, `AGE_YEARS`, `EMPLOYED_YEARS`, `DAYS_EMPLOYED_ANOM`, `EXT_SOURCE_MISSING_COUNT`, `IS_THIN_FILE`) **ne sont pas calculées par l'artefact**. Elles sont produites par les fonctions partagées d'Areej, dans cet ordre : `crediscore.data.featurize.fix_days_employed(df)` puis `add_derived_features(df)`. Ces fonctions sont appelées par l'étape `featurize` **et** par l'API avant le modèle. Un test de parité (les deux chemins donnent des colonnes identiques) est **à écrire** quand l'API d'Iyed existera ; il n'existe pas encore.
- Thin file (D10, définition d'Areej adoptée le 24/11/2026) : `IS_THIN_FILE = 1` si au moins 2 des 3 `EXT_SOURCE` manquent, ou si `AMT_REQ_CREDIT_BUREAU_YEAR` est manquant ou nul.
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
- Valeur de base : calculée à partir du modèle, par robustesse : marge brute du classifieur sur une ligne nulle (LightGBM `booster_.predict(raw_score=True)`, XGBoost `get_booster().predict(output_margin=True)`, LogReg `decision_function` ; probabilité pour RandomForest) moins la somme de ses contributions. Raison : avec shap 0.51, l'`expected_value` de XGBoost 3 peut être décalée d'une constante qui dépend du nombre d'arbres (+0,009 et −0,023 log-odds mesurés par la relecture) ; elle est nulle avec `scale_pos_weight`, le chemin de production. L'additivité (base + somme = marge, ou probabilité pour RandomForest) est testée pour les quatre modèles.
- LogReg : la référence de `LinearExplainer` est la moyenne d'entraînement des numériques mises à l'échelle (0) et « aucune modalité » pour le one-hot, ou la moyenne d'un échantillon `background` s'il est fourni.
- Cible de latence : < 50 ms par ligne (mesurée en local, T9).

## 7. Tags et métadonnées du run

- Tags de la **version du modèle** (lus par l'API avec le modèle) : `threshold_low`, `threshold_high` (pleine précision), `shap_unit`.
- Tags du **run** : `model_type`, `sklearn_version`, `git_sha`, `raw_data_hash` (md5 DVC du CSV brut, lu dans `data/raw/application_train.csv.dvc`), `train_data_hash` (md5 du fichier réellement lu par `train` : le parquet `--train` ou le fichier `--sample`), `serialization_format` (`cloudpickle`).
- Métriques de validation du run : ROC-AUC, PR-AUC, Brier, recall, part en étude, coût.
- Les métriques de **test** sont écrites par `evaluate`, seul lecteur du test pour la sélection.
- `crediscore.models.registry` copie à l'enregistrement `threshold_low`, `threshold_high`, `shap_unit`, `model_type`, `train_data_hash`, `raw_data_hash` et `git_sha` du run vers la version. Alias `champion` s'il n'y en a pas encore, sinon `challenger` (la promotion est au lot 4).
- **Version réelle (lot 3, 24/11/2026) : `crediscore` v1, alias `champion`**, run `1c3f3b8712aa43d491ec3882bf64e70e` (LightGBM réglé par Optuna, 25 essais). Tags de la version :

| Tag | Valeur |
|---|---|
| `threshold_low` | `0.09405060544577634` |
| `threshold_high` | `0.13294897717543763` |
| `shap_unit` | `log_odds` |
| `model_type` | `lightgbm` |
| `train_data_hash` | `d0178dcfbc4973fc61524219c72ec849` (md5 de `data/processed/train.parquet`) |
| `raw_data_hash` | `793a017f41fbac1dc28176b26dbab30e` |
| `git_sha` | `b76421118a572ee338de2b6031a2b3d56e848cdb` |

- `models:/crediscore@champion` rechargé depuis l'hôte donne exactement les mêmes probabilités que `models/model.joblib` produit par `dvc repro train` (écart max 0,0 sur les 20 000 lignes du test).

## 8. Référence baseline et champion (lot 3, `reports/metrics.json`)

Mesures de `evaluate` sur le split de test d'Areej (20 000 lignes), la baseline LogReg étant réentraînée à l'identique sur le même `train.parquet` :

| Test | Baseline LogReg | Champion LightGBM v1 | Cible | Statut |
|---|---|---|---|---|
| ROC-AUC | 0,735 | 0,747 | ≥ 0,75 | **manquée** (−0,003) |
| PR-AUC | 0,208 | 0,227 | ≥ 0,25 | **manquée** (−0,023) |
| Brier | 0,0692 | 0,0685 | ≤ 0,07 | atteinte |
| Recall (D8) | 69,4 % | 66,9 % | ≥ 65 % | atteint |
| Part en étude | 14,4 % | 14,5 % | ≤ 15 % (validation) | — |
| Coût par dossier | 0,352 | 0,320 | −15 % | **manqué** : −9,0 % (écart 6,0 pts) |
| Seuils low / high | 0,0763 / 0,1145 | 0,0941 / 0,1329 | | |

- Le KPI −15 % se lit uniquement contre cette baseline recalculée (0,352 par dossier), plus contre le chiffre du notebook (0,325 sur un autre split).
- Équité (D5) : écart selon le sexe 7,7 pts brut / 6,3 pts égalité des chances (baseline 8,4 / 7,0) ; écart d'âge 34,6 / 33,1 pts (baseline 39,7 / 38,7, limite 40,7). D5 respecté.
- Thin file (D10, 39,4 % du test) : `kpi.targets.thin_file`.
  - **KPI du SYNOPSIS, lu à la lettre (indicateur de réussite) : atteint.** 4 946 bons payeurs thin file acceptés, contre 0 pour « refus systématique sans historique ».
  - Compagnons, champion contre baseline : bons payeurs thin file acceptés **68,4 % contre 61,8 %** ; taux de défaut parmi les thin files acceptés **3,96 % contre 3,79 %** (+0,17 pt : « sans augmenter les pertes » tient à 0,2 pt près, pas strictement).
  - Informatif seulement : ratio des bons payeurs acceptés sur tout le portefeuille contre la scorecard, 1,62 (baseline 1,59).

## 9. Passation à Iyed : squelette « marchant » (lot 2)

En attendant le Registry (lot 3), l'API peut servir le modèle produit par `train` à partir de deux fichiers locaux. Ce sont les sorties DVC de `train`, non versionnées dans git :

```
poetry run python -m crediscore.models.train --model logreg     # -> models/model.joblib, models/thresholds.json, models/metrics.json
```

**Chemin de service (D2), dans cet ordre :** données brutes de la requête → `fix_days_employed` → `add_derived_features` → modèle → `decide`.

```python
import json

import joblib
import pandas as pd

from crediscore.config import load_params
from crediscore.data.featurize import add_derived_features, fix_days_employed
from crediscore.models.threshold import decide

params = load_params()
model = joblib.load("models/model.joblib")              # CalibratedClassifierCV
th = json.loads(open("models/thresholds.json", encoding="utf-8").read())

raw = pd.DataFrame([request_fields])                     # champs bruts validés par Pydantic, None si absent
X = add_derived_features(fix_days_employed(raw))[params.features.all]
X[params.features.numeric] = X[params.features.numeric].astype("float64")
proba = model.predict_proba(X)[:, 1]                     # float64, NON arrondie
decision = decide(proba, th["low"], th["high"])[0]       # "ACCEPTE" / "A_ETUDIER" / "REFUSE"
response = {"proba": round(float(proba[0]), 4), "decision": str(decision)}   # arrondi APRÈS la décision
```

- **Toutes les colonnes brutes doivent être présentes** dans le DataFrame de la requête : l'API remplit un champ absent avec `None` ; sinon `add_derived_features` ou le modèle lèvent `KeyError`.
- **Entrée brute attendue par `add_derived_features`** : les 23 colonnes brutes de `params.features.numeric`, plus `DAYS_BIRTH` et `DAYS_EMPLOYED` (à la place des 8 dérivées), plus les 10 catégorielles. Le modèle ignore toute autre colonne (`CODE_GENDER`, `SK_ID_CURR`…).
- `low` et `high` sont lus tels quels (float64), jamais arrondis. Si `params.threshold.low/high` est renseigné (forçage manuel, D9), l'API l'utilise à la place et le journalise.
- L'image de l'API doit installer `crediscore` (pour `add_derived_features`, `decide`, et la fonction `none_to_nan` référencée par le pickle) et exactement la version de scikit-learn de `poetry.lock`.
- Depuis le lot 3 (version 1 enregistrée), `joblib.load` peut être remplacé par `mlflow.sklearn.load_model("models:/crediscore@champion")`, et les seuils seront lus dans les tags de la version (`threshold_low`, `threshold_high`, `shap_unit`). Accès au serveur MLflow :
  - **depuis un conteneur du réseau compose** (l'API) : `MLFLOW_TRACKING_URI=http://mlflow:5000`. Cela fonctionne parce que le serveur liste `mlflow:5000` dans `--allowed-hosts` (sans cela, MLflow 3.17 répond 403 « Invalid Host header ») ; les URL présignées vers `http://minio:9000` sont résolues dans le réseau. Vérifié : un conteneur du réseau charge les deux modèles, mêmes probabilités que `model.joblib` ;
  - **depuis la machine hôte** : `MLFLOW_TRACKING_URI=http://localhost:5000` **et** `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` (voir `.env.example`), sinon `load_model` reste bloqué sur `http://minio:9000`, que l'hôte ne résout pas.

## 10. SHAP et décision pour `/predict` (passation à Iyed, lot 4)

`crediscore.explain.shap_explainer.Explainer` produit `top_factors`. **On le construit une seule fois, au démarrage de l'API** (environ 0,3 s pour le champion), juste après le chargement du modèle ; on l'appelle ensuite à chaque requête.

**Ordre exact de service :** champs bruts de la requête → `fix_days_employed` → `add_derived_features` → colonnes `params.features.all`, numériques en float64 → `model.predict_proba` → `decide` (sur la proba float64 non arrondie) → `explainer.explain` → arrondi de `proba` pour la réponse.

- `explain(X)` rend, pour chaque ligne, une liste de **3** `Factor(feature, value, contribution)`, triés par |contribution| décroissante. `f._asdict()` donne directement l'objet `{feature, value, contribution}` du contrat 3.
- `feature` : toujours une colonne d'origine de `params.features.all` (les one-hot sont ré-additionnés : `OCCUPATION_TYPE`, jamais `cat__OCCUPATION_TYPE_Laborers`). Une feature exclue du modèle n'apparaît jamais.
- `value` : la valeur d'entrée de la colonne (float, int ou str), `None` si elle manquait.
- `contribution` : unité lue dans le tag `shap_unit` de la version (section 6) : **log-odds** du modèle non calibré pour LogReg, LightGBM et XGBoost (champion v1 : `log_odds`) ; **probabilité** non calibrée pour RandomForest. Ne s'additionne pas à `proba`.
- Sens pour l'interface : contribution **> 0 « augmente le risque »**, **< 0 « diminue le risque »**. Précision : les contributions agissent sur la **marge non calibrée** ; la `proba` affichée en est une fonction seulement **non décroissante** (paliers de l'isotonique). Un facteur qui augmente le risque peut donc laisser `proba` inchangée (même palier), jamais la faire baisser.
- Seuils : `thresholds.json` de la version fait foi ; les tags `threshold_low/high` en sont une copie exacte (vérifiée par `promote`, échec `thresholds_mismatch` sinon). L'extrait lit les tags, ce qui donne les mêmes valeurs.
- **Champs bruts que `/predict` doit accepter** (tous présents dans le DataFrame, `None` si absent ; ce sont toutes les entrées de `fix_days_employed`, `add_derived_features` et du modèle) :
  - 23 numériques : `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `REGION_POPULATION_RELATIVE`, `REGION_RATING_CLIENT_W_CITY`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`, `REG_CITY_NOT_WORK_CITY`, `LIVE_CITY_NOT_WORK_CITY`, `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_DOCUMENT_3`, `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE`, `AMT_REQ_CREDIT_BUREAU_YEAR`, `TOTALAREA_MODE` ;
  - 2 numériques brutes qui remplacent des dérivées : **`DAYS_BIRTH`** (→ `AGE_YEARS`) et **`DAYS_EMPLOYED`** (→ `EMPLOYED_YEARS`, `DAYS_EMPLOYED_ANOM`) ;
  - 10 catégorielles : `NAME_CONTRACT_TYPE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `NAME_TYPE_SUITE`, `ORGANIZATION_TYPE`.
  - Les 8 dérivées (`ANNUITY_INCOME_RATIO`, `CREDIT_INCOME_RATIO`, `CREDIT_TERM`, `AGE_YEARS`, `EMPLOYED_YEARS`, `DAYS_EMPLOYED_ANOM`, `EXT_SOURCE_MISSING_COUNT`, `IS_THIN_FILE`) ne sont **pas** envoyées : elles sont calculées. `CODE_GENDER` et `SK_ID_CURR` ne sont jamais des entrées du modèle.
- Latence mesurée en local sur 200 lignes réelles du test, une ligne par appel, champion v1 : `explain` p50 13,1 ms, **p95 14,9 ms** (cible < 50 ms) ; `predict_proba` + `explain` p95 33,2 ms.

Extrait exécutable (vérifié le 24/11/2026 sur la ligne brute `SK_ID_CURR = 100295` du jeu de test, `OCCUPATION_TYPE` mis à `None` ; depuis l'hôte : `MLFLOW_TRACKING_URI=http://localhost:5000` et `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`, section 9) :

```python
import mlflow
import mlflow.sklearn
import pandas as pd

from crediscore.config import load_params
from crediscore.data.featurize import add_derived_features, fix_days_employed
from crediscore.explain.shap_explainer import Explainer
from crediscore.models.threshold import decide

# Une fois, au démarrage
params = load_params()
client = mlflow.MlflowClient()
version = client.get_model_version_by_alias("crediscore", "champion")
model = mlflow.sklearn.load_model("models:/crediscore@champion")
low = float(version.tags["threshold_low"])          # pleine précision, D9
high = float(version.tags["threshold_high"])
unit = version.tags["shap_unit"]                    # "log_odds" pour v1
explainer = Explainer(model)                        # construit UNE fois

# À chaque requête
raw = pd.DataFrame([request_fields])                # tous les champs bruts, None si absent
X = add_derived_features(fix_days_employed(raw))[params.features.all]
X[params.features.numeric] = X[params.features.numeric].astype("float64")
proba = model.predict_proba(X)[:, 1]                # float64 non arrondie
decision = str(decide(proba, low, high)[0])
factors = explainer.explain(X)[0]                   # 3 Factor, par |contribution|
response = {
    "proba": round(float(proba[0]), 4),             # arrondi APRÈS la décision
    "decision": decision,
    "top_factors": [f._asdict() for f in factors],
    "model_version": version.version,
}
```

Résultat sur cette ligne (défaut réel, `TARGET = 1`) :

```
{'proba': 0.0978, 'decision': 'A_ETUDIER', 'top_factors': [
  {'feature': 'EXT_SOURCE_2', 'value': 0.3023938171668298, 'contribution': 0.3960677271526772},
  {'feature': 'EXT_SOURCE_1', 'value': 0.2620051155519113, 'contribution': 0.21113877101579187},
  {'feature': 'REGION_POPULATION_RELATIVE', 'value': 0.072508, 'contribution': -0.16480913253840848}],
 'model_version': '1'}
```

Lecture : un `EXT_SOURCE_2` bas (0,30) augmente le risque (+0,40 log-odds) ; une région plus peuplée le diminue (−0,16). Le graphique de synthèse SHAP du champion (barres, moyenne des |contributions| sur 2 000 lignes du train) est `reports/figures/shap_summary.png`, aussi enregistré dans MLflow (run `a146f2b6842d42adac5da1979fa5d0cd`, tag `kind=shap_summary`, `model_run_id` = run du champion).

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
