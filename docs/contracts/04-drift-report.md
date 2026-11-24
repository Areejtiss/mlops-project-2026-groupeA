# Contrat 4 — Rapport de drift (JSON)

- **Propriétaires :** Achraf (calcul, `src/crediscore/monitoring/drift.py`) et Areej (lots dérivés, `monitoring_flow`). **Consommateurs :** Iyed (jauges Prometheus), Areej (`monitoring_flow` : réentraînement si alerte).
- **Statut :** **proposition du 24/11/2026, en attente de relecture par Areej et Iyed.** Tout changement passe par une PR relue par les deux autres membres.
- **Paramètres :** section `monitoring:` de `params.yaml` (nouvelle, PR pour Areej). Elle est distincte de `drift:`, qui configure le simulateur d'Areej.

## 1. Produire le rapport

```
poetry run python -m crediscore.monitoring.drift \
    --reference data/processed/test.parquet \
    --current data/simulated/economic_crisis.parquet \
    [--model models/lightgbm | --model models:/crediscore@champion] \
    [--out-dir reports/drift] [--params params.yaml]
```

- Écrit `<out-dir>/<AAAA-MM-JJ_HHMMSS>.json` (ce contrat), `<out-dir>/<AAAA-MM-JJ_HHMMSS>.html` (rapport Evidently) et `<out-dir>/latest.json` (copie du dernier JSON). L'horodatage du nom est celui du champ `timestamp`, en **UTC**.
- Code de sortie **0 quelle que soit l'alerte** : l'alerte est dans le JSON. Un code non nul signale une vraie erreur (fichier absent, colonne manquante, lot labellisé d'une seule classe).
- Étape DVC `detect_drift` (après `simulate_crisis`) : test contre crise économique, sans modèle (`auc_drop` = `null`) ; `reports/drift/latest.json` est déclaré `cache: false`. Le HTML reste hors DVC. `reports/drift/` n'est pas versionné dans Git.
- En Python (flow d'Areej) : `compute_drift_report(reference, current, params, auc_drop=auc_drop(model, reference, current, params))` renvoie le dict ; `write_html(reference, current, params, path)` écrit le HTML.

## 2. Champs

Les clés sont **exactement** celles-ci, dans cet ordre. JSON strict : pas de `NaN`, pas d'`Infinity`.

| Champ | Type | Unité / domaine | Définition |
|---|---|---|---|
| `timestamp` | chaîne | ISO 8601 UTC, à la seconde (`2026-11-24T19:40:32+00:00`) | Heure du calcul |
| `drift_share` | nombre | [0, 1], sans unité | Part des features du modèle (`params.features.all`, 41) dont la p-value KS ou χ² est **strictement** inférieure à `alpha` |
| `psi` | objet `{feature: nombre}` | ≥ 0, fini | PSI de chaque feature du modèle (numérique **et** catégorielle), 41 entrées |
| `ks` | objet `{feature: nombre ou null}` | p-value dans [0, 1] | Test de Kolmogorov-Smirnov à deux échantillons, features **numériques** (31 entrées) |
| `chi2` | objet `{feature: nombre}` | p-value dans [0, 1] | Test du χ² d'indépendance lot × catégorie, features **catégorielles** (10 entrées) |
| `auc_drop` | nombre ou `null` | points d'AUC (0,03 = 3 points) | AUC du modèle sur la référence − AUC du même modèle sur le lot courant. Positif = le modèle se dégrade. `null` sans modèle ou sans aucun label dans le lot courant ; les lignes courantes sans label sont ignorées |
| `alert` | booléen | | Règle de la section 5 |

`ks` et `chi2` contiennent des **p-values**, pas des statistiques : plus la valeur est petite, plus la dérive est certaine. Les colonnes hors modèle (`SK_ID_CURR`, `TARGET`, `CODE_GENDER`) ne sont jamais comparées.

## 3. Tests et `drift_share`

- **α = 0,05** (`monitoring.alpha`).
- **Numériques : KS** (`scipy.stats.ks_2samp`) sur les valeurs **non manquantes**. Si un des deux lots n'a aucune valeur, la p-value est `null` et la feature ne compte **pas** comme dérivée ; le PSI la montre quand même (bin « manquant », section 4).
- **Catégorielles : χ²** (`scipy.stats.chi2_contingency`) sur la table 2 × catégories ; **manquant = une catégorie**, une catégorie absente de la référence aussi. Une seule catégorie au total : p-value = 1.
- `drift_share` = (nombre de features avec p-value < α) / 41. Une p-value `null` compte comme non dérivée.
- Sur de gros lots, KS et χ² détectent de très petits écarts : `drift_share` dit « combien de features ont bougé », le PSI dit « de combien ».

## 4. PSI (écrit à la main)

PSI = Σ sur les bins de (part courante − part référence) × ln(part courante / part référence).

- **Numériques.** Bornes = les `psi_bins + 1` quantiles de la **référence** (`psi_bins` = 10 : les déciles, min et max compris), **dédoublonnés** (T10). Chaque valeur de quantile est un bin, chaque intervalle ouvert entre deux quantiles aussi, plus les deux bins extérieurs ouverts (sous le min, au-dessus du max) : au plus 2 × 11 + 1 = 23 bins. Ainsi une colonne 0/1 garde un bin « 0 » et un bin « 1 » (ses quantiles se réduisent à {0, 1}), et une colonne constante sépare « en dessous » et « au-dessus ».
- **Catégorielles.** Un bin par catégorie présente dans l'un ou l'autre lot (catégorie inconnue de la référence = son propre bin).
- **Manquants.** Toujours un bin à part (numériques et catégorielles).
- **Epsilon = 1e-4.** Chaque part est plancher à 1e-4 avant le calcul, pour qu'un bin vide donne un terme fini. Un bin vide des deux côtés donne 0.
- **Sens.** La référence fixe les bins : `psi(ref, cur)` diffère en général de `psi(cur, ref)`. La référence est toujours le premier argument (`--reference`).
- Lecture usuelle : moins de 0,1 stable, 0,1 à 0,2 à surveiller, 0,2 et plus dérive.

## 5. Alerte

```
alert = drift_share >= 0.30                                (monitoring.drift_share_alert)
     OU PSI >= 0.20 sur une feature clé                     (monitoring.psi_alert)
     OU auc_drop > 0.03, quand auc_drop n'est pas null      (monitoring.auc_drop_alert)
```

- **Features clés** (`monitoring.key_features`) : `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `ANNUITY_INCOME_RATIO`, `EXT_SOURCE_2`. Ce doivent être des features du modèle (vérifié par `config.py`).
- Bornes testées : `drift_share` = 0,30 déclenche ; PSI = 0,20 sur une feature clé déclenche ; un PSI élevé sur une feature **non** clé ne déclenche pas seul ; `auc_drop` = 0,03 ne déclenche pas, 0,0301 oui ; une AUC qui s'améliore (`auc_drop` négatif) ne déclenche pas.
- Le JSON ne dit pas quelle règle a déclenché (clés fixes) : on le relit dans `drift_share`, `psi` et `auc_drop`.

## 6. Qui lit quoi

| Consommateur | Lit | Pour |
|---|---|---|
| Iyed, `monitoring/metrics.py` | `drift_share`, `psi` (une jauge `psi{feature}` par entrée), `alert` | Jauges Prometheus et règles d'alerte (`drift_share ≥ 0,30`, `psi ≥ 0,20`). Les seuils sont ceux de `params.yaml`, section `monitoring:` |
| Areej, `flows/monitoring_flow.py` | `alert` | `alert = true` → `run_deployment` du `training_flow` ; archive le JSON et le HTML |
| Rapport et démo (2:30) | le HTML Evidently, `psi`, `auc_drop` | Analyse de la dérive par feature |

## 7. Rapport HTML Evidently

- `evidently==0.7.23`. API 0.7 : `Report([DataDriftPreset(...)]).run(current, reference)` sur des `Dataset.from_pandas(..., data_definition=DataDefinition(...))`, puis `snapshot.save_html(path)`. `ColumnMapping` et `evidently.metric_preset` de la 0.4 ne sont plus que dans `evidently.legacy` : ne pas copier un exemple 0.4.
- Mêmes colonnes (les 41 features), même α, KS pour les numériques (`num_method="ks"`), χ² pour les catégorielles (`cat_method="chisquare"`), seuil de `drift_share` de l'aperçu = 0,30.
- **Écart connu :** le « chi-square » d'Evidently est un test d'adéquation qui prend les proportions de la référence comme exactes ; notre χ² est le test d'indépendance à deux échantillons. Sur les catégorielles à beaucoup de modalités, Evidently signale plus souvent (test contre test : 3 catégorielles signalées par Evidently, 0 par nous). **Le JSON fait foi** ; le HTML sert à l'exploration. Sur les numériques, les deux donnent les mêmes features (testé).
- Taille : environ 6 Mo par rapport (20 000 lignes, 41 colonnes).

## 8. Exemple réel

Lot « crise économique » d'Areej (`dvc repro detect_drift`, `drift.seed = 7`) contre `data/processed/test.parquet` (20 000 lignes chacun), avec un modèle LightGBM local (`--model models/lightgbm`, artefact de travail : ses niveaux d'AUC ne font pas foi, seule la baisse compte). Valeurs arrondies pour la lecture ; le fichier réel garde la précision complète.

```json
{
  "timestamp": "2026-11-24T19:40:32+00:00",
  "drift_share": 0.1707,
  "psi": {
    "AMT_INCOME_TOTAL": 2.9707,
    "AMT_CREDIT": 0.5788,
    "AMT_ANNUITY": 0.1836,
    "AMT_GOODS_PRICE": 0.0,
    "EXT_SOURCE_1": 0.0,
    "EXT_SOURCE_2": 0.0,
    "EXT_SOURCE_3": 0.0,
    "CNT_CHILDREN": 0.0,
    "CNT_FAM_MEMBERS": 0.0,
    "REGION_POPULATION_RELATIVE": 0.0,
    "REGION_RATING_CLIENT_W_CITY": 0.0,
    "DAYS_REGISTRATION": 0.0,
    "DAYS_ID_PUBLISH": 0.0,
    "DAYS_LAST_PHONE_CHANGE": 0.0,
    "REG_CITY_NOT_WORK_CITY": 0.0,
    "LIVE_CITY_NOT_WORK_CITY": 0.0,
    "FLAG_WORK_PHONE": 0.0,
    "FLAG_PHONE": 0.0,
    "FLAG_DOCUMENT_3": 0.0,
    "OBS_30_CNT_SOCIAL_CIRCLE": 0.0,
    "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 0.0,
    "TOTALAREA_MODE": 0.0,
    "ANNUITY_INCOME_RATIO": 0.6769,
    "CREDIT_INCOME_RATIO": 0.4112,
    "CREDIT_TERM": 0.7025,
    "AGE_YEARS": 0.0,
    "EMPLOYED_YEARS": 0.1006,
    "DAYS_EMPLOYED_ANOM": 0.0,
    "EXT_SOURCE_MISSING_COUNT": 0.0,
    "IS_THIN_FILE": 0.0,
    "NAME_CONTRACT_TYPE": 0.0,
    "NAME_INCOME_TYPE": 0.0,
    "NAME_EDUCATION_TYPE": 0.0,
    "NAME_FAMILY_STATUS": 0.0,
    "NAME_HOUSING_TYPE": 0.0,
    "OCCUPATION_TYPE": 0.0,
    "FLAG_OWN_CAR": 0.0,
    "FLAG_OWN_REALTY": 0.0,
    "NAME_TYPE_SUITE": 0.0,
    "ORGANIZATION_TYPE": 0.0
  },
  "ks": {
    "AMT_INCOME_TOTAL": 0.0,
    "AMT_CREDIT": 1.15e-181,
    "AMT_ANNUITY": 3.23e-289,
    "AMT_GOODS_PRICE": 1.0,
    "EXT_SOURCE_1": 1.0,
    "EXT_SOURCE_2": 1.0,
    "EXT_SOURCE_3": 1.0,
    "CNT_CHILDREN": 1.0,
    "CNT_FAM_MEMBERS": 1.0,
    "REGION_POPULATION_RELATIVE": 1.0,
    "REGION_RATING_CLIENT_W_CITY": 1.0,
    "DAYS_REGISTRATION": 1.0,
    "DAYS_ID_PUBLISH": 1.0,
    "DAYS_LAST_PHONE_CHANGE": 1.0,
    "REG_CITY_NOT_WORK_CITY": 1.0,
    "LIVE_CITY_NOT_WORK_CITY": 1.0,
    "FLAG_WORK_PHONE": 1.0,
    "FLAG_PHONE": 1.0,
    "FLAG_DOCUMENT_3": 1.0,
    "OBS_30_CNT_SOCIAL_CIRCLE": 1.0,
    "DEF_30_CNT_SOCIAL_CIRCLE": 1.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
    "TOTALAREA_MODE": 1.0,
    "ANNUITY_INCOME_RATIO": 0.0,
    "CREDIT_INCOME_RATIO": 0.0,
    "CREDIT_TERM": 2.85e-25,
    "AGE_YEARS": 1.0,
    "EMPLOYED_YEARS": 3e-138,
    "DAYS_EMPLOYED_ANOM": 1.0,
    "EXT_SOURCE_MISSING_COUNT": 1.0,
    "IS_THIN_FILE": 1.0
  },
  "chi2": {
    "NAME_CONTRACT_TYPE": 1.0,
    "NAME_INCOME_TYPE": 1.0,
    "NAME_EDUCATION_TYPE": 1.0,
    "NAME_FAMILY_STATUS": 1.0,
    "NAME_HOUSING_TYPE": 1.0,
    "OCCUPATION_TYPE": 1.0,
    "FLAG_OWN_CAR": 1.0,
    "FLAG_OWN_REALTY": 1.0,
    "NAME_TYPE_SUITE": 1.0,
    "ORGANIZATION_TYPE": 1.0
  },
  "auc_drop": 0.0328,
  "alert": true
}
```

Lecture : 7 features dérivées sur 41 (`drift_share` 0,171, sous 0,30), mais PSI ≥ 0,20 sur trois features clés (revenu 2,97, crédit 0,58, ratio annuité/revenu 0,68) et `auc_drop` 0,033 au-dessus de 0,03 : `alert = true`. `EXT_SOURCE_2`, que le simulateur ne touche pas, reste à 0.

Pour comparaison (mêmes commandes, chiffres dans le plan, lot 5) : deux moitiés aléatoires du test l'une contre l'autre donnent `drift_share` 0,024, PSI max 0,011, `auc_drop` 0,003, `alert = false` ; le scénario C (30 % des labels « Commercial associate » inversés) donne `drift_share` 0, tous les PSI à 0, `auc_drop` 0,162, `alert = true` : seule l'AUC voit une dérive de concept.
