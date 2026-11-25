# Contrat 4 — Rapport de drift (JSON)

- **Propriétaires :** Achraf (calcul, `src/crediscore/monitoring/drift.py`) et Areej (lots dérivés, `monitoring_flow`). **Consommateurs :** Iyed (jauges Prometheus), Areej (`monitoring_flow` : réentraînement si alerte).
- **Statut :** **proposition du 24/11/2026, révisée après relecture le même jour, en attente de relecture par Areej et Iyed.** Tout changement passe par une PR relue par les deux autres membres.
- **Ajout du 25/11/2026 (lot 6), proposition pour relecture d'Areej et d'Iyed :** deux clés **optionnelles**, `prediction` et `fairness`, ajoutées **à la fin** du JSON (section 9). Elles valent `null` sans `--model`. Elles ne changent **pas** la règle d'alerte.
- **Paramètres :** section `monitoring:` de `params.yaml` (nouvelle, PR pour Areej). Elle est distincte de `drift:`, qui configure le simulateur d'Areej.

## 1. Produire le rapport

```
poetry run python -m crediscore.monitoring.drift \
    --reference data/processed/test.parquet \
    --current data/simulated/economic_crisis.parquet \
    [--model models | --model models:/crediscore@champion] \
    [--metrics reports/metrics.json] \
    [--out-dir reports/drift] [--params params.yaml]
```

`--metrics` (lot 6) : le `metrics.json` d'`evaluate`, d'où est lue l'ancre d'âge D5 de la section 9 ; utilisé seulement avec `--model`.

- Écrit `<out-dir>/<AAAA-MM-JJ_HHMMSS>.json` (ce contrat), `<out-dir>/<AAAA-MM-JJ_HHMMSS>.html` (rapport Evidently) et `<out-dir>/latest.json` (copie du dernier JSON). L'horodatage du nom est celui du champ `timestamp`, en **UTC**.
- Code de sortie **0 quelle que soit l'alerte** : l'alerte est dans le JSON. Un code non nul signale une vraie erreur : fichier absent, colonne manquante, lot vide (`ValueError("empty batch…")`). Un lot labellisé d'une seule classe n'est **pas** une erreur : `auc_drop` vaut alors `null` (section 2).
- Étape DVC `detect_drift` (après `simulate_crisis`) : test contre crise économique, **sans modèle** (`auc_drop` = `null`). `reports/drift/latest.json` est déclaré `cache: false` et ignoré par Git. C'est une sortie **non reproductible** à l'octet près (elle porte l'heure du calcul) : DVC la régénère à chaque exécution. Le HTML reste hors DVC.
- En Python (flow d'Areej) : `compute_drift_report(reference, current, params, auc_drop=auc_drop(model, reference, current, params))` renvoie le dict ; `write_html(reference, current, params, path)` écrit le HTML. `now`, s'il est donné, doit porter un fuseau horaire (un `datetime` naïf lève `ValueError`) ; un `auc_drop` NaN ou infini lève aussi `ValueError`.

## 2. Champs

Les clés sont **exactement** celles-ci, dans cet ordre, suivies des deux clés du lot 6 (`prediction`, `fairness`, section 9), toujours présentes, `null` quand elles ne sont pas calculées. JSON strict : pas de `NaN`, pas d'`Infinity`.

| Champ | Type | Unité / domaine | Définition |
|---|---|---|---|
| `timestamp` | chaîne | ISO 8601 UTC, à la seconde, sans microsecondes (`2026-11-24T21:33:35+00:00`) | Heure du calcul |
| `drift_share` | nombre | [0, 1], sans unité | Part des features du modèle (`params.features.all`, 41) dont la p-value KS ou χ² est **strictement** inférieure à `alpha` |
| `psi` | objet `{feature: nombre}` | ≥ 0, fini | PSI de chaque feature du modèle (numérique **et** catégorielle), 41 entrées |
| `ks` | objet `{feature: nombre ou null}` | p-value dans [0, 1] | Test de Kolmogorov-Smirnov à deux échantillons, features **numériques** (31 entrées) |
| `chi2` | objet `{feature: nombre}` | p-value dans [0, 1] | Test du χ² d'indépendance lot × catégorie, features **catégorielles** (10 entrées) |
| `auc_drop` | nombre ou `null` | points d'AUC (0,03 = 3 points) | AUC du modèle sur la référence − AUC du même modèle sur le lot courant. Positif = le modèle se dégrade. Les lignes sans label sont ignorées. `null` = **non calculable** : pas de modèle, pas de colonne label, ou, dans l'un des deux lots, moins de `monitoring.min_labelled` (1 000) lignes labellisées, moins de `monitoring.min_positives` (50) défauts, ou une seule classe |
| `alert` | booléen | | Règle de la section 5 |

`ks` et `chi2` contiennent des **p-values**, pas des statistiques : plus la valeur est petite, plus la dérive est certaine. Les colonnes hors modèle (`SK_ID_CURR`, `TARGET`, `CODE_GENDER`) ne sont jamais comparées.

## 3. Tests et `drift_share`

- **α = 0,05** (`monitoring.alpha`).
- **Numériques : KS** (`scipy.stats.ks_2samp`) sur les valeurs **non manquantes**. Si **un seul** des deux lots n'a aucune valeur, la p-value vaut **0** : la feature compte comme dérivée (toutes ses valeurs ont disparu, ou sont apparues). Si **aucun** des deux n'a de valeur, la p-value est `null` et la feature ne compte pas comme dérivée.
- **Catégorielles : χ²** (`scipy.stats.chi2_contingency`, réglage par défaut) sur la table 2 × catégories ; **manquant = une catégorie**, une catégorie absente de la référence aussi. Avec deux catégories (1 degré de liberté), scipy applique la **correction de Yates** (testé). Une seule catégorie au total : p-value = 1.
- `drift_share` = (nombre de features avec p-value < α) / 41. Une p-value `null` compte comme non dérivée.
- Sur de gros lots, KS et χ² détectent de très petits écarts : `drift_share` dit « combien de features ont bougé », le PSI dit « de combien ».
- **Petits lots.** Les p-values et les PSI sont toujours calculés et publiés. Mais si le plus petit des deux lots a moins de `monitoring.min_rows` (500) lignes, `drift_share` et le PSI **ne déclenchent pas** l'alerte (section 5) : sur 100 lignes, le PSI des features clés vaut déjà environ 0,14 sans aucune dérive.

## 4. PSI (écrit à la main)

PSI = Σ sur les bins de (part courante − part référence) × ln(part courante / part référence).

- **Numériques : des intervalles, dont aucune borne ne tombe sur une valeur de la référence.**
  - On prend les quantiles intérieurs de la **référence** (`psi_bins` = 10 : les 9 déciles intérieurs, méthode `lower`, donc des valeurs réelles de la référence), **dédoublonnés**.
  - Chaque borne intérieure est placée **à mi-chemin entre ce quantile et la valeur distincte suivante** de la référence. Exemple réel (revenu, test) : décile à 135 000, valeur distincte suivante 136 350 → borne à 135 675 ; les 11 bornes vont de 26 707,5 à 3 937 500.
  - Deux bins extérieurs ouverts : sous le minimum et au-dessus du maximum, à une demi-distance au-delà (min − (2ᵉ valeur − min)/2 ; max + (max − avant-dernière)/2).
  - Le **dédoublonnage** fusionne les déciles qui tombent sur la même valeur, très fréquente (revenus ronds, 0/1) : une seule borne au lieu de plusieurs. Une colonne 0/1 a une seule borne intérieure, à 0,5 : deux bins « 0 » et « 1 », plus les deux bins extérieurs.
  - Colonne constante c : un bin [c − 1e-6·max(|c|, 1) ; c + 1e-6·max(|c|, 1)[, plus les deux bins extérieurs.
  - **Pourquoi :** une valeur fréquente (37 % du revenu de test tombe exactement sur un décile) reste dans son bin si on la décale d'un centime, d'un facteur 1 ± 1e-9 ou d'un aller-retour float32. La version précédente faisait de chaque quantile un bin d'un seul point : test contre test × (1 + 1e-9) donnait un PSI de 2,9 sur le revenu et une alerte (corrigé et testé).
- **Catégorielles.** Un bin par catégorie présente dans l'un ou l'autre lot (catégorie inconnue de la référence = son propre bin).
- **Manquants.** Toujours un bin à part (numériques et catégorielles) ; les manquants de la référence n'entrent jamais dans les quantiles.
- **Epsilon = 1e-4.** Chaque part est plancher à 1e-4 avant le calcul, pour qu'un bin vide donne un terme fini. Un bin vide des deux côtés donne 0.
- **Sens.** La référence fixe les bins : `psi(ref, cur)` diffère en général de `psi(cur, ref)`, mais reste du même ordre (testé : rapport entre 0,5 et 2 pour une loi normale décalée de 0,5). La référence est toujours le premier argument (`--reference`).
- Lecture usuelle : moins de 0,1 stable, 0,1 à 0,2 à surveiller, 0,2 et plus dérive.

## 5. Alerte

```
n = nombre de lignes du plus petit des deux lots
alert = (n >= 500 ET (drift_share >= 0.30 OU PSI >= 0.20 sur une feature clé))
     OU auc_drop > 0.03, quand auc_drop n'est pas null
```

Seuils : `monitoring.min_rows`, `monitoring.drift_share_alert`, `monitoring.psi_alert`, `monitoring.auc_drop_alert`.

- **Features clés** (`monitoring.key_features`) : `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `ANNUITY_INCOME_RATIO`, `EXT_SOURCE_2`. Ce doivent être des features du modèle (vérifié par `config.py`).
- Bornes testées : `drift_share` = 0,30 déclenche ; PSI = 0,20 sur une feature clé déclenche ; un PSI élevé sur une feature **non** clé ne déclenche pas seul ; `auc_drop` = 0,03 ne déclenche pas, 0,0301 oui ; une AUC qui s'améliore (`auc_drop` négatif) ne déclenche pas ; 499 lignes : ni `drift_share` ni le PSI ne déclenchent, 500 lignes oui ; la règle `auc_drop` s'applique quelle que soit la taille (elle a déjà son propre minimum, `min_labelled` = 1 000 lignes labellisées et `min_positives` = 50 défauts).
- **Petits lots labellisés.** Sur le test réel, avec 1 000 lignes labellisées, l'écart type de `auc_drop` sans dérive est de 0,021 (1 fausse alerte sur 40 tirages) ; avec 2 000, 0,014 (aucune). En dessous de 1 000, `auc_drop` est `null`.
- Le JSON ne dit pas quelle règle a déclenché (clés fixes) ; les consommateurs lisent les jauges (section 6) et **ne réimplémentent pas** la règle.

## 6. Qui lit quoi

| Consommateur | Lit | Pour |
|---|---|---|
| Iyed, `monitoring/metrics.py` | toutes les valeurs, en jauges : `drift_alert` (0/1, depuis `alert`), `drift_auc_drop` (absente si `null`), `drift_share`, `psi{feature}` | **Les règles d'alerte Prometheus portent sur la jauge `alert` (0/1) et sur `auc_drop`**, pas sur chaque `psi{feature}` : la règle complète (petits lots, features clés) est déjà appliquée par `drift.py`. Une règle par feature, si on en veut une pour l'affichage, se limite aux features clés : `psi{feature=~"AMT_INCOME_TOTAL|AMT_CREDIT|ANNUITY_INCOME_RATIO|EXT_SOURCE_2"}`. Les autres jauges servent aux tableaux de bord |
| Areej, `flows/monitoring_flow.py` | `alert` | `alert = true` → `run_deployment` du `training_flow` ; archive le JSON et le HTML |
| Rapport et démo (2:30) | le HTML Evidently, `psi`, `auc_drop` | Analyse de la dérive par feature |

## 7. Rapport HTML Evidently

- `evidently==0.7.23`. API 0.7 : `Report([DataDriftPreset(...)]).run(current, reference)` (le lot **courant en premier**) sur des `Dataset.from_pandas(..., data_definition=DataDefinition(...))`, puis `snapshot.save_html(path)`. `ColumnMapping` et `evidently.metric_preset` de la 0.4 ne sont plus que dans `evidently.legacy` : ne pas copier un exemple 0.4.
- Mêmes colonnes (les 41 features), même α, KS pour les numériques (`num_method="ks"`), χ² pour les catégorielles (`cat_method="chisquare"`), seuil de `drift_share` de l'aperçu = 0,30 (le HTML affiche « Dataset drift detection threshold is 0.3 », testé).
- **Le bandeau du HTML n'est pas notre alerte.** Il applique la seule règle d'Evidently : « drift du jeu de données si ≥ 30 % des colonnes dérivent ». Sur la crise, il affiche « Dataset Drift is NOT detected » (7 colonnes sur 41, 17 %) alors que notre JSON dit `alert = true` (PSI des features clés et `auc_drop`). **Le JSON fait foi** ; à la démo, montrer le PSI par feature et le JSON, pas le bandeau.
- **Écart connu :** le « chi-square » d'Evidently est un test d'adéquation qui prend les proportions de la référence comme exactes ; notre χ² est le test d'indépendance à deux échantillons. Sur les catégorielles à beaucoup de modalités, Evidently signale plus souvent (test contre test : 3 catégorielles signalées par Evidently, 0 par nous). Sur les numériques, les deux donnent les mêmes features (testé).
- Taille : environ 6 Mo par rapport (20 000 lignes, 41 colonnes).

## 8. Exemple réel

Produit par la **CLI avec `--model models`** (pas par l'étape DVC, qui tourne sans modèle et donne donc `auc_drop` = `null`) : lot « crise économique » d'Areej (`drift.seed = 7`) contre `data/processed/test.parquet`, 20 000 lignes chacun. Le modèle est le **champion du lot 3** (`models/model.joblib` de `dvc repro -s train`, md5 `66def444…`, celui de `reports/metrics.json`) : AUC 0,747 sur le test, 0,738 sur la crise (recalculé le 25/11/2026 ; l'ancien exemple venait d'un artefact local du 25/10/2026). Valeurs arrondies pour la lecture ; le fichier réel garde la précision complète.

```json
{
  "timestamp": "2026-11-25T00:14:29+00:00",
  "drift_share": 0.1707,
  "psi": {
    "AMT_INCOME_TOTAL": 0.2624,
    "AMT_CREDIT": 0.2314,
    "AMT_ANNUITY": 0.1776,
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
    "ANNUITY_INCOME_RATIO": 0.6758,
    "CREDIT_INCOME_RATIO": 0.4093,
    "CREDIT_TERM": 0.089,
    "AGE_YEARS": 0.0,
    "EMPLOYED_YEARS": 0.0992,
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
  "auc_drop": 0.0092,
  "alert": true
}
```

Lecture : 7 features dérivées sur 41 (`drift_share` 0,171, sous 0,30), mais PSI ≥ 0,20 sur trois features clés (revenu 0,26, crédit 0,23, ratio annuité/revenu 0,68) : `alert = true`. `auc_drop` 0,009 reste **sous** 0,03 : le champion classe presque aussi bien les dossiers de la crise ; l'alerte vient de la dérive des données, pas de la performance. `EXT_SOURCE_2`, que le simulateur ne touche pas, reste à 0.

Pour comparaison (mêmes commandes et même champion, chiffres dans le plan, lot 5) : deux moitiés aléatoires du test l'une contre l'autre donnent `drift_share` 0,024, PSI max 0,011 (`ORGANIZATION_TYPE`), `auc_drop` 0,005, `alert = false` ; le scénario C (30 % des labels « Commercial associate » inversés) donne `drift_share` 0, tous les PSI à 0, `auc_drop` 0,126 (AUC 0,747 → 0,621), `alert = true` : seule l'AUC voit une dérive de concept.

## 9. Ajouts du lot 6 : dérive des prédictions et équité dans le temps (proposition, pour relecture d'Areej et d'Iyed)

Deux clés **optionnelles**, ajoutées **après `alert`** (les sept clés existantes gardent leur ordre et leur sens). Elles sont **toujours présentes** dans le JSON, et valent **`null`** quand elles ne sont pas calculées : un consommateur teste `is None`, il ne teste jamais l'absence de la clé. Choix : des clés fixes restent la règle du contrat (section 2). Ni l'une ni l'autre ne **change la règle d'alerte** (section 5) : elles sont informatives, publiées pour les jauges d'Iyed et le rapport.

### 9.1 `prediction` (dérive des prédictions)

`null` sans `--model`. Avec `--model`, calculé sur **toutes** les lignes des deux lots (pas besoin de label) :

| Champ | Type | Définition |
|---|---|---|
| `psi_proba` | nombre ≥ 0 | PSI des probabilités de défaut du modèle, lot courant contre référence, **même méthode** que pour les features (section 4, `monitoring.psi_bins`, bornes sur la référence). Les probabilités isotoniques sont en paliers : le dédoublonnage des quantiles s'applique |
| `acceptance_rate_ref` | nombre dans [0, 1] | Part de `ACCEPTE` sur la référence, avec les seuils du `thresholds.json` du modèle (D8 : `A_ETUDIER` n'est pas accepté) |
| `acceptance_rate_cur` | nombre dans [0, 1] | Idem sur le lot courant |
| `acceptance_rate_delta_pts` | nombre | `100 × (acceptance_rate_cur − acceptance_rate_ref)`, en points ; négatif = le modèle accepte moins |

### 9.2 `fairness` (équité dans le temps)

`null` sans `--model`, sans colonne `TARGET` ou `CODE_GENDER` dans le lot courant, ou quand les lignes labellisées du lot courant ne passent pas **la même garde que `auc_drop`** : moins de `monitoring.min_labelled` (1 000) lignes labellisées, moins de `monitoring.min_positives` (50) défauts, ou une seule classe. Sinon : audit `crediscore.fairness.audit` (le même que `evaluate`) sur les lignes **labellisées** du lot courant, avec les **décisions du modèle** (seuils de son `thresholds.json`).

| Champ | Type | Définition |
|---|---|---|
| `n_labelled` | entier | Lignes labellisées auditées |
| `gender` | objet | `raw_gap_pts`, `eo_gap_pts` (points), `acceptance_by_group`, `acceptance_good_payers_by_group` (parts), `excluded_rows` (`XNA` ou vide) : exactement les champs de `reports/metrics.json` |
| `age` | objet | Mêmes champs, tranches `<30`, `30-50`, `≥50` |
| `d5` | objet ou `null` | `{age_anchor_pts, checks, pass, reason}` : les trois conditions D5 (`gender_raw_gap`, `gender_eo_gap`, `age_eo_gap`, chacune `{value, limit, pass}`, bornes incluses, tolérances `params.promotion`) contre l'ancre d'âge lue dans `--metrics` (`baseline.fairness.age.eo_gap_pts`) ; `pass` = les trois, `reason` = `null`. **Sans comparaison possible selon le sexe** (moins de 2 groupes de sexe, `XNA` exclu, ayant chacun au moins `promotion.min_group_rows` = 500 lignes labellisées : la règle de `promote`) : `pass` = `null`, `reason` = `"gender_groups_missing"`, `checks` et écarts toujours publiés (un seul groupe donnerait un écart de 0, donc un faux « respecté »). `d5` = `null` si le fichier ou l'ancre manque |

Un `d5.pass = false` n'est **pas** une alerte et ne déclenche aucun réentraînement : c'est un signal pour l'équipe (et le rapport). La promotion, elle, refuse toujours un challenger qui échoue D5 (contrat 2 §1).

### 9.3 Qui lit quoi (ajout à la section 6)

| Consommateur | Lit | Proposition |
|---|---|---|
| Iyed, `monitoring/metrics.py` | `prediction`, `fairness` | Jauges `drift_psi_proba`, `acceptance_rate` (le SYNOPSIS la cite) depuis `acceptance_rate_cur`, `fairness_gap_pts{group="gender"\|"age", kind="raw"\|"eo"}`, `fairness_d5_pass` (0/1). Jauge absente quand la clé vaut `null`. **Pas de règle d'alerte** dessus |
| Areej, `monitoring_flow` | rien de plus | La décision de réentraîner reste `alert` |

### 9.4 Exemple réel (25/11/2026)

Même commande que la section 8, sur un serveur MLflow jetable (`--model models:/crediscore@champion`, `--metrics reports/metrics.json`). Le champion y est obtenu par l'amorçage documenté (passation, `docs/handoff/boucle-reentrainement.md`) : `train` avec les paramètres figés sur `train.parquet`, puis `registry --select` et `promote`. C'est le même modèle que `models/model.joblib` (entraînement déterministe : AUC de validation 0,7670, mêmes seuils) ; la relecture des lots 6-7 a rejoué cet amorçage et retrouvé les mêmes chiffres. Les sept premières clés sont identiques à la section 8 :

```json
"prediction": {
  "psi_proba": 0.0305,
  "acceptance_rate_ref": 0.67315,
  "acceptance_rate_cur": 0.62465,
  "acceptance_rate_delta_pts": -4.85
},
"fairness": {
  "n_labelled": 20000,
  "gender": {"raw_gap_pts": 7.77, "eo_gap_pts": 6.46, "...": "..."},
  "age": {"raw_gap_pts": 35.46, "eo_gap_pts": 34.00, "...": "..."},
  "d5": {"age_anchor_pts": 38.675, "checks": {"age_eo_gap": {"value": 34.00, "limit": 40.675, "pass": true}, "...": "..."}, "pass": true, "reason": null}
}
```

Lecture : la crise ne déplace presque pas la distribution des scores (`psi_proba` 0,03, « stable »), mais le modèle **accepte 4,85 points de dossiers en moins** (67,3 % → 62,5 %) : les revenus baissent et l'endettement monte, les scores glissent juste au-dessus du seuil `low`. L'équité reste dans D5 (sexe 7,8 / 6,5 pts ; âge en égalité des chances 34,0 pts, limite 40,7). `acceptance_rate_ref` = 0,67315 est exactement l'`accept_share` du champion sur le test dans `reports/metrics.json`.
