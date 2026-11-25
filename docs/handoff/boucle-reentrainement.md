# Passation : boucle de réentraînement (lot 6, Achraf → Areej, Iyed)

- **Pour :** Areej (`flows/monitoring_flow.py`, `flows/training_flow.py`, `prefect.yaml`), Iyed (déploiement Blue/Green après promotion).
- **Statut :** proposition du 25/11/2026, à relire par Areej et Iyed. Les flows Prefect n'existent pas encore : ce document donne la **séquence exacte de commandes**, leurs **codes de sortie** et les **fichiers** à lire, pour que les flows n'aient qu'à les enchaîner. Le câblage Prefect (tâches, retries, CRON, `run_deployment`) est le travail d'Areej.
- **Preuve que la chaîne marche :**
  - `tests/test_monitoring_loop.py` : la séquence complète (premier modèle promu par `promote` → alerte de dérive → réentraînement → `registry --select` → `promote` → alias, puis code 3 quand rien n'a changé) sur un MLflow sqlite et des données synthétiques, en environ 20 s ;
  - exécution réelle du 25/11/2026 sur un serveur MLflow jetable (section 6).

## 1. Prérequis

- `MLFLOW_TRACKING_URI` défini (ex. `http://mlflow:5000` dans un conteneur, `http://localhost:5000` depuis l'hôte avec `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`, contrat 2 §9). Toutes les commandes passent par `poetry run` (ou l'environnement de l'image).
- Un **champion** existe déjà (`models:/crediscore@champion`) : c'est lui que l'API sert et que la dérive compare. **Amorçage** (une seule fois, au démarrage du projet) : étapes 2, 3 et 4 ci-dessous sans dérive ; le premier modèle est enregistré `@challenger` comme les autres, et `promote` le juge sur l'équité seule (pas de champion à battre). Code 0 = le déployer.
- `reports/metrics.json` (sortie d'`evaluate`, versionnée dans Git) : il porte l'**ancre d'âge D5** (`baseline.fairness.age.eo_gap_pts`, 38,7 pts) lue par la dérive et par la promotion. S'il manque, la promotion **refuse** (`age_anchor_missing`).
- `data/processed/holdout.parquet` (étape DVC `holdout`, 20 000 lignes étiquetées hors échantillon) : le **lot de promotion**. Jamais `test.parquet` ni `economic_crisis.parquet` pour promouvoir (contrat 2 §1).

## 2. La séquence

| Étape | Commande | Code de sortie | Écrit |
|---|---|---|---|
| 1. Dérive (`monitoring_flow`) | `python -m crediscore.monitoring.drift --reference data/processed/test.parquet --current <lot courant>.parquet --model models:/crediscore@champion --metrics reports/metrics.json --out-dir reports/drift` | **0 quelle que soit l'alerte** ; ≠ 0 = vraie erreur (fichier absent, colonne manquante, lot vide) | `reports/drift/<AAAA-MM-JJ_HHMMSS>.json`, `.html` (Evidently), `reports/drift/latest.json` |
| Décision | lire `reports/drift/latest.json`, clé **`alert`** | | |
| → `alert = false` | fin du flow (archiver le JSON et le HTML) | | |
| → `alert = true` | **si les données d'entraînement ont changé** (remarque 2 ci-dessous) : `run_deployment` du `training_flow` ; sinon fin du flow (rien à réentraîner) | | |
| 2. Entraînement (`training_flow`) | `python -m crediscore.models.train --model lightgbm --model-params configs/champion_model_params.json --train <fichier d'entraînement> --output-dir models/challenger` | 0 ; ≠ 0 = erreur | `models/challenger/model.joblib`, `thresholds.json`, `metrics.json`, `model_params.json` (**jamais `models/`** : c'est la sortie DVC de l'étape `train`) ; un run MLflow `lightgbm` (expérience `crediscore`) avec le modèle journalisé et ses tags |
| 3. Enregistrement | `python -m crediscore.models.registry --select --train <le même fichier>` | **0** = une version créée ; **3** = le run choisi est déjà celui du champion : **rien à faire, pas de déploiement**, fin du flow ; **1** (message sur la sortie d'erreur) = aucun candidat valide, ou le gagnant est une expérience (`no_age`) | une nouvelle version de `crediscore`, **toujours** alias **`challenger`**, même pour le tout premier modèle ; la sortie standard finit par `registered crediscore version N as @challenger` (code 3 : `nothing to register: selected run is already champion`) |
| 4. Promotion | `python -m crediscore.models.promote --data data/processed/holdout.parquet` | **0 = promu, déployer** (y compris le premier modèle) ; **1 = on garde le modèle actuel** (gate échoué, garde-fou, ou erreur) | `reports/promotion.json` (réécrit à chaque exécution, ignoré par Git) ; si promu : tags de la version, alias `challenger` supprimé, alias `champion` déplacé |
| → code 0 | appeler le déploiement Blue/Green d'Iyed (`scripts/deploy_blue_green.py`, commande exacte à fixer par Iyed) ; l'API recharge `models:/crediscore@champion` | | |
| → code 1 | rien : l'ancien champion reste servi ; journaliser `failed` de `promotion.json` | | |

Remarques sur chaque étape :

1. **Dérive.**
   - `--reference` : le lot de référence du modèle en service. Aujourd'hui `data/processed/test.parquet` (comme l'étape DVC `detect_drift`).
   - Avec `--model`, le JSON porte aussi `prediction` et `fairness` (contrat 4 §9), informatifs : **seule `alert` décide** du réentraînement.
   - L'étape DVC `detect_drift` tourne **sans** modèle (`auc_drop`, `prediction`, `fairness` à `null`) ; le flow doit appeler la CLI avec `--model` pour avoir le concept drift.
   - En Python, si le flow préfère importer : `compute_drift_report(...)`, `auc_drop(...)`, `prediction_drift(...)`, `fairness_drift(...)`, `write_html(...)` (docstring de `crediscore.monitoring.drift`).
2. **Entraînement : paramètres figés du champion, pas `--tune`.**
   - On réentraîne le **même modèle** (LightGBM) avec les hyperparamètres Optuna **figés** dans `configs/champion_model_params.json`, exactement comme l'étape DVC `train`.
   - Pourquoi :
     - c'est reproductible et rapide (environ 1 min sur 80 000 lignes, plan lot 3 ; `--tune 25` ajoute 25 validations croisées à 5 plis, bien plus longues, non chronométrées ici) ;
     - le challenger ne diffère du champion **que par les données**, donc la comparaison de la promotion se lit simplement ;
     - le réglage Optuna (`--tune 25`) reste une expérience ponctuelle, relue, qui change `configs/champion_model_params.json` par un commit (plan, lot 3).
   - Le **fichier d'entraînement** est le choix d'Areej (données rafraîchies avec les lots labellisés récents, via son pipeline DVC).
   - **Ne réentraîner que si ce fichier a changé** : comparer son md5 (`crediscore.models.train.file_md5`) au tag `train_data_hash` de la version `@champion` (`MlflowClient().get_model_version_by_alias("crediscore", "champion").tags`). Égal = mêmes données, donc même modèle ; sans cette garde, une dérive qui persiste relancerait la boucle à chaque passage. `registry` sort de toute façon avec le code 3 si le run choisi est celui du champion.
   - `--output-dir models/challenger`, jamais `models/` : `models/` est la sortie DVC de l'étape `train` (le champion versionné).
3. **Enregistrement.**
   - `--select` choisit parmi les runs `train` **terminés, de premier niveau, entraînés sur le fichier `--train`** (son md5 doit être égal au tag `train_data_hash`), avec les coûts et contraintes du `params.yaml` courant (règle L3-1 : AUC de validation ≥ 0,75 si possible, puis coût de validation le plus bas).
   - **Passer le même `--train` qu'à l'étape 2.** Sinon, `--select` cherche des runs sur `data/processed/train.parquet` et peut choisir un run plus ancien.
   - **Pas de retry automatique :** chaque appel réussi crée une **nouvelle version**. Si l'étape échoue avant la ligne `registered …`, la relance est sans risque.
   - Une panne entre la création de la version et la pose de l'alias laisse une **version orpheline** (sans alias) : `promote` l'ignore (il ne lit que les alias), et la relance de `registry` en crée une nouvelle. Sans danger ; elle reste visible dans le Registry.
4. **Promotion.**
   - Champion et challenger sont évalués sur le **même** lot (`holdout.parquet`), chacun avec ses propres seuils.
   - Le challenger est promu si : AUC ≥ champion, coût ≤ champion, écarts selon le sexe ≤ 10 pts (brut et égalité des chances), écart d'âge en égalité des chances ≤ ancre + 2 pts (contrat 2 §1).
   - La relance est **sans risque** : une fois promu, l'alias `challenger` n'existe plus, donc une relance répond `no_challenger` (code 1) sans rien changer.

## 3. Cas particuliers

| Cas | Ce qui se passe | Que faire dans le flow |
|---|---|---|
| Fichier d'entraînement **inchangé** | Le flow ne lance pas le `training_flow` (garde sur `train_data_hash`, section 2). Sans la garde, le nouveau run est identique à celui du champion : à égalité parfaite (coût et AUC), `--select` garde le **run du champion**, donc `registry` sort avec le **code 3** sans rien enregistrer (rejoué le 25/11 : deux cycles, une seule version). La garde évite surtout un entraînement inutile | Garde recommandée ; code 3 = fin du flow, pas de déploiement |
| Challenger meilleur mais **injuste** | `promote` : code 1, `failed: ["age_eo_gap"]` (par exemple), alias inchangés : le challenger reste `challenger` | Garder le modèle actuel ; signaler à l'équipe. Le test de bout en bout le montre sur données synthétiques |
| Lot de promotion trop petit | `batch_too_small` (< 5 000 lignes) ou `gender_groups_missing` (< 2 groupes de 500) | Code 1 ; attendre un lot plus grand |
| Toute erreur (MLflow injoignable, fichier absent…) | `promote` écrit `failed: ["error", "<message>"]`, code 1 ; rien n'a changé, sauf si la panne arrive en cours d'écriture (ligne suivante, contrat 2 §1) | Retry Prefect possible |
| Panne **entre** la suppression de `challenger` et le déplacement de `champion` | L'ancien champion reste servi, mais l'alias `challenger` est perdu : la relance répond `no_challenger` (code 1) | Reposer l'alias à la main sur la version `challenger_version` de `promotion.json` (`MlflowClient().set_registered_model_alias("crediscore", "challenger", "<N>")`), puis relancer `promote` |

## 4. Lire `reports/promotion.json`

```json
{
  "timestamp": "2026-11-25T00:42:41+00:00",
  "dry_run": false,
  "data": "data\\processed\\holdout.parquet",
  "data_md5": "8bc410e828b2db35adcdcd61366f0b55",
  "n_rows": 20000,
  "model": "crediscore",
  "challenger_version": "2",
  "champion_version": "1",
  "age_anchor_pts": 38.67539487501559,
  "applied": true,
  "scores": {"challenger": {"roc_auc": 0.7591, "cost_per_file": 0.3141, "fairness": "…"}, "champion": "…"},
  "promote": true,
  "failed": [],
  "checks": {
    "auc": {"value": 0.7591, "limit": 0.7591, "pass": true},
    "cost": {"value": 0.3141, "limit": 0.3141, "pass": true},
    "gender_raw_gap": {"value": 7.92, "limit": 10.0, "pass": true},
    "gender_eo_gap": {"value": 6.87, "limit": 10.0, "pass": true},
    "age_eo_gap": {"value": 32.18, "limit": 40.68, "pass": true}
  }
}
```

- **`promote`** : la décision (le code de sortie la reflète). **`applied`** : `true` si les alias ont vraiment bougé (`false` avec `--dry-run`).
- **`failed`** : liste vide si promu ; sinon les conditions en échec (`auc`, `cost`, `gender_raw_gap`, `gender_eo_gap`, `age_eo_gap`) ou un code de garde-fou (`no_challenger`, `already_champion`, `age_anchor_missing`, `batch_too_small`, `gender_groups_missing`, `thresholds_mismatch`), ou `["error", "<message>"]`.
- **`checks`** : chaque condition `{value, limit, pass}` ; pour `auc` et `cost`, `limit` est la valeur du champion.
- **`scores`** : toutes les métriques des deux modèles sur le lot (AUC, PR-AUC, Brier, coût, recall, part en étude, audit d'équité complet).
- Version à déployer : `challenger_version` si `promote` est vrai (c'est le nouveau `champion`).

## 5. Esquisse pour les flows (à adapter par Areej)

```python
import json
import subprocess
from pathlib import Path

def run(*args: str) -> int:
    return subprocess.run(["python", "-m", *args]).returncode

# monitoring_flow
assert run("crediscore.monitoring.drift", "--reference", "data/processed/test.parquet",
           "--current", current, "--model", "models:/crediscore@champion",
           "--metrics", "reports/metrics.json", "--out-dir", "reports/drift") == 0
if json.loads(Path("reports/drift/latest.json").read_text())["alert"]:
    ...  # run_deployment("training_flow/...")

# training_flow
import mlflow
from crediscore.models.train import file_md5

champion = mlflow.MlflowClient().get_model_version_by_alias("crediscore", "champion")
if file_md5(Path(train_file)) == champion.tags["train_data_hash"]:
    return  # mêmes données : rien à réentraîner
assert run("crediscore.models.train", "--model", "lightgbm",
           "--model-params", "configs/champion_model_params.json",
           "--train", train_file, "--output-dir", "models/challenger") == 0
code = run("crediscore.models.registry", "--select", "--train", train_file)  # pas de retry
if code == 3:
    return  # le run choisi est déjà le champion : rien à faire, pas de déploiement
assert code == 0
if run("crediscore.models.promote", "--data", "data/processed/holdout.parquet") == 0:
    ...  # déploiement Blue/Green d'Iyed (aussi pour le tout premier modèle)
else:
    decision = json.loads(Path("reports/promotion.json").read_text())
    ...  # journaliser decision["failed"] ; garder le modèle actuel
```

## 6. Vérification réelle (25/11/2026, serveur MLflow jetable)

Serveur jetable sqlite sur le port 5341 (jamais celui de l'équipe), arrêté après coup :

1. v1 `champion` = le champion du lot 3. L'amorçage documenté (section 1) est `train` avec les paramètres figés sur `train.parquet`, puis `registry --select` et `promote` ; la relecture des lots 6-7 l'a rejoué et a obtenu le même modèle (validation AUC 0,7670, mêmes seuils, mêmes chiffres de dérive ci-dessous). Lors de notre exécution, `models/model.joblib` avait été enregistré directement par un script jetable, avant que `registry` ne passe toujours par `@challenger` ;
2. dérive, lot « crise économique » contre `test.parquet` avec `--model models:/crediscore@champion` : `drift_share` 0,171, PSI ratio annuité/revenu 0,676, revenu 0,262, crédit 0,231, `auc_drop` 0,009, **`alert = true`** ; `prediction` : `psi_proba` 0,031, acceptation 67,3 % → 62,5 % ; `fairness` : D5 respecté ;
3. `train --model lightgbm --model-params configs/champion_model_params.json` sur `train.parquet` : validation AUC 0,7670, coût 0,3070 par dossier (mêmes chiffres que le champion, données inchangées) ;
4. `registry --select` : deux candidats à égalité, le run le plus récent est choisi, **v2 `challenger`**. C'est la boucle sans fin corrigée depuis : la garde sur `train_data_hash` n'aurait pas lancé ce réentraînement, et `--select` garde désormais le run du champion sur une égalité parfaite (code 3) ;
5. `promote --data data/processed/holdout.parquet` : **code 0, promu** (AUC 0,7591 = 0,7591, coût 0,3141 = 0,3141, sexe 7,9 / 6,9 pts, âge en égalité des chances 32,2 ≤ 40,7) ; Registry ensuite : **`champion` → v2**, plus d'alias `challenger`, v2 porte `promotion_previous_champion = 1` et `promoted_at`.

Le challenger est ici le même modèle que le champion (mêmes données, mêmes paramètres, entraînement déterministe) : la vérification prouve la **chaîne**, pas un gain. Un vrai gain demande des données rafraîchies (étape 2).

**Rejeu après correction (25/11/2026, serveur jetable, port 5341, arrêté ensuite)** : amorçage `train` → `registry --select` (**v1 `@challenger`**) → `promote` (**code 0**, équité seule : sexe 7,9 / 6,9 pts, âge en égalité des chances 32,2 ≤ 40,7) → `champion` = v1 ; dérive de la crise : `alert = true` ; réentraînement sur le même `train.parquet` puis `registry --select` : égalité, run du champion gardé, **code 3**, `nothing to register: selected run is already champion` ; un second cycle donne encore le code 3. À la fin : une seule version, `champion` = v1.
