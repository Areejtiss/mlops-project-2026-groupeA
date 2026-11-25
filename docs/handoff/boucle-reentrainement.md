# Passation : boucle de réentraînement (lot 6, Achraf → Areej, Iyed)

- **Pour :** Areej (`flows/monitoring_flow.py`, `flows/training_flow.py`, `prefect.yaml`), Iyed (déploiement Blue/Green après promotion).
- **Statut :** proposition du 25/11/2026, à relire par Areej et Iyed. Les flows Prefect n'existent pas encore : ce document donne la **séquence exacte de commandes**, leurs **codes de sortie** et les **fichiers** à lire, pour que les flows n'aient qu'à les enchaîner. Le câblage Prefect (tâches, retries, CRON, `run_deployment`) est le travail d'Areej.
- **Preuve que la chaîne marche :**
  - `tests/test_monitoring_loop.py` : la séquence complète (champion initial → alerte de dérive → réentraînement → `registry --select` → `promote` → alias) sur un MLflow sqlite et des données synthétiques, en environ 13 s ;
  - exécution réelle du 25/11/2026 sur un serveur MLflow jetable (section 6).

## 1. Prérequis

- `MLFLOW_TRACKING_URI` défini (ex. `http://mlflow:5000` dans un conteneur, `http://localhost:5000` depuis l'hôte avec `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`, contrat 2 §9). Toutes les commandes passent par `poetry run` (ou l'environnement de l'image).
- Un **champion** existe déjà (`models:/crediscore@champion`) : c'est lui que l'API sert et que la dérive compare.
- `reports/metrics.json` (sortie d'`evaluate`, versionnée dans Git) : il porte l'**ancre d'âge D5** (`baseline.fairness.age.eo_gap_pts`, 38,7 pts) lue par la dérive et par la promotion. S'il manque, la promotion **refuse** (`age_anchor_missing`).
- `data/processed/holdout.parquet` (étape DVC `holdout`, 20 000 lignes étiquetées hors échantillon) : le **lot de promotion**. Jamais `test.parquet` ni `economic_crisis.parquet` pour promouvoir (contrat 2 §1).

## 2. La séquence

| Étape | Commande | Code de sortie | Écrit |
|---|---|---|---|
| 1. Dérive (`monitoring_flow`) | `python -m crediscore.monitoring.drift --reference data/processed/test.parquet --current <lot courant>.parquet --model models:/crediscore@champion --metrics reports/metrics.json --out-dir reports/drift` | **0 quelle que soit l'alerte** ; ≠ 0 = vraie erreur (fichier absent, colonne manquante, lot vide) | `reports/drift/<AAAA-MM-JJ_HHMMSS>.json`, `.html` (Evidently), `reports/drift/latest.json` |
| Décision | lire `reports/drift/latest.json`, clé **`alert`** | | |
| → `alert = false` | fin du flow (archiver le JSON et le HTML) | | |
| → `alert = true` | `run_deployment` du `training_flow` | | |
| 2. Entraînement (`training_flow`) | `python -m crediscore.models.train --model lightgbm --model-params configs/champion_model_params.json --train <fichier d'entraînement> --output-dir models` | 0 ; ≠ 0 = erreur | `models/model.joblib`, `models/thresholds.json`, `models/metrics.json`, `models/model_params.json` ; un run MLflow `lightgbm` (expérience `crediscore`) avec le modèle journalisé et ses tags |
| 3. Enregistrement | `python -m crediscore.models.registry --select --train <le même fichier>` | 0 = une version créée ; ≠ 0 (message sur la sortie d'erreur) = aucun candidat valide, ou le gagnant est une expérience (`no_age`) | une nouvelle version de `crediscore`, alias **`challenger`** (ou **`champion`** s'il n'y en avait pas, section 3) ; la sortie standard finit par `registered crediscore version N as @challenger` |
| 4. Promotion | `python -m crediscore.models.promote --data data/processed/holdout.parquet` | **0 = promu** ; **1 = on garde le modèle actuel** (gate échoué, garde-fou, ou erreur) | `reports/promotion.json` (réécrit à chaque exécution, ignoré par Git) ; si promu : tags de la version, alias `challenger` supprimé, alias `champion` déplacé |
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
   - Le **fichier d'entraînement** est le choix d'Areej (données rafraîchies avec les lots labellisés récents, via son pipeline DVC). `dvc repro train` est équivalent, si `MLFLOW_TRACKING_URI` est défini dans l'environnement du flow ; DVC saute l'étape si `train.parquet` n'a pas changé.
3. **Enregistrement.**
   - `--select` choisit parmi les runs `train` **terminés, de premier niveau, entraînés sur le fichier `--train`** (son md5 doit être égal au tag `train_data_hash`), avec les coûts et contraintes du `params.yaml` courant (règle L3-1 : AUC de validation ≥ 0,75 si possible, puis coût de validation le plus bas).
   - **Passer le même `--train` qu'à l'étape 2.** Sinon, `--select` cherche des runs sur `data/processed/train.parquet` et peut choisir un run plus ancien.
   - **Pas de retry automatique :** chaque appel réussi crée une **nouvelle version**. Si l'étape échoue avant la ligne `registered …`, la relance est sans risque.
4. **Promotion.**
   - Champion et challenger sont évalués sur le **même** lot (`holdout.parquet`), chacun avec ses propres seuils.
   - Le challenger est promu si : AUC ≥ champion, coût ≤ champion, écarts selon le sexe ≤ 10 pts (brut et égalité des chances), écart d'âge en égalité des chances ≤ ancre + 2 pts (contrat 2 §1).
   - La relance est **sans risque** : une fois promu, l'alias `challenger` n'existe plus, donc une relance répond `no_challenger` (code 1) sans rien changer.

## 3. Cas particuliers

| Cas | Ce qui se passe | Que faire dans le flow |
|---|---|---|
| **Pas encore de champion** (démarrage du projet seulement) | `registry` pose directement l'alias **`champion`** sur la nouvelle version ; `promote` répond ensuite `failed: ["no_challenger"]`, code 1 | Lire la ligne `… as @champion` de `registry` (ou `champion_version` de `promotion.json`) et déployer cette version ; ce n'est **pas** un échec |
| Fichier d'entraînement **inchangé** | Le nouveau run a les mêmes métriques de validation que celui du champion ; à égalité, `--select` prend le plus récent (constaté le 25/11, section 6). Le challenger est le même modèle sous un nouveau numéro : AUC et coût égaux, donc **promu** (bornes incluses) | Sans danger (même modèle). Pour l'éviter, ne lancer le `training_flow` que si les données ont changé |
| Challenger meilleur mais **injuste** | `promote` : code 1, `failed: ["age_eo_gap"]` (par exemple), alias inchangés : le challenger reste `challenger` | Garder le modèle actuel ; signaler à l'équipe. Le test de bout en bout le montre sur données synthétiques |
| Lot de promotion trop petit | `batch_too_small` (< 5 000 lignes) ou `gender_groups_missing` (< 2 groupes de 500) | Code 1 ; attendre un lot plus grand |
| Toute erreur (MLflow injoignable, fichier absent…) | `promote` écrit `failed: ["error", "<message>"]`, code 1 ; rien n'a changé, sauf si la panne arrive après le déplacement de l'alias (contrat 2 §1) | Retry Prefect possible |

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
assert run("crediscore.models.train", "--model", "lightgbm",
           "--model-params", "configs/champion_model_params.json",
           "--train", train_file, "--output-dir", "models") == 0
assert run("crediscore.models.registry", "--select", "--train", train_file) == 0  # pas de retry
if run("crediscore.models.promote", "--data", "data/processed/holdout.parquet") == 0:
    ...  # déploiement Blue/Green d'Iyed
else:
    decision = json.loads(Path("reports/promotion.json").read_text())
    ...  # journaliser decision["failed"] ; garder le modèle actuel
```

## 6. Vérification réelle (25/11/2026, serveur MLflow jetable)

Serveur jetable sqlite sur le port 5341 (jamais celui de l'équipe), arrêté après coup :

1. `models/model.joblib` (champion du lot 3, `dvc repro -s train`) journalisé comme run `train` et enregistré : **v1 `champion`** ;
2. dérive, lot « crise économique » contre `test.parquet` avec `--model models:/crediscore@champion` : `drift_share` 0,171, PSI ratio annuité/revenu 0,676, revenu 0,262, crédit 0,231, `auc_drop` 0,009, **`alert = true`** ; `prediction` : `psi_proba` 0,031, acceptation 67,3 % → 62,5 % ; `fairness` : D5 respecté ;
3. `train --model lightgbm --model-params configs/champion_model_params.json` sur `train.parquet` : validation AUC 0,7670, coût 0,3070 par dossier (mêmes chiffres que le champion, données inchangées) ;
4. `registry --select` : deux candidats à égalité, le run le plus récent est choisi, **v2 `challenger`** ;
5. `promote --data data/processed/holdout.parquet` : **code 0, promu** (AUC 0,7591 = 0,7591, coût 0,3141 = 0,3141, sexe 7,9 / 6,9 pts, âge en égalité des chances 32,2 ≤ 40,7) ; Registry ensuite : **`champion` → v2**, plus d'alias `challenger`, v2 porte `promotion_previous_champion = 1` et `promoted_at`.

Le challenger est ici le même modèle que le champion (mêmes données, mêmes paramètres, entraînement déterministe) : la vérification prouve la **chaîne**, pas un gain. Un vrai gain demande des données rafraîchies (étape 2).
