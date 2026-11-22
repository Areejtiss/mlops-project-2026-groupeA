# CrediScore Inclusion

Plateforme MLOps de scoring de crédit explicable pour l'inclusion financière
(dataset Home Credit Default Risk). Voir [SYNOPSIS.md](SYNOPSIS.md) pour le cadrage du projet.

Équipe : Areej Tiss (Data & DVC), Achraf Gharbi (Modeling & MLflow), Iyed Ouerghi (Serving & CI/CD).

## Arborescence

```
src/crediscore/
  config.py        # lecture et validation de params.yaml (contrat 1)
  data/            # download, validate, featurize, split
  models/          # train, threshold, cost, evaluate, registry, promote
  explain/         # SHAP
  fairness/        # audit d'équité
  monitoring/      # simulateur de dérive, drift, métriques
  flows/           # flows Prefect
  api/             # FastAPI
tests/             # pytest (tests/fixtures/ : échantillon CI)
data/raw/          # données brutes, versionnées par DVC (fichiers .dvc dans Git)
data/processed/    # sorties du pipeline DVC (features, train, test)
reports/           # validation.json (métriques DVC)
dvc.yaml, dvc.lock # pipeline validate -> featurize -> split
params.yaml        # paramètres du pipeline (contrat 1)
docker-compose.yml # MinIO (remote DVC)
```

## Reproduire le projet

### Prérequis

| Outil | Version | Vérification |
|---|---|---|
| Git | récent | `git --version` |
| Python | 3.11 | `python --version` (ou `py -3.11 --version` sous Windows) |
| Poetry | 2.x | `poetry --version` |
| Docker Desktop | récent, démarré | `docker compose version` |
| Compte Kaggle | vérifié par téléphone | uniquement pour retélécharger les données brutes |

### 1. Cloner le dépôt

```bash
git clone https://github.com/Areejtiss/mlops-project-2026-groupeA.git
cd mlops-project-2026-groupeA
```

### 2. Installer l'environnement Python

```bash
poetry install                  # crée .venv/ et installe les versions exactes de poetry.lock
poetry run pre-commit install   # active les vérifications avant chaque commit
```

### 3. Créer le fichier `.env` (secrets, jamais commité)

```bash
cp .env.example .env            # PowerShell : Copy-Item .env.example .env
```

Puis remplir `.env` :

- `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` : identifiants MinIO (mot de passe de 8 caractères minimum).
- `MINIO_API_PORT` / `MINIO_CONSOLE_PORT` : laisser `9000` / `9001`, sauf si ces ports sont déjà pris.
- Kaggle : **soit** `KAGGLE_API_TOKEN` (token qui commence par `KGAT_`), **soit** `KAGGLE_USERNAME` + `KAGGLE_KEY` (fichier `kaggle.json`).

### 4. Démarrer MinIO

```bash
docker compose up -d
docker compose ps -a
```

Attendu : `crediscore-minio` est `Up (healthy)` et `minio-init` est `Exited (0)`.
Console web : <http://localhost:9001>, le bucket `dvc-store` existe.

### 5. Donner les identifiants MinIO à DVC (en local uniquement)

```bash
poetry run dvc remote modify --local minio access_key_id     <MINIO_ROOT_USER>
poetry run dvc remote modify --local minio secret_access_key <MINIO_ROOT_PASSWORD>
# seulement si MINIO_API_PORT n'est pas 9000 :
poetry run dvc remote modify --local minio endpointurl http://localhost:<MINIO_API_PORT>
```

Ces valeurs vont dans `.dvc/config.local`, ignoré par Git.

### 6. Récupérer les données

```bash
poetry run dvc pull
```

Attendu : `data/raw/application_train.csv` (307 511 lignes × 122 colonnes),
`data/raw/HomeCredit_columns_description.csv`, et les sorties du pipeline dans
`data/processed/` (`features.parquet`, `train.parquet`, `test.parquet`).

> **Bucket vide ?** `dvc pull` télécharge depuis le MinIO configuré. Si ce MinIO ne contient pas
> encore les données (premier lancement sur une machine), on les récupère depuis Kaggle puis on
> les envoie dans MinIO :
>
> ```bash
> poetry run python -m crediscore.data.download   # idempotent, vérifie 307 511 x 122
> poetry run dvc push
> ```
>
> Pour télécharger, il faut avoir accepté les règles de la compétition :
> <https://www.kaggle.com/competitions/home-credit-default-risk/rules>

### 7. Rejouer le pipeline de données

```bash
poetry run dvc repro    # ne relance que les étapes dont une entrée a changé
poetry run dvc push     # après une modification : envoie les nouvelles sorties dans MinIO
```

Le pipeline (`dvc.yaml`) enchaîne trois étapes :

| Étape | Rôle | Sortie |
|---|---|---|
| `validate` | 307 511 × 122, colonnes requises, `TARGET` binaire (~8 %), ids uniques. Code de sortie 1 en cas d'échec : le pipeline s'arrête. | `reports/validation.json` (Git) |
| `featurize` | Échantillon stratifié de 100 000 lignes, suppression des colonnes > 60 % manquantes, `DAYS_EMPLOYED = 365243` → NaN + `DAYS_EMPLOYED_ANOM`, 8 features dérivées. | `data/processed/features.parquet` |
| `split` | Train/test stratifié 80/20, sans client commun, et fixture CI de 5 000 lignes. | `train.parquet`, `test.parquet`, `tests/fixtures/ci_sample.parquet` (Git) |

Chaque étape déclare les clés de `params.yaml` qu'elle lit : modifier `data.test_size` relance
uniquement `split`. Les empreintes de toutes les entrées et sorties sont dans `dvc.lock` (commité).

`IS_THIN_FILE = 1` quand au moins 2 des 3 `EXT_SOURCE` manquent, ou qu'aucune demande au bureau
de crédit n'a été faite dans l'année (`AMT_REQ_CREDIT_BUREAU_YEAR` nul ou manquant) : le client
est peu connu des bureaux de crédit, c'est la population visée par l'inclusion (~40 % de l'échantillon).

`tests/fixtures/ci_sample.parquet` est commité dans Git (et non dans MinIO) car la CI GitHub ne peut
pas accéder à MinIO : CI et CML s'entraînent uniquement sur ce fichier.

### 8. Lancer les tests

```bash
poetry run pytest
```

Attendu : tous les tests passent.

## Dépannage

| Symptôme | Cause | Solution |
|---|---|---|
| `Bind for 0.0.0.0:9000 failed: port is already allocated` | Un autre service utilise le port | Changer `MINIO_API_PORT` / `MINIO_CONSOLE_PORT` dans `.env`, puis l'`endpointurl` de l'étape 5 |
| `poetry` introuvable | Dossier de Poetry absent du `PATH` | Ajouter le dossier au `PATH`, puis **redémarrer** le terminal (et VS Code) |
| `The request signature we calculated does not match` | Mauvais `secret_access_key` dans `.dvc/config.local` | Refaire l'étape 5 avec le mot de passe de `.env` |
| Kaggle `401` | Token absent ou mal rangé | Un token `KGAT_...` va dans `KAGGLE_API_TOKEN`, pas dans `KAGGLE_KEY` |
| Kaggle `403` | Règles de la compétition non acceptées | Accepter les règles sur Kaggle (compte vérifié par téléphone) |
