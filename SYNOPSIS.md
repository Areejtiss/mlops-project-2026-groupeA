# FICHE DE CADRAGE : SYNOPSIS DU PROJET MLOPS (SESSION 2026–2027)

---

## 1. Informations Générales & Équipe

* **Titre du Projet :** CrediScore Inclusion — Plateforme MLOps de scoring de crédit explicable pour l'inclusion financière
* **Taille du groupe :** 3 étudiants (Trinôme)
* **Membres de l'équipe :**
  1. **Étudiante 1 (Lead Data & DVC) :** Erij Tiss — Email : `erijetis@gmail.com` *(+ orchestration Prefect et simulation de dérive)*
  2. **Étudiant 2 (Lead Modeling & MLflow) :** Achraf Gharbi — Email : `gharbiachraf6@gmail.com` *(+ explicabilité SHAP et audit d'équité)*
  3. **Étudiant 3 (Lead Serving & CI/CD) :** Mohamed Iyed Ouerghi — Email : `mohamediyed.ouerghi@outlook.com` *(+ déploiement Blue/Green et export Prometheus)*
* **Lien vers le Dépôt GitHub :** `https://github.com/Areejtiss/mlops-project-2026-groupeA`

---

## 2. Problématique Métier & Cas d'Usage

* **Contexte Business :** Secteur **fintech / crédit à la consommation**. Les systèmes de scoring classiques reposent sur l'historique de crédit : les profils sans historique (« thin file » : jeunes actifs, indépendants, secteur informel) sont refusés par défaut, faute de pouvoir être évalués. Le prêteur perd des clients solvables et ces personnes se tournent vers le crédit informel. Le projet vise à estimer le risque de défaut à partir de **données alternatives** (revenus, emploi, foyer, logement, scores externes partiels) afin d'**accepter davantage de bons profils sans augmenter les pertes**.
* **Décision assistée ou automatisée :** **Décision assistée.** Pour chaque demande de prêt, l'API renvoie une probabilité de défaut, une décision (`ACCEPTÉ` / `À ÉTUDIER` / `REFUSÉ`) obtenue par un seuil optimisé sur un coût métier asymétrique, les 3 facteurs principaux de la décision (SHAP) et la version du modèle utilisée. L'analyste crédit reste décisionnaire, notamment sur la zone grise « À ÉTUDIER ».
* **Objectif métier quantifié (KPI) :**
  * Réduire le **coût métier total** (pertes sur défauts + manque à gagner sur refus de bons clients) d'au moins **15 %** par rapport à la baseline (régression logistique) ;
  * Détecter au moins **65 %** des défauts (recall classe 1) au seuil retenu ;
  * Sur le segment **thin file**, accepter au moins **2 fois plus** de bons payeurs qu'une politique « refus systématique sans historique » ;
  * Écart de taux d'acceptation entre groupes (sexe, tranche d'âge) **≤ 10 points**.

---

## 3. Formulation Machine Learning & Choix du Dataset

* **Type de problème ML :** Classification binaire supervisée, classes déséquilibrées (~8 % de positifs).
* **Variable Cible (`Target`) :** `TARGET` — `1` = client ayant eu des difficultés de paiement (retards significatifs sur les premières échéances) ; `0` = remboursement normal.
* **Source du Dataset & Licence :** Home Credit Default Risk (Kaggle) — `https://www.kaggle.com/competitions/home-credit-default-risk` — données réelles d'un prêteur ciblant les populations peu bancarisées, diffusées sous les règles de la compétition Kaggle (usage académique). Dataset proposé par le guide du module (Sujet 6 — Crédit).
* **Volume et dimensions :** Table source `application_train.csv` (307 511 lignes, 122 colonnes). Pour respecter les critères du guide (≤ 200 000 lignes, fichier < 100 Mo, quotas GitHub Actions), le pipeline DVC produit un **échantillon stratifié de 100 000 lignes** (taux de défaut préservé, ~8 %) et une sélection d'**environ 40 variables** (≥ 25 numériques, ≥ 8 catégorielles), après suppression des colonnes massivement vides. Un sous-échantillon de 5 000 lignes sert aux tests CI.
* **Variables prédictives clés :**
  * *Variables numériques :* Revenu annuel (`AMT_INCOME_TOTAL`), Montant du crédit (`AMT_CREDIT`), Mensualité (`AMT_ANNUITY`), Prix du bien (`AMT_GOODS_PRICE`), Âge (`DAYS_BIRTH`), Ancienneté d'emploi (`DAYS_EMPLOYED`), Scores externes (`EXT_SOURCE_1/2/3`), Ratios dérivés (mensualité/revenu, crédit/revenu).
  * *Variables catégorielles :* Type de contrat (`NAME_CONTRACT_TYPE`), Type de revenu (`NAME_INCOME_TYPE`), Niveau d'études (`NAME_EDUCATION_TYPE`), Statut matrimonial (`NAME_FAMILY_STATUS`), Logement (`NAME_HOUSING_TYPE`), Métier (`OCCUPATION_TYPE`).
  * *Attribut sensible :* `CODE_GENDER` **exclu des features**, conservé uniquement pour l'audit d'équité.
* **Métriques ML cibles :** ROC-AUC **≥ 0.75**, PR-AUC **≥ 0.25** (baseline aléatoire ≈ 0.08), Brier score ≤ 0.07 (probabilités calibrées), Latence d'inférence p95 **< 150 ms**.

---

## 4. Architecture MLOps Prévisionnelle

| Composant MLOps | Outil & Configuration retenue |
|---|---|
| **Versioning Données** | DVC, remote **MinIO** (S3 local via Docker Compose) ; pipeline `dvc.yaml` : `validate → featurize → split → train → evaluate`, paramètres dans `params.yaml` |
| **Tracking & Registry** | MLflow Tracking Server (`http://127.0.0.1:5000`), comparaison LogReg / RandomForest / LightGBM / XGBoost (hyperparamètres, métriques, matrices de confusion, courbes ROC/PR, SHAP) ; Model Registry : promotion formelle Staging → Production conditionnée à l'AUC et à l'équité |
| **Microservice Inférence** | FastAPI : `/health`, `/model-info`, `/predict` (unitaire : score + décision + SHAP), `/predict/batch` (lot de demandes), Swagger auto sur `/docs` |
| **Validation des données** | Schémas Pydantic v2 stricts (bornes, `Literal` pour les catégories, valeurs manquantes autorisées sur `ext_source_*`) ; erreur 422 explicite |
| **Packaging & Isolation** | Image Docker multi-stage `python:3.11-slim`, utilisateur non-root, `HEALTHCHECK` ; `docker-compose.yml` : API, MLflow, MinIO, Prometheus |
| **Intégration Continue (CI)** | GitHub Actions : `flake8` + `black --check` + `mypy` (typage statique) + Pytest (unitaires, schéma, inférence, API) avec couverture **> 75 %** + build Docker ; pre-commit en local ; dépendances Poetry exportées en `requirements.txt` |
| **Rapports CML** | À chaque PR : entraînement sur échantillon stratifié, tableau des métriques + courbe ROC postés en commentaire |
| **Orchestration** | Prefect 3.x : `training_flow` en DAG (`@task` atomiques avec retries, validation bloquante) et `monitoring_flow` planifié en **CRON** ; déploiement déclaratif `prefect.yaml`, Work Pool + worker process |
| **Déploiement Continu** | Stratégie **Blue/Green** : deux conteneurs API derrière un reverse proxy Nginx ; le nouveau modèle est déployé sur l'environnement inactif, validé par `/health` et tests de fumée, puis bascule du trafic sans interruption (rollback immédiat possible) |
| **Surveillance de Dérive** | Module maison + Evidently AI : **Kolmogorov-Smirnov** (numériques), **χ²** (catégorielles), **PSI** calculé formellement par feature (implémentation testée) ; **dashboard HTML interactif** Evidently ; suivi de la dérive des prédictions, de la performance (concept drift) et de l'équité |
| **Export Métriques** | Endpoint `/metrics` au format OpenMetrics (`prometheus-fastapi-instrumentator`) : latence, requêtes, erreurs, + jauges `drift_share`, `psi{feature}`, `acceptance_rate` ; scrapé par Prometheus, règle d'alerte sur le drift ; tableau de bord Grafana (bonus) |
| **Boucle de Rétroaction** | Si `drift_share >= 0.30` ou PSI ≥ 0.20 sur une feature clé : alerte Prometheus + déclenchement automatique du `training_flow` Prefect ; le nouveau modèle n'est promu que s'il bat le `champion` |

---

## 5. Scénario de Dérive de Données (Data Drift) Simulé

* **Quelles variables vont dériver en production ?**
  * *Scénario « crise économique » :* baisse des revenus (`AMT_INCOME_TOTAL`), hausse des montants demandés (`AMT_CREDIT`), donc hausse du ratio mensualité/revenu ; ancienneté d'emploi raccourcie.
  * *Scénario « campagne d'inclusion » :* afflux de jeunes demandeurs (< 30 ans) et de profils thin file (hausse des `EXT_SOURCE` manquants).
  * *Scénario « concept drift » :* hausse du taux de défaut d'un secteur (`NAME_INCOME_TYPE = Commercial associate`) sans changement des distributions.
* **Méthode d'injection de la dérive :** Module `DriftSimulator` appliqué au jeu de test : décalage multiplicatif et bruit gaussien sur les variables financières, ré-échantillonnage pondéré (âge, profils thin file), injection de valeurs manquantes, bascule de labels sur un sous-groupe pour le concept drift. Les lots dérivés sont envoyés à l'API comme trafic de production simulé.
* **Seuil d'alerte :** PSI **≥ 0.20** sur une feature clé (revenu, crédit, ratio d'endettement, `EXT_SOURCE_2`) **ou** proportion de features en dérive `drift_share` **≥ 0.30** ; baisse de ROC-AUC > 0.03 lorsque les labels sont disponibles.
* **Comportement autonome du système :** Le `monitoring_flow` Prefect génère le rapport Evidently (HTML archivé), met à jour les jauges Prometheus et déclenche une alerte ; il lance ensuite le `training_flow` de réentraînement. Le nouveau modèle est enregistré en `challenger`, promu `champion` s'il respecte les critères de performance et d'équité, puis déployé en Blue/Green.

---

## 6. Planning Prévisionnel & Validation Enseignant

* [ ] **Jalon 1 (S2) :** Fiche de cadrage `SYNOPSIS.md` soumise, dépôt GitHub créé, dataset validé
* [ ] **Jalon 2 (S4) :** Environnement (Poetry, pre-commit, structure `src/`), dataset sous DVC avec remote MinIO, pipeline DVC + MLflow fonctionnels (baseline + LightGBM + XGBoost, seuil de coût), 1ers tests Pytest
* [ ] **Jalon 3 (S5) :** API FastAPI opérationnelle (SHAP, Pydantic), Dockerfile validé, Workflow CI + CML actif
* [ ] **Jalon 4 (S6) :** Flow Prefect orchestré, Détection de drift (KS, χ², PSI) & Prometheus, Blue/Green
* [ ] **Jalon Final (S8) :** Tag `v1.0-release`, Rapport PDF et Soutenance

> **Avis de l'Enseignant :** `[  ] VALIDÉ  /  [  ] À REVOIR`
> **Commentaires :**
