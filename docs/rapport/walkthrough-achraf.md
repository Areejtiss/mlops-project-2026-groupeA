# Walkthrough d'Achraf — préparation aux questions individuelles

Pour chaque module : ce qu'il fait, le choix de conception clé et pourquoi, puis 2 ou 3 questions probables avec une réponse courte. Les chiffres viennent de `docs/rapport/modelisation.md` (sources citées là-bas). À la fin : les questions MLflow et Evidently du guide (§9.3).

---

## `models/cost.py` — coût métier

- **Fait :** `business_cost(y, decisions, *, cost_fn, cost_fp, cost_review)` compte les défauts acceptés (× 5), les bons clients refusés (× 1) et chaque dossier en étude (× 0,3). C'est une fonction pure : les coûts viennent de `params.yaml`.
- **Choix :** les trois coûts sont des arguments **obligatoires**. Un appel à 3 zones ne peut donc jamais compter l'étude comme gratuite par oubli.
- **Q : Pourquoi 5:1 ?** Un défaut fait perdre une grande part du capital ; un bon client refusé ne fait perdre que la marge. La justification est dans `02_baseline.ipynb`.
- **Q : Pourquoi pas l'accuracy ?** Avec 8 % de défauts, prédire « 0 » partout donne 92 % d'exactitude et ne détecte aucun défaut. On mesure le coût, le recall, l'AUC et la PR-AUC.
- **Q : Un défaut envoyé en étude est-il « détecté » ?** Oui pour le recall, non pour l'acceptation (D8).

## `models/split.py` — validation (D1)

- **Fait :** découpe 20 % du train (`data.val_size`), stratifié sur `TARGET`, avec la graine `data.random_state`.
- **Choix :** les seuils et la sélection se font sur cette validation, jamais sur le test, qui n'est lu qu'une fois par `evaluate`.
- **Q : Pourquoi ne pas régler les seuils sur le test ?** Le test deviendrait optimiste : on aurait choisi ce qui marche sur ces lignes-là.
- **Q : Comment garantir qu'il n'y a pas de fuite ?** Un test vérifie qu'aucun `SK_ID_CURR` n'est commun ; la stratification est tenue à ±0,5 pt.

## `models/pipeline.py` — préparation + modèle

- **Fait :** `build_pipeline(name, …)` construit un `ColumnTransformer` (numériques : médiane + mise à l'échelle ; catégorielles : `None` → NaN, mode, one-hot `handle_unknown="ignore"`), puis LogReg, RandomForest, LightGBM ou XGBoost. `calibrate()` l'enveloppe dans `CalibratedClassifierCV(isotonic, cv=5, ensemble=False)`.
- **Choix :** préparation et modèle forment **un seul artefact** : l'API ne peut pas préparer les données autrement. Avec `ensemble=False`, il n'y a qu'un pipeline réajusté, que SHAP explique.
- **Q : Pourquoi calibrer ?** Les seuils et le Brier supposent de vraies probabilités. `class_weight="balanced"` déforme les probabilités ; l'isotonique les remet à l'échelle.
- **Q : Une catégorie inconnue en production ?** `handle_unknown="ignore"` : toutes ses colonnes one-hot valent 0, sans erreur (testé).
- **Q : Pourquoi pas SMOTE ?** Il invente des clients. Le poids des classes suffit, puis on calibre (T5).

## `models/threshold.py` — seuils à 3 zones

- **Fait :** `optimize_thresholds` cherche exactement (sur toutes les probabilités distinctes de la validation) le couple `low ≤ high` de coût minimal, avec recall ≥ 65 % et au plus 15 % en étude. `decide` applique `proba < low` → `ACCEPTE`, `< high` → `A_ETUDIER`, sinon `REFUSE`.
- **Choix :** les coupures sont placées à **mi-chemin** entre deux paliers isotoniques. Un léger écart numérique (float32, JSON) ne change donc pas la décision.
- **Q : Pourquoi 3 zones ?** Un seul seuil donne 38 % de recall ; sans plafond, la moitié des dossiers part en étude. Les 3 zones plafonnées respectent les deux contraintes au coût le plus bas.
- **Q : Où vivent les seuils ?** Avec la version du modèle (`thresholds.json` et tags), jamais dans `params.yaml`, ce qui évite une boucle DVC (D9).
- **Q : `proba == low` ?** `A_ETUDIER` : les zones sont fermées à gauche (testé).

## `models/train.py` — entraînement tracé

- **Fait :** découpe la validation, entraîne et calibre, optimise les seuils, écrit `models/model.joblib`, `thresholds.json` et `metrics.json`. Si `MLFLOW_TRACKING_URI` est défini, journalise un run : paramètres, métriques `val_*`, courbes, modèle (`cloudpickle`, signature) et tags (`train_data_hash`, `git_sha`, seuils).
- **Choix :** l'étape DVC `train` utilise les hyperparamètres Optuna **figés** dans `configs/champion_model_params.json` : reproductible, environ 1 min.
- **Q : Comment retrouver les données d'un modèle ?** Les tags `train_data_hash` (md5 du parquet) et `raw_data_hash` (md5 DVC du CSV), plus `git_sha`.
- **Q : `--exclude AGE_YEARS` ?** Il exige `--experiment` : le run est une mesure, `registry` refuse de l'enregistrer seul.

## `models/tune.py` — Optuna

- **Fait :** TPE avec graine, 25 essais, objectif = AUC en validation croisée à 5 plis sur la partie fit ; chaque essai est un run MLflow imbriqué.
- **Choix :** la validation croisée se fait dans le train, pour ne pas user la validation (déjà utilisée pour les seuils).
- **Q : Pourquoi l'AUC et pas le coût comme objectif ?** Le coût dépend des seuils, optimisés ensuite ; l'AUC mesure le classement, indépendamment du seuil.
- **Q : Le réglage est-il refait à chaque réentraînement ?** Non : c'est une expérience ponctuelle, relue (handoff §2).

## `models/evaluate.py` — le seul lecteur du test

- **Fait :** charge le modèle et ses seuils, **réentraîne la baseline LogReg à l'identique**, mesure les deux sur le test et écrit `reports/metrics.json` : métriques, coût, KPI −15 %, thin file, équité D5, chaque cible avec son écart.
- **Choix :** la baseline est recalculée sur le même split. Le −15 % ne se compare qu'à elle, jamais à un chiffre de notebook.
- **Q : Vos KPI sont-ils atteints ?** Recall, Brier, thin file et sexe : oui. AUC 0,747, PR-AUC 0,227 et coût −9 % : non, et c'est publié tel quel.
- **Q : Pourquoi la baisse validation → test ?** Environ 0,02 d'AUC sur les trois modèles mesurés : c'est le split, pas le modèle.

## `models/registry.py` — enregistrement

- **Fait :** `--select` choisit le meilleur run `train` (même fichier d'entraînement, mêmes coûts et contraintes, AUC ≥ 0,75 si possible, coût de validation minimal), l'enregistre comme version de `crediscore` et copie les tags. Alias `champion` s'il n'y en a pas, sinon `challenger`.
- **Choix :** **alias plutôt que stages** (dépréciés). Le run gagnant d'une expérience (`no_age`) n'est pas enregistré sans `--allow-experiment`.
- **Q : Différence Tracking / Registry ?** Le Tracking garde les runs (paramètres, métriques, artefacts). Le Registry donne un nom, des versions et des alias aux modèles retenus.
- **Q : Pourquoi filtrer sur `train_data_hash` ?** Comparer des runs entraînés sur des données différentes n'aurait pas de sens.

## `models/promote.py` — challenger → champion

- **Fait :** évalue champion et challenger sur le **même lot récent** (`holdout.parquet`), chacun avec ses seuils. Le challenger est promu si AUC ≥ champion, coût ≤ champion, écarts selon le sexe ≤ 10 pts et écart d'âge en égalité des chances ≤ baseline + 2. Code 0 = promu, 1 = non ; décision dans `reports/promotion.json`.
- **Choix :**
  - **règle relative** : « AUC ≥ 0,75 » bloquerait tout, le champion fait 0,747 ;
  - **échec fermé** : lot trop petit, ancre absente, seuils différents des tags, ou erreur → refus ;
  - ordre d'écriture tags → suppression de `challenger` → déplacement de `champion` → `promoted_at` : une panne laisse l'ancien champion en service.
- **Q : Comment revenir à la version précédente ?** Remettre l'alias `champion` sur la version notée dans `promotion_previous_champion` ; l'API recharge `models:/crediscore@champion`.
- **Q : Pourquoi pas le test pour promouvoir ?** Il a servi à mesurer le champion. Le hold-out (20 000 lignes hors échantillon) n'a été lu par aucune sélection.
- **Q : La boucle marche-t-elle ?** Oui : test de bout en bout sur un MLflow sqlite, et exécution réelle le 25/11 (v2 promue, `promotion_previous_champion = 1`).

## `data/holdout.py` — lot de promotion (pour relecture d'Areej)

- **Fait :** tire 20 000 lignes étiquetées (stratifiées, avec graine) parmi les lignes brutes **absentes** de `features.parquet`, avec exactement le traitement de `featurize`.
- **Choix :** un lot neuf, jamais lu par la sélection, pour que la promotion compare honnêtement.
- **Q : Pourquoi pas le lot de crise ?** Il réutilise les lignes du test : il sert à la démo de dérive, jamais à la promotion.
- **Q : Comment savoir qu'il n'y a pas de recouvrement ?** Un test vérifie qu'aucun `SK_ID_CURR` n'est commun avec `features.parquet` ; le taux de défaut est 8,075 % contre 8,073 % dans l'échantillon.

## `fairness/audit.py` — audit d'équité

- **Fait :** taux d'acceptation (`ACCEPTE` seulement), égalité des chances (acceptation des bons payeurs) et recall, par sexe (`XNA` exclu et compté) et par tranche d'âge ; écarts max − min en points ; `d5_checks` compare aux limites.
- **Choix :** c'est la seule source des chiffres d'équité, utilisée par `evaluate`, `promote` et la dérive (lot 6).
- **Q : Pourquoi le sexe n'est-il pas une variable ?** C'est un attribut protégé : il ne sert qu'à mesurer l'équité. Un test vérifie que changer `CODE_GENDER` ne change rien au score.
- **Q : Pourquoi l'égalité des chances ?** Elle compare des clients qui ont **réellement** remboursé : l'écart ne s'explique pas par un risque différent.
- **Q : Pourquoi ne pas retirer l'âge ?** Sans `AGE_YEARS`, l'écart d'âge monte (33,1 → 35,6 pts) et le coût aussi : l'âge passe par d'autres variables.

## `explain/shap_explainer.py` — SHAP

- **Fait :** `Explainer(model)`, construit une fois ; `explain(X)` rend les 3 facteurs `(feature, value, contribution)` triés par |contribution|, sur les colonnes d'origine.
- **Choix :** il explique le pipeline **non calibré** contenu dans le modèle servi, en log-odds (`shap_unit`). La valeur de base est recalculée depuis la marge brute du modèle, pour que l'additivité tienne.
- **Q : Les contributions s'additionnent-elles à la proba ?** Non, à la marge non calibrée. La calibration est monotone, donc le sens est juste, mais pas les valeurs en points de probabilité.
- **Q : Latence ?** p95 14,9 ms par ligne (cible 50) : l'explainer est construit au démarrage, pas à chaque requête.

## `monitoring/drift.py` — dérive

- **Fait :** KS (numériques), χ² (catégorielles), PSI écrit à la main, `drift_share`, `auc_drop` ; règle d'alerte du contrat 4 ; HTML Evidently. Depuis le lot 6 : `prediction` (PSI des probabilités, taux d'acceptation) et `fairness` (audit du lot courant, statut D5), informatifs.
- **Choix :**
  - bins PSI en **intervalles**, dont les bornes ne tombent jamais sur une valeur réelle (bug corrigé : 1e-9 de décalage donnait un PSI de 2,97) ;
  - gardes des petits lots (500 lignes ; 1 000 lignes labellisées et 50 défauts) ;
  - le cœur reste pur : le CLI charge le modèle et passe les résultats.
- **Q : Pourquoi l'alerte alors que l'AUC ne baisse que de 0,009 sur la crise ?** Les entrées dérivent (revenus, ratios), mais pas les scores externes ni les labels : dérive des données, pas de concept. L'acceptation baisse pourtant de 4,85 pts.
- **Q : Pourquoi PSI en plus de KS ?** Sur 20 000 lignes, KS signale de tout petits écarts. Le PSI dit « de combien » ça a bougé.
- **Q : Que se passe-t-il à l'alerte ?** Le `monitoring_flow` lance le `training_flow` : réentraînement, `registry`, `promote`, puis Blue/Green si promu (handoff).

---

## Questions du guide (§9.3) : MLflow

- **Différence entre Tracking et Model Registry ?** Le Tracking enregistre chaque run (paramètres, métriques, courbes, modèle). Le Registry garde les modèles retenus sous un nom (`crediscore`), avec des versions numérotées et des alias (`champion`, `challenger`).
- **Comment l'API sait-elle quel modèle charger ?** Elle charge `models:/crediscore@champion`, et lit les seuils et `shap_unit` dans les tags de cette version. Promouvoir = déplacer l'alias : l'URI de l'API ne change jamais.
- **Comment revenir à la version précédente ?** `MlflowClient().set_registered_model_alias("crediscore", "champion", <version précédente>)`. Le numéro est dans le tag `promotion_previous_champion` de la version actuelle. Puis l'API recharge le modèle (bascule Blue/Green d'Iyed).

## Questions du guide (§9.3) : Evidently

- **Data drift ou concept drift ?** Data drift : la distribution des entrées change (crise : revenus, crédits). Concept drift : la relation entrées → défaut change (scénario C : les labels d'un secteur changent, les entrées non). Le premier se voit sans labels (PSI, KS, χ²) ; le second demande des labels (`auc_drop`).
- **Pourquoi le concept drift est-il difficile à détecter en crédit ?** Le label (défaut ou non) n'arrive que des mois après l'octroi. On surveille donc d'abord les entrées et les prédictions (`prediction` : PSI des scores, taux d'acceptation), et l'AUC quand les labels arrivent (au moins 1 000 lignes labellisées).
- **Quel test pour une numérique, et pour une catégorielle ?** Kolmogorov-Smirnov pour les numériques, χ² pour les catégorielles, avec α = 0,05, chez nous comme dans le rapport Evidently. Nuance : le « chi-square » d'Evidently est un test d'adéquation aux proportions de la référence, le nôtre un test d'indépendance à deux échantillons.
- **Que faites-vous quand une dérive est détectée ?** `alert = true` dans le JSON → jauge Prometheus → le `monitoring_flow` lance le `training_flow`. On réentraîne avec les paramètres figés, on enregistre un `challenger`, et on le promeut seulement s'il bat le champion sans dégrader l'équité. Sinon, on garde le modèle actuel.
