# Décisions modélisation (Achraf) — 24/10/2026

Ces décisions cadrent les contrats 1 à 4. Les lignes marquées **Équipe** restent
ouvertes à objection d'Areej et d'Iyed jusqu'à l'approbation de la PR qui les porte.
Plan détaillé : `docs/plans/2026-10-24-achraf-roadmap.md`.

## Résumé

| # | Décision | Pourquoi | Impact |
|---|---|---|---|
| D1 | Un **jeu de validation** de 20 % est découpé dans le train (stratifié, même seed). Nouvelle clé `data.val_size: 0.2` | Régler seuils et calibration sans toucher au test | Areej : clé dans `params.yaml` + `config.py` |
| D2 | **Équipe.** Les features dérivées (ratios, `AGE_YEARS`…) sont calculées par **une seule fonction partagée**, appelée par l'étape `featurize` d'Areej **et** par l'API avant le modèle. Le modèle reçoit exactement `params.features.all` | Éviter que l'API calcule les ratios autrement qu'à l'entraînement | **Livré par Areej (PR #2, `dev`)** : `crediscore.data.featurize.fix_days_employed` puis `add_derived_features` sont LES fonctions partagées ; notre copie `data/features.py` est supprimée (24/11/2026). Iyed : les appeler dans `/predict`, dans cet ordre. Un test vérifie que les deux chemins donnent les mêmes colonnes |
| D3 | **3 zones** optimisées sur le coût : défaut raté = 5, bon client refusé = 1, **étude manuelle = 0,3**, avec **au plus 15 %** des dossiers en « à étudier » | Sans plafond, l'optimum envoie environ la moitié des dossiers en étude | Areej : clés `cost.cost_review: 0.3`, `threshold.max_review_share: 0.15` |
| D4 | **Équipe.** Codes de décision dans le JSON : `ACCEPTE` / `A_ETUDIER` / `REFUSE` (sans accents ni espaces). Libellés accentués dans le rapport et l'affichage uniquement | Code machine stable (Pydantic `Literal`, scripts) | Iyed : `Literal` du contrat 3. Areej : corriger le commentaire `params.yaml:85` (`A ETUDIER`) |
| D5 | **Équipe. Tranchée le 25/10/2026 (humain), après mesure sur la baseline.** **Sexe : seuil absolu.** L'écart de taux d'acceptation (`ACCEPTE` seulement) entre F et M, `XNA` exclu, doit être ≤ 10 pts, et l'écart d'égalité des chances (parmi les clients qui ont remboursé) aussi ≤ 10 pts. Baseline : 7,8 / 6,6 pts. **Tranches d'âge (<30, 30-50, ≥50) : pas de seuil absolu.** Règle de non-régression **ancrée sur la baseline** (précisée le 25/10/2026, validée par l'humain) : l'ancre est l'écart d'égalité des chances selon l'âge de la baseline LogReg, recalculé par `evaluate` sur le jeu d'évaluation qu'on lui donne ; l'écart du challenger doit rester ≤ cette valeur + 2 pts. « Fixe » veut dire liée à la baseline, jamais au champion : elle ne peut pas dériver. Valeur actuelle 38,7 pts, donc limite 40,7 pts (historique : 34,9 pts était la limite provisoire du lot 1, sur un autre split). Elle s'applique dès la première promotion et ne peut pas dériver de 2 pts à chaque promotion, comme le ferait « ≤ champion + 2 ». Les deux écarts d'âge sont publiés à chaque évaluation et suivis dans le temps. **Tolérances dans `params.yaml` (24/11/2026)** : `promotion.max_gender_gap_pts: 10`, `promotion.age_gap_margin_pts: 2`. Elles servent à la promotion (règle relative au champion, voir « Contrat 2 ») | L'âge est un vrai facteur de risque dans ces données (les moins de 30 ans font environ 2 fois plus défaut que les 50 ans et plus : 11,4 % contre 5,7 %) et une feature du modèle. Sur la baseline, l'écart d'âge est de 34,5 pts brut et 32,9 pts en égalité des chances : un seuil absolu de 10 pts bloquerait toute promotion | Achraf : prévenir l'enseignant (KPI du SYNOPSIS). Lot 3 : une expérience MLflow LightGBM **sans `AGE_YEARS`** pour mesurer le compromis AUC/coût contre écart d'âge |
| D6 | **Équipe. Révisée le 24/11/2026 :** les PR visent **`dev`** (pratique de l'équipe : PR #2 et #3 fusionnées dans `dev`) ; `dev` est ensuite livré vers `main` | L'équipe intègre sur `dev` | Iyed : la CI doit tourner sur les PR vers `dev` ; ajouter `feature/**` au déclencheur `push` et corriger `cache: poetry` (placé avant l'installation de Poetry, ce qui fait échouer la CI) |
| D8 | **Équipe.** Lecture des KPI avec 3 zones : un défaut envoyé en « à étudier » compte comme **détecté** (recall) ; un dossier « à étudier » ne compte **pas** comme accepté (taux d'acceptation, équité). L'optimiseur impose **recall ≥ 65 %** | Avec un seul seuil, le recall serait d'environ 50 % | Areej : clé `threshold.min_recall: 0.65` |
| D9 | Les **seuils sont un résultat** de l'entraînement : ils voyagent avec la version du modèle (tags MLflow + `thresholds.json`). `threshold.low/high` de `params.yaml` devient un **forçage manuel optionnel** | Écrire les seuils dans `params.yaml` crée une boucle DVC et deux copies qui divergent | Areej : documenter ce rôle. Iyed : l'API lit les seuils avec le modèle |
| D10 | **Équipe. Définition révisée le 24/11/2026 : on adopte celle du code fusionné d'Areej.** Thin file (`IS_THIN_FILE`) = **au moins 2 des 3 `EXT_SOURCE` manquants, ou `AMT_REQ_CREDIT_BUREAU_YEAR` manquant ou nul** (`crediscore.data.featurize`). KPI thin file : comparaison avec une **politique « scorecard classique »** (refus de tout thin file, sinon notre seuil) | « 2× plus qu'un refus systématique » compare à zéro : toujours vrai | Sur le split réel, 39,7 % du train est thin file (11,9 % avec l'ancienne définition « `EXT_SOURCE_1` et `EXT_SOURCE_3` manquants », utilisée par les notebooks 01/02). Les chiffres qui font foi viennent de `evaluate` (lot 3) |

**Aussi tranchées :** D7, pas de dates fixes : les lots s'enchaînent au plus vite. D11, l'e-mail des commits d'Achraf est bien relié à son compte GitHub.

## Contrat 2 (artefact modèle), points actés

- Nom `crediscore`, alias `champion` / `challenger` (les stages sont dépréciés).
- Chargement en `mlflow.sklearn`, objet calibré (`CalibratedClassifierCV`, isotonique, `ensemble=False`). SHAP explique le pipeline non calibré contenu dans cet objet (`calibrated_classifiers_[0].estimator`) : pas d'artefact séparé.
- Entrée : les colonnes `params.features.all` (D2). Sortie : probabilité de défaut.
- `top_factors` : contributions SHAP en log-odds du modèle non calibré. Elles ne s'additionnent **pas** à `proba`.
- Promotion (**règle relative, révisée le 24/11/2026, validée par l'humain**) : champion et challenger évalués **sur le même lot récent**, chacun avec ses seuils. Le challenger gagne s'il a une AUC ≥ celle du champion, un coût par dossier ≤ celui du champion, des écarts selon le sexe ≤ 10 pts (brut et égalité des chances), et un écart d'âge en égalité des chances ≤ celui de la baseline + 2 pts (D5 : l'ancre est l'écart d'égalité des chances selon l'âge de la baseline LogReg, recalculé par `evaluate` sur le jeu d'évaluation qu'on lui donne et lu dans `reports/metrics.json` ; absente = refus. « Fixe » veut dire liée à la baseline, jamais au champion. Valeur actuelle 38,7 pts, donc limite 40,7 pts). L'ancienne condition « AUC ≥ 0,75 » est retirée : le champion actuel fait 0,7469 sur le test, elle bloquerait toute promotion. Les cibles 0,75 / 0,25 / −15 % restent des KPI publiés. Sans champion, seules les conditions d'équité s'appliquent. Code de sortie 0 si promu, 1 sinon ; décision dans `reports/promotion.json`.
- Promotion, compléments du 24/11/2026 (relecture du lot 4, option recommandée retenue) : le lot (`--data`, **obligatoire**) doit être un lot récent étiqueté qui n'est **pas** le split de test (lot de production ou lot simulé d'Areej ; le test seulement pour une démo manuelle, nommé explicitement). **Échec fermé** si le lot a moins de **5 000** lignes (`promotion.min_rows`) ou moins de 2 groupes de sexe d'au moins **500** lignes chacun (`promotion.min_group_rows`) : les écarts seraient trop bruités. `thresholds.json` fait foi pour les seuils (tags différents = refus). Ordre des écritures : tags, puis suppression de l'alias `challenger`, puis déplacement de l'alias `champion`, puis le tag `promoted_at` (relecture 2, R3 : une promotion avortée ne porte jamais `promoted_at`) ; un challenger déjà champion ne fait rien.

## Actions

**Areej**
- [ ] Ajouter dans `params.yaml` / `config.py` : `data.val_size: 0.2`, `cost.cost_review: 0.3`, `threshold.max_review_share: 0.15`, `threshold.min_recall: 0.65`.
- [ ] Documenter `threshold.low/high` comme forçage manuel optionnel.
- [ ] Corriger le libellé `A ETUDIER` → `A_ETUDIER`.
- [x] Exposer `add_derived_features(df)` (pure, testée) : `crediscore.data.featurize`, PR #2.
- [x] `IS_THIN_FILE` : D10 adopte sa définition (24/11/2026).
- [ ] Relire le commit « (for Areej's review) » : conversion en float64 des entrées brutes, pour les requêtes d'une ligne avec `None`.

**Iyed**
- [ ] CI : déclencheur `push` sur `feature/**` ; `cache: poetry` après l'installation de Poetry.
- [ ] Contrat 3 : `decision: Literal["ACCEPTE", "A_ETUDIER", "REFUSE"]` ; `/predict` appelle `add_derived_features` ; seuils lus avec le modèle.

**Achraf**
- [x] Mesurer l'écart d'équité sur la baseline (D5) : sexe 7,8 / 6,6 pts, âge 34,5 / 32,9 pts (brut / égalité des chances, test).
- [ ] Prévenir l'enseignant : le KPI du SYNOPSIS « écart ≤ 10 points » s'applique désormais au **sexe** ; l'âge est « mesuré, expliqué, sans dégradation » (D5).
- [x] Lot 3 : expérience LightGBM sans `AGE_YEARS` (compromis AUC/coût contre écart d'âge, pour le rapport). Résultat : retirer l'âge **augmente** l'écart d'âge (33,1 → 35,6 pts en égalité des chances) et le coût (0,320 → 0,327 par dossier) ; détail dans le plan, lot 3.
- [ ] Écrire `docs/contracts/02-model-artifact.md`.
- [ ] Livrer en début de S3 un modèle squelette avec `thresholds.json` et `decide()`.
