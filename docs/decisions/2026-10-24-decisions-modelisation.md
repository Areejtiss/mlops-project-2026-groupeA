# Décisions modélisation (Achraf) — 24/10/2026

Ces décisions cadrent les contrats 1 à 4. Les lignes marquées **Équipe** restent
ouvertes à objection d'Areej et d'Iyed jusqu'à l'approbation de la PR qui les porte.
Plan détaillé : `docs/plans/2026-10-24-achraf-roadmap.md`.

## Résumé

| # | Décision | Pourquoi | Impact |
|---|---|---|---|
| D1 | Un **jeu de validation** de 20 % est découpé dans le train (stratifié, même seed). Nouvelle clé `data.val_size: 0.2` | Régler seuils et calibration sans toucher au test | Areej : clé dans `params.yaml` + `config.py` |
| D2 | **Équipe.** Les features dérivées (ratios, `AGE_YEARS`…) sont calculées par **une seule fonction partagée**, appelée par l'étape `featurize` d'Areej **et** par l'API avant le modèle. Le modèle reçoit exactement `params.features.all` | Éviter que l'API calcule les ratios autrement qu'à l'entraînement | Areej : exposer `add_derived_features(df)` pure et testée. Iyed : l'appeler dans `/predict`. Un test vérifie que les deux chemins donnent les mêmes colonnes |
| D3 | **3 zones** optimisées sur le coût : défaut raté = 5, bon client refusé = 1, **étude manuelle = 0,3**, avec **au plus 15 %** des dossiers en « à étudier » | Sans plafond, l'optimum envoie environ la moitié des dossiers en étude | Areej : clés `cost.cost_review: 0.3`, `threshold.max_review_share: 0.15` |
| D4 | **Équipe.** Codes de décision dans le JSON : `ACCEPTE` / `A_ETUDIER` / `REFUSE` (sans accents ni espaces). Libellés accentués dans le rapport et l'affichage uniquement | Code machine stable (Pydantic `Literal`, scripts) | Iyed : `Literal` du contrat 3. Areej : corriger le commentaire `params.yaml:85` (`A ETUDIER`) |
| D5 | **Équipe. Tranchée le 25/10/2026 (humain), après mesure sur la baseline.** **Sexe : seuil absolu.** L'écart de taux d'acceptation (`ACCEPTE` seulement) entre F et M, `XNA` exclu, doit être ≤ 10 pts, et l'écart d'égalité des chances (parmi les clients qui ont remboursé) aussi ≤ 10 pts. Baseline : 7,8 / 6,6 pts. **Tranches d'âge (<30, 30-50, ≥50) : pas de seuil absolu.** Règle de non-régression **ancrée sur la baseline** (précisée le 25/10/2026, validée par l'humain) : l'écart d'égalité des chances du challenger ≤ celui de la **baseline** + 2 pts, soit ≤ 34,9 pts. La référence est la baseline LogReg recalculée par `evaluate` au lot 3 sur le même jeu de test ; d'ici là, 32,9 pts. Elle est fixe : elle s'applique dès la première promotion et ne peut pas dériver de 2 pts à chaque promotion, comme le ferait « ≤ champion + 2 ». Les deux écarts d'âge sont publiés à chaque évaluation et suivis dans le temps | L'âge est un vrai facteur de risque dans ces données (les moins de 30 ans font environ 2 fois plus défaut que les 50 ans et plus : 11,4 % contre 5,7 %) et une feature du modèle. Sur la baseline, l'écart d'âge est de 34,5 pts brut et 32,9 pts en égalité des chances : un seuil absolu de 10 pts bloquerait toute promotion | Achraf : prévenir l'enseignant (KPI du SYNOPSIS). Lot 3 : une expérience MLflow LightGBM **sans `AGE_YEARS`** pour mesurer le compromis AUC/coût contre écart d'âge |
| D6 | **Équipe.** Les PR visent `main` | La CI ne tourne que sur les PR vers `main` | Iyed : ajouter `feature/**` au déclencheur `push` et corriger `cache: poetry` (placé avant l'installation de Poetry, ce qui fait échouer la CI) |
| D8 | **Équipe.** Lecture des KPI avec 3 zones : un défaut envoyé en « à étudier » compte comme **détecté** (recall) ; un dossier « à étudier » ne compte **pas** comme accepté (taux d'acceptation, équité). L'optimiseur impose **recall ≥ 65 %** | Avec un seul seuil, le recall serait d'environ 50 % | Areej : clé `threshold.min_recall: 0.65` |
| D9 | Les **seuils sont un résultat** de l'entraînement : ils voyagent avec la version du modèle (tags MLflow + `thresholds.json`). `threshold.low/high` de `params.yaml` devient un **forçage manuel optionnel** | Écrire les seuils dans `params.yaml` crée une boucle DVC et deux copies qui divergent | Areej : documenter ce rôle. Iyed : l'API lit les seuils avec le modèle |
| D10 | **Équipe.** KPI thin file : comparaison avec une **politique « scorecard classique »** (refus si `EXT_SOURCE_1` et `EXT_SOURCE_3` manquants, sinon notre seuil). Thin file = `EXT_SOURCE_1` et `EXT_SOURCE_3` manquants | « 2× plus qu'un refus systématique » compare à zéro : toujours vrai | Areej : même définition dans `IS_THIN_FILE` |

**Aussi tranchées :** D7, pas de dates fixes : les lots s'enchaînent au plus vite. D11, l'e-mail des commits d'Achraf est bien relié à son compte GitHub.

## Contrat 2 (artefact modèle), points actés

- Nom `crediscore`, alias `champion` / `challenger` (les stages sont dépréciés).
- Chargement en `mlflow.sklearn`, objet calibré (`CalibratedClassifierCV`, isotonique, `ensemble=False`). SHAP explique le pipeline non calibré contenu dans cet objet (`calibrated_classifiers_[0].estimator`) : pas d'artefact séparé.
- Entrée : les colonnes `params.features.all` (D2). Sortie : probabilité de défaut.
- `top_factors` : contributions SHAP en log-odds du modèle non calibré. Elles ne s'additionnent **pas** à `proba`.
- Promotion : champion et challenger évalués **sur le même jeu récent**. Le challenger gagne s'il a une AUC ≥ celle du champion et ≥ 0,75, des écarts selon le sexe ≤ 10 pts (brut et égalité des chances), et un écart d'âge en égalité des chances ≤ celui de la baseline + 2 pts (≤ 34,9 pts, référence fixe, D5).

## Actions

**Areej**
- [ ] Ajouter dans `params.yaml` / `config.py` : `data.val_size: 0.2`, `cost.cost_review: 0.3`, `threshold.max_review_share: 0.15`, `threshold.min_recall: 0.65`.
- [ ] Documenter `threshold.low/high` comme forçage manuel optionnel.
- [ ] Corriger le libellé `A ETUDIER` → `A_ETUDIER`.
- [ ] Exposer `add_derived_features(df)` (pure, testée).
- [ ] Aligner `IS_THIN_FILE` sur D10.

**Iyed**
- [ ] CI : déclencheur `push` sur `feature/**` ; `cache: poetry` après l'installation de Poetry.
- [ ] Contrat 3 : `decision: Literal["ACCEPTE", "A_ETUDIER", "REFUSE"]` ; `/predict` appelle `add_derived_features` ; seuils lus avec le modèle.

**Achraf**
- [x] Mesurer l'écart d'équité sur la baseline (D5) : sexe 7,8 / 6,6 pts, âge 34,5 / 32,9 pts (brut / égalité des chances, test).
- [ ] Prévenir l'enseignant : le KPI du SYNOPSIS « écart ≤ 10 points » s'applique désormais au **sexe** ; l'âge est « mesuré, expliqué, sans dégradation » (D5).
- [ ] Lot 3 : expérience LightGBM sans `AGE_YEARS` (compromis AUC/coût contre écart d'âge, pour le rapport).
- [ ] Écrire `docs/contracts/02-model-artifact.md`.
- [ ] Livrer en début de S3 un modèle squelette avec `thresholds.json` et `decide()`.
