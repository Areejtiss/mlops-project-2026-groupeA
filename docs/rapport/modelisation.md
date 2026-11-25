# Rapport — Modélisation, MLflow, SHAP, équité, dérive (sections d'Achraf)

Ces sections couvrent les chapitres attribués à Achraf dans la roadmap d'équipe : résultats comparatifs MLflow, choix du seuil et coût métier, explicabilité SHAP, audit d'équité, analyse de la dérive. Rédigées le 25/11/2026 pour être reprises dans le rapport PDF.

**Règle de lecture :** chaque chiffre renvoie à un fichier versionné, entre crochets.
- [M] : `reports/metrics.json` (sortie d'`evaluate`, test de 20 000 lignes).
- [P] : `docs/plans/2026-10-24-achraf-roadmap.md` (plan, avec la section du lot).
- [C2] : `docs/contracts/02-model-artifact.md`.
- [C4] : `docs/contracts/04-drift-report.md`.
- [H] : `docs/handoff/boucle-reentrainement.md`.
- [B] : `reports/baseline_metrics.json` (notebook `02_baseline`).

---

## 1. Comparaison des modèles dans MLflow

### 1.1 Protocole

- **Données :** le `train.parquet` d'Areej (80 000 lignes, échantillon stratifié de 100 000 parmi 307 511) ; 20 % du train mis de côté en **validation** (16 000 lignes, D1) [P, lot 3].
- **Calibration :** `CalibratedClassifierCV(isotonic, cv=5, ensemble=False)` sur la partie fit ; la validation ne sert qu'aux seuils et à la sélection (T2) [P, §2].
- **Réglage :** Optuna, 25 essais pour LightGBM et XGBoost, sur la validation croisée à 5 plis de la partie fit (chaque essai est un run MLflow imbriqué) [P, lot 3, L3-2].
- **Sélection (L3-1) :** sur la validation uniquement. Parmi les runs qui respectent les contraintes (recall ≥ 65 %, au plus 15 % en étude), on garde ceux d'AUC ≥ 0,75, puis le **coût par dossier le plus bas**. Le test n'est lu qu'une fois, par `evaluate`, après la sélection [P, lot 3].
- **Déséquilibre :** `class_weight="balanced"` / `scale_pos_weight`, jamais SMOTE (T5) [P, §2].

### 1.2 Tableau comparatif (validation, expérience MLflow `crediscore`) [P, lot 3]

| Run | ROC-AUC | PR-AUC | Brier | Recall (D8) | En étude | Coût / dossier |
|---|---|---|---|---|---|---|
| LogReg (baseline) | 0,7558 | 0,2382 | 0,0680 | 71,9 % | 14,8 % | 0,3395 |
| RandomForest (300 arbres, fixe) | 0,7480 | 0,2337 | 0,0682 | 69,3 % | 15,0 % | 0,3326 |
| **LightGBM, Optuna 25 essais** | **0,7670** | **0,2574** | **0,0670** | 69,7 % | 14,9 % | **0,3070** |
| XGBoost, Optuna 25 essais | 0,7648 | 0,2543 | 0,0671 | 71,0 % | 11,4 % | 0,3428 |
| LightGBM sans `AGE_YEARS` (expérience D5) | 0,7649 | 0,2507 | 0,0672 | 69,4 % | 14,0 % | 0,3136 |

« Tout accepter » coûte 0,4037 par dossier sur la validation [P, lot 3].

- **Lecture :** RandomForest tombe sous le plancher de 0,75. XGBoost a presque la même AUC que LightGBM mais un coût plus élevé : ses seuils envoient moins de dossiers en étude (11,4 %) et il refuse davantage. **Champion : LightGBM**, enregistré comme `crediscore` **v1**, alias `champion` (run `1c3f3b8712aa43d491ec3882bf64e70e`) [P, lot 3 ; C2 §7].
- **Courbes :** chaque run `train` contient dans MLflow `roc_curve.png`, `pr_curve.png`, `calibration_curve.png` et `confusion_matrix.png` (validation), et les métriques `val_*`. Pour les slides, faire les captures dans l'UI MLflow : vue « Compare » des 5 runs, puis le Registry (`crediscore`, alias `champion`). Les courbes de la baseline sont aussi dans `reports/figures/baseline_roc_pr_calibration.png`.

### 1.3 Champion sur le test (`evaluate`, seul lecteur du test) [M]

La baseline LogReg est **réentraînée à l'identique sur le même `train.parquet`** et évaluée sur le même test (L3-4).

| Test (20 000 lignes) | Baseline LogReg | Champion LightGBM v1 |
|---|---|---|
| ROC-AUC | 0,7346 | 0,7469 |
| PR-AUC | 0,2079 | 0,2267 |
| Brier | 0,0692 | 0,0685 |
| Recall (D8) | 69,4 % | 66,9 % |
| En étude | 14,4 % | 14,5 % |
| Coût / dossier | 0,3519 | 0,3203 |
| Seuils low / high | 0,0763 / 0,1145 | 0,0941 / 0,1329 |

- **Écart validation → test :** les trois modèles évalués sur le test perdent environ 0,02 d'AUC (LogReg 0,756 → 0,735 ; LightGBM 0,767 → 0,747 ; sans âge 0,765 → 0,746). La baisse est commune, elle ne vient donc pas du modèle choisi [P, lot 3].
- On ne règle **rien** sur le test (L3-6) : il ne sert qu'à mesurer.

## 2. Seuils et coût métier (3 zones, D3 et D8)

### 2.1 Le coût

- Défaut non détecté (accepté) = **5**, bon client refusé = **1**, dossier envoyé en étude manuelle = **0,3** (D3) [`params.yaml` `cost` ; décisions D3].
- **Pourquoi 5:1 :** un défaut coûte une grande part du capital prêté ; un bon client refusé ne coûte que la marge perdue. La justification est écrite dans `notebooks/02_baseline.ipynb` [P, lot 1].
- **L'accuracy n'est jamais utilisée :** prédire « 0 » partout donne environ 92 % d'exactitude, car il y a 8 % de défauts.

### 2.2 Trois zones

- `proba < low` → `ACCEPTE` ; `low ≤ proba < high` → `A_ETUDIER` ; `proba ≥ high` → `REFUSE` (D4) [C2 §5].
- `low` et `high` minimisent le coût sur la **validation**, sous deux contraintes :
  - **recall ≥ 65 %** (D8 : un défaut envoyé en étude compte comme détecté) ;
  - **au plus 15 % des dossiers en étude** (D3 : capacité des analystes).
- Les seuils sont une **sortie** de l'entraînement : ils voyagent avec la version du modèle (`thresholds.json` et tags MLflow, D9).

### 2.3 Pourquoi trois zones : les chiffres de la baseline (lot 1) [P, lot 1]

| Politique (baseline, test du lot 1) | Recall | En étude | Coût / dossier |
|---|---|---|---|
| Un seul seuil optimal (0,1707) | 38,6 % | 0 % | 0,340 |
| 3 zones sans plafond (0,0571 / 0,7500) | — | 48,7 % | — |
| **3 zones D3 (0,0926 / 0,1378)** | 66,5 % | 14,2 % | **0,325** |

- Avec **un seul seuil**, l'optimum ne détecte que 38 % des défauts. Imposer 65 % de recall avec un seul seuil coûte même plus cher que tout accepter (0,409 contre 0,404 par dossier) [P, lot 1].
- **Sans plafond**, l'optimum envoie la moitié des dossiers en étude, ce qui est irréaliste.
- Les 3 zones avec plafond respectent les deux contraintes au coût le plus bas.
- Figure : `reports/figures/baseline_cost_vs_threshold.png`.

### 2.4 Résultat du champion [M]

- Coût **0,3203** par dossier, contre **0,3519** pour la baseline : **−9,0 %**. La cible du SYNOPSIS est −15 % (≤ 0,2991) : **manquée de 6,0 points**.
- Contre « tout accepter » (0,4037) : −20,7 %.
- Recall 66,9 % (cible 65 %, atteinte), 14,5 % des dossiers en étude.

## 3. Explicabilité SHAP

### 3.1 Méthode [C2 §6 ; P, T3]

- SHAP explique le **pipeline non calibré** contenu dans le modèle servi : `calibrated_classifiers_[0].estimator`. Il n'y a pas d'artefact séparé.
- Explainer : `TreeExplainer` (LightGBM, XGBoost, RandomForest), `LinearExplainer` (LogReg). Il est construit **une seule fois** au démarrage de l'API.
- Les contributions des colonnes one-hot sont ré-additionnées par **colonne d'origine** (`OCCUPATION_TYPE`, pas `cat__OCCUPATION_TYPE_Laborers`).
- **Unités** (tag `shap_unit` de la version) : **log-odds** du modèle non calibré pour LogReg, LightGBM et XGBoost (champion : `log_odds`) ; probabilité pour RandomForest. Signe > 0 = augmente le risque.

### 3.2 Réserve due à la calibration (à dire au jury)

- Les contributions s'additionnent à la **marge non calibrée**, **pas** à la `proba` affichée.
- Avec `ensemble=False`, la `proba` servie est une fonction **monotone non décroissante** de cette marge (l'isotonique fait des paliers). Un facteur qui augmente la marge peut donc laisser la `proba` sur le même palier, mais jamais la faire baisser.
- L'**ordre** des facteurs est donc une approximation fidèle du sens de la décision. Leurs **valeurs** ne sont pas des points de probabilité.

### 3.3 Vue globale

Figure `reports/figures/shap_summary.png` : barres de la moyenne des |contributions| sur 2 000 lignes du train, aussi enregistrée dans MLflow (run `a146f2b6842d42adac5da1979fa5d0cd`, tag `kind=shap_summary`). **Premières variables : `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1`, `AMT_GOODS_PRICE`, `CREDIT_TERM`** [P, lot 4]. Les scores externes dominent, comme le laissait prévoir l'EDA (corrélations avec `TARGET` de 0,16 à 0,18, `params.yaml`).

### 3.4 Exemple local [C2 §10]

Dossier brut `SK_ID_CURR = 100295` (défaut réel), `OCCUPATION_TYPE` mis à `None` : **proba 0,0978 → `A_ETUDIER`**.

| Facteur | Valeur | Contribution (log-odds) |
|---|---|---|
| `EXT_SOURCE_2` | 0,302 | +0,40 (augmente le risque) |
| `EXT_SOURCE_1` | 0,262 | +0,21 |
| `REGION_POPULATION_RELATIVE` | 0,0725 | −0,16 (diminue le risque) |

Lecture pour l'analyste : deux scores externes bas poussent vers le défaut ; la région en atténue un peu l'effet. Le modèle n'accepte pas ce dossier : il l'envoie en étude, et c'était bien un défaut.

### 3.5 Latence [P, lot 4]

Champion v1, 200 lignes réelles du test, une ligne par appel : `explain` p50 13,1 ms, **p95 14,9 ms** (cible < 50 ms) ; `predict_proba` + `explain` p95 33,2 ms ; construction de l'explainer 0,35 s.

## 4. Audit d'équité

### 4.1 Règle D5 (tranchée le 25/10/2026) [décisions D5 ; C2 §1]

- **Sexe : seuil absolu.** L'écart de taux d'acceptation entre F et M doit être ≤ 10 pts **et** l'écart d'égalité des chances (acceptation parmi les bons payeurs) aussi ≤ 10 pts. Seul `ACCEPTE` compte comme accepté (D8) ; `XNA` est exclu (4 lignes sur 307 511).
- **Âge (<30, 30-50, ≥50) : non-régression.** L'écart d'égalité des chances doit rester ≤ celui de la baseline LogReg + 2 pts. L'ancre est recalculée par `evaluate` : **38,7 pts**, donc une limite de **40,7 pts**. Elle est liée à la baseline, jamais au champion, et ne peut pas dériver d'une promotion à l'autre.
- **Pourquoi pas 10 pts pour l'âge :**
  - l'âge est un vrai facteur de risque dans ces données (les moins de 30 ans font défaut 11,4 % du temps, contre 5,7 % pour les 50 ans et plus) ;
  - l'écart d'âge de la baseline est déjà de 34,5 pts brut et 32,9 pts en égalité des chances (lot 1).
  - Un seuil de 10 pts bloquerait toute promotion. Le KPI « écart ≤ 10 pts » du SYNOPSIS s'applique donc au **sexe** ; l'âge est « mesuré, expliqué, sans dégradation ». Il faut le dire à l'enseignant [P, §2].
- `CODE_GENDER` n'entre jamais dans le modèle : il ne sert qu'à l'audit (test d'invariance dans `tests/test_model_behaviour.py`).

### 4.2 Résultats sur le test [M]

| Écart (pts) | Baseline | Champion | Limite | Statut |
|---|---|---|---|---|
| Sexe, brut | 8,4 | 7,7 | 10 | respecté |
| Sexe, égalité des chances | 7,0 | 6,3 | 10 | respecté |
| Âge, brut | 39,7 | 34,6 | (publié) | — |
| Âge, égalité des chances | 38,7 | 33,1 | 40,7 | respecté |

Acceptation par tranche, champion : <30 ans 46,9 %, 30-50 ans 63,9 %, ≥50 ans 81,5 % [M]. Le champion réduit **tous** les écarts par rapport à la baseline.

### 4.3 L'expérience « sans âge » (D5, lot 3) [P, lot 3]

| | Avec `AGE_YEARS` (v1) | Sans | Baseline |
|---|---|---|---|
| ROC-AUC / PR-AUC | 0,7469 / 0,2267 | 0,7463 / 0,2251 | 0,7346 / 0,2079 |
| Coût / dossier (vs baseline) | 0,3203 (−9,0 %) | 0,3267 (−7,1 %) | 0,3519 |
| Écart d'âge brut / égalité des chances | 34,6 / 33,1 | **37,1 / 35,6** | 39,7 / 38,7 |
| Acceptation <30 / 30-50 / ≥50 | 46,9 / 63,9 / 81,5 % | 42,9 / 65,9 / 80,0 % | 37,3 / 61,0 / 76,9 % |

- **Retirer l'âge n'améliore pas l'équité selon l'âge : l'écart augmente** (+2,5 pts en égalité des chances), l'acceptation des moins de 30 ans baisse de 4 pts et le coût monte.
- **Lecture probable** (non prouvée ici) : l'âge reste visible au travers d'autres variables (ancienneté d'emploi, `DAYS_*`, famille). Le modèle perd en revanche le signal direct qui distingue les jeunes bons payeurs.
- **Décision :** on **garde `AGE_YEARS`**. C'est à la fois moins cher et plus équitable selon l'âge. La règle D5 reste la non-régression par rapport à la baseline.

### 4.4 Thin file (D10) [M]

- 39,4 % du test est thin file : au moins 2 `EXT_SOURCE` manquants, ou aucune demande au bureau de crédit sur un an.
- **KPI du SYNOPSIS lu à la lettre : atteint.** 4 946 bons payeurs thin file acceptés, contre 0 pour « refus systématique sans historique ».
- Indicateurs compagnons, champion contre baseline :
  - **68,4 % contre 61,8 %** des bons payeurs thin file acceptés ;
  - taux de défaut parmi les thin files acceptés **3,96 % contre 3,79 %** (+0,17 pt) : « sans augmenter les pertes » tient à 0,2 pt près, pas strictement.
- AUC sur le segment thin file : 0,755.

### 4.5 Équité dans le temps (lot 6) [C4 §9]

- À chaque lot surveillé, le rapport de dérive publie l'audit des **décisions du champion** sur les lignes labellisées du lot (clé `fairness`) et son statut D5. C'est informatif : cela ne déclenche pas l'alerte.
- **Sur la crise économique :** sexe 7,8 / 6,5 pts ; âge en égalité des chances 34,0 pts (limite 40,7) ; **D5 respecté** [C4 §9.4].

## 5. Analyse de la dérive

### 5.1 Méthode [C4 §2-5]

- **Tests :** KS pour les 31 numériques, χ² pour les 10 catégorielles (manquant = une catégorie) ; `drift_share` = part des 41 features dont la p-value est < 0,05.
- **PSI écrit à la main :** bornes sur les quantiles de la **référence**, dédoublonnées ; bins extérieurs ouverts ; un bin à part pour les manquants ; epsilon 1e-4 par bin.
- **Alerte :** `drift_share` ≥ 0,30, **ou** PSI ≥ 0,20 sur une feature clé (revenu, crédit, ratio annuité/revenu, `EXT_SOURCE_2`), **ou** `auc_drop` > 0,03. Sur moins de 500 lignes, ni `drift_share` ni le PSI ne déclenchent ; `auc_drop` demande au moins 1 000 lignes labellisées et 50 défauts.
- **Rapport HTML :** Evidently 0.7.23 (`DataDriftPreset`, mêmes tests et même α). **Attention, à la démo :** son bandeau applique la règle d'Evidently (≥ 30 % de colonnes) et affiche « Dataset Drift is NOT detected » sur la crise. **Le JSON fait foi** [C4 §7].

### 5.2 Résultats (champion du lot 3) [P, lot 5 ; C4 §8]

| Lot courant contre référence | `drift_share` | PSI clés (revenu / crédit / ratio / `EXT_SOURCE_2`) | AUC réf. → cour. | `auc_drop` | Alerte |
|---|---|---|---|---|---|
| Crise économique contre test | 0,171 (7/41) | **0,262 / 0,231 / 0,676** / 0,000 | 0,747 → 0,738 | 0,009 | **oui** (PSI) |
| Moitié B contre moitié A du test | 0,024 (1/41) | 0,004 / 0,003 / 0,003 / 0,003 | 0,749 → 0,744 | 0,005 | non |
| Scénario C : 30 % des labels « Commercial associate » inversés | 0,000 | 0 / 0 / 0 / 0 | **0,747 → 0,621** | **0,126** | **oui** (AUC) |

- Sur la crise, les 7 features dérivées sont le revenu, le crédit, l'annuité, les 3 ratios et `EMPLOYED_YEARS`. Le PSI le plus fort est celui du ratio annuité/revenu (0,676) [P, lot 5].
- **Dérive des prédictions sur la crise (lot 6)** [C4 §9.4] : `psi_proba` **0,031** (stable), mais l'acceptation passe de **67,3 % à 62,5 % (−4,85 pts)**.

### 5.3 Pourquoi la crise fait si peu baisser l'AUC alors que le PSI alerte

- **Ce que touche le simulateur :** il ne modifie que le revenu, le crédit, l'annuité (et donc les ratios) et l'ancienneté d'emploi (`params.yaml`, `drift`). Il réutilise les lignes du test **avec leurs labels d'origine**.
- **Data drift sans concept drift :** le risque réel de chaque dossier n'a pas changé. Seules les entrées ont bougé. C'est une dérive des données (P(X) change), pas une dérive de concept (P(y|X) inchangée).
- **Ce qui classe les dossiers :** les variables qui comptent le plus pour le classement sont les `EXT_SOURCE` (SHAP, §3.3). Le simulateur ne les touche pas (PSI de `EXT_SOURCE_2` : 0,000). Le **classement** des dossiers bouge donc peu : AUC −0,009.
- **Ce qui change quand même :** les ratios d'endettement montent, donc les scores glissent un peu vers le risque. L'acceptation baisse de 4,85 pts. C'est un **impact métier réel**, invisible dans l'AUC.
- **Le scénario C est l'inverse :** les distributions ne bougent pas (`drift_share` 0, PSI 0), seuls les labels changent. Seul `auc_drop` le voit.
- **Conclusion :** il faut les deux familles d'indicateurs, données et performance. En crédit, la performance arrive en retard : les labels n'existent que des mois après l'octroi.

### 5.4 Leçon : le bug du PSI trouvé en relecture (lot 5) [P, lot 5, B1 ; C4 §4]

- **Le bug :** la première version faisait de chaque quantile **un bin d'un seul point**. Or 37 % des revenus du test tombent exactement sur un décile (revenus ronds). Multiplier le test par (1 + 1e-9), soit un décalage invisible, sortait ces valeurs de leur bin : **PSI de 2,97 sur le revenu, et une fausse alerte**.
- **Trouvé par :** une sonde de la relecture adversariale (gigue × (1 ± 1e-9), aller-retour float32).
- **Correction :** des **intervalles**, dont chaque borne est placée à mi-chemin entre un quantile et la valeur distincte suivante de la référence. Aucune borne ne tombe donc sur une valeur réelle. Les mêmes sondes donnent alors un PSI de 0,000.
- **Effet sur la crise :** PSI du revenu 2,971 → 0,262, du crédit 0,579 → 0,231. L'alerte reste vraie, pour la bonne raison.
- **Leçon :** un indicateur « écrit à la main et testé » doit être testé sur des **valeurs réelles en tas** (heaped values) et sur des perturbations minuscules, pas seulement sur des lois normales. Un test « PSI = 0 sur des données identiques » ne suffisait pas à trouver ce bug.

## 6. Pourquoi des alias plutôt que Staging / Production

- Le SYNOPSIS prévoyait « promotion Staging → Production ». **MLflow a déprécié les stages** (depuis la 2.9). La version utilisée, MLflow 3.17, recommande les **alias** et les tags [C2 §1].
- **Deux alias :** `champion` (servi par l'API : `models:/crediscore@champion`) et `challenger` (candidat).
- **Avantages :**
  - l'API ne change jamais d'URI : déplacer l'alias suffit à servir le nouveau modèle ;
  - **le retour arrière est immédiat** : on remet `champion` sur la version précédente, notée dans le tag `promotion_previous_champion` ;
  - un alias désigne une seule version à la fois, sans la sémantique figée des 4 stages ;
  - l'historique est porté par les tags de chaque version (`promotion_*`, `promoted_at`).
- **La promotion est relative :** même lot récent, chacun avec ses seuils ; AUC ≥ champion, coût ≤ champion, D5 [C2 §1].
  - Un seuil absolu « AUC ≥ 0,75 » aurait bloqué toute promotion : le champion fait 0,7469 sur le test.
  - Les cibles 0,75 / 0,25 / −15 % restent des **KPI publiés**, pas des conditions de promotion.
- **Boucle vérifiée le 25/11/2026** sur un serveur jetable [H §6] : dérive de la crise → alerte → réentraînement → `challenger` v2 → promotion sur `holdout.parquet` (code 0) → `champion` passe sur v2, avec l'historique `promotion_previous_champion = 1` et `promoted_at`.

## 7. État des KPI (sans arrondi favorable)

| KPI / cible (SYNOPSIS) | Mesure | Statut | Source |
|---|---|---|---|
| ROC-AUC ≥ 0,75 | **0,747** | **manqué** (−0,003) | [M] `kpi.targets.roc_auc` |
| PR-AUC ≥ 0,25 | **0,227** | **manqué** (−0,023) | [M] `kpi.targets.pr_auc` |
| Brier ≤ 0,07 | 0,0685 | atteint (marge 0,0015) | [M] |
| Recall (classe 1) ≥ 65 % | 66,9 % | atteint (+1,9 pt) | [M] |
| Coût métier −15 % vs baseline | **−9,0 %** (0,3203 contre 0,3519) | **manqué** (6,0 pts) | [M] `kpi.cost_vs_baseline` |
| Thin file : ≥ 2 × les bons payeurs de « refus sans historique » | 4 946 contre 0 | atteint (lecture littérale) ; compagnons au §4.4 | [M] `kpi.targets.thin_file` |
| Écart d'acceptation ≤ 10 pts (sexe) | 7,7 brut / 6,3 égalité des chances | atteint | [M] |
| Écart selon l'âge (D5 : ≤ baseline + 2) | 33,1 (limite 40,7) ; brut 34,6 | règle D5 respectée ; **10 pts non atteint** | [M] |
| SHAP < 50 ms par ligne | p95 14,9 ms | atteint | [P, lot 4] |
| API p95 < 150 ms | mesuré par Iyed | — | (section d'Iyed) |

- **Ce que l'on dit au jury :** l'AUC, la PR-AUC et le coût −15 % sont **manqués et publiés comme tels**. Aucune cible n'a été assouplie, et rien n'a été réglé sur le test (L3-6).
- **Ce qui a été tenté :** 4 familles de modèles, Optuna (25 essais par modèle), l'expérience sans âge.
- **Ce qui limiterait encore l'écart :**
  - les tables `bureau` et `previous_application`, non téléchargées : le dataset n'utilise que `application_train` ;
  - l'échantillon de 100 000 lignes, imposé par le guide.
- **La baisse validation → test (≈ 0,02 d'AUC) est commune aux trois modèles mesurés** : elle tient au split, pas au modèle choisi.
