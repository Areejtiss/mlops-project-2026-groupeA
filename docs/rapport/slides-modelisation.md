# Slides — partie d'Achraf (environ 3 minutes, 6 slides)

Plan des slides et notes orales. Les chiffres viennent de `docs/rapport/modelisation.md`, qui cite leurs sources (`reports/metrics.json`, le plan, les contrats 2 et 4). Les créneaux de la démo (roadmap d'équipe, section 5) sont à la fin.

---

## Slide 1 — Résultats et KPI (40 s)

**À l'écran :** le tableau KPI (rapport §7), avec des pastilles vert / rouge.
- Atteints : recall 66,9 % (≥ 65 %), Brier 0,0685 (≤ 0,07), thin file 4 946 bons payeurs acceptés contre 0, écart selon le sexe 7,7 / 6,3 pts (≤ 10), SHAP p95 14,9 ms (< 50).
- **Manqués :** ROC-AUC 0,747 (cible 0,75), PR-AUC 0,227 (cible 0,25), coût −9,0 % contre la baseline (cible −15 %).

**Notes orales :**
- « Nous avons trois cibles manquées et nous les montrons telles quelles : rien n'a été assoupli, rien n'a été réglé sur le test. »
- « Le coût baisse quand même de 9 % contre la régression logistique, et de 21 % contre "tout accepter". »
- « La baisse validation → test d'environ 0,02 d'AUC touche les trois modèles mesurés : elle tient au split, pas au choix du modèle. »

## Slide 2 — MLflow : 4 modèles comparés, un champion (35 s)

**À l'écran :** capture de la vue « Compare » de MLflow (5 runs) et du Registry (`crediscore`, alias `champion`). Tableau réduit : LogReg 0,756 / 0,340 ; RandomForest 0,748 / 0,333 ; **LightGBM 0,767 / 0,307** ; XGBoost 0,765 / 0,343 (AUC / coût par dossier, validation).

**Notes orales :**
- « La sélection se fait sur la validation, découpée dans le train. Le test n'est lu qu'une fois, par `evaluate`. »
- « Règle : AUC ≥ 0,75, puis le coût le plus bas sous les contraintes. XGBoost a presque la même AUC, mais il refuse plus : il coûte plus cher. »
- « Optuna : 25 essais par modèle, chacun visible comme run imbriqué. »

## Slide 3 — Seuils et coût métier : 3 zones (30 s)

**À l'écran :** la figure `reports/figures/baseline_cost_vs_threshold.png` et le schéma `ACCEPTE < 0,094 ≤ A_ETUDIER < 0,133 ≤ REFUSE`.

**Notes orales :**
- « Coûts : défaut raté 5, bon client refusé 1, étude manuelle 0,3. »
- « Un seul seuil ne détecte que 38 % des défauts. Trois zones avec au plus 15 % de dossiers en étude atteignent 66,9 % de recall. »
- « Les seuils sont une sortie de l'entraînement : ils voyagent avec la version du modèle. »

## Slide 4 — SHAP : pourquoi cette décision (30 s)

**À l'écran :** `reports/figures/shap_summary.png` et l'exemple 100295 : proba 0,098 → `A_ETUDIER` ; `EXT_SOURCE_2` +0,40, `EXT_SOURCE_1` +0,21, `REGION_POPULATION_RELATIVE` −0,16 (log-odds).

**Notes orales :**
- « Les trois scores externes dominent. »
- « Réserve honnête : SHAP explique le modèle **avant calibration**, en log-odds. Les contributions ne s'additionnent pas à la probabilité affichée. Mais la calibration est monotone, donc le sens des facteurs reste juste. »

## Slide 5 — Équité : la règle D5 et l'expérience « sans âge » (35 s)

**À l'écran :** sexe 7,7 / 6,3 pts (limite 10) ; âge en égalité des chances 33,1 pts (limite 40,7 = baseline 38,7 + 2) ; mini-tableau avec / sans `AGE_YEARS` : écart 33,1 → 35,6, coût −9,0 % → −7,1 %.

**Notes orales :**
- « `CODE_GENDER` n'entre jamais dans le modèle : il sert seulement à l'audit. »
- « Pour l'âge, un écart de 10 points est impossible : les moins de 30 ans font défaut deux fois plus. Notre règle : ne jamais faire pire que la baseline + 2 points. »
- « Retirer l'âge du modèle **augmente** l'écart d'âge et le coût : nous le gardons. »
- « L'audit tourne aussi sur chaque lot surveillé : c'est le suivi de l'équité dans le temps. »

## Slide 6 — Dérive : ce que le module voit (30 s)

**À l'écran :** le tableau crise / moitiés / scénario C (rapport §5.2) et une capture de `latest.json` avec `"alert": true`.

**Notes orales :**
- « Crise économique : le PSI du ratio annuité/revenu vaut 0,68, le revenu 0,26, le crédit 0,23 : alerte. »
- « L'AUC ne baisse que de 0,009 : le simulateur ne touche ni les scores externes ni les labels. C'est une dérive des données, pas une dérive de concept. »
- « En revanche, le modèle accepte 4,9 points de dossiers en moins : c'est un impact métier, visible grâce à la dérive des prédictions. »
- « Le scénario C fait l'inverse : rien ne bouge dans les données, seule l'AUC voit la dérive (−0,126). »
- « Leçon : notre premier PSI donnait une fausse alerte sur un décalage de 1e-9, à cause des revenus ronds. La relecture l'a trouvé, nous l'avons corrigé et testé. »

---

## Créneaux de démo (script d'équipe, 5 minutes)

### 1:00 — MLflow : comparaison des 4 modèles, champion dans le Registry (Achraf, environ 45 s)

1. Ouvrir l'UI MLflow, expérience `crediscore`, cocher les runs LogReg, RandomForest, LightGBM et XGBoost, puis « Compare » : montrer `val_roc_auc` et `val_cost_per_file`.
2. Ouvrir le run LightGBM : onglet Artifacts, `roc_curve.png` et `confusion_matrix.png` ; tags `threshold_low` / `threshold_high`.
3. Onglet Models : `crediscore`, alias `champion` sur la version servie ; dire que l'API charge `models:/crediscore@champion`.
- **À préparer avant :** l'UI déjà ouverte sur l'expérience ; aucun run de test récent qui encombre la liste.

### 2:30 — Rapport Evidently et PSI par feature, puis l'alerte dans Prometheus (Achraf → Iyed, environ 45 s)

1. Ouvrir le HTML Evidently du dernier passage du `monitoring_flow` (`reports/drift/<date>.html`) : montrer les 7 colonnes en dérive (revenu, crédit, annuité, ratios, ancienneté d'emploi).
2. **Ne pas lire le bandeau** « Dataset Drift is NOT detected » : c'est la règle d'Evidently (≥ 30 % de colonnes), pas la nôtre. Ouvrir `reports/drift/latest.json` : PSI par feature, `alert: true`, puis `prediction` (acceptation 67,3 % → 62,5 %) et `fairness` (D5 respecté).
3. Passer la parole à Iyed : la jauge `drift_alert` passe à 1 dans Prometheus.
- **À préparer avant :** le lot de crise déjà simulé (Areej, 1:45), le JSON et le HTML générés avec `--model` (sinon `auc_drop`, `prediction` et `fairness` valent `null`).

### Répétition

- Chronométrer les deux créneaux (cible 45 s chacun) ; garder les captures en secours, au cas où l'UI MLflow serait lente.
- À 4:00 (Iyed), la promotion montre `champion` sur la nouvelle version : préparer la réponse à « pourquoi le nouveau modèle a-t-il gagné ? » avec `reports/promotion.json` (`checks`).
