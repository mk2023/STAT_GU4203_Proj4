# Part 3 — Supervised Modelling

This document covers tasks 4 (Model Development) and 5 (Model Comparison & Selection) of the project rubric. It walks through the experimental matrix, the rationale for the chosen final model, calibration and threshold decisions, the sensitivity-analysis findings, and the handoff to Part 4.

## TL;DR — start here

```
Winner:               L2 Logistic Regression (C=0.01, class_weight='balanced')
Feature set:          A (full, 80 columns; lifestyle_risk_score_no_dx dropped)
Calibration:          Platt scaling (5-fold CV on train)
Operating threshold:  0.10 (screening criterion: recall >= 80% on OOF train)

Cross-validated:      ROC-AUC = 0.824 +/- 0.010   PR-AUC = 0.381
Test set (held out):  ROC-AUC = 0.813              PR-AUC = 0.391
                      Brier  = 0.092 (Platt) vs 0.175 raw -- 47% reduction
                      Recall = 0.852  Precision = 0.259  F1 = 0.397 @ t=0.10
```

To reproduce from scratch (~12 min on a 16-thread CPU):

```bash
pip install xgboost catboost shap   # in addition to the Part 1-2 stack; see README
python3 src/supervised_models.py
```

To consume the model from a downstream notebook or dashboard:

```python
from src.predict_api import predict_with_threshold
preds, probs = predict_with_threshold(X_already_preprocessed)
```

## What's in this folder

`outputs/models/` is created by `src/supervised_models.py`:

| File | Description |
|---|---|
| `cv_results.csv`           | 12 rows (4 models x 3 feature subsets) of mean ± std CV metrics |
| `final_model.joblib`       | Calibrated winning estimator, retrained on full train |
| `final_model_card.json`    | Model name, feature subset, hyperparameters, threshold, test metrics, full feature list used |
| `test_metrics.json`        | Test ROC-AUC, PR-AUC, Brier (raw / Platt / isotonic), F1 |
| `roc_pr_curves.png`        | All 12 models on the test set; winner highlighted |
| `calibration.png`          | Reliability diagram comparing raw / Platt / isotonic probabilities |
| `threshold_analysis.csv`   | Out-of-fold precision / recall / F1 / specificity at thresholds 0.05–0.95 |
| `shap_summary.png`         | SHAP beeswarm — per-row feature attribution on the test set |
| `shap_bar.png`             | SHAP global mean(\|SHAP\|) ranking |

## Modelling matrix

The rubric requires at least three distinct supervised models. We trained **four** (linear baseline + non-linear ensembles + two boosting libraries) on **three** feature subsets defined by Part 2's sensitivity_feature_exclusions.json — yielding **12 experiments** in total.

### Models

| Model | Library | Imbalance handling | Tuning |
|---|---|---|---|
| Logistic Regression | sklearn 1.8 | `class_weight='balanced'`, L2 penalty (`solver='liblinear'`) | GridSearchCV over `C` ∈ {0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100} |
| Random Forest | sklearn | `class_weight='balanced_subsample'` | RandomizedSearchCV (`n_iter=20`) over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| XGBoost | xgboost 3.2 | `scale_pos_weight=6.945` | RandomizedSearchCV (`n_iter=20`) over `learning_rate`, `max_depth`, `n_estimators`, `reg_lambda`, `subsample`, `colsample_bytree` |
| CatBoost | catboost 1.2 | `auto_class_weights='Balanced'` | RandomizedSearchCV (`n_iter=20`) over `depth`, `learning_rate`, `l2_leaf_reg`, `iterations` |

All searches use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for inner CV, optimise mean ROC-AUC, and record PR-AUC, F1, and Brier (via `neg_brier_score`) in the same pass. All seeds = 42. All inner estimators use `n_jobs=1` / `thread_count=1`; the outer search uses `n_jobs=-1` to parallelise across folds and hyperparameter samples without oversubscribing the CPU.

### Feature subsets

Per Part 2's exclusion JSON:

- **A (full, 80 cols):** all 81 columns minus `lifestyle_risk_score_no_dx`. The `_no_dx` version is highly correlated with the full lifestyle score (r ≈ 0.84) and only used when the full version is excluded.
- **B (no diagnosis history, 79 cols):** drop `diagnosis_history_columns` (`lifestyle_risk_score`, `diabetes_borderline_flag`); keep ACC/AHA / ADA threshold bins.
- **C (raw measurements only, 72 cols):** drop everything in `columns_to_drop` (diagnosis history *and* threshold bins); keep only raw labs/vitals plus engineered transformations of raw values.

The A→B AUC drop quantifies the predictive contribution of prior diagnosis. The B→C drop quantifies the additional value of clinical-guideline threshold encodings beyond the raw measurements.

## Cross-validated results

```
                                 ROC-AUC                 PR-AUC      F1     Brier   fit (s)
Model         Set  n_feat   mean    std         mean         mean    mean    mean
LogReg          A      80   0.8243 ± 0.0097     0.3813      0.413    0.178      6
RandomForest    A      80   0.8154 ± 0.0113     0.3565      0.419    0.131     41
XGBoost         A      80   0.8117 ± 0.0076     0.3560      0.411    0.144     14
CatBoost        A      80   0.8240 ± 0.0089     0.3772      0.425    0.157     98
LogReg          B      79   0.8225 ± 0.0079     0.3797      0.410    0.179      2
RandomForest    B      79   0.8132 ± 0.0104     0.3533      0.418    0.132     40
XGBoost         B      79   0.8086 ± 0.0100     0.3604      0.418    0.151     13
CatBoost        B      79   0.8222 ± 0.0081     0.3824      0.416    0.164     97
LogReg          C      72   0.8214 ± 0.0083     0.3792      0.407    0.180      2
RandomForest    C      72   0.8140 ± 0.0104     0.3553      0.416    0.134     38
XGBoost         C      72   0.8088 ± 0.0102     0.3615      0.416    0.151     13
CatBoost        C      72   0.8228 ± 0.0087     0.3814      0.420    0.158     96
```

All metrics are means over the 5 outer folds at the best hyperparameter setting. Standard deviations on PR-AUC, F1, and Brier are in `cv_results.csv`.

## Final-model selection

We rank by **mean CV ROC-AUC** with ties broken by (i) higher PR-AUC, (ii) simpler model class (LogReg < RF < XGBoost < CatBoost). The top three are:

1. **LogReg on A — 0.8243 ± 0.010** ← winner
2. CatBoost on A — 0.8240 ± 0.009 (Δ = 0.0003, statistically indistinguishable)
3. CatBoost on C — 0.8228 ± 0.009

Why LogReg wins on a virtual tie:

- **Interpretability.** Coefficients map directly to feature contributions on the log-odds scale; the rubric explicitly weights "interpretability and robustness" alongside performance. CatBoost would require post-hoc SHAP for the same story.
- **Stability and runtime.** LogReg trains in ~2 s end-to-end; CatBoost takes ~95 s. For repeated experimentation and dashboard inference, the linear model is preferable.
- **Regularisation.** The chosen `C = 0.01` corresponds to strong L2 shrinkage. With 80 features and 6,245 rows, this controls both multicollinearity (BMI ↔ waist, glucose ↔ HbA1c, total_chol ↔ LDL) and overfitting.

The implication is non-trivial: **on this dataset, sophisticated tree ensembles do not improve over a regularised linear baseline.** This is consistent with NHANES-type data being approximately log-linear in CVD risk once Part 2's transformations (log1p, interactions, threshold bins) are applied — and it argues against deploying a heavier model when the simpler one matches it.

## Calibration

Boundary probabilities from `class_weight='balanced'` LogReg are pushed toward 0.5 to compensate for imbalance, so raw output is poorly calibrated for absolute-probability use. We compare three calibration paths on the test set:

| Calibration | Brier (test) |
|---|---|
| Raw (no calibration)         | 0.1747 |
| **Platt (sigmoid, 5-fold)**  | **0.0915** ← chosen |
| Isotonic (5-fold)            | 0.0918 |

Platt and isotonic are essentially tied; we pick Platt because its parametric form is monotonic and more conservative on out-of-distribution rows than the piecewise-constant isotonic mapping.

For reference, the trivial "always predict prevalence (0.126)" baseline has a Brier of 0.110. **Raw LogReg (0.175) is worse than this baseline**; **Platt-calibrated LogReg (0.092) is better.** Calibration is therefore not optional for this model. See `calibration.png` for the reliability diagram across all three.

## Operating threshold

We do not use the default 0.5 cut-off. Instead we sweep thresholds on **out-of-fold training probabilities from the calibrated model** (5-fold CV) and pick the threshold with the **highest specificity subject to recall ≥ 0.80** (a screening criterion: capture at least 80% of true cases at the cheapest false-positive rate that still does so).

Threshold sweep highlights from `threshold_analysis.csv`:

```
threshold  precision  recall   F1     specificity  n_pos_pred
   0.05      0.204     0.947   0.336    0.470         3640
   0.10      0.256     0.854   0.394    0.642         2624     ← chosen
   0.15      0.298     0.744   0.425    0.747         1966
   0.20      0.344     0.640   0.447    0.824         1464
   0.30      0.404     0.375   0.389    0.920          730
   0.50      0.539     0.075   0.131    0.989          110
   0.80      1.000     0.003   0.005    1.000            2
```

At 0.10 the model captures 85% of positives in OOF training data. Applied to the held-out test set with the calibrated probabilities, we measure:

- **Recall = 0.852**
- **Precision = 0.259**
- **F1 = 0.397**

A higher-precision operating point (e.g. for triage instead of screening) is also straightforward: at threshold 0.20, recall drops to 0.64 but precision rises to 0.34. The full sweep is in `threshold_analysis.csv`.

> **Threshold-calibration matching.** The threshold is selected on OOF probabilities from the *same* `final_estimator` we apply to the test set (the Platt-calibrated model, not the raw one). Selecting on raw OOF probs and applying on calibrated test probs silently shifts the distribution and collapses recall — `predict_api.py` consumes the same `final_estimator` so this stays consistent.

## Sensitivity analysis — the research story

Per-model Δ ROC-AUC across feature subsets:

```
Model            A         B         C       A−B       A−C
LogReg         0.8243    0.8225    0.8214   +0.0018   +0.0029
CatBoost       0.8240    0.8222    0.8228   +0.0018   +0.0012
RandomForest   0.8154    0.8132    0.8140   +0.0021   +0.0014
XGBoost        0.8117    0.8086    0.8088   +0.0031   +0.0029
```

**The largest gap between any subset is 0.003 ROC-AUC, smaller than the CV standard deviation (~0.01) for every model.** Consequences:

- **Diagnosis-history features add essentially no information** (A→B drop ≤ 0.003). This is striking given that two of the dropped columns — `lifestyle_risk_score` and `diabetes_borderline_flag` — encode prior diabetes diagnosis directly. The implication is that the residual signal in those columns, after age and metabolic measurements, is captured elsewhere.
- **Guideline-threshold encodings add nothing on top of raw labs and vitals** (B→C drop ≤ 0.002). The ACC/AHA blood-pressure stages and ADA HbA1c categories carry the same information the raw `sbp_avg` / `hba1c` columns already provide, plus or minus a sigmoid.

**The actionable takeaway** is that a CVD screening model on NHANES-style routine measurements does *not* need access to a participant's prior diagnosis history to perform within 0.003 ROC-AUC of the full model. This is encouraging for screening use cases where diagnosis history is unreliable or unavailable — the deployable system can run on raw labs/vitals alone.

For the final pickled model we ship feature subset A (full) because it has marginally the best CV ROC-AUC and there is no operational reason to artificially restrict features. The robustness across A/B/C is itself the headline.

## Feature attribution

### Top 15 features by |LogReg coefficient| (averaged across the 5 calibration folds)

```
+ age_years                  +0.538     <-- ranked first; Cohen's d = 0.95 in Part 2 EDA
+ metabolic_burden_z         +0.332
- total_chol                 -0.284     <-- statin reversal (see below)
- age_group_30_44            -0.263     <-- younger groups protective vs. reference
- education_college_grad     -0.259
+ age_x_hba1c                +0.245     <-- age × glycaemia interaction
- sex_female                 -0.225
- race_ethnicity_mexican_am. -0.187
+ is_former_smoker           +0.186
+ age_x_sbp                  +0.175     <-- age × systolic BP interaction
+ lifestyle_risk_score       +0.174
- bp_stage_elevated          -0.173     <-- one-hot referent effect; not a contradiction
+ waist_cm                   +0.167
- income_poverty_ratio       -0.155
+ is_current_smoker          +0.150
```

These align with established CVD epidemiology — age, metabolic burden, smoking, hypertension, central adiposity — and reproduce Part 2's Cohen's-d ranking with `age_years` first.

### Cholesterol reversal (handoff item 5)

The handoff document explicitly warned that `total_chol` and `ldl` would carry **negative** coefficients because diagnosed CVD patients are typically on statin therapy, which suppresses measured cholesterol below their pre-diagnosis baseline. Our fitted coefficients confirm:

| Feature | Coefficient | Interpretation |
|---|---|---|
| `total_chol`   | **-0.284** | strongly negative — statin effect |
| `ldl`          | **-0.119** | negative |
| `chol_hdl_ratio` | -0.024  | weakly negative |
| `hdl`          | -0.013     | near zero |

These coefficients are **not bugs**. The model is correctly capturing a treatment-induced association in the data. **They cannot be interpreted as "lower cholesterol causes CVD"** — they reflect the joint distribution of measured cholesterol and diagnosed status under standard care.

### SHAP

`shap_summary.png` (beeswarm) and `shap_bar.png` (global mean(|SHAP|)) recover the same ordering on test-set data, with `age_years` dominating, followed by metabolic-burden composites and individual measurement features. Global rankings agree with the LogReg coefficients above; SHAP is included primarily because for non-linear deployments the same plotting code drops in unchanged for tree models.

## Caveats and known limitations

1. **Effective ceiling around ROC-AUC 0.82–0.83.** All four model families converge to within 0.013 of each other. With current features this is the natural ceiling; further gains require additional information (medication history, family history, longitudinal follow-up) rather than fancier models.
2. **Test ROC-AUC (0.813) is below the CV mean (0.824).** A 0.011 generalisation gap is well within one standard deviation and is consistent with normal optimism in CV vs held-out evaluation. There is no evidence of leakage.
3. **PR-AUC is 0.391 against a prevalence baseline of 0.126** — a 3.1× lift. Given an absolute prevalence this low, screening models published on similar data report PR-AUC in the 0.3–0.5 range; ours is in the middle of that.
4. **Operating threshold is selected on OOF training data, not on the test set.** This is intentional (test set is reserved for final evaluation), but means the test-set recall (0.852) is a mildly optimistic estimate of the screening sensitivity in deployment. A pre-deployment sensitivity audit on a fresh NHANES cycle would be the next step.
5. **K-Means cluster as a feature was not used.** Doing it correctly requires re-fitting clusters inside each CV fold (the supplied `kmeans_pipeline.joblib` was fit on the full sample), and Part 2's silhouette of 0.188 with 4 phenotypes suggests low marginal information beyond existing features. Optional extension only.
6. **Imbalance was handled via `class_weight='balanced'` and `scale_pos_weight`, not SMOTE.** SMOTE was considered but adds risk (synthetic minority neighbours can leak across folds if applied incorrectly) without observed benefit on this dataset.
7. **`xgboost >= 3.0` is required.** sklearn 1.7+ enforces strict estimator-type checks that XGBoost 2.x's wrapper does not satisfy (raises `XGBClassifier should either be a classifier ... Got a regressor`). The CLAUDE.md and README pin this.

## Handoff to Part 4

`src/predict_api.py` is the dashboard's entry point. It exposes three functions, all driven by `outputs/models/final_model.joblib` + `final_model_card.json`:

```python
from src.predict_api import (
    load_artifacts,           # (preprocessor, model, card) — cached
    predict_proba,            # already-preprocessed DataFrame -> P(CVD)
    predict_with_threshold,   # -> (binary_predictions, probabilities)
    predict_from_raw,         # raw engineered features -> P(CVD), runs preprocessor
)
```

Default behaviour:

- `predict_with_threshold(X)` uses the card's `recommended_threshold = 0.10` (recall ≥ 80% screening criterion).
- Pass an explicit `threshold` to move to a higher-precision operating point.
- `predict_from_raw(X_raw)` accepts engineered-but-unscaled rows (matching `data/processed/analytic_dataset_engineered_*.csv`) and pushes them through the saved preprocessor before scoring — useful if the dashboard ingests raw NHANES rows.

Smoke test:

```bash
python3 src/predict_api.py
# Loaded model: LogReg / A
#   recommended_threshold = 0.1
#   test ROC-AUC = 0.8130
#   row 0: pred=1, proba=0.1528
#   ...
```

The dashboard does **not** need to retrain anything or know about Parts 1–2's transformations. It only needs to call one of the four functions above.

## Reproducibility

- All random seeds = 42 (`StratifiedKFold`, `RandomizedSearchCV`, every estimator).
- Final outputs deterministic to within floating-point noise across machines.
- `outputs/models/final_model_card.json` records the exact best hyperparameters, feature columns used, and training timestamp for audit.
- Verify Parts 1–2 are still consistent after any change here: `python3 src/verify_part2.py` should print `ALL CHECKS PASSED`.

## Open questions for Part 4

- Do you want the dashboard to surface the calibrated probability, the binary screening prediction, or both? (The interface supports both — recommendation: show the probability with a colour-coded screening flag, and let the user move the threshold slider.)
- Should the SHAP per-row explanation be available in the dashboard? (Cheap for LogReg via the saved coefficients; for the tree models we'd need to ship the SHAP explainer object.)
- Is there interest in a "what-if" panel that lets users perturb a single feature and see how the predicted probability moves? (The infrastructure is in place — `predict_proba` accepts arbitrary rows.)
