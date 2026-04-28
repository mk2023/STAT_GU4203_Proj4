# Handoff to Part 3 (Modelling Team)

This document tells you exactly what's ready, what's in each file, and the
methodological rules you need to follow to keep the pipeline leakage-free.

## TL;DR — start here

```python
import pandas as pd
import joblib

# Pre-built model matrices (already imputed, scaled, one-hot encoded).
X_train = pd.read_csv("outputs/features/X_train.csv")    # (6245, 81)
X_test  = pd.read_csv("outputs/features/X_test.csv")     # (1562, 81)
y_train = pd.read_csv("outputs/features/y_train.csv")["cvd"]   # (6245,)
y_test  = pd.read_csv("outputs/features/y_test.csv")["cvd"]    # (1562,)

# That's all you need to start training. Don't refit any preprocessing on
# the full data — see "Rules" below.
```

## What you're getting

| File | Description |
|---|---|
| `outputs/features/X_train.csv` | Model matrix, 6245 rows × 81 columns |
| `outputs/features/X_test.csv` | Model matrix, 1562 rows × 81 columns |
| `outputs/features/y_train.csv` | Binary CVD label, 6245 rows |
| `outputs/features/y_test.csv` | Binary CVD label, 1562 rows |
| `outputs/features/feature_names.json` | Ordered list of 81 column names |
| `outputs/features/preprocessor.joblib` | Fitted `ColumnTransformer` (median-impute + indicators + scale + one-hot) |
| `outputs/features/train_fit_stats.json` | All train-only statistics for full reproducibility |
| `outputs/features/sensitivity_feature_exclusions.json` | Columns to drop for the screening-only sensitivity model |
| `outputs/clusters/kmeans_pipeline.joblib` | Fitted scaler + KMeans for OPTIONAL cluster feature use |
| `data/processed/analytic_dataset_engineered_train.csv` | Pre-pipeline engineered train data (reference) |
| `data/processed/analytic_dataset_engineered_test.csv` | Pre-pipeline engineered test data (reference) |

## Class balance

- Train: 6245 rows, CVD positive rate **12.6%**
- Test:  1562 rows, CVD positive rate **12.6%**
- Stratified split, `random_state=42`, `test_size=0.20`

## Rules — read these before training

### 1. Don't refit any preprocessing on the full data
The split was made before any preprocessing. If you concatenate X_train and
X_test and re-scale, you've leaked test info into train. Just use what's
provided, or call `preprocessor.transform(new_data)` for any new participants.

### 2. Class imbalance — don't rely on accuracy
With ~12.6% positive rate, a "predict 0 always" baseline gets 87% accuracy.
Use these metrics instead:
- **Primary:** ROC-AUC, PR-AUC (these handle imbalance natively)
- **Threshold-based:** sensitivity, specificity, F1
- For sklearn models: `class_weight='balanced'`
- For XGBoost: `scale_pos_weight = 6.94` (= 87.4 / 12.6)
- Or: SMOTE on the training set only (`imblearn` package)

### 3. Sensitivity analysis — strongly recommended
The exclusion list is split into two categories (see
`sensitivity_feature_exclusions.json`):

- **Diagnosis-history features** — encode prior diagnosis directly:
  `lifestyle_risk_score` (includes diagnosed-diabetes component) and
  `diabetes_borderline_flag`.
- **Guideline-threshold features** — derived from routine measurements
  via ACC/AHA / ADA cutoffs, but not prior diagnosis themselves:
  `bp_stage_*` and `hba1c_status_*`.

Recommended sensitivity contrasts:

- **Model A (full):** all 81 columns **except** `lifestyle_risk_score_no_dx`
  — that diagnosis-free score is highly correlated with the full
  `lifestyle_risk_score` (r ≈ 0.84) and only used when the full version
  is dropped. So Model A uses 80 columns in practice.
- **Model B (no diagnosis history):** drop the columns listed in
  `diagnosis_history_columns` of the JSON, AND swap
  `lifestyle_risk_score` → `lifestyle_risk_score_no_dx`. Keeps the
  threshold bins.
- **Model C (raw measurements only):** drop everything in `columns_to_drop`,
  AND swap `lifestyle_risk_score` → `lifestyle_risk_score_no_dx`. Forces
  the model to use only raw lab/vital measurements.

The A → B AUC drop tells you how much signal comes from "prior diagnosis";
the B → C drop tells you how much extra the threshold encoding adds beyond
the raw measurements. This 3-way contrast directly answers the original
research question and gives a much stronger story than a simple
full-vs-screening comparison.

### 4. Multicollinearity
The correlation heatmap shows several near-duplicate pairs:
- `bmi` ↔ `waist_cm`
- `sbp_avg` ↔ `pulse_pressure`
- `fasting_glucose` ↔ `hba1c`
- `total_chol` ↔ `ldl`
- `lifestyle_risk_score` ↔ `lifestyle_risk_score_no_dx` (r ≈ 0.84)

The last pair is by design — `_no_dx` is the diagnosis-free version of the
full score, intentionally provided so the screening-only model has a
substitute. As noted above: in Model A drop `lifestyle_risk_score_no_dx`,
in Models B/C drop `lifestyle_risk_score` and keep the `_no_dx` version.

**Logistic regression:** use L2 (Ridge) regularization or ElasticNet.
**Tree models (RF, XGBoost):** ignore — they're robust to collinearity.

### 5. Reverse causation on cholesterol
`total_chol` and `ldl` are LOWER in CVD-positive participants in this dataset
(Cohen's d = -0.56, -0.68). This is the statin effect: diagnosed CVD patients
are typically on lipid-lowering therapy, which suppresses measured cholesterol
below their pre-treatment baseline. Negative coefficients on these features
are not a bug — but mention it explicitly in your model interpretation.

### 6. Optional: K-Means cluster as a feature
The cluster column in `analytic_dataset_with_clusters.csv` was fit on the
full sample. **Don't use it directly** — that's leakage. If you want
`cluster` as a supervised feature, refit KMeans inside each CV fold:

```python
from joblib import load
pipe = load("outputs/clusters/kmeans_pipeline.joblib")
# pipe contains: scaler, kmeans, feature_order, imputation_medians, min_observed_frac
# Use these as a template — refit on each train fold.
```

In practice, with only 4 clusters and overlapping phenotypes (silhouette
0.188), this is unlikely to add much over the existing features. Try it
last, after baseline models are working.

## Recommended modelling sequence

1. **Baseline:** `LogisticRegression(class_weight='balanced', penalty='l2')`
   on the full feature set. Report ROC-AUC, PR-AUC.
2. **Random Forest:** `RandomForestClassifier(class_weight='balanced')` with
   modest tuning (n_estimators, max_depth).
3. **XGBoost:** `XGBClassifier(scale_pos_weight=6.94)` with cross-validated
   tuning of `learning_rate`, `max_depth`, `n_estimators`.
4. **Sensitivity:** repeat 1–3 on the screening-only feature set.
5. **Calibration:** for the chosen final model, plot a calibration curve and
   apply `CalibratedClassifierCV` if predictions are miscalibrated.
6. **Threshold:** in the report, pick a threshold based on a clinically
   defensible criterion (e.g., 80% sensitivity for screening) rather than
   the default 0.5.

## Reproducibility

- All random seeds in this stage = 42.
- All train-fit statistics are in `train_fit_stats.json`.
- Verify with: `python3 src/verify_part2.py` — should print
  `ALL CHECKS PASSED`.

## Questions on Part 2 work — see

- `docs/feature_engineering_report.md` — full methodology
- `outputs/eda/` — statistical tests, effect sizes, audit
- `outputs/clusters/` — K-Means + PCA + robustness
- `outputs/features/` — pipeline outputs + audits

If anything below isn't clear or breaks on your end, ping me before assuming
the data is wrong. Most "weird" patterns in this dataset are real (e.g.,
the cholesterol reversal) and worth flagging in your modelling report.
