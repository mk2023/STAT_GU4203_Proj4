# Part 2 — Unsupervised EDA, Feature Engineering & Preprocessing

This document covers the second stage of the CVD prediction pipeline: it
takes the cleaned analytic dataset from Part 1 and produces (1) an
unsupervised view of the patient population, (2) formal statistical
testing of group differences, (3) clinically grounded engineered features,
and (4) a leakage-safe preprocessing pipeline that delivers a model-ready
training matrix to the modelling team in Part 3.

## 1. Inputs and outputs

**Input**
- `data/processed/analytic_dataset.csv` (11,933 rows × 29 columns; from Part 1)

**New datasets**
- `data/processed/analytic_dataset_with_clusters.csv` — original data with
  cluster labels appended (EDA use only — see §3.5)
- `data/processed/analytic_dataset_engineered_train.csv`,
  `analytic_dataset_engineered_test.csv` — winsorized + engineered features,
  split-aware

**Output artifacts** (everything written to `outputs/`)
- `outputs/clusters/` — K-Means selection, profiles, PCA, robustness, fitted pipeline
- `outputs/features/` — model matrices, fitted preprocessor, audits, exclusion list
- `outputs/eda/` — statistical tests, effect-size plots, categorical audit

**Handoff document**
- `docs/handoff_to_part3.md` — quick-start guide for the modelling team

## 2. Advanced EDA — statistical hypothesis testing

Beyond the descriptive statistics in Part 1, we ran formal tests of the
hypothesis that each feature differs between CVD-positive and CVD-negative
participants.

### 2.1 Categorical level audit (preprocessing for chi-square)

Before running chi-square tests we audited the levels of each candidate
categorical NHANES variable (`outputs/eda/categorical_level_audit.csv`).
The audit revealed three variables that look categorical by name but
actually contain duration/frequency-style numeric responses rather than
a small set of clean nominal categories:

| Variable | Source NHANES code | n_unique | Reason it's not categorical |
|---|---|---|---|
| `moderate_ltpa_minutes` | PAD800 | 55 | minutes of moderate leisure-time physical activity (LTPA) |
| `vigorous_ltpa_minutes` | PAD820 | 38 | minutes of vigorous leisure-time physical activity (LTPA) |
| `drinking_frequency_code` | ALQ121 | 11 | drinking frequency code (0=never, 1=every day, 10=1-2/year) |

Running chi-square on a 50-level "category" inflates degrees of freedom and
yields uninterpretable results, so **we excluded these from chi-square
testing and routed them into the numeric/continuous testing family
(activity variables as numeric minutes, ALQ121 as an ordinal frequency
code), where Welch's t-test is the appropriate tool**.

### 2.2 Tests and corrections

- **Numeric features (19 total):** Welch's t-test (no equal-variance
  assumption), with Cohen's *d* as the effect size.
- **Categorical features (6 cleaned):** Chi-square test of independence,
  with Cramér's *V* as the effect size and minimum expected cell count
  reported as an assumption check (all ≥ 34, well above the conventional 5).
- **Multiple-testing correction:** Benjamini–Hochberg FDR at *q* = 0.05
  within each family.

### 2.3 Headline findings

- **16 of 19 numeric features** are significant after FDR correction;
  all 6 cleaned categorical features are significant, although effect
  sizes range from small (Cramér's V = 0.06 for race/ethnicity) to
  moderate (V = 0.27 for self-reported hypertension).
- The single largest effect is **age** (Cohen's *d* = 0.95, *p* < 10⁻²²⁰),
  classified as a "large" effect by Cohen's convention.
- **Reverse-causation signal:** total cholesterol and LDL are *lower* in
  CVD-positive participants (*d* = −0.56, −0.68). This is consistent with
  clinical reality — patients with diagnosed CVD are typically on statin
  therapy, which suppresses measured cholesterol below their pre-treatment
  baseline. The modelling team should interpret negative cholesterol
  coefficients accordingly (see §6).
- The previously over-significant activity variables drop out after re-test:
  neither vigorous nor moderate activity differs between groups in this
  sample (p > 0.2).
- Strongest categorical predictors (by Cramér's V): self-reported
  hypertension (V = 0.27), diabetes (V = 0.23), lifetime smoking (V = 0.15).

> **Statistical caveat:** NHANES uses a complex survey design with
> sampling weights, strata, and PSUs. We did not incorporate survey
> weights, so these results describe the analytic sample rather than
> population-level prevalences.

See `effect_sizes_numeric.png` for the full ranking and
`top_feature_distributions.png` for the empirical distributions of the
six largest effects.

## 3. Unsupervised EDA — K-Means risk phenotypes

### 3.1 Setup

- **Clustering features (11):** age + body composition + BP + lipids + glucose.
- Engineered ratios (`chol_hdl_ratio`, `met_syn_score`) were excluded from
  the clustering input to avoid double-counting their components.
- **Sample restriction:** rows with at least 60% of clustering features
  observed (6,948 of 11,933 participants).
- **Preprocessing:** median imputation followed by `StandardScaler` so
  Euclidean distance is meaningful across heterogeneous units.

### 3.2 Choosing k

We swept k from 2 to 8 and inspected both inertia (elbow) and silhouette
score. Silhouette peaked at k = 4 (0.188); the inertia curve bends at the
same point.

> **Tone caveat:** A silhouette of 0.188 is modest in absolute terms.
> It indicates *overlapping but clinically interpretable* phenotypes
> rather than sharply separated disease subtypes. We treat the partition
> as a useful descriptive layer, not a definitive taxonomy.

### 3.3 Robustness of the partition

A single fit can be misleading if the result depends on initialization or
algorithm choice. Two complementary checks:

| Check | Method | Result | Interpretation |
|---|---|---|---|
| Bootstrap stability | 50 resamples → re-fit K-Means → ARI vs original | **0.948** | The centroid-based solution is highly stable under bootstrap resampling |
| Cross-algorithm | Agglomerative (Ward) on a 3,000-row subsample | 0.542 | Moderate agreement — expected since Ward and K-Means optimize different objectives |

### 3.4 Resulting phenotypes

| Cluster | n | Age | BMI | SBP | HbA1c | Triglycerides | CVD rate | Phenotype |
|---|---|---|---|---|---|---|---|---|
| 0 | 2,679 | 30 | 23 | 108 | 5.3 | 81 | 5% | younger / low-risk |
| 1 | 2,310 | 50 | 35 | 117 | 5.8 | 142 | 13% | younger / obese |
| 2 |   201 | 59 | 33 | 127 | 9.6 | 227 | 26% | older / obese / dysglycemic / dyslipidemic |
| 3 | 1,758 | 66 | 28 | 140 | 5.8 | 110 | 16% | older / hypertensive |

Key observations:

- CVD prevalence climbs monotonically with phenotype severity, from 5% in
  the young/healthy cluster to 26% in the metabolically severe cluster —
  even though CVD labels were never used to form the clusters.
- **Cluster 2 caveat:** with only 201 participants and HbA1c near 10%,
  this cluster is small and likely driven by extreme glycemic outliers.
  We interpret it as a high-risk *exploratory* phenotype rather than a
  definitive population subgroup.
- Cluster 3 captures isolated systolic hypertension with elevated pulse
  pressure (62 mmHg), consistent with arterial stiffness in older adults.

### 3.5 Note on using `cluster` as a supervised feature

The `cluster` column in `analytic_dataset_with_clusters.csv` was fit on the
**full sample**, which is appropriate for EDA but would constitute test-set
leakage if used as a supervised model feature. We saved the fitted pipeline
(scaler, KMeans, imputation medians, feature ordering) to
`outputs/clusters/kmeans_pipeline.joblib`. If the modelling team chooses to
use cluster membership as a feature, they should re-fit KMeans within each
training fold using this pipeline as the template.

### 3.6 PCA — interpretable risk axes

PCA on the same standardized matrix revealed a low-dimensional structure
(5 components explain 80% of variance):

- **PC1 (28%) — overall metabolic load**: all variables load in the same direction.
- **PC2 (17%) — hypertensive vs central-obesity contrast**: SBP/HDL load
  positive, BMI/waist load negative.
- **PC3 (13%) — glycemic axis**: dominated by fasting glucose (0.54) and
  HbA1c (0.45).
- **PC4 (11%) — cholesterol axis**: total cholesterol loads at 0.70.

Loadings are saved to `pca_loadings.csv` and visualized in `pca_loadings.png`.

## 4. Feature engineering

### 4.1 Outlier handling — IQR winsorization (train-only fit)

Several lab values had extreme right tails (max triglycerides = 1745 mg/dL,
max fasting glucose = 561 mg/dL). We applied IQR-based winsorization with
k = 3 (more conservative than the standard k = 1.5), capping values that
fall outside Q1 − 3·IQR or Q3 + 3·IQR.

> **Leakage prevention:** IQR thresholds are computed on the **training
> set only** and then applied identically to the test set. This applies
> to every train-dependent statistic in this stage.

The full audit (per-feature bounds + train-side clip counts) is in
`outputs/features/winsorization_log.csv`.

### 4.2 Categorical bins from clinical guidelines

Continuous clinical measurements were translated into the categories used by
practising physicians:

- `age_group` — 5 bins: <30, 30–44, 45–59, 60–74, 75+
- `bmi_category` — WHO categories (under / normal / over / obese)
- `bp_stage` — ACC/AHA 2017 staging (normal / elevated / Stage 1 / Stage 2)
- `hba1c_status` — ADA cut-offs (normal / prediabetes / diabetes)
- `is_current_smoker`, `is_former_smoker` — derived from smoking history
- `diabetes_borderline_flag` — separates the NHANES "borderline" diabetes
  code (DIQ010 == 3) from the "Yes" code (== 1) so the model can use them
  independently
- Human-readable labels for `sex`, `race_ethnicity`, `education`

These give downstream models a guideline-aligned representation that's both
interpretable and clinically defensible.

### 4.3 Lifestyle composite — `lifestyle_risk_score` (0-4)

Four binary risk components are summed:

1. **Current smoker** (from smoking history flag)
2. **Low activity** — both vigorous and moderate self-reported activity
   measures equal zero. Importantly, this is **only flagged when at least
   one activity value is observed**; if both are missing, the component is
   left as NaN rather than silently coded as low activity. NHANES activity
   variables have heavy skip-pattern missingness.
3. **Frequent drinking** — ALQ121 drinking-frequency code in {1, 2, 3, 4, 5},
   corresponding to weekly-or-more drinking. Note that NHANES ALQ121 is
   coded with **lower codes meaning higher frequency** (1 = every day,
   10 = 1–2 times in the last year, 0 = never in the last year), so a
   simple ≥-threshold is the wrong direction.
4. **Diagnosed diabetes** — DIQ010 == 1 only (the borderline code, == 3,
   is exposed separately as `diabetes_borderline_flag` — folding it into a
   binary score with arbitrary integer weight 3 would have been a coding
   bug rather than a clinical signal)

The score uses `min_count=2` (a row with more than two missing components
becomes NaN). The empirical distribution audit is saved to
`outputs/features/lifestyle_risk_score_distribution.csv`; in our training
sample it ranges 0–3 (no participants triggered all four risk factors),
with ~55% scoring 0, ~38% scoring 1, ~6% scoring 2, ~0.4% scoring 3.

> **Diagnosis-free variant for sensitivity analysis.** Because component 4
> uses prior diabetes diagnosis, a model that uses `lifestyle_risk_score`
> in the "screening-only" sensitivity analysis (see §6) would still leak
> diagnosis-history signal. We therefore additionally expose
> `lifestyle_risk_score_no_dx` (0–3 scale, components 1–3 only) and list
> the diagnosis-including variant in
> `outputs/features/sensitivity_feature_exclusions.json` as a column to
> drop, with `lifestyle_risk_score_no_dx` as its substitute.

### 4.4 Non-linear transforms

Right-skewed labs (triglycerides, fasting glucose, alcohol exposure) are
additionally log-transformed via `log1p`. Both raw and log versions are
kept; regularization in the linear models will pick the more useful
representation. Side-by-side distributions are in `transform_before_after.png`.

### 4.5 Interactions and composite index

We added **three multiplicative interactions plus one composite index**,
all clinically motivated:

- `age_x_sbp` — hypertension is far more dangerous in older adults
- `bmi_x_glucose` — central adiposity amplifies dysglycemia risk
- `age_x_hba1c` — duration-of-diabetes effect proxy
- `metabolic_burden_z` — z-score average of age, SBP, HbA1c, and chol/HDL
  ratio, serving as a Framingham-style summary index. Z-score parameters
  are fit on the training set only, then applied to the test set.

## 5. Preprocessing pipeline

### 5.1 Leakage-safe split-then-fit ordering

The whole stage is structured to avoid any test-set information bleeding
into train-time decisions:

1. Drop rows missing `cvd` (n_modelling = 7,807; positive rate 12.6%).
2. **Stratified 80/20 train/test split first** (`random_state=42`).
3. **Fit** winsorization bounds on train → apply to both.
4. Apply deterministic engineering (clinical bins, log1p, interactions,
   lifestyle score) on both — these don't depend on training-set statistics.
5. **Fit** z-score statistics for `metabolic_burden_z` on train → apply to both.
6. **Fit** the sklearn `ColumnTransformer` on train → transform both.

Train-fit statistics are persisted to `outputs/features/train_fit_stats.json`
for full reproducibility.

### 5.2 Pipeline composition

- **Numeric features (29):** median imputation **with missingness indicators**,
  then `StandardScaler`.
- **Categorical features (7):** most-frequent imputation, then one-hot encoding.

> **Why missingness indicators?** NHANES missingness is rarely random.
> Many labs are only measured on a fasting subsample; questionnaire fields
> use skip patterns (e.g., "smoke now?" is only asked of lifetime smokers).
> Treating missingness as informative — rather than silently filling with
> the median — preserves real signal that the modelling team can choose
> to use or ignore.

### 5.3 Final shapes

- Train shape: (6,245, **81**) — 29 numeric columns + 24 missingness
  indicators + 28 from one-hot encoding the 7 categorical features.
- Test shape: (1,562, 81).

The fitted `ColumnTransformer` is persisted as
`outputs/features/preprocessor.joblib` so the modelling team can reuse it
without re-fitting (essential for preventing test-set leakage downstream).

## 6. Notes for the modelling stage

The full handoff guide is in `docs/handoff_to_part3.md`. The five most
important points:

- **Class imbalance** (≈1:7) — use `class_weight='balanced'`, SMOTE, or
  `scale_pos_weight=6.94` (XGBoost). Avoid raw accuracy; ROC-AUC and PR-AUC
  are the correct primary metrics.

- **Multicollinearity** — the correlation heatmap and PCA show several
  near-duplicate pairs (`bmi`/`waist_cm`, `sbp_avg`/`pulse_pressure`,
  `fasting_glucose`/`hba1c`, `total_chol`/`ldl`). For logistic regression,
  L2 regularization is essential. Tree-based models are robust to this.

- **Reverse causation** — interpret negative coefficients on
  `total_chol`/`ldl` with care; many CVD-positive participants are on
  lipid-lowering therapy.

- **Sensitivity analysis — strongly recommended.** The exclusion list is
  split into two categories: (1) self-reported diagnosis features
  (`lifestyle_risk_score`, `diabetes_borderline_flag`) and (2) guideline-
  threshold features derived from routine measurements (`bp_stage_*`,
  `hba1c_status_*`). A 3-way comparison (full / drop diagnosis only / drop
  diagnosis + thresholds) gives clean attribution: the first gap measures
  prior-diagnosis leakage, the second measures whether the binning encoding
  helps beyond raw measurements. See `docs/handoff_to_part3.md` for the
  full protocol.

- **Cluster as a feature** — see §3.5. If used, re-fit KMeans within each
  training fold using the saved pipeline template.

## 7. Reproducibility

- Run order: `cluster_eda.py` → `feature_engineering.py` → `advanced_eda.py`
- All random seeds = 42.
- All train-fit statistics persisted (`train_fit_stats.json`).
- Verification script: `python3 src/verify_part2.py` — should print
  `ALL CHECKS PASSED` on a fresh run.
