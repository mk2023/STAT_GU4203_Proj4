# Predicting Cardiovascular Disease Risk from NHANES

End-to-end machine learning project predicting cardiovascular disease (CVD)
status from the U.S. CDC's NHANES 2021-2023 survey, using a combination of
unsupervised exploration, clinically grounded feature engineering, and
supervised classification. Includes an interactive Streamlit dashboard
(`streamlit run src/dashboard.py`) that walks through the entire pipeline
and exposes the calibrated model for live prediction.

## Project structure

```
STAT_GU4203_Proj4-main/
├── data/
│   └── processed/              # Cleaned + engineered datasets
├── docs/                       # Reports
│   ├── preparation_report.md           (Part 1)
│   ├── feature_engineering_report.md   (Part 2)
│   └── handoff_to_part3.md             (Part 2 -> Part 3 guide)
├── outputs/
│   ├── eda/                    # Descriptive + statistical EDA results
│   ├── clusters/               # K-Means risk phenotypes
│   ├── features/               # Model-ready train/test matrices + pipeline
│   └── models/                 # Part 3 supervised models + diagnostics
├── rawfiles/                   # Original NHANES .xpt files (Part 1 input)
├── src/                        # Python scripts
│   ├── preprocess_eda.py             (Part 1)
│   ├── cluster_eda.py                (Part 2)
│   ├── feature_engineering.py        (Part 2)
│   ├── advanced_eda.py               (Part 2)
│   ├── verify_part2.py               (automated verification)
│   ├── supervised_models.py          (Part 3: 4 models x 3 feature sets)
│   ├── predict_api.py                (Part 3: inference wrapper for dashboards)
│   ├── dashboard.py                  (Part 4: Streamlit dashboard entry point)
│   ├── dashboard_helpers.py          (Part 4: Plotly factories + raw-input pipeline)
│   └── dashboard_styles.py           (Part 4: CSS + hero / KPI / callout helpers)
└── README.md
```

## How to reproduce

### Requirements

- Python 3.9+
- Packages: `pandas`, `numpy<2`, `scikit-learn`, `scipy`, `statsmodels`,
  `matplotlib`, `seaborn`, `joblib`, `pyreadstat`, and for Part 3:
  `xgboost>=3`, `catboost`, `shap`. Part 4 dashboard adds: `streamlit`,
  `streamlit-option-menu`, `plotly`.

Install:
```bash
pip install "numpy<2" pandas scikit-learn scipy statsmodels matplotlib seaborn joblib pyreadstat
pip install xgboost catboost shap                       # Part 3
pip install streamlit streamlit-option-menu plotly      # Part 4 dashboard
```

The `numpy<2` pin avoids ABI mismatches with anaconda-shipped scipy on
Windows (numpy 2.x triggers `ImportError: numpy.core.multiarray failed
to import` on scipy/sklearn compiled against numpy 1.x).

### Run from scratch

From the project root:

```bash
# Part 1: build analytic_dataset.csv from raw NHANES .xpt files
python3 src/preprocess_eda.py

# Part 2: unsupervised EDA + feature engineering + preprocessing
python3 src/cluster_eda.py            # K-Means risk phenotypes
python3 src/feature_engineering.py    # Engineering + leakage-safe pipeline
python3 src/advanced_eda.py           # Statistical hypothesis testing

# Verification (32 automated checks for Parts 1-2)
python3 src/verify_part2.py

# Part 3: 4 models x 3 feature sets, calibration, threshold, SHAP (~12 min on 16-thread CPU)
python3 src/supervised_models.py
```

Expected final output of `verify_part2.py`:
`ALL CHECKS PASSED — outputs match expectations.`

`supervised_models.py` writes nine artifacts to `outputs/models/`:

| File | Contents |
|---|---|
| `cv_results.csv` | 12 rows (4 models x 3 feature sets) of mean ± std CV metrics |
| `final_model.joblib` | Winning model retrained on full train, with calibration |
| `final_model_card.json` | Model name, feature set, hyperparameters, threshold, test metrics |
| `test_metrics.json` | Test ROC-AUC, PR-AUC, Brier (raw / Platt / isotonic), F1 |
| `roc_pr_curves.png` | All 12 models on test set; winner thickened |
| `calibration.png` | Reliability diagram (raw vs Platt vs isotonic) |
| `threshold_analysis.csv` | Out-of-fold precision / recall / F1 / specificity by threshold |
| `shap_summary.png`, `shap_bar.png` | Global feature attribution on test |

All scripts use `random_state=42`, so results are deterministic.

## Run the dashboard (Part 4)

The dashboard reads only what `outputs/` already contains — it never refits
the model — so as long as Parts 1–3 have been run (or the artefacts are
checked in), launching the dashboard takes one command from the project root:

```bash
pip install streamlit streamlit-option-menu plotly      # one-time
streamlit run src/dashboard.py
```

Default URL: <http://localhost:8501>. Five pages, navigated from the sidebar:

| Page | What's there |
|---|---|
| **Overview** | Project framing, headline metrics, class-balance donut. |
| **EDA & Clustering** | Cohen's d effect-size bar chart, correlation heatmap, K-Means PCA scatter, cluster CVD-rate bar, cluster profile heatmap. |
| **Model Comparison** | 4 KPI cards for the best of each metric, switchable grouped bar chart of all 12 CV runs (4 algorithms × 3 feature sets), full sortable table. |
| **Final Model** | Test ROC + PR + calibration curves (Plotly), top-15 LogReg coefficients, SHAP beeswarm, and a threshold slider that recomputes precision/recall/F1/specificity from the saved sweep. |
| **Predict** | Three input paths — pick a row from the test set, upload a CSV with the 80 preprocessed columns, or fill a manual form (16 raw fields, training-set median/mode for the rest). Output: gauge + binary screening flag + per-row LogReg local SHAP + What-if slider for age / SBP / BMI / HbA1c / total cholesterol. |

The Predict page goes through `src/predict_api.py`, so it always uses the
calibrated probability and never refits anything. The manual-form path
reuses `engineer_deterministic` and `add_metabolic_burden` from
`src/feature_engineering.py` so derived features are recomputed using the
same train-fit statistics as the deployed model.

## Key results

- **Sample:** 11,933 NHANES participants; 7,807 with known CVD status
- **CVD prevalence:** 12.6% (imbalanced)
- **K-Means risk phenotypes (k=4):**
  - Cluster 0 (n=2,679): younger / low-risk — 5% CVD
  - Cluster 1 (n=2,310): younger / obese — 13% CVD
  - Cluster 2 (n=201): older / obese / dysglycemic / dyslipidemic — 26% CVD
  - Cluster 3 (n=1,758): older / hypertensive — 16% CVD
  - Bootstrap stability ARI = 0.948
- **Strongest predictor:** age (Cohen's d = 0.95, "large" effect)
- **Modeling matrix:** train (6,245 × 81), test (1,562 × 81) — stratified
  80/20 split
- **Part 3 winner:** L2-regularized logistic regression on the full feature
  set (80 cols, dropping the diagnosis-free lifestyle score) — CV ROC-AUC
  0.824 ± 0.010, test ROC-AUC 0.813, test PR-AUC 0.391
- **Calibration:** Platt scaling reduces Brier from 0.175 → 0.092
- **Operating point:** decision threshold 0.10 yields test recall 85.2% and
  precision 25.9% (clinical screening criterion: recall ≥ 80%)
- **Sensitivity analysis:** ROC-AUC differs by ≤0.003 across the full /
  no-diagnosis / raw-only feature subsets — predictive signal lives in
  raw lab and vital measurements, not in prior-diagnosis labels

## Data

Raw NHANES files are from the CDC public repository
(<https://wwwn.cdc.gov/nchs/nhanes/>), August 2021 – August 2023 cycle.

## Team contributions

- **Part 1 (Data Acquisition & Preparation + basic EDA):** [Teammate 1]
- **Part 2 (Unsupervised EDA + Feature Engineering & Preprocessing):**
  Qiujun Zhang
- **Part 3 (Supervised Modeling):** Junye
- **Part 4 (Communication — interactive dashboard):** Junye
- **Part 4 (Final Report):** Justine Dugger-Ades

See `docs/feature_engineering_report.md` for full Part 2 methodology.
