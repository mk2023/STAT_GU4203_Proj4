# Predicting Cardiovascular Disease Risk from NHANES

End-to-end machine learning project predicting cardiovascular disease (CVD)
status from the U.S. CDC's NHANES 2021-2023 survey, using a combination of
unsupervised exploration, clinically grounded feature engineering, and
supervised classification.

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
│   └── features/               # Model-ready train/test matrices + pipeline
├── rawfiles/                   # Original NHANES .xpt files (Part 1 input)
├── src/                        # Python scripts
│   ├── preprocess_eda.py             (Part 1)
│   ├── cluster_eda.py                (Part 2)
│   ├── feature_engineering.py        (Part 2)
│   ├── advanced_eda.py               (Part 2)
│   └── verify_part2.py               (automated verification)
└── README.md
```

## How to reproduce

### Requirements

- Python 3.9+
- Packages: `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`,
  `matplotlib`, `seaborn`, `joblib`, `pyreadstat`

Install:
```bash
pip install pandas numpy scikit-learn scipy statsmodels matplotlib seaborn joblib pyreadstat
```

### Run from scratch

From the project root:

```bash
# Part 1: build analytic_dataset.csv from raw NHANES .xpt files
python3 src/preprocess_eda.py

# Part 2: unsupervised EDA + feature engineering + preprocessing
python3 src/cluster_eda.py            # K-Means risk phenotypes
python3 src/feature_engineering.py    # Engineering + leakage-safe pipeline
python3 src/advanced_eda.py           # Statistical hypothesis testing

# Verification (32 automated checks)
python3 src/verify_part2.py
```

Expected final output: `ALL CHECKS PASSED — outputs match expectations.`

All scripts use `random_state=42`, so results are deterministic.

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

## Data

Raw NHANES files are from the CDC public repository
(<https://wwwn.cdc.gov/nchs/nhanes/>), August 2021 – August 2023 cycle.

## Team contributions

- **Part 1 (Data Acquisition & Preparation + basic EDA):** [Teammate 1]
- **Part 2 (Unsupervised EDA + Feature Engineering & Preprocessing):**
  Qiujun Zhang
- **Part 3 (Supervised Modeling):** [Teammate 3]
- **Part 4 (Final Report & Communication):** [Teammate 4]

See `docs/feature_engineering_report.md` for full Part 2 methodology.
