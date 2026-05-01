# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

End-to-end ML project for Columbia STAT GU4203 (Project 4): predict cardiovascular disease (CVD) status from CDC NHANES 2021–2023 survey data. The pipeline is split into four sequential parts owned by different teammates; Parts 1–2 are complete, Part 3 (supervised modelling) is the active work, Part 4 (final report + presentation) is pending. Final deliverable due **2026-05-05 23:59**.

## Common commands

All commands run from the repo root with Python 3.9+. Random seeds are pinned to `42` everywhere — runs are deterministic.

```bash
# One-time install
pip install "numpy<2" pandas scikit-learn scipy statsmodels matplotlib seaborn joblib pyreadstat
pip install xgboost catboost shap   # Part 3 only

# Full reproduction (Part 1 → Part 2 → Part 3)
python3 src/preprocess_eda.py          # Part 1: builds data/processed/analytic_dataset.csv from rawfiles/*.xpt
python3 src/cluster_eda.py             # Part 2: K-Means risk phenotypes
python3 src/feature_engineering.py     # Part 2: engineering + leakage-safe preprocessing pipeline
python3 src/advanced_eda.py            # Part 2: Welch's t-tests + chi-square with FDR correction
python3 src/supervised_models.py       # Part 3: 4 models x 3 feature sets, calibration, threshold, SHAP (~12 min)

# Verify Parts 1–2 outputs (32 automated checks)
python3 src/verify_part2.py
# Expected final line: "ALL CHECKS PASSED — outputs match expectations."

# Smoke-test the dashboard inference wrapper
python3 src/predict_api.py
```

There is no test framework or linter configured — `verify_part2.py` is the project's correctness gate for Parts 1–2; `predict_api.py`'s `__main__` block doubles as a smoke test for Part 3 artifacts. Run them after any change that touches the corresponding parts.

**numpy 2.x note:** scipy/sklearn shipped with anaconda is compiled against numpy 1.x. Installing numpy 2.x in the user site-packages causes `ImportError: numpy.core.multiarray failed to import`. Pin `numpy<2` until anaconda's binaries are rebuilt.

**XGBoost compatibility:** scikit-learn ≥ 1.7 enforces strict estimator-type checks that XGBoost 2.x does not pass (raises `XGBClassifier should either be a classifier ... Got a regressor`). Use **xgboost ≥ 3.0**.

## Architecture: the leakage contract is the architecture

The pipeline has one non-obvious invariant that touches every file: **the train/test split happens before any train-dependent statistic is fit.** Any change that violates this leaks test info into train and silently inflates downstream model metrics.

Concretely, the order is:

1. `preprocess_eda.py` produces `data/processed/analytic_dataset.csv` (29 cols, 11,933 rows; 7,807 with known CVD).
2. `feature_engineering.py` does the stratified 80/20 split first (`random_state=42`, `test_size=0.20`, stratified on `cvd`), **then** fits in this order:
   - IQR winsorization bounds (k=3) — train only → applied to both
   - Deterministic engineering (clinical bins, log1p, interactions, lifestyle score) — no fit needed
   - Z-score parameters for `metabolic_burden_z` — train only → applied to both
   - sklearn `ColumnTransformer` (median impute + missingness indicators + scale + one-hot) — train only → applied to both
3. All train-fit statistics are persisted to `outputs/features/train_fit_stats.json` for reproducibility, and the fitted transformer to `outputs/features/preprocessor.joblib`. **Never refit either on combined train+test data.**

Final model matrices: `outputs/features/X_train.csv` (6245 × 81), `X_test.csv` (1562 × 81), `y_train.csv`/`y_test.csv` (CVD positive rate ≈ 12.6%, identical in train and test).

`outputs/clusters/kmeans_pipeline.joblib` was fit on the **full** sample (Parts 1–2 used clusters for descriptive EDA only). Using the cluster column as a supervised feature directly is leakage; if Part 3 wants cluster membership as a feature, KMeans must be re-fit inside each CV fold using this pipeline as a template.

## Part 3 modelling (implemented)

Part 3 lives in `src/supervised_models.py` (training pipeline) and `src/predict_api.py` (dashboard inference wrapper). Artifacts land in `outputs/models/`; the winning model card (`final_model_card.json`) is the source of truth for which feature set, threshold, and calibration the deployed model uses.

The handoff document `docs/handoff_to_part3.md` is the authoritative spec. Highlights that affect coding decisions:

- **Class imbalance (≈ 1:7).** Don't use raw accuracy — predicting all-zero gets 87%. Primary metrics are **ROC-AUC** and **PR-AUC**. Use `class_weight='balanced'` for sklearn estimators or `scale_pos_weight=6.94` for XGBoost. SMOTE is acceptable but **only on the training fold**.
- **Three-way sensitivity analysis (required).** Build each model on three feature sets, controlled by `outputs/features/sensitivity_feature_exclusions.json`:
  - **Model A (full):** all 81 columns **except** `lifestyle_risk_score_no_dx` (kept only as a substitute when the full score is dropped). Effective: 80 columns.
  - **Model B (no diagnosis history):** drop the columns in `diagnosis_history_columns`, **and** swap `lifestyle_risk_score` → `lifestyle_risk_score_no_dx`. Keeps guideline threshold bins.
  - **Model C (raw measurements only):** drop everything in `columns_to_drop`, **and** swap `lifestyle_risk_score` → `lifestyle_risk_score_no_dx`.
  The A→B AUC drop measures prior-diagnosis signal; B→C measures whether threshold encodings add anything beyond raw labs. This is the project's research story.
- **Multicollinearity** is by design (e.g. `lifestyle_risk_score` ↔ `lifestyle_risk_score_no_dx`, r ≈ 0.84). Use L2 / ElasticNet for linear models; trees ignore it.
- **Cholesterol reversal is not a bug.** `total_chol` and `ldl` are *lower* in CVD-positive participants (Cohen's d = −0.56, −0.68) because diagnosed patients are on statins. Negative coefficients on these features are real and must be flagged in the model writeup, not "fixed".
- **Calibration + threshold.** For the chosen final model, plot a calibration curve, apply `CalibratedClassifierCV` if needed, and pick a decision threshold from a clinical criterion (e.g. 80% sensitivity for screening) — not the default 0.5.
- **Threshold must match the calibration**: select the operating threshold on out-of-fold probabilities from the *same* `final_estimator` you'll apply on test (calibrated or raw, whichever wins on Brier). Selecting on raw OOF probs and applying on Platt-calibrated test probs silently changes the distribution and collapses recall — there is a regression test for this in `predict_api.py`.

## Path conventions

All scripts assume the **repo root** is the current working directory. They write to `outputs/` and `data/processed/` using relative paths. Don't `cd` into `src/` before running.

`docs/preparation_report.md` contains a stale absolute path (`/Users/minseulkim/...`) from Part 1's author machine; it's documentation only and doesn't affect execution.

## Team contributions (per README)

- Part 1 (Data Acquisition & Preparation): Teammate 1
- Part 2 (Unsupervised EDA + Feature Engineering): Qiujun Zhang
- Part 3 (Supervised Modelling): Junye (active)
- Part 4 (Final Report & Communication): Teammate 4
