"""
Feature engineering + preprocessing for the CVD prediction task.

LEAKAGE-SAFE PIPELINE:
All train-dependent transformations (winsorization thresholds, z-score
composite statistics, imputation values, scaling parameters, one-hot
mappings) are estimated on the TRAINING set only and then applied to
the test set. This is the strict approach required to avoid biased
test-set evaluation downstream.

Pipeline:
1. Drop rows missing the CVD outcome.
2. Stratified 80/20 train/test split FIRST (before any preprocessing).
3. Fit winsorization bounds on train; apply to both.
4. Fit z-score statistics on train (for metabolic_burden_z); apply to both.
5. Engineer deterministic features (clinical bins, log transforms,
   interactions) on both — these don't depend on training-set statistics.
6. Build sklearn ColumnTransformer that:
     - median-imputes numeric features with missing indicators + standardizes,
     - mode-imputes categorical features and one-hot encodes them.
7. Fit on train, transform both, persist all artifacts.

Outputs:
- outputs/features/X_train.csv, X_test.csv, y_train.csv, y_test.csv
- outputs/features/feature_names.json (column order)
- outputs/features/preprocessor.joblib (fitted transformer for reuse)
- outputs/features/winsorization_log.csv (audit of bounds + clip counts)
- outputs/features/transform_before_after.png (diagnostic plot)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_DIR = ROOT / "outputs" / "features"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# IQR-based winsorization with k=3 (more conservative than the standard k=1.5).
# Caps only the most extreme outliers — typically lab errors or rare extreme
# pathologies that distort linear models without adding generalizable signal.
WINSORIZE_FEATURES = [
    "bmi", "waist_cm", "sbp_avg", "dbp_avg",
    "total_chol", "hdl", "ldl", "triglycerides",
    "fasting_glucose", "hba1c",
]
IQR_K = 3.0

# Right-skewed labs benefit from log1p so linear models see less leverage from
# the long tail. We keep both raw and log versions; regularization can pick.
LOG_TRANSFORM_FEATURES = ["triglycerides", "fasting_glucose", "avg_drinks_per_day"]

# Components of the Framingham-style metabolic burden composite. Each is
# z-scored using TRAINING-set mean/std and the four are averaged.
METABOLIC_BURDEN_COLS = ["age_years", "sbp_avg", "hba1c", "chol_hdl_ratio"]


def ensure_dirs() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "analytic_dataset_with_clusters.csv"
    if not path.exists():
        path = PROCESSED_DIR / "analytic_dataset.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Train-only-fit helpers (winsorization + z-score composite)
# ---------------------------------------------------------------------------

def fit_winsor_bounds(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compute IQR-based winsorization bounds from TRAINING data only."""
    bounds = {}
    rows = []
    for col in WINSORIZE_FEATURES:
        if col not in df.columns:
            continue
        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        bounds[col] = (lo, hi)
        rows.append({
            "feature": col,
            "lower": round(lo, 2),
            "upper": round(hi, 2),
            "clipped_low_train": int((s < lo).sum()),
            "clipped_high_train": int((s > hi).sum()),
        })
    return bounds, pd.DataFrame(rows)


def apply_winsor_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def fit_zscore_stats(df: pd.DataFrame, cols: list[str]) -> dict:
    return {
        col: (float(df[col].mean()), float(df[col].std()))
        for col in cols
        if col in df.columns
    }


def add_metabolic_burden(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    components = []
    for col, (mu, sd) in stats.items():
        if col in out.columns and sd != 0:
            components.append((out[col] - mu) / sd)
    if components:
        out["metabolic_burden_z"] = pd.concat(components, axis=1).mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Deterministic feature engineering (no training-set fitting required)
# ---------------------------------------------------------------------------

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [-np.inf, 30, 45, 60, 75, np.inf]
    labels = ["lt30", "30_44", "45_59", "60_74", "75plus"]
    out["age_group"] = pd.cut(out["age_years"], bins=bins, labels=labels, right=False)
    return out


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [-np.inf, 18.5, 25, 30, np.inf]
    labels = ["underweight", "normal", "overweight", "obese"]
    out["bmi_category"] = pd.cut(out["bmi"], bins=bins, labels=labels, right=False)
    return out


def add_bp_stage(df: pd.DataFrame) -> pd.DataFrame:
    # ACC/AHA 2017 staging.
    out = df.copy()
    sbp = out["sbp_avg"]
    dbp = out["dbp_avg"]
    stage = pd.Series(index=out.index, dtype="object")
    stage[(sbp < 120) & (dbp < 80)] = "normal"
    stage[(sbp.between(120, 129, inclusive="left")) & (dbp < 80)] = "elevated"
    stage[((sbp.between(130, 139, inclusive="both")) | (dbp.between(80, 89, inclusive="both")))] = "htn_stage1"
    stage[(sbp >= 140) | (dbp >= 90)] = "htn_stage2"
    stage[sbp.isna() | dbp.isna()] = np.nan
    out["bp_stage"] = stage
    return out


def add_hba1c_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [-np.inf, 5.7, 6.5, np.inf]
    labels = ["normal", "prediabetes", "diabetes"]
    out["hba1c_status"] = pd.cut(out["hba1c"], bins=bins, labels=labels, right=False)
    return out


def add_smoker_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    smoked_ever = out["smoked_100_life_code"].fillna(0).astype(int)
    smoke_now = out["smoke_now_code"]
    current_smoker = smoke_now.isin([1, 2]).astype(int)
    former_smoker = ((smoked_ever == 1) & (smoke_now == 3)).astype(int)
    out["is_current_smoker"] = current_smoker
    out["is_former_smoker"] = former_smoker
    return out


def add_lifestyle_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Composite 0-4 lifestyle risk score using corrected NHANES semantics.

    Per the CDC NHANES 2021-2023 codebook, the underlying columns encode:

    - `moderate_ltpa_minutes`         -> PAD800: minutes of moderate leisure-time physical activity
    - `vigorous_ltpa_minutes`         -> PAD820: minutes of vigorous leisure-time physical activity
    - `avg_drinks_per_day`            -> ALQ130: average drinks/day past 12 months
    - `drinking_frequency_code`       -> ALQ121: drinking frequency code
    - `diabetes_told_code`            -> DIQ010: 0=No, 1=Yes, 3=Borderline

    ALQ121 codebook (CDC NHANES 2021-2023, verified against ALQ_L.htm):
        0  = Never in the last year
        1  = Every day                (most frequent)
        2  = Nearly every day
        3  = 3 to 4 times a week
        4  = 2 times a week
        5  = Once a week
        6  = 2 to 3 times a month
        7  = Once a month
        8  = 7 to 11 times a year
        9  = 3 to 6 times a year
        10 = 1 to 2 times a year      (least frequent)
    Lower codes = higher frequency. We define "frequent drinker" as
    weekly-or-more, i.e. codes {1, 2, 3, 4, 5}.

    Components (each 0/1, summed to 0-4):
    1. current_smoker       — from add_smoker_flag
    2. low_activity         — both PAD800 and PAD820 are 0 minutes (when observed)
    3. frequent_drinking    — ALQ121 in {1,2,3,4,5} (weekly or more)
    4. diabetes_history     — DIQ010 == 1 (Yes; borderline kept SEPARATELY,
                              not folded into the 0-4 score because it has its
                              own clinical meaning)

    We also expose `lifestyle_risk_score_no_dx`, a 0-3 score that omits the
    diabetes-history component. The modelling team should use this version
    in their "screening-only" sensitivity model so prior diabetes diagnosis
    does not leak into a feature labeled as lifestyle.

    Missing-value policy:
    - Activity: if BOTH PAD800 and PAD820 are missing, low_activity is NaN.
    - Drinking: missing -> 0 (lifetime non-drinker assumption common in NHANES
      where ALQ121 is skipped for never-drinkers).
    - Diabetes: missing -> 0 (no diagnosis on record).
    - Final score uses min_count=2: a row with >2 missing components becomes NaN.
    """
    out = df.copy()

    # 1. current smoker
    smoker = out.get("is_current_smoker", pd.Series(0, index=out.index)).fillna(0).astype(int)

    # 2. low activity — only flag if at least one activity value is observed
    work_min = out["moderate_ltpa_minutes"]
    rec_min = out["vigorous_ltpa_minutes"]
    activity_observed = work_min.notna() | rec_min.notna()
    low_activity_obs = ((work_min.fillna(0) == 0) & (rec_min.fillna(0) == 0)).astype(float)
    low_activity = np.where(activity_observed, low_activity_obs, np.nan)

    # 3. frequent drinking — ALQ121 codes 1..5 = weekly-or-more
    # (1=every day, 2=nearly every day, 3=3-4/week, 4=2/week, 5=once/week).
    # Codes 6..10 are less frequent; code 0 is "never in the last year".
    freq_drink = out["drinking_frequency_code"].isin([1, 2, 3, 4, 5]).astype(int)

    # 4. diagnosed diabetes (Yes only; borderline excluded from this binary)
    diabetes_yes = (out["diabetes_told_code"] == 1).astype(int)

    components_df = pd.DataFrame({
        "smoker": smoker,
        "low_activity": low_activity,
        "freq_drink": freq_drink,
        "diabetes": diabetes_yes,
    })

    # Full score (0-4): includes diagnosed diabetes.
    # min_count=2: require at least 2 non-missing components to compute a score.
    out["lifestyle_risk_score"] = components_df.sum(axis=1, min_count=2)

    # Diagnosis-free score (0-3): for the "screening-only" sensitivity model
    # so that prior diabetes diagnosis does not leak through this feature.
    out["lifestyle_risk_score_no_dx"] = components_df[
        ["smoker", "low_activity", "freq_drink"]
    ].sum(axis=1, min_count=2)

    # Also expose a binary "borderline diabetes" flag separately so the model
    # can use it without conflating with the diagnosed-yes group.
    out["diabetes_borderline_flag"] = (out["diabetes_told_code"] == 3).astype(int)

    return out


def map_categoricals_to_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sex_map = {1: "male", 2: "female"}
    race_map = {
        1: "mexican_american", 2: "other_hispanic", 3: "non_hispanic_white",
        4: "non_hispanic_black", 6: "non_hispanic_asian", 7: "other_or_multi",
    }
    edu_map = {
        1: "lt_9th", 2: "9_to_11", 3: "high_school",
        4: "some_college", 5: "college_grad",
    }
    out["sex"] = out["sex_code"].map(sex_map)
    out["race_ethnicity"] = out["race_ethnicity_code"].map(race_map)
    out["education"] = out["education_code"].map(edu_map)
    return out


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in LOG_TRANSFORM_FEATURES:
        if col in out.columns:
            out[f"{col}_log"] = np.log1p(out[col].clip(lower=0))
    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clinically motivated multiplicative interactions. These capture risk
    that is greater than the sum of individual factors — e.g., hypertension
    is far more dangerous in older adults."""
    out = df.copy()
    out["age_x_sbp"] = out["age_years"] * out["sbp_avg"]
    out["bmi_x_glucose"] = out["bmi"] * out["fasting_glucose"]
    out["age_x_hba1c"] = out["age_years"] * out["hba1c"]
    return out


def engineer_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all train-independent feature engineering. Does NOT include
    metabolic_burden_z — that needs train-only z-score statistics."""
    out = df.copy()
    out = add_age_group(out)
    out = add_bmi_category(out)
    out = add_bp_stage(out)
    out = add_hba1c_status(out)
    out = add_smoker_flag(out)
    out = add_lifestyle_risk_score(out)
    out = map_categoricals_to_labels(out)
    out = add_log_transforms(out)
    out = add_interaction_features(out)
    return out


# ---------------------------------------------------------------------------
# Modelling matrix assembly
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age_years", "bmi", "waist_cm",
    "sbp_avg", "dbp_avg", "pulse_pressure", "pulse_avg",
    "total_chol", "hdl", "ldl", "triglycerides",
    "fasting_glucose", "hba1c",
    "chol_hdl_ratio", "met_syn_score",
    "income_poverty_ratio", "avg_drinks_per_day",
    "lifestyle_risk_score", "lifestyle_risk_score_no_dx",
    "is_current_smoker", "is_former_smoker",
    "diabetes_borderline_flag",
    # Engineered non-linear / interaction / composite features.
    "triglycerides_log", "fasting_glucose_log", "avg_drinks_per_day_log",
    "age_x_sbp", "bmi_x_glucose", "age_x_hba1c", "metabolic_burden_z",
]

CATEGORICAL_FEATURES = [
    "sex", "race_ethnicity", "education",
    "age_group", "bmi_category", "bp_stage", "hba1c_status",
]


def select_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """ColumnTransformer with missingness indicators on numeric features.

    NHANES missingness reflects survey skip patterns and subsample
    eligibility, not random noise — so we preserve missingness as signal
    via SimpleImputer(add_indicator=True) instead of silently filling it in.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def plot_distribution_before_after(
    df_before: pd.DataFrame, df_after: pd.DataFrame, feature_dir: Path
) -> None:
    sns.set_theme(style="whitegrid")
    cols = [c for c in ["triglycerides", "fasting_glucose"] if c in df_before.columns]
    fig, axes = plt.subplots(len(cols), 2, figsize=(11, 3.2 * len(cols)))
    if len(cols) == 1:
        axes = axes[None, :]

    for i, c in enumerate(cols):
        sns.histplot(df_before[c].dropna(), kde=True, ax=axes[i, 0], color="steelblue")
        axes[i, 0].set_title(f"{c} — raw (training set)")

        log_col = f"{c}_log"
        if log_col in df_after.columns:
            sns.histplot(df_after[log_col].dropna(), kde=True, ax=axes[i, 1], color="darkorange")
            axes[i, 1].set_title(f"{c} — winsorized + log1p (training set)")

    plt.tight_layout()
    plt.savefig(feature_dir / "transform_before_after.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_data()
    print(f"Loaded {len(df):,} rows")

    # Drop rows missing the outcome BEFORE anything else.
    df = df.dropna(subset=["cvd"]).copy()
    df["cvd"] = df["cvd"].astype(int)
    print(f"Modelling rows (cvd known): {len(df):,}; positive rate: {df['cvd'].mean():.3f}")

    # =============================================================
    # CRITICAL: split FIRST, then fit any train-dependent transforms.
    # =============================================================
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["cvd"],
    )
    print(f"Train rows: {len(df_train):,}; Test rows: {len(df_test):,}")

    # Snapshot raw training distributions for the diagnostic plot.
    df_train_pre_winsorize = df_train.copy()

    # ---- Step 1: fit winsorization bounds on TRAIN, apply to both ----
    bounds, winsorize_log = fit_winsor_bounds(df_train)
    df_train = apply_winsor_bounds(df_train, bounds)
    df_test = apply_winsor_bounds(df_test, bounds)
    winsorize_log.to_csv(FEATURE_DIR / "winsorization_log.csv", index=False)
    print("\nWinsorization bounds (fit on TRAIN only):")
    print(winsorize_log.to_string(index=False))

    # ---- Step 2: deterministic feature engineering on both ----
    df_train = engineer_deterministic(df_train)
    df_test = engineer_deterministic(df_test)

    # ---- Step 3: fit z-score stats on TRAIN, build metabolic_burden_z ----
    z_stats = fit_zscore_stats(df_train, METABOLIC_BURDEN_COLS)
    df_train = add_metabolic_burden(df_train, z_stats)
    df_test = add_metabolic_burden(df_test, z_stats)

    # ---- Step 4: assemble model matrices ----
    numeric_cols, categorical_cols = select_columns(df_train)
    print(f"\nNumeric features: {len(numeric_cols)}; categorical features: {len(categorical_cols)}")

    X_train_raw = df_train[numeric_cols + categorical_cols].copy()
    X_test_raw = df_test[numeric_cols + categorical_cols].copy()
    y_train = df_train["cvd"]
    y_test = df_test["cvd"]

    # ---- Step 5: fit ColumnTransformer on TRAIN, transform both ----
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    Xt_train = preprocessor.fit_transform(X_train_raw)
    Xt_test = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.get_feature_names_out().tolist()

    # ---- Step 6: persist ----
    pd.DataFrame(Xt_train, columns=feature_names).to_csv(
        FEATURE_DIR / "X_train.csv", index=False
    )
    pd.DataFrame(Xt_test, columns=feature_names).to_csv(
        FEATURE_DIR / "X_test.csv", index=False
    )
    y_train.to_csv(FEATURE_DIR / "y_train.csv", index=False)
    y_test.to_csv(FEATURE_DIR / "y_test.csv", index=False)

    with open(FEATURE_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    joblib.dump(preprocessor, FEATURE_DIR / "preprocessor.joblib")

    # Save the fitted train-stats so the modelling team can reproduce.
    with open(FEATURE_DIR / "train_fit_stats.json", "w") as f:
        json.dump(
            {
                "winsor_bounds": {k: list(v) for k, v in bounds.items()},
                "metabolic_burden_zscore_stats": z_stats,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
            },
            f, indent=2,
        )

    plot_distribution_before_after(df_train_pre_winsorize, df_train, FEATURE_DIR)

    # ---- Audit: lifestyle_risk_score distribution (verify 0-4 range) ----
    score_dist = (
        df_train["lifestyle_risk_score"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("score")
        .reset_index(name="count_train")
    )
    score_dist["pct_train"] = (score_dist["count_train"] / len(df_train) * 100).round(2)
    score_dist.to_csv(FEATURE_DIR / "lifestyle_risk_score_distribution.csv", index=False)
    print("\nlifestyle_risk_score distribution (train):")
    print(score_dist.to_string(index=False))

    # ---- Sensitivity feature set ----
    # We identify columns to drop in a "screening-only" sensitivity analysis.
    # The exclusion list contains TWO categories of features, each with a
    # different rationale:
    #
    # (A) Self-reported diagnosis history — directly encodes prior diagnosis,
    #     and may leak post-treatment state (e.g., diagnosed CVD patients on
    #     statins have lower measured cholesterol than they would otherwise):
    #         hypertension_told_*, diabetes_told_*, diabetes_borderline_flag,
    #         lifestyle_risk_score (includes diagnosed-diabetes component)
    #
    # (B) Guideline-threshold features derived from routine measurements —
    #     these are NOT prior diagnosis, but they encode clinical
    #     decision points (ACC/AHA, ADA cutoffs). Including or excluding
    #     them tests whether binning helps beyond the raw measurements:
    #         bp_stage_*, hba1c_status_*
    #
    # The full vs screening-only AUC gap quantifies how much signal comes
    # from "prior diagnosis" + "guideline-threshold encoding" vs from raw
    # routine measurements alone. The modelling team can also do a 3-way
    # comparison (full / no diagnosis history / no diagnosis + no thresholds)
    # for a cleaner attribution.
    diagnosis_history_cols = [
        n for n in feature_names
        if any(tag in n for tag in [
            "hypertension_told", "diabetes_told", "diabetes_borderline_flag",
            "lifestyle_risk_score",
        ])
        # Keep the diagnosis-free lifestyle variant.
        and "lifestyle_risk_score_no_dx" not in n
    ]
    threshold_derived_cols = [
        n for n in feature_names
        if any(tag in n for tag in ["bp_stage", "hba1c_status"])
    ]
    columns_to_drop = diagnosis_history_cols + threshold_derived_cols
    with open(FEATURE_DIR / "sensitivity_feature_exclusions.json", "w") as f:
        json.dump(
            {
                "description": (
                    "Columns to drop in the 'screening-only' sensitivity "
                    "analysis. Two categories: prior-diagnosis features and "
                    "guideline-threshold features derived from routine labs."
                ),
                "columns_to_drop": columns_to_drop,
                "diagnosis_history_columns": diagnosis_history_cols,
                "threshold_derived_columns": threshold_derived_cols,
                "screening_only_substitutes": {
                    "lifestyle_risk_score": "lifestyle_risk_score_no_dx",
                },
                "rationale": (
                    "Self-reported diagnoses can leak post-treatment state "
                    "into the predictor set (e.g., diagnosed CVD patients "
                    "are on medication that lowers cholesterol). Threshold "
                    "features (bp_stage_*, hba1c_status_*) are not prior "
                    "diagnosis but are derived from routine measurements via "
                    "ACC/AHA and ADA cutoffs; excluding them lets the model "
                    "rely on raw measurements only. The modelling team should "
                    "report at minimum the full vs screening-only contrast; a "
                    "3-way contrast (full / drop diagnosis only / drop "
                    "diagnosis + thresholds) gives cleaner attribution. "
                    "lifestyle_risk_score_no_dx is the diagnosis-free 0-3 "
                    "version of lifestyle_risk_score and should be "
                    "substituted in the screening-only model."
                ),
            },
            f, indent=2,
        )

    # Save the engineered (post-pipeline-input) frames for inspection.
    df_train.to_csv(PROCESSED_DIR / "analytic_dataset_engineered_train.csv", index=False)
    df_test.to_csv(PROCESSED_DIR / "analytic_dataset_engineered_test.csv", index=False)

    print(f"\nTrain X shape: {Xt_train.shape}; Test X shape: {Xt_test.shape}")
    print(f"Final encoded feature count: {len(feature_names)}")
    print(f"Saved transformed matrices and preprocessor to: {FEATURE_DIR}")
    print(f"Saved train-fit stats to:                       {FEATURE_DIR / 'train_fit_stats.json'}")
    print(f"Saved sensitivity feature exclusion list to:    {FEATURE_DIR / 'sensitivity_feature_exclusions.json'}")


if __name__ == "__main__":
    main()
