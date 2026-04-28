"""
Advanced EDA: statistical hypothesis testing of CVD vs non-CVD groups.

For each numeric feature, runs a Welch's t-test and reports effect size
(Cohen's d). For each categorical feature, runs a chi-square test of
independence. Multiple-testing correction is applied via Benjamini-Hochberg
to control the false discovery rate at q=0.05.

This complements the descriptive EDA from preprocess_eda.py with formal
statistical evidence of which features differ between CVD-positive and
CVD-negative participants — which directly informs feature selection for
the modelling stage.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
EDA_DIR = ROOT / "outputs" / "eda"


NUMERIC_TEST_FEATURES = [
    "age_years",
    "bmi",
    "waist_cm",
    "sbp_avg",
    "dbp_avg",
    "pulse_pressure",
    "pulse_avg",
    "total_chol",
    "hdl",
    "ldl",
    "triglycerides",
    "fasting_glucose",
    "hba1c",
    "chol_hdl_ratio",
    "met_syn_score",
    "income_poverty_ratio",
]

CATEGORICAL_TEST_FEATURES = [
    "sex_code",
    "race_ethnicity_code",
    "education_code",
    "smoked_100_life_code",
    "diabetes_told_code",
    "hypertension_told_code",
]

# These NHANES variables look categorical by name but the underlying coding
# is a count/frequency (e.g., minutes of activity, drinking frequency code),
# not a finite category. We exclude them from chi-square testing — running a
# chi-square on 50+ levels inflates degrees of freedom and yields
# uninterpretable results.
CATEGORICAL_EXCLUDED_AS_NUMERIC = [
    "moderate_ltpa_minutes",        # NHANES PAD800: moderate LTPA (minutes)
    "vigorous_ltpa_minutes",  # NHANES PAD820: vigorous LTPA (minutes)
    "drinking_frequency_code",      # NHANES ALQ121: drinking frequency code (0-10)
]


def audit_categorical_levels(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """For each candidate categorical, report the number of unique levels and
    the top 10 most-common values. NHANES often packs counts/frequencies into
    fields that look categorical; this audit catches those cases."""
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False).head(10)
        rows.append({
            "feature": col,
            "n_unique": int(df[col].nunique(dropna=True)),
            "top_levels": vc.to_dict(),
        })
    return pd.DataFrame(rows)


def ensure_dirs() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "analytic_dataset.csv")


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled


def numeric_tests(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    feats = features if features is not None else NUMERIC_TEST_FEATURES
    rows = []
    for col in feats:
        if col not in df.columns:
            continue
        pos = df.loc[df["cvd"] == 1, col].dropna().values
        neg = df.loc[df["cvd"] == 0, col].dropna().values
        if len(pos) < 5 or len(neg) < 5:
            continue
        # Welch's t-test does not assume equal variances.
        t_stat, p_val = stats.ttest_ind(pos, neg, equal_var=False)
        d = cohens_d(pos, neg)
        rows.append({
            "feature": col,
            "n_cvd_pos": len(pos),
            "n_cvd_neg": len(neg),
            "mean_cvd_pos": round(float(np.mean(pos)), 3),
            "mean_cvd_neg": round(float(np.mean(neg)), 3),
            "t_statistic": round(float(t_stat), 3),
            "p_value": float(p_val),
            "cohens_d": round(float(d), 3),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        # BH correction across the family of numeric tests.
        _, p_adj, _, _ = multipletests(out["p_value"].values, alpha=0.05, method="fdr_bh")
        out["p_value_bh"] = p_adj
        out = out.sort_values("p_value_bh").reset_index(drop=True)
    return out


def categorical_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_TEST_FEATURES:
        if col not in df.columns:
            continue
        sub = df[[col, "cvd"]].dropna()
        if sub[col].nunique() < 2:
            continue
        ct = pd.crosstab(sub[col], sub["cvd"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        # Cramer's V effect size.
        n = ct.values.sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n > 0 else np.nan
        # Chi-square assumption check: all expected cell counts should be >= 5.
        min_expected = float(expected.min())
        rows.append({
            "feature": col,
            "n": int(n),
            "chi2": round(float(chi2), 3),
            "dof": int(dof),
            "p_value": float(p),
            "cramers_v": round(float(v), 3),
            "min_expected_cell": round(min_expected, 2),
            "assumption_ok": min_expected >= 5,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        _, p_adj, _, _ = multipletests(out["p_value"].values, alpha=0.05, method="fdr_bh")
        out["p_value_bh"] = p_adj
        out = out.sort_values("p_value_bh").reset_index(drop=True)
    return out


def plot_effect_sizes(numeric_results: pd.DataFrame) -> None:
    if numeric_results.empty:
        return
    sns.set_theme(style="whitegrid")
    plot_df = numeric_results.assign(abs_d=lambda d: d["cohens_d"].abs()).sort_values("abs_d")

    plt.figure(figsize=(8, 5))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in plot_df["cohens_d"]]
    plt.barh(plot_df["feature"], plot_df["cohens_d"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.axvline(0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.axvline(-0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.xlabel("Cohen's d  (CVD+ vs CVD-)")
    plt.title("Standardized mean differences across CVD status\n(dashed lines: small=0.2, medium=0.5)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "effect_sizes_numeric.png", dpi=180)
    plt.close()


def plot_top_feature_distributions(df: pd.DataFrame, numeric_results: pd.DataFrame, top_n: int = 6) -> None:
    if numeric_results.empty:
        return
    sns.set_theme(style="whitegrid")
    top = (
        numeric_results.assign(abs_d=lambda d: d["cohens_d"].abs())
        .sort_values("abs_d", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )
    cvd_df = df[df["cvd"].notna()].copy()
    cvd_df["CVD status"] = cvd_df["cvd"].map({0: "No CVD", 1: "CVD"})

    n = len(top)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i, feat in enumerate(top):
        sns.kdeplot(
            data=cvd_df, x=feat, hue="CVD status", ax=axes[i],
            common_norm=False, fill=True, alpha=0.4,
        )
        axes[i].set_title(feat)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Top {n} features by |Cohen's d|: distribution by CVD status", y=1.02)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "top_feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_data()
    cvd_known = df[df["cvd"].notna()]
    print(f"Sample with known CVD: {len(cvd_known):,}")
    print(f"CVD prevalence: {cvd_known['cvd'].mean():.3f}\n")

    # Audit categorical variables before testing — flags variables that look
    # categorical by name but have too many levels to be valid chi-square inputs.
    audit_cols = CATEGORICAL_TEST_FEATURES + CATEGORICAL_EXCLUDED_AS_NUMERIC
    audit = audit_categorical_levels(df, audit_cols)
    audit.to_csv(EDA_DIR / "categorical_level_audit.csv", index=False)
    print("Categorical level audit:")
    print(audit.to_string(index=False))
    print(f"\nExcluded from chi-square (treated as numeric/ordinal features): {CATEGORICAL_EXCLUDED_AS_NUMERIC}\n")

    # Add the excluded numeric/ordinal variables to the numeric tests so we don't lose
    # the signal — they're better suited to t-tests anyway.
    extended_numeric = NUMERIC_TEST_FEATURES + [
        c for c in CATEGORICAL_EXCLUDED_AS_NUMERIC if c in df.columns
    ]

    print("Running Welch's t-tests on numeric features...")
    num_res = numeric_tests(df, features=extended_numeric)
    num_res.to_csv(EDA_DIR / "stat_tests_numeric.csv", index=False)
    print(num_res.to_string(index=False))

    print("\nRunning chi-square tests on cleaned categorical features...")
    cat_res = categorical_tests(df)
    cat_res.to_csv(EDA_DIR / "stat_tests_categorical.csv", index=False)
    print(cat_res.to_string(index=False))

    plot_effect_sizes(num_res)
    plot_top_feature_distributions(df, num_res)

    n_sig_num = int((num_res["p_value_bh"] < 0.05).sum()) if not num_res.empty else 0
    n_sig_cat = int((cat_res["p_value_bh"] < 0.05).sum()) if not cat_res.empty else 0
    print(f"\nNumeric features significant (BH q<0.05): {n_sig_num}/{len(num_res)}")
    print(f"Categorical features significant (BH q<0.05): {n_sig_cat}/{len(cat_res)}")
    print(f"\nResults saved to: {EDA_DIR}")


if __name__ == "__main__":
    main()
