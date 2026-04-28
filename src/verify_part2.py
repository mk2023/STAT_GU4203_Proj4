"""
Verification script — run this AFTER cluster_eda.py and feature_engineering.py.

It compares your local outputs against the expected values and tells you
whether everything is fine, has acceptable numerical drift, or genuinely differs.

Usage:
    python3 src/verify_part2.py
"""
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = ROOT / "outputs" / "clusters"
FEATURE_DIR = ROOT / "outputs" / "features"


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}{(' — ' + detail) if detail else ''}")
    return condition


def main() -> None:
    print("=" * 60)
    print("Part 2 verification")
    print("=" * 60)

    all_pass = True

    # ------------------------------------------------------------
    # 1. Files exist
    # ------------------------------------------------------------
    print("\n--- Files exist ---")
    expected_files = [
        CLUSTER_DIR / "k_selection_metrics.csv",
        CLUSTER_DIR / "cluster_profiles.csv",
        CLUSTER_DIR / "k_selection.png",
        CLUSTER_DIR / "cluster_profile_heatmap.png",
        CLUSTER_DIR / "cluster_pca_scatter.png",
        CLUSTER_DIR / "cvd_rate_by_cluster.png",
        CLUSTER_DIR / "pca_scree.png",
        CLUSTER_DIR / "pca_loadings.png",
        CLUSTER_DIR / "pca_loadings.csv",
        CLUSTER_DIR / "robustness_check.csv",
        CLUSTER_DIR / "kmeans_pipeline.joblib",
        FEATURE_DIR / "X_train.csv",
        FEATURE_DIR / "X_test.csv",
        FEATURE_DIR / "y_train.csv",
        FEATURE_DIR / "y_test.csv",
        FEATURE_DIR / "feature_names.json",
        FEATURE_DIR / "preprocessor.joblib",
        FEATURE_DIR / "winsorization_log.csv",
        FEATURE_DIR / "transform_before_after.png",
        FEATURE_DIR / "train_fit_stats.json",
        FEATURE_DIR / "lifestyle_risk_score_distribution.csv",
        FEATURE_DIR / "sensitivity_feature_exclusions.json",
        ROOT / "outputs" / "eda" / "stat_tests_numeric.csv",
        ROOT / "outputs" / "eda" / "stat_tests_categorical.csv",
        ROOT / "outputs" / "eda" / "categorical_level_audit.csv",
        ROOT / "outputs" / "eda" / "effect_sizes_numeric.png",
        ROOT / "docs" / "handoff_to_part3.md",
        ROOT / "docs" / "feature_engineering_report.md",
    ]
    for f in expected_files:
        all_pass &= check(f"exists: {f.relative_to(ROOT)}", f.exists())

    if not all_pass:
        print("\nSome files are missing. Did both scripts finish without errors?")
        sys.exit(1)

    # ------------------------------------------------------------
    # 2. K-selection: k=4 should be optimal by silhouette
    # ------------------------------------------------------------
    print("\n--- K-Means clustering ---")
    metrics = pd.read_csv(CLUSTER_DIR / "k_selection_metrics.csv")
    best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    sil_at_4 = float(metrics.loc[metrics["k"] == 4, "silhouette"].iloc[0])

    all_pass &= check("best k by silhouette = 4", best_k == 4, f"got k={best_k}")
    all_pass &= check(
        "silhouette at k=4 in [0.17, 0.20]",
        0.17 <= sil_at_4 <= 0.20,
        f"got {sil_at_4:.4f}",
    )

    # ------------------------------------------------------------
    # 3. Cluster profiles: 4 clusters, n totals add up
    # ------------------------------------------------------------
    profiles = pd.read_csv(CLUSTER_DIR / "cluster_profiles.csv")
    all_pass &= check("4 cluster rows", len(profiles) == 4, f"got {len(profiles)} rows")
    total_n = int(profiles["n"].sum())
    all_pass &= check(
        "total cluster n in [6900, 7000]",
        6900 <= total_n <= 7000,
        f"got n={total_n}",
    )

    # Each cluster should have a CVD rate; the spread tells us phenotypes are real.
    cvd_min = profiles["cvd_rate"].min()
    cvd_max = profiles["cvd_rate"].max()
    all_pass &= check(
        "CVD rates differ across clusters (max - min >= 0.10)",
        (cvd_max - cvd_min) >= 0.10,
        f"min={cvd_min:.2f}, max={cvd_max:.2f}",
    )

    # The 4 phenotype shapes should be recognizable (regardless of cluster id).
    # Sort by age to identify them.
    by_age = profiles.sort_values("age_years").reset_index(drop=True)
    youngest = by_age.iloc[0]
    oldest = by_age.iloc[-1]
    all_pass &= check(
        "youngest cluster avg age < 35 (low-risk group)",
        youngest["age_years"] < 35,
        f"got {youngest['age_years']:.1f}",
    )
    all_pass &= check(
        "oldest cluster avg age > 60",
        oldest["age_years"] > 60,
        f"got {oldest['age_years']:.1f}",
    )

    # There should be a small dysglycemic cluster (HbA1c >= 8).
    high_hba1c = profiles[profiles["hba1c"] >= 8.0]
    all_pass &= check(
        "exactly one severe-dysglycemic cluster (hba1c >= 8)",
        len(high_hba1c) == 1,
        f"got {len(high_hba1c)} cluster(s)",
    )

    # ------------------------------------------------------------
    # 4. Feature engineering output shapes
    # ------------------------------------------------------------
    print("\n--- Feature engineering ---")
    X_train = pd.read_csv(FEATURE_DIR / "X_train.csv")
    X_test = pd.read_csv(FEATURE_DIR / "X_test.csv")
    y_train = pd.read_csv(FEATURE_DIR / "y_train.csv")
    y_test = pd.read_csv(FEATURE_DIR / "y_test.csv")

    all_pass &= check(
        "X_train shape == (6245, 81)",
        X_train.shape == (6245, 81),
        f"got {X_train.shape}",
    )
    all_pass &= check(
        "X_test shape == (1562, 81)",
        X_test.shape == (1562, 81),
        f"got {X_test.shape}",
    )
    all_pass &= check(
        "y_train rows == 6245",
        len(y_train) == 6245,
        f"got {len(y_train)}",
    )
    all_pass &= check(
        "y_test rows == 1562",
        len(y_test) == 1562,
        f"got {len(y_test)}",
    )

    # Stratified split should preserve the ~12.6% CVD rate in both sets.
    train_rate = y_train.iloc[:, 0].mean()
    test_rate = y_test.iloc[:, 0].mean()
    all_pass &= check(
        "y_train CVD rate ~ 0.126 (+/-0.005)",
        abs(train_rate - 0.126) <= 0.005,
        f"got {train_rate:.4f}",
    )
    all_pass &= check(
        "y_test CVD rate ~ 0.126 (+/-0.005)",
        abs(test_rate - 0.126) <= 0.005,
        f"got {test_rate:.4f}",
    )

    # Numeric features should be standardized: mean ~0, std ~1.
    num_cols = [c for c in X_train.columns if c.startswith("num__")]
    means = X_train[num_cols].mean().abs().max()
    stds_diff = (X_train[num_cols].std() - 1).abs().max()
    all_pass &= check(
        "numeric features have |mean| < 0.01",
        means < 0.01,
        f"max |mean| = {means:.4f}",
    )
    all_pass &= check(
        "numeric features have std ~ 1 (within 0.01)",
        stds_diff < 0.01,
        f"max |std-1| = {stds_diff:.4f}",
    )

    # Feature names should exist.
    feature_names = json.load(open(FEATURE_DIR / "feature_names.json"))
    all_pass &= check(
        "feature_names.json has 81 entries (with missingness indicators)",
        len(feature_names) == 81,
        f"got {len(feature_names)}",
    )

    # Check that we actually have missing-indicator columns (advanced preprocessing).
    indicator_cols = [n for n in feature_names if "missingindicator" in n.lower()]
    all_pass &= check(
        ">=1 missingness indicator column present",
        len(indicator_cols) >= 1,
        f"found {len(indicator_cols)} indicator columns",
    )

    # Lifestyle risk score should be in the valid 0-4 range (no diabetes_code==3 leak).
    score_dist = pd.read_csv(FEATURE_DIR / "lifestyle_risk_score_distribution.csv")
    max_score = score_dist["score"].dropna().max()
    all_pass &= check(
        "lifestyle_risk_score max <= 4 (no coding bug)",
        max_score <= 4,
        f"got max={max_score}",
    )

    # Diagnosis-free lifestyle score must exist (for screening-only sensitivity model).
    has_no_dx = any("lifestyle_risk_score_no_dx" in n for n in feature_names)
    all_pass &= check(
        "lifestyle_risk_score_no_dx column present (for screening-only model)",
        has_no_dx,
        "missing — sensitivity model can't be built without diagnosis-free score",
    )

    # Sensitivity exclusion list must list lifestyle_risk_score (without no_dx).
    excl = json.load(open(FEATURE_DIR / "sensitivity_feature_exclusions.json"))
    excl_cols = excl.get("columns_to_drop", [])
    has_lifestyle_in_exclusion = any(
        "lifestyle_risk_score" in c and "no_dx" not in c for c in excl_cols
    )
    no_dx_kept = not any("lifestyle_risk_score_no_dx" in c for c in excl_cols)
    all_pass &= check(
        "sensitivity exclusion drops lifestyle_risk_score (it includes diabetes diagnosis)",
        has_lifestyle_in_exclusion,
        "lifestyle_risk_score not in exclusion list",
    )
    all_pass &= check(
        "sensitivity exclusion KEEPS lifestyle_risk_score_no_dx (diagnosis-free)",
        no_dx_kept,
        "no_dx variant should NOT be excluded",
    )

    # ------------------------------------------------------------
    # 5. Advanced EDA: statistical tests
    # ------------------------------------------------------------
    print("\n--- Advanced EDA (statistical tests) ---")
    EDA_DIR = ROOT / "outputs" / "eda"
    num_tests = pd.read_csv(EDA_DIR / "stat_tests_numeric.csv")
    cat_tests = pd.read_csv(EDA_DIR / "stat_tests_categorical.csv")
    n_sig_num = (num_tests["p_value_bh"] < 0.05).sum()
    n_sig_cat = (cat_tests["p_value_bh"] < 0.05).sum()
    all_pass &= check(
        ">=10 numeric features significant after BH",
        n_sig_num >= 10,
        f"got {n_sig_num}/{len(num_tests)}",
    )
    all_pass &= check(
        ">=6 categorical features significant after BH (cleaned set)",
        n_sig_cat >= 6,
        f"got {n_sig_cat}/{len(cat_tests)}",
    )
    age_d = num_tests.loc[num_tests["feature"] == "age_years", "cohens_d"].iloc[0]
    all_pass &= check(
        "age_years has large effect size (Cohen's d >= 0.8)",
        age_d >= 0.8,
        f"got d={age_d:.3f}",
    )

    # Chi-square assumption check (all expected cell counts should be >= 5).
    if "assumption_ok" in cat_tests.columns:
        all_assumptions_met = bool(cat_tests["assumption_ok"].all())
        all_pass &= check(
            "all chi-square tests have min expected cell >= 5",
            all_assumptions_met,
            f"min expected = {cat_tests['min_expected_cell'].min():.2f}",
        )

    # ------------------------------------------------------------
    # 6. Robustness of clustering
    # ------------------------------------------------------------
    print("\n--- Cluster robustness ---")
    rob = pd.read_csv(CLUSTER_DIR / "robustness_check.csv")
    bootstrap_ari = rob["bootstrap_ari_mean"].iloc[0]
    all_pass &= check(
        "bootstrap stability ARI >= 0.85 (clusters reproducible)",
        bootstrap_ari >= 0.85,
        f"got {bootstrap_ari:.3f}",
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED — outputs match expectations.")
    else:
        print("SOME CHECKS FAILED — see [FAIL] lines above.")
        print("Numerical drift on the order of 1e-4 is normal across machines;")
        print("only worry if the structural checks (file existence, shapes,")
        print("number of clusters, k=4) failed.")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
