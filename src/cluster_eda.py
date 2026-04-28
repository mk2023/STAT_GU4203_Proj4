"""
Unsupervised EDA: identify CVD risk phenotypes via K-Means clustering.

Pipeline:
1. Load analytic_dataset.csv (output of preprocess_eda.py).
2. Select clinically meaningful numeric features for clustering.
3. Median-impute and standardize.
4. Choose k via the elbow method (inertia) and silhouette score.
5. Fit final K-Means and assign cluster labels back to participants.
6. Profile each cluster (means + CVD prevalence) and visualize via PCA.

Outputs are written to outputs/clusters/ and the labelled dataset is saved
to data/processed/analytic_dataset_with_clusters.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
CLUSTER_DIR = ROOT / "outputs" / "clusters"

# Features used for clustering: clinical measurements + age.
# Engineered ratios (chol_hdl_ratio, met_syn_score) are excluded to avoid
# double-counting variables already represented by their components.
CLUSTER_FEATURES = [
    "age_years",
    "bmi",
    "waist_cm",
    "sbp_avg",
    "dbp_avg",
    "pulse_pressure",
    "total_chol",
    "hdl",
    "triglycerides",
    "fasting_glucose",
    "hba1c",
]

K_RANGE = range(2, 9)
FINAL_K = 4
RANDOM_STATE = 42


def ensure_dirs() -> None:
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "analytic_dataset.csv")


def prepare_cluster_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    # Keep only rows with at least 60% of cluster features observed; this
    # avoids feeding mostly-imputed rows into K-Means while still retaining
    # most of the sample.
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    obs_frac = df[available].notna().mean(axis=1)
    keep_mask = obs_frac >= 0.6
    sub = df.loc[keep_mask, available].copy()

    # Median-impute remaining NaNs using the cluster sample's own medians.
    medians = sub.median(numeric_only=True)
    sub_filled = sub.fillna(medians)

    scaler = StandardScaler()
    X = scaler.fit_transform(sub_filled.values)

    return df.loc[keep_mask].copy(), X, scaler


def choose_k(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        # Silhouette is expensive on large samples; subsample for stability.
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
        sil = silhouette_score(X[idx], labels[idx])
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def plot_k_selection(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(data=metrics, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("Elbow Method (Inertia vs k)")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Within-cluster sum of squares")

    sns.lineplot(data=metrics, x="k", y="silhouette", marker="o", color="darkorange", ax=axes[1])
    axes[1].set_title("Silhouette Score vs k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")

    plt.tight_layout()
    plt.savefig(CLUSTER_DIR / "k_selection.png", dpi=180)
    plt.close()


def fit_final_kmeans(X: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X)
    return km, labels


def profile_clusters(df_sub: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    out = df_sub.copy()
    out["cluster"] = labels

    available = [c for c in CLUSTER_FEATURES if c in out.columns]
    profile = out.groupby("cluster")[available].mean(numeric_only=True)
    profile["n"] = out.groupby("cluster").size()

    # CVD prevalence within each cluster (only for participants with known status).
    cvd_known = out[out["cvd"].notna()]
    cvd_rate = cvd_known.groupby("cluster")["cvd"].mean().rename("cvd_rate")
    profile = profile.join(cvd_rate)

    return profile.round(2)


def label_phenotypes(profile: pd.DataFrame) -> dict[int, str]:
    # Heuristic naming based on each cluster's relative position on key axes.
    # We rank clusters on age, bmi, sbp_avg, fasting_glucose to assign descriptive
    # labels — useful for the report narrative, not for downstream modelling.
    names = {}
    for cid, row in profile.iterrows():
        tags = []
        if row.get("age_years", 0) >= profile["age_years"].median():
            tags.append("older")
        else:
            tags.append("younger")

        if row.get("sbp_avg", 0) >= 130 or row.get("dbp_avg", 0) >= 85:
            tags.append("hypertensive")
        if row.get("bmi", 0) >= 30:
            tags.append("obese")
        if row.get("fasting_glucose", 0) >= 110 or row.get("hba1c", 0) >= 6.0:
            tags.append("dysglycemic")
        if row.get("triglycerides", 0) >= 150 or row.get("hdl", 100) < 40:
            tags.append("dyslipidemic")

        if len(tags) == 1:
            tags.append("low-risk")
        names[cid] = " / ".join(tags)
    return names


def plot_cluster_pca(X: np.ndarray, labels: np.ndarray, feature_names: list[str]) -> None:
    # Full PCA so we can inspect all components, not just the first two.
    pca = PCA(random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)

    # ----- 2D scatter colored by cluster -----
    plot_df = pd.DataFrame(
        {"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": labels.astype(str)}
    )

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=plot_df, x="PC1", y="PC2", hue="cluster", s=12, alpha=0.5, palette="tab10"
    )
    var = pca.explained_variance_ratio_
    plt.title(f"K-Means Clusters in PCA Space (PC1 {var[0]:.0%}, PC2 {var[1]:.0%})")
    plt.tight_layout()
    plt.savefig(CLUSTER_DIR / "cluster_pca_scatter.png", dpi=180)
    plt.close()

    # ----- Scree plot: explained variance per component + cumulative -----
    fig, ax1 = plt.subplots(figsize=(8, 4))
    pcs = np.arange(1, len(var) + 1)
    ax1.bar(pcs, var, color="steelblue", alpha=0.7, label="Per-component")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Variance explained")
    ax1.set_xticks(pcs)

    ax2 = ax1.twinx()
    ax2.plot(pcs, np.cumsum(var), "o-", color="darkorange", label="Cumulative")
    ax2.set_ylabel("Cumulative variance")
    ax2.axhline(0.80, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax2.text(len(var) - 0.5, 0.81, "80%", color="red", fontsize=9)

    plt.title("PCA Scree Plot")
    fig.tight_layout()
    plt.savefig(CLUSTER_DIR / "pca_scree.png", dpi=180)
    plt.close()

    # ----- Loadings heatmap (clinical interpretation of PCs) -----
    n_show = min(5, pca.components_.shape[0])
    loadings = pd.DataFrame(
        pca.components_[:n_show].T,
        index=feature_names,
        columns=[f"PC{i+1} ({var[i]:.0%})" for i in range(n_show)],
    )
    loadings.to_csv(CLUSTER_DIR / "pca_loadings.csv")

    plt.figure(figsize=(7, 6))
    sns.heatmap(loadings, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("PCA Loadings — feature contribution to top components")
    plt.tight_layout()
    plt.savefig(CLUSTER_DIR / "pca_loadings.png", dpi=180)
    plt.close()


def plot_profile_heatmap(profile: pd.DataFrame) -> None:
    feat_cols = [c for c in CLUSTER_FEATURES if c in profile.columns]
    # Z-score columns so the heatmap shows relative profile shape across clusters.
    standardized = profile[feat_cols].apply(
        lambda col: (col - col.mean()) / col.std(ddof=0)
    )

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        standardized,
        cmap="coolwarm",
        center=0,
        annot=profile[feat_cols].round(1),
        fmt="",
        cbar_kws={"label": "z-score across clusters"},
    )
    plt.title("Cluster Phenotype Profiles (cell text = raw mean, color = z-score)")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(CLUSTER_DIR / "cluster_profile_heatmap.png", dpi=180)
    plt.close()


def plot_cvd_by_cluster(profile: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    cvd_pct = (profile["cvd_rate"] * 100).round(1)
    ax = sns.barplot(x=cvd_pct.index.astype(str), y=cvd_pct.values, color="steelblue")
    for i, v in enumerate(cvd_pct.values):
        ax.text(i, v + 0.3, f"{v}%", ha="center", fontsize=10)
    plt.title("CVD Prevalence by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("CVD prevalence (%)")
    plt.tight_layout()
    plt.savefig(CLUSTER_DIR / "cvd_rate_by_cluster.png", dpi=180)
    plt.close()


def attach_clusters_to_full_dataset(df_full: pd.DataFrame, df_sub: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    out = df_full.copy()
    cluster_series = pd.Series(labels, index=df_sub.index, name="cluster")
    out["cluster"] = cluster_series
    return out


def hierarchical_robustness_check(X: np.ndarray, kmeans_labels: np.ndarray, k: int) -> dict:
    """
    Two robustness checks for the K-Means partition:

    (a) Bootstrap stability: re-fit K-Means on 50 bootstrap resamples and
        measure the average Adjusted Rand Index between each bootstrap's
        labels (transferred via nearest-centroid prediction) and the
        original labels. High ARI => clusters are reproducible.

    (b) Cross-algorithm agreement: fit Agglomerative (Ward) on a 3000-row
        subsample and report ARI vs K-Means on the same subsample. We
        expect modest agreement (Ward and K-Means use different objectives),
        but the dominant clusters should align.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans as _KMeans
    from sklearn.metrics import adjusted_rand_score

    # ---- (a) bootstrap stability ----
    rng = np.random.default_rng(RANDOM_STATE)
    aris_boot = []
    for _ in range(50):
        idx = rng.choice(len(X), size=len(X), replace=True)
        km_b = _KMeans(n_clusters=k, random_state=rng.integers(0, 1_000_000), n_init=10)
        km_b.fit(X[idx])
        # Predict on original X using the bootstrap centroids — this gives
        # comparable labels on the same points across runs.
        pred = km_b.predict(X)
        aris_boot.append(adjusted_rand_score(kmeans_labels, pred))
    bootstrap_ari = float(np.mean(aris_boot))

    # ---- (b) cross-algorithm agreement ----
    idx2 = rng.choice(len(X), size=min(3000, len(X)), replace=False)
    agglo = AgglomerativeClustering(n_clusters=k, linkage="ward")
    agg_labels = agglo.fit_predict(X[idx2])
    cross_ari = float(adjusted_rand_score(kmeans_labels[idx2], agg_labels))

    return {"bootstrap_ari_mean": round(bootstrap_ari, 3),
            "cross_algo_ari": round(cross_ari, 3)}


def main() -> None:
    ensure_dirs()
    df = load_data()
    print(f"Loaded {len(df):,} rows from analytic_dataset.csv")

    df_sub, X, scaler = prepare_cluster_matrix(df)
    print(f"Cluster sample: {len(df_sub):,} rows after >=60% feature-completeness filter")

    metrics = choose_k(X)
    metrics.to_csv(CLUSTER_DIR / "k_selection_metrics.csv", index=False)
    plot_k_selection(metrics)
    print("k-selection metrics:")
    print(metrics.to_string(index=False))

    km, labels = fit_final_kmeans(X, FINAL_K)
    print(f"Final K-Means fitted with k={FINAL_K}")

    # Robustness: two complementary checks.
    rob = hierarchical_robustness_check(X, labels, FINAL_K)
    print(f"Robustness — bootstrap stability ARI: {rob['bootstrap_ari_mean']:.3f}")
    print(f"Robustness — cross-algorithm agreement (Agglomerative Ward) ARI: {rob['cross_algo_ari']:.3f}")
    pd.DataFrame([rob]).to_csv(CLUSTER_DIR / "robustness_check.csv", index=False)

    profile = profile_clusters(df_sub, labels)
    names = label_phenotypes(profile)
    profile["phenotype"] = profile.index.map(names)
    profile.to_csv(CLUSTER_DIR / "cluster_profiles.csv")
    print("\nCluster profiles:")
    print(profile.to_string())

    cluster_features = [c for c in CLUSTER_FEATURES if c in df_sub.columns]
    plot_cluster_pca(X, labels, cluster_features)
    plot_profile_heatmap(profile)
    plot_cvd_by_cluster(profile)

    df_with_clusters = attach_clusters_to_full_dataset(df, df_sub, labels)
    df_with_clusters.to_csv(PROCESSED_DIR / "analytic_dataset_with_clusters.csv", index=False)

    # Persist the fitted scaler + KMeans + medians + feature list so the
    # modelling team can produce train-only cluster labels if they want to
    # use `cluster` as a supervised feature without leaking test info.
    import joblib
    medians = df.loc[df_sub.index, cluster_features].median(numeric_only=True).to_dict()
    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": km,
            "feature_order": cluster_features,
            "imputation_medians": medians,
            "min_observed_frac": 0.6,
        },
        CLUSTER_DIR / "kmeans_pipeline.joblib",
    )

    print(f"\nSaved cluster outputs to: {CLUSTER_DIR}")
    print(f"Saved labelled dataset:    {PROCESSED_DIR / 'analytic_dataset_with_clusters.csv'}")
    print(f"Saved fitted pipeline:     {CLUSTER_DIR / 'kmeans_pipeline.joblib'}")
    print("\nNOTE for the modelling team:")
    print("  The 'cluster' column in analytic_dataset_with_clusters.csv was")
    print("  fit on the FULL sample. If you plan to use cluster as a supervised")
    print("  feature, re-fit KMeans within each train fold instead — the")
    print("  saved pipeline shows the exact preprocessing steps to reuse.")


if __name__ == "__main__":
    main()
