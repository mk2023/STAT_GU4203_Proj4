from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "rawfiles"
PROCESSED_DIR = ROOT / "data" / "processed"
EDA_DIR = ROOT / "outputs" / "eda"

SPECIAL_MISSING_CODES = {
    7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999
}


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def read_xpt(filename: str, columns: list[str]) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_sas(path, format="xport", encoding="utf-8")
    keep_cols = [c for c in columns if c in df.columns]
    return df[keep_cols].copy()


def normalize_numeric_missing(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    exclude = exclude or []
    out = df.copy()
    for col in out.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].replace(list(SPECIAL_MISSING_CODES), np.nan)
    return out


def yes_no_to_binary(series: pd.Series) -> pd.Series:
    # NHANES coding convention for many items: 1=Yes, 2=No
    return series.replace({1: 1, 2: 0})


def build_demo() -> pd.DataFrame:
    demo = read_xpt(
        "DEMO_L.xpt",
        ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"],
    )
    demo = normalize_numeric_missing(demo, exclude=["SEQN"])
    demo = demo.rename(
        columns={
            "RIAGENDR": "sex_code",
            "RIDAGEYR": "age_years",
            "RIDRETH3": "race_ethnicity_code",
            "DMDEDUC2": "education_code",
            "INDFMPIR": "income_poverty_ratio",
        }
    )
    return demo


def build_cvd_target() -> pd.DataFrame:
    # CVD composite from medical conditions:
    # CHF, CHD, angina, heart attack, stroke.
    mcq = read_xpt(
        "MCQ_L.xpt",
        ["SEQN", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
    )
    mcq = normalize_numeric_missing(mcq, exclude=["SEQN"])
    cvd_cols = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
    for col in cvd_cols:
        mcq[col] = yes_no_to_binary(mcq[col])

    mcq["cvd_positive_count"] = mcq[cvd_cols].fillna(0).sum(axis=1)
    mcq["cvd"] = np.where(mcq["cvd_positive_count"] > 0, 1, 0)

    # Keep unknowns as missing if all components are missing.
    all_missing = mcq[cvd_cols].isna().all(axis=1)
    mcq.loc[all_missing, "cvd"] = np.nan

    return mcq[["SEQN", "cvd"]]


def build_blood_pressure() -> pd.DataFrame:
    bpx = read_xpt(
        "BPXO_L.xpt",
        [
            "SEQN",
            "BPXOSY1",
            "BPXOSY2",
            "BPXOSY3",
            "BPXODI1",
            "BPXODI2",
            "BPXODI3",
            "BPXOPLS1",
            "BPXOPLS2",
            "BPXOPLS3",
        ],
    )
    bpx = normalize_numeric_missing(bpx, exclude=["SEQN"])
    bpx["sbp_avg"] = bpx[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1, skipna=True)
    bpx["dbp_avg"] = bpx[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1, skipna=True)
    bpx["pulse_avg"] = bpx[["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"]].mean(axis=1, skipna=True)
    return bpx[["SEQN", "sbp_avg", "dbp_avg", "pulse_avg"]]


def build_body_measures() -> pd.DataFrame:
    bmx = read_xpt("BMX_L.xpt", ["SEQN", "BMXBMI", "BMXWAIST"])
    bmx = normalize_numeric_missing(bmx, exclude=["SEQN"])
    return bmx.rename(columns={"BMXBMI": "bmi", "BMXWAIST": "waist_cm"})


def build_labs() -> pd.DataFrame:
    tchol = read_xpt("TCHOL_L.xpt", ["SEQN", "LBXTC"])
    hdl = read_xpt("HDL_L.xpt", ["SEQN", "LBDHDD"])
    trigly = read_xpt("TRIGLY_L.xpt", ["SEQN", "LBXTLG", "LBDLDL"])
    glu = read_xpt("GLU_L.xpt", ["SEQN", "LBXGLU"])
    ghb = read_xpt("GHB_L.xpt", ["SEQN", "LBXGH"])

    labs = tchol.merge(hdl, on="SEQN", how="outer")
    labs = labs.merge(trigly, on="SEQN", how="outer")
    labs = labs.merge(glu, on="SEQN", how="outer")
    labs = labs.merge(ghb, on="SEQN", how="outer")
    labs = normalize_numeric_missing(labs, exclude=["SEQN"])

    return labs.rename(
        columns={
            "LBXTC": "total_chol",
            "LBDHDD": "hdl",
            "LBXTLG": "triglycerides",
            "LBDLDL": "ldl",
            "LBXGLU": "fasting_glucose",
            "LBXGH": "hba1c",
        }
    )


def build_lifestyle() -> pd.DataFrame:
    smq = read_xpt("SMQ_L.xpt", ["SEQN", "SMQ020", "SMQ040"])
    alq = read_xpt("ALQ_L.xpt", ["SEQN", "ALQ121", "ALQ130"])
    paq = read_xpt("PAQ_L.xpt", ["SEQN", "PAD800", "PAD820"])
    diq = read_xpt("DIQ_L.xpt", ["SEQN", "DIQ010"])
    bpq = read_xpt("BPQ_L.xpt", ["SEQN", "BPQ020"])

    lifestyle = smq.merge(alq, on="SEQN", how="outer")
    lifestyle = lifestyle.merge(paq, on="SEQN", how="outer")
    lifestyle = lifestyle.merge(diq, on="SEQN", how="outer")
    lifestyle = lifestyle.merge(bpq, on="SEQN", how="outer")
    lifestyle = normalize_numeric_missing(lifestyle, exclude=["SEQN"])

    lifestyle = lifestyle.rename(
        columns={
            "SMQ020": "smoked_100_life_code",
            "SMQ040": "smoke_now_code",
            # CDC labels:
            # ALQ121 = drinking frequency in past 12 months (ordinal code)
            # ALQ130 = average alcoholic drinks/day in past 12 months
            "ALQ121": "drinking_frequency_code",
            "ALQ130": "avg_drinks_per_day",
            # CDC labels:
            # PAD800 = minutes moderate LTPA, PAD820 = minutes vigorous LTPA
            "PAD800": "moderate_ltpa_minutes",
            "PAD820": "vigorous_ltpa_minutes",
            "DIQ010": "diabetes_told_code",
            "BPQ020": "hypertension_told_code",
        }
    )

    binary_cols = [
        "smoked_100_life_code",
        "diabetes_told_code",
        "hypertension_told_code",
    ]
    for col in binary_cols:
        lifestyle[col] = yes_no_to_binary(lifestyle[col])

    return lifestyle


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pulse_pressure"] = out["sbp_avg"] - out["dbp_avg"]
    out["chol_hdl_ratio"] = np.where(out["hdl"] > 0, out["total_chol"] / out["hdl"], np.nan)

    met_syn_components = [
        (out["waist_cm"] >= 102),
        (out["sbp_avg"] >= 130) | (out["dbp_avg"] >= 85),
        (out["triglycerides"] >= 150),
        (out["hdl"] < 40),
        (out["fasting_glucose"] >= 100),
    ]
    out["met_syn_score"] = np.sum(np.column_stack(met_syn_components), axis=1)
    out.loc[
        out[["waist_cm", "sbp_avg", "dbp_avg", "triglycerides", "hdl", "fasting_glucose"]]
        .isna()
        .all(axis=1),
        "met_syn_score",
    ] = np.nan

    return out


def build_dataset() -> pd.DataFrame:
    demo = build_demo()
    target = build_cvd_target()
    bp = build_blood_pressure()
    body = build_body_measures()
    labs = build_labs()
    lifestyle = build_lifestyle()

    merged = demo.merge(target, on="SEQN", how="left")
    merged = merged.merge(bp, on="SEQN", how="left")
    merged = merged.merge(body, on="SEQN", how="left")
    merged = merged.merge(labs, on="SEQN", how="left")
    merged = merged.merge(lifestyle, on="SEQN", how="left")

    merged = engineer_features(merged)
    return merged


def run_eda(df: pd.DataFrame) -> None:
    numeric_cols = [
        "age_years",
        "bmi",
        "waist_cm",
        "sbp_avg",
        "dbp_avg",
        "pulse_pressure",
        "total_chol",
        "hdl",
        "ldl",
        "triglycerides",
        "fasting_glucose",
        "hba1c",
        "chol_hdl_ratio",
        "met_syn_score",
    ]
    available_num = [c for c in numeric_cols if c in df.columns]

    summary = df[available_num].describe().T
    summary.to_csv(EDA_DIR / "numeric_summary.csv")

    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_pct")
        .to_frame()
    )
    missing.to_csv(EDA_DIR / "missingness_summary.csv")

    cvd_df = df[df["cvd"].notna()].copy()
    cvd_rates = cvd_df["cvd"].value_counts(normalize=True).rename("proportion").to_frame()
    cvd_rates.to_csv(EDA_DIR / "cvd_class_balance.csv")

    by_cvd = cvd_df.groupby("cvd")[available_num].mean(numeric_only=True).T
    by_cvd.to_csv(EDA_DIR / "means_by_cvd.csv")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=cvd_df, x="cvd")
    plt.title("CVD Class Count")
    plt.xlabel("CVD (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "cvd_class_count.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=cvd_df, x="age_years", hue="cvd", bins=30, kde=True, element="step")
    plt.title("Age Distribution by CVD Status")
    plt.xlabel("Age (years)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "age_by_cvd.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=cvd_df, x="cvd", y="sbp_avg")
    plt.title("Systolic BP by CVD Status")
    plt.xlabel("CVD (0=No, 1=Yes)")
    plt.ylabel("Average Systolic BP")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "sbp_by_cvd.png", dpi=180)
    plt.close()

    corr_cols = [c for c in available_num if df[c].notna().sum() > 500]
    corr = df[corr_cols].corr(numeric_only=True)
    corr.to_csv(EDA_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()


def write_preparation_report(df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Data Acquisition & Preparation Report")
    lines.append("")
    lines.append(f"- Input folder: `{RAW_DIR}`")
    lines.append(f"- Output dataset: `{PROCESSED_DIR / 'analytic_dataset.csv'}`")
    lines.append(f"- Total participants (rows): {len(df):,}")
    lines.append(f"- Total variables (columns): {len(df.columns):,}")
    lines.append(f"- Participants with known CVD status: {df['cvd'].notna().sum():,}")
    lines.append("")
    lines.append("## Key preprocessing decisions")
    lines.append("- Merged modules by `SEQN` using left joins from demographics.")
    lines.append("- Recoded NHANES special missing codes (7/9, 77/99, 777/999, etc.) to NaN.")
    lines.append("- Averaged repeated blood pressure and pulse measures.")
    lines.append("- Built CVD composite outcome from MCQ cardiovascular condition variables.")
    lines.append("- Engineered pulse pressure, cholesterol ratio, and metabolic syndrome score.")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("- Cleaned dataset: `data/processed/analytic_dataset.csv`")
    lines.append("- EDA summaries and plots: `outputs/eda/`")

    (ROOT / "docs").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "preparation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = build_dataset()
    df = df.sort_values("SEQN").reset_index(drop=True)
    df.to_csv(PROCESSED_DIR / "analytic_dataset.csv", index=False)
    run_eda(df)
    write_preparation_report(df)

    print(f"Saved: {PROCESSED_DIR / 'analytic_dataset.csv'}")
    print(f"EDA outputs: {EDA_DIR}")
    print(f"Report: {ROOT / 'docs' / 'preparation_report.md'}")


if __name__ == "__main__":
    main()
