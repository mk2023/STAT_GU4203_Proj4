"""STAT 5243 Project 4 — interactive dashboard.

Run from the repo root:

    streamlit run src/dashboard.py

Five pages walk an audience through the full ML pipeline (Overview, EDA &
Clustering, Model Comparison, Final Model & Calibration, Predict). The
Predict page exposes the calibrated LogReg model from Part 3 via three
input paths (test-set sample, CSV upload, manual form), with per-row local
explanations and a What-if simulator.

The dashboard never refits anything — it only reads ``outputs/`` artefacts
and calls ``predict_api`` for inference, so reported metrics stay valid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dashboard_helpers import (  # noqa: E402
    DIABETES_OPTIONS,
    DRINKING_FREQ_OPTIONS,
    EDUCATION_OPTIONS,
    HTN_OPTIONS,
    RACE_OPTIONS,
    SEX_OPTIONS,
    SMOKE_NOW_OPTIONS,
    build_raw_input_row,
    cached_artifacts,
    get_train_baseline,
    load_csv,
    load_engineered_train,
    load_test_matrix,
    local_shap_logreg,
    plot_calibration,
    plot_cluster_cvd_rate,
    plot_cv_results,
    plot_donut_class_balance,
    plot_effect_sizes,
    plot_gauge,
    plot_roc_pr,
    plot_shap_global,
    plot_shap_local,
    plot_threshold_sweep,
    plot_whatif_curve,
    predict_engineered_row,
)
from dashboard_styles import (  # noqa: E402
    PRIMARY,
    callout,
    flag_card,
    inject_global_css,
    kpi_card,
    kpi_row,
    render_hero,
    section_title,
    sidebar_snapshot,
)
from predict_api import predict_proba  # noqa: E402


st.set_page_config(
    page_title="CVD Risk Dashboard · STAT 5243",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()


# ---------------------------------------------------------------------------
# Cached top-level data
# ---------------------------------------------------------------------------

PRE, MODEL, CARD = cached_artifacts()
THRESHOLD = float(CARD["recommended_threshold"])
ROC_AUC = float(CARD["roc_auc"])
PR_AUC = float(CARD["pr_auc"])
BRIER_PLATT = float(CARD["brier_platt"])
RECALL_AT = float(CARD["test_recall_at_recommended"])
PRECISION_AT = float(CARD["test_precision_at_recommended"])
F1_AT = float(CARD["test_f1_at_recommended"])
N_TRAIN = int(CARD["n_train"])
N_TEST = int(CARD["n_test"])
TRAIN_TS = str(CARD.get("training_timestamp", "—"))[:10]


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="padding:0.4rem 0 0.5rem 0; font-weight:700; font-size:1.05rem;'
        f' color:{PRIMARY};">CVD Risk Dashboard</div>'
        '<div style="font-size:0.78rem; color:#64748B; margin-bottom:0.6rem;">'
        "STAT 5243 · Project 4</div>",
        unsafe_allow_html=True,
    )
    selected = option_menu(
        menu_title=None,
        options=["Overview", "EDA & Clustering", "Model Comparison",
                 "Final Model", "Predict"],
        icons=["house", "search", "bar-chart", "sliders", "activity"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": PRIMARY, "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "padding": "0.55rem 0.75rem",
                "border-radius": "8px",
                "margin": "2px 0",
                "color": "#0F172A",
            },
            "nav-link-selected": {
                "background-color": PRIMARY,
                "color": "white",
                "font-weight": "600",
            },
        },
    )
    sidebar_snapshot([
        ("Model", CARD.get("model", "—")),
        ("Feature set", CARD.get("feature_set", "—")),
        ("Threshold", f"{THRESHOLD:.2f}"),
        ("Test ROC-AUC", f"{ROC_AUC:.3f}"),
        ("Trained", TRAIN_TS),
    ])
    st.markdown(
        '<div style="font-size:0.72rem; color:#94A3B8; margin-top:1rem;">'
        "All artefacts loaded from <code>outputs/</code>; no refitting happens here."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------

def page_overview() -> None:
    render_hero(
        title="Predicting cardiovascular disease in NHANES",
        subtitle=(
            "End-to-end ML pipeline on CDC NHANES 2021–2023: data cleaning, "
            "unsupervised phenotyping, four supervised models compared on three "
            "feature subsets, calibrated logistic regression chosen as the "
            "operating point at 80% sensitivity."
        ),
        kpis=[
            ("Labelled rows", f"{N_TRAIN + N_TEST:,}", f"{N_TEST:,} held out"),
            ("Test ROC-AUC", f"{ROC_AUC:.3f}", "L2 LogReg"),
            ("Recall @ thr=0.10", f"{RECALL_AT:.1%}", "screening criterion"),
            ("Brier (calibrated)", f"{BRIER_PLATT:.3f}", "Platt scaled"),
        ],
    )

    left, right = st.columns([1.05, 1])
    with left:
        section_title("What this project does",
                      "From raw .xpt files to a deployable risk model")
        st.markdown(
            """
            <div class="card">
            <h3>Pipeline at a glance</h3>
            <ol style="margin: 0.4rem 0 0 1.1rem; padding: 0; color: #0F172A; font-size: 0.95rem; line-height: 1.65;">
              <li><b>Part 1 — Data acquisition.</b> Joined six NHANES 2021–2023
                  XPT files (demographics, BMX, BPX, TCHOL, GHB, MCQ) on SEQN.
                  11,933 rows, 7,807 with known CVD status.</li>
              <li><b>Part 2 — EDA + clustering.</b> Welch's t-tests with FDR;
                  K-Means with k=4 on standardised vitals/labs; PCA for visual
                  inspection. Designed leakage-safe feature pipeline.</li>
              <li><b>Part 3 — Supervised modelling.</b> 4 models × 3 sensitivity
                  feature sets, 5-fold stratified CV with 20-iter random search,
                  Platt calibration, threshold from clinical screening criterion,
                  global + local SHAP.</li>
              <li><b>Part 4 — Communication.</b> This dashboard plus the final
                  written report.</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        callout(
            "Predictive signal lives in routine vitals and labs, not in prior "
            "diagnosis labels: ROC-AUC differs by ≤ 0.003 across the three "
            "feature subsets we tested.",
            kind="success",
            title="Headline finding",
        )
        callout(
            "L2 logistic regression beat tree ensembles after calibration "
            "(Brier 0.092 vs RF 0.131, XGB 0.144). Simple linear models are "
            "still competitive on tabular health data when properly regularised.",
            kind="info",
            title="Why a linear model wins",
        )

    with right:
        section_title("Class distribution", "n = 7,807 labelled rows")
        balance = load_csv("outputs/eda/cvd_class_balance.csv")
        positive_rate = float(balance.loc[balance["cvd"] == 1.0, "proportion"].iloc[0])
        st.plotly_chart(
            plot_donut_class_balance(positive_rate, n=7807),
            use_container_width=True,
        )
        kpi_row([
            ("Total NHANES rows", 11933, "raw participants", "accent"),
            ("With known CVD", 7807, "after dropping unlabelled", "success"),
        ])
        kpi_row([
            ("Train / Test split", "6,245 / 1,562", "stratified 80/20", "accent"),
            ("Positive rate", f"{positive_rate:.1%}", "≈ 1:7 imbalance", "warn"),
        ])


# ---------------------------------------------------------------------------
# Page 2 — EDA & clustering
# ---------------------------------------------------------------------------

def page_eda() -> None:
    section_title("Univariate effect sizes",
                  "Cohen's d for top numeric features (CVD positive vs negative)")
    eff_df = load_csv("outputs/eda/stat_tests_numeric.csv")
    st.plotly_chart(plot_effect_sizes(eff_df, top_k=12), use_container_width=True)

    callout(
        "<b>Cholesterol reversal:</b> total_chol, LDL, and triglycerides are "
        "<i>lower</i> in CVD-positive participants — a statin-treatment artefact. "
        "Negative coefficients on these features in the final model are real, not a bug.",
        kind="warn",
        title="Watch out for this in the model",
    )

    section_title("Variable correlations",
                  "Correlation heatmap on the analytic dataset")
    st.image(
        str(ROOT / "outputs/eda/correlation_heatmap.png"),
        use_column_width=True,
        caption="Pearson correlation across the engineered numeric features.",
    )

    section_title("K-Means risk phenotypes",
                  "k=4 chosen via silhouette + elbow on the standardised matrix")

    left, right = st.columns([1, 1])
    with left:
        st.image(
            str(ROOT / "outputs/clusters/cluster_pca_scatter.png"),
            use_column_width=True,
            caption="Samples projected onto the first two principal components, coloured by cluster.",
        )
    with right:
        profiles = load_csv("outputs/clusters/cluster_profiles.csv")
        st.plotly_chart(plot_cluster_cvd_rate(profiles), use_container_width=True)
        st.dataframe(
            profiles[["cluster", "n", "cvd_rate", "phenotype"]].assign(
                cvd_rate=lambda d: (d["cvd_rate"] * 100).round(1).astype(str) + "%"
            ),
            hide_index=True,
            use_container_width=True,
        )

    section_title("Cluster profile heatmap",
                  "Z-scored cluster means across vitals, labs, and demographics")
    st.image(
        str(ROOT / "outputs/clusters/cluster_profile_heatmap.png"),
        use_column_width=True,
    )


# ---------------------------------------------------------------------------
# Page 3 — Model comparison
# ---------------------------------------------------------------------------

def page_models() -> None:
    section_title("12 candidate models",
                  "4 algorithms × 3 feature subsets, 5-fold stratified CV with 20-iter random search")

    cv = load_csv("outputs/models/cv_results.csv")
    best_roc = cv.loc[cv["roc_auc_mean"].idxmax()]
    best_pr = cv.loc[cv["pr_auc_mean"].idxmax()]
    best_brier = cv.loc[cv["brier_mean"].idxmin()]
    best_f1 = cv.loc[cv["f1_mean"].idxmax()]

    kpi_row([
        ("Best ROC-AUC", f"{best_roc['roc_auc_mean']:.3f}",
         f"{best_roc['model']} (set {best_roc['feature_set']})", "accent"),
        ("Best PR-AUC", f"{best_pr['pr_auc_mean']:.3f}",
         f"{best_pr['model']} (set {best_pr['feature_set']})", "success"),
        ("Best Brier", f"{best_brier['brier_mean']:.3f}",
         f"{best_brier['model']} (set {best_brier['feature_set']})", "warn"),
        ("Best F1", f"{best_f1['f1_mean']:.3f}",
         f"{best_f1['model']} (set {best_f1['feature_set']})", "danger"),
    ])

    metric = st.radio(
        "Metric",
        options=["roc_auc", "pr_auc", "f1", "brier"],
        format_func=lambda x: x.replace("_", " ").upper(),
        horizontal=True,
        index=0,
    )
    st.plotly_chart(plot_cv_results(cv, metric=metric), use_container_width=True)

    callout(
        "Feature sets: <b>A</b> = full (80 cols, includes prior-diagnosis labels), "
        "<b>B</b> = no diagnosis history (79 cols, swap to <code>lifestyle_risk_score_no_dx</code>), "
        "<b>C</b> = raw measurements only (72 cols, drop guideline bins too). "
        "The A → B → C drop in CV ROC-AUC is at most 0.003 — predictive signal "
        "lives in raw labs and vitals, not in prior-diagnosis labels.",
        kind="success",
        title="Sensitivity story",
    )

    section_title("Full results table", "Sortable; click a column header")
    display = cv.copy()
    for col in ["roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std",
                "f1_mean", "f1_std", "brier_mean", "brier_std"]:
        if col in display.columns:
            display[col] = display[col].round(4)
    if "fit_seconds" in display.columns:
        display["fit_seconds"] = display["fit_seconds"].round(1)
    # height = (header 38px) + (12 rows × 35px) + buffer → all 12 rows show
    # without an inner scrollbar.
    st.dataframe(
        display, hide_index=True, use_container_width=True,
        height=38 + 35 * len(display) + 4,
    )


# ---------------------------------------------------------------------------
# Page 4 — Final model & calibration
# ---------------------------------------------------------------------------

def page_final() -> None:
    render_hero(
        title="Final model — L2 logistic regression",
        subtitle=(
            "Calibrated with Platt scaling, operating at threshold 0.10 to "
            "satisfy a screening sensitivity ≥ 80% criterion. Below: held-out "
            "test set ROC / PR / calibration, plus the threshold sweep."
        ),
        kpis=[
            ("Test ROC-AUC", f"{ROC_AUC:.3f}", None),
            ("Test PR-AUC", f"{PR_AUC:.3f}", None),
            ("Brier raw", f"{CARD['brier_raw']:.3f}", "before calibration"),
            ("Brier calibrated", f"{BRIER_PLATT:.3f}", f"-{CARD['brier_raw']-BRIER_PLATT:.3f}"),
        ],
    )

    X_test, y_test = load_test_matrix()
    proba = predict_proba(X_test)

    left, right = st.columns([1.1, 1])
    with left:
        section_title("ROC and Precision-Recall (test set)",
                      "AUC and AP read off the calibrated probabilities")
        st.plotly_chart(plot_roc_pr(y_test.values, proba), use_container_width=True)
    with right:
        section_title("Calibration curve",
                      "10-bin quantile binning; closer to diagonal = better calibrated")
        st.plotly_chart(plot_calibration(y_test.values, proba), use_container_width=True)

    callout(
        "<b>Cholesterol reversal warning.</b> total_chol, LDL, and "
        "triglycerides have <i>negative</i> coefficients. This is real: "
        "diagnosed CVD patients are on statins, lowering their measured "
        "cholesterol below pre-treatment baseline. Do not interpret "
        "negative coefficients as 'lower cholesterol causes CVD.'",
        kind="danger",
        title="Reading the coefficients",
    )

    section_title("Global feature importance",
                  "Top 15 LogReg coefficients, signed (red = pushes risk up)")
    left2, right2 = st.columns([1.05, 1])
    with left2:
        st.plotly_chart(plot_shap_global(top_k=15), use_container_width=True)
    with right2:
        st.image(
            str(ROOT / "outputs/models/shap_summary.png"),
            use_column_width=True,
            caption="SHAP beeswarm (Part 3 saved figure) — sample-level effect spread per feature.",
        )

    section_title("Threshold sweep",
                  "Recall, precision, and F1 across decision thresholds")
    thr_df = load_csv("outputs/models/threshold_analysis.csv")
    selected_thr = st.slider(
        "Move the decision threshold",
        min_value=float(thr_df["threshold"].min()),
        max_value=float(thr_df["threshold"].max()),
        value=THRESHOLD,
        step=0.01,
    )
    sub = thr_df.iloc[(thr_df["threshold"] - selected_thr).abs().argsort().iloc[:1]]
    if not sub.empty:
        row = sub.iloc[0]
        kpi_row([
            ("Recall", f"{row['recall']:.1%}", "true positive rate", "danger"),
            ("Precision", f"{row['precision']:.1%}", "PPV", "accent"),
            ("F1", f"{row['f1']:.3f}", None, "warn"),
            ("Specificity", f"{row['specificity']:.1%}", "1 − FPR", "success"),
        ])
    st.plotly_chart(
        plot_threshold_sweep(thr_df, current_threshold=selected_thr),
        use_container_width=True,
    )
    st.caption(
        f"Recommended operating point in the model card: threshold = {THRESHOLD:.2f} "
        f"(recall ≈ {RECALL_AT:.1%}, precision ≈ {PRECISION_AT:.1%}, F1 ≈ {F1_AT:.3f})."
    )


# ---------------------------------------------------------------------------
# Page 5 — Predict
# ---------------------------------------------------------------------------

WHATIF_FEATURES = {
    "age_years": dict(label="Age (years)", min=20.0, max=85.0, step=1.0),
    "sbp_avg": dict(label="Systolic BP (mmHg)", min=80.0, max=200.0, step=2.0),
    "bmi": dict(label="BMI (kg/m²)", min=15.0, max=50.0, step=0.5),
    "hba1c": dict(label="HbA1c (%)", min=4.0, max=12.0, step=0.1),
    "total_chol": dict(label="Total cholesterol (mg/dL)", min=100.0, max=320.0, step=5.0),
}


def _row_to_preprocessed(row: pd.DataFrame) -> pd.DataFrame:
    """Return a 1-row preprocessed (81-col num__/cat__) DataFrame for SHAP /
    predict_proba. Detects whether the input is already preprocessed by looking
    for the ``num__`` / ``cat__`` prefixes; otherwise runs the raw pipeline
    through the saved preprocessor."""
    feat_cols = CARD["feature_columns_used"]
    looks_preprocessed = all(c in row.columns for c in feat_cols)
    if looks_preprocessed:
        return row
    import json as _json
    arr = PRE.transform(row)
    full_cols = _json.loads(
        (ROOT / "outputs/features/feature_names.json").read_text()
    )
    return pd.DataFrame(arr, columns=full_cols, index=row.index)


def _render_prediction_output(row: pd.DataFrame, raw_for_whatif: dict | None) -> None:
    """Standard right-side output: gauge + flag card + local SHAP + What-if.

    ``row`` may be either a preprocessed 1-row matrix (sample / CSV paths) or
    an engineered raw row (manual form path). _row_to_preprocessed normalises
    both before predict_proba / SHAP.
    """
    preprocessed = _row_to_preprocessed(row)
    proba = float(predict_proba(preprocessed)[0])

    left, right = st.columns([1, 1.05])
    with left:
        st.plotly_chart(plot_gauge(proba, THRESHOLD), use_container_width=True)
        flag_card(proba, THRESHOLD)
    with right:
        section_title("Why this prediction",
                      "LogReg local contributions = coef × (x − train mean)")
        contribs = local_shap_logreg(preprocessed.iloc[0])
        st.plotly_chart(plot_shap_local(contribs, top_k=12), use_container_width=True)

    if raw_for_whatif is not None:
        section_title("What-if simulator",
                      "Hold every other field constant; sweep one feature")
        feat_key = st.selectbox(
            "Feature to vary",
            options=list(WHATIF_FEATURES.keys()),
            format_func=lambda k: WHATIF_FEATURES[k]["label"],
        )
        meta = WHATIF_FEATURES[feat_key]
        grid = np.arange(meta["min"], meta["max"] + meta["step"] / 2, meta["step"])
        baseline = get_train_baseline()
        probas: list[float] = []
        for v in grid:
            user_overrides = {**raw_for_whatif, feat_key: float(v)}
            row = build_raw_input_row(user_overrides, baseline)
            probas.append(predict_engineered_row(row))
        current = float(raw_for_whatif.get(feat_key, baseline.get(feat_key, np.nan)))
        st.plotly_chart(
            plot_whatif_curve(
                feature_label=meta["label"],
                grid=grid,
                probas=probas,
                current_value=current,
                current_proba=proba,
                threshold=THRESHOLD,
            ),
            use_container_width=True,
        )


def _path_sample() -> None:
    X_test, y_test = load_test_matrix()
    idx = st.slider("Pick a row from the held-out test set",
                    min_value=0, max_value=len(X_test) - 1, value=0)
    row = X_test.iloc[[idx]]
    truth = int(y_test.iloc[idx])
    st.caption(f"Selected row index {idx} · ground-truth label = "
               f"{'CVD positive' if truth == 1 else 'CVD negative'}")
    _render_prediction_output(row, raw_for_whatif=None)


def _path_csv() -> None:
    feat_cols = CARD["feature_columns_used"]
    st.markdown(
        f'<div class="callout">Upload a CSV containing the {len(feat_cols)} '
        "preprocessed feature columns (the same shape as <code>outputs/features/X_test.csv</code>). "
        "Each row gets a calibrated probability and a screening flag.</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return
    df = pd.read_csv(uploaded)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        st.error(
            f"CSV is missing {len(missing)} required columns. First few: {missing[:5]}"
        )
        return
    proba = predict_proba(df[feat_cols])
    out = df.copy()
    out["probability"] = proba
    out["screening_flag"] = (proba >= THRESHOLD).astype(int)

    kpi_row([
        ("Rows scored", f"{len(df):,}", None, "accent"),
        ("Mean probability", f"{proba.mean():.3f}", None, "success"),
        ("Flagged positive",
         f"{int((out['screening_flag'] == 1).sum()):,}",
         f"{(out['screening_flag'] == 1).mean():.1%}", "danger"),
        ("Threshold", f"{THRESHOLD:.2f}", "screening criterion", "warn"),
    ])

    show_cols = ["probability", "screening_flag"] + feat_cols[:4]
    st.dataframe(
        out[show_cols].head(50).style.format({"probability": "{:.3f}"}),
        use_container_width=True,
    )

    pick = st.number_input("Inspect a single row from the upload",
                           min_value=0, max_value=len(df) - 1, value=0, step=1)
    _render_prediction_output(df.iloc[[int(pick)]][feat_cols], raw_for_whatif=None)


def _path_form() -> None:
    baseline = get_train_baseline()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age", min_value=18, max_value=100,
                              value=int(baseline.get("age_years", 53) or 53))
        sex_label = st.selectbox("Sex", list(SEX_OPTIONS.keys()),
                                 index=0 if (baseline.get("sex_code") in (1, 1.0)) else 1)
        race_label = st.selectbox("Race / ethnicity", list(RACE_OPTIONS.keys()), index=2)
        edu_label = st.selectbox("Education", list(EDUCATION_OPTIONS.keys()), index=3)
        ipr = st.number_input("Income-poverty ratio", min_value=0.0, max_value=5.0,
                              value=float(baseline.get("income_poverty_ratio", 2.5) or 2.5),
                              step=0.1)
    with col2:
        st.markdown("**Vitals & body**")
        sbp = st.number_input("Systolic BP", min_value=60, max_value=220,
                              value=int(baseline.get("sbp_avg", 122) or 122))
        dbp = st.number_input("Diastolic BP", min_value=40, max_value=140,
                              value=int(baseline.get("dbp_avg", 73) or 73))
        pulse = st.number_input("Pulse (bpm)", min_value=40, max_value=140,
                                value=int(baseline.get("pulse_avg", 72) or 72))
        bmi_v = st.number_input("BMI", min_value=12.0, max_value=60.0,
                                value=float(baseline.get("bmi", 28.5) or 28.5),
                                step=0.1)
        waist = st.number_input("Waist (cm)", min_value=50.0, max_value=180.0,
                                value=float(baseline.get("waist_cm", 99.0) or 99.0),
                                step=0.5)
    with col3:
        st.markdown("**Labs**")
        tchol = st.number_input("Total cholesterol (mg/dL)", min_value=80, max_value=400,
                                value=int(baseline.get("total_chol", 187) or 187))
        hdl = st.number_input("HDL", min_value=15, max_value=120,
                              value=int(baseline.get("hdl", 52) or 52))
        ldl = st.number_input("LDL", min_value=20, max_value=300,
                              value=int(baseline.get("ldl", 110) or 110))
        trig = st.number_input("Triglycerides", min_value=20, max_value=900,
                               value=int(baseline.get("triglycerides", 110) or 110))
        fbg = st.number_input("Fasting glucose", min_value=50, max_value=400,
                              value=int(baseline.get("fasting_glucose", 102) or 102))
        hba1c_v = st.number_input("HbA1c (%)", min_value=3.5, max_value=15.0,
                                  value=float(baseline.get("hba1c", 5.6) or 5.6),
                                  step=0.1)

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**Lifestyle**")
        smoke_label = st.selectbox("Currently smokes", list(SMOKE_NOW_OPTIONS.keys()), index=2)
        smoked_lifetime = st.checkbox("Smoked ≥ 100 cigarettes lifetime",
                                      value=bool(baseline.get("smoked_100_life_code", 0) == 1))
        drink_freq = st.selectbox("Drinking frequency", list(DRINKING_FREQ_OPTIONS.keys()), index=0)
        drinks = st.number_input("Avg. drinks per day", min_value=0.0, max_value=20.0,
                                 value=float(baseline.get("avg_drinks_per_day", 1.0) or 1.0),
                                 step=0.5)
        mod_min = st.number_input("Moderate activity (min/week)", min_value=0, max_value=600,
                                  value=int(baseline.get("moderate_ltpa_minutes", 30) or 30))
        vig_min = st.number_input("Vigorous activity (min/week)", min_value=0, max_value=600,
                                  value=int(baseline.get("vigorous_ltpa_minutes", 0) or 0))
    with col5:
        st.markdown("**Medical history**")
        diab_label = st.selectbox("Told had diabetes", list(DIABETES_OPTIONS.keys()), index=0)
        htn_label = st.selectbox("Told had hypertension", list(HTN_OPTIONS.keys()), index=0)
        st.caption("All other fields use the training-set median / mode; "
                   "edit any of the 16 inputs above to override.")

    user_inputs = {
        "age_years": float(age),
        "sex_code": SEX_OPTIONS[sex_label],
        "race_ethnicity_code": RACE_OPTIONS[race_label],
        "education_code": EDUCATION_OPTIONS[edu_label],
        "income_poverty_ratio": float(ipr),
        "sbp_avg": float(sbp),
        "dbp_avg": float(dbp),
        "pulse_avg": float(pulse),
        "bmi": float(bmi_v),
        "waist_cm": float(waist),
        "total_chol": float(tchol),
        "hdl": float(hdl),
        "ldl": float(ldl),
        "triglycerides": float(trig),
        "fasting_glucose": float(fbg),
        "hba1c": float(hba1c_v),
        "smoked_100_life_code": 1 if smoked_lifetime else 0,
        "smoke_now_code": SMOKE_NOW_OPTIONS[smoke_label],
        "drinking_frequency_code": DRINKING_FREQ_OPTIONS[drink_freq],
        "avg_drinks_per_day": float(drinks),
        "moderate_ltpa_minutes": float(mod_min),
        "vigorous_ltpa_minutes": float(vig_min),
        "diabetes_told_code": DIABETES_OPTIONS[diab_label],
        "hypertension_told_code": HTN_OPTIONS[htn_label],
    }

    row = build_raw_input_row(user_inputs, baseline)
    _render_prediction_output(row, raw_for_whatif=user_inputs)


def page_predict() -> None:
    render_hero(
        title="Score a participant",
        subtitle=(
            "Three input paths into the calibrated final model. The output is "
            "the calibrated probability of CVD-positive plus a binary "
            "screening flag at the recommended threshold."
        ),
        kpis=[
            ("Threshold", f"{THRESHOLD:.2f}", "screening criterion"),
            ("Test recall @ thr", f"{RECALL_AT:.1%}", None),
            ("Test precision @ thr", f"{PRECISION_AT:.1%}", None),
            ("Test F1 @ thr", f"{F1_AT:.3f}", None),
        ],
    )

    tab_sample, tab_csv, tab_form = st.tabs(
        ["Sample from test set", "Upload CSV", "Manual form"]
    )
    with tab_sample:
        _path_sample()
    with tab_csv:
        _path_csv()
    with tab_form:
        _path_form()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "EDA & Clustering": page_eda,
    "Model Comparison": page_models,
    "Final Model": page_final,
    "Predict": page_predict,
}

PAGES[selected]()
