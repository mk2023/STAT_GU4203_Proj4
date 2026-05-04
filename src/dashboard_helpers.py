"""Data + plotting helpers for the Streamlit dashboard.

Keeps the main app file focused on layout. All artefact loading, raw-input
construction, local SHAP, and Plotly chart factories live here. Anything
expensive is wrapped in ``st.cache_data`` / ``st.cache_resource`` so the app
stays snappy when users move between pages or drag sliders.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import streamlit as st

# allow ``from feature_engineering import ...`` / ``from predict_api import ...``
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from feature_engineering import (  # noqa: E402  (sys.path tweak above)
    add_metabolic_burden,
    apply_winsor_bounds,
    engineer_deterministic,
)
from predict_api import (  # noqa: E402
    load_artifacts,
    predict_from_raw,
    predict_proba,
)

import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from dashboard_styles import PLOTLY_COLORWAY, PRIMARY, ACCENT, DANGER, SLATE_500  # noqa: E402


# ---------------------------------------------------------------------------
# Cached artefact loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cached_artifacts():
    """Returns (preprocessor, model, card)."""
    return load_artifacts()


@st.cache_resource(show_spinner=False)
def cached_train_fit_stats() -> dict:
    return json.loads((ROOT / "outputs/features/train_fit_stats.json").read_text())


@st.cache_data(show_spinner=False)
def load_csv(rel_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel_path)


@st.cache_data(show_spinner=False)
def load_json(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text())


@st.cache_data(show_spinner=False)
def load_engineered_train() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/processed/analytic_dataset_engineered_train.csv")


@st.cache_data(show_spinner=False)
def load_test_matrix() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(ROOT / "outputs/features/X_test.csv")
    y = pd.read_csv(ROOT / "outputs/features/y_test.csv").squeeze("columns")
    return X, y


# ---------------------------------------------------------------------------
# Manual form helpers
# ---------------------------------------------------------------------------

# Raw column space the preprocessor expects (numerics it will impute / scale,
# categoricals it will OHE). Sourced from feature_engineering.NUMERIC_FEATURES
# and CATEGORICAL_FEATURES, plus the upstream raw columns the deterministic
# engineer functions read from.
RAW_INPUT_COLUMNS: list[str] = [
    "sex_code", "age_years", "race_ethnicity_code", "education_code",
    "income_poverty_ratio",
    "sbp_avg", "dbp_avg", "pulse_avg",
    "bmi", "waist_cm",
    "total_chol", "hdl", "triglycerides", "ldl",
    "fasting_glucose", "hba1c",
    "smoked_100_life_code", "smoke_now_code",
    "drinking_frequency_code", "avg_drinks_per_day",
    "moderate_ltpa_minutes", "vigorous_ltpa_minutes",
    "diabetes_told_code", "hypertension_told_code",
]


SEX_OPTIONS = {"male": 1, "female": 2}
RACE_OPTIONS = {
    "Mexican American": 1, "Other Hispanic": 2, "Non-Hispanic White": 3,
    "Non-Hispanic Black": 4, "Non-Hispanic Asian": 6, "Other / Multi": 7,
}
EDUCATION_OPTIONS = {
    "< 9th grade": 1, "9th-11th grade": 2, "High school": 3,
    "Some college": 4, "College graduate": 5,
}
SMOKE_NOW_OPTIONS = {"Every day": 1, "Some days": 2, "Not at all": 3, "Never smoked 100 lifetime": 0}
DIABETES_OPTIONS = {"No": 0, "Yes": 1, "Borderline": 3}
HTN_OPTIONS = {"No": 0, "Yes": 1}
DRINKING_FREQ_OPTIONS = {
    "Never (last year)": 0, "Every day": 1, "Nearly every day": 2,
    "3-4/week": 3, "2/week": 4, "1/week": 5,
    "2-3/month": 6, "1/month": 7, "<= 1/year": 10,
}


@st.cache_data(show_spinner=False)
def get_train_baseline() -> dict:
    """Return median (numeric) / mode (categorical/code) values from the
    engineered training set, used to fill in fields the user does not edit."""
    df = load_engineered_train()
    baseline: dict = {}
    for col in RAW_INPUT_COLUMNS:
        if col not in df.columns:
            baseline[col] = np.nan
            continue
        s = df[col]
        if s.dtype.kind in {"i", "u", "f"}:
            baseline[col] = float(s.median())
        else:
            mode = s.mode(dropna=True)
            baseline[col] = mode.iloc[0] if not mode.empty else np.nan
    return baseline


def _recompute_simple_derived(row: dict) -> dict:
    """Pulse pressure / chol-HDL ratio / metabolic syndrome score follow
    closed-form definitions in preprocess_eda.engineer_features. Mirror them
    here so manual-form inputs propagate correctly without rerunning Part 1.
    """
    out = dict(row)
    sbp, dbp = out.get("sbp_avg"), out.get("dbp_avg")
    if sbp is not None and dbp is not None and not (pd.isna(sbp) or pd.isna(dbp)):
        out["pulse_pressure"] = float(sbp) - float(dbp)
    else:
        out["pulse_pressure"] = np.nan

    total_chol, hdl = out.get("total_chol"), out.get("hdl")
    if hdl and hdl > 0 and not pd.isna(total_chol):
        out["chol_hdl_ratio"] = float(total_chol) / float(hdl)
    else:
        out["chol_hdl_ratio"] = np.nan

    waist = out.get("waist_cm")
    trig = out.get("triglycerides")
    fbg = out.get("fasting_glucose")
    components = []
    components.append(int(waist >= 102) if waist is not None and not pd.isna(waist) else np.nan)
    components.append(
        int((sbp is not None and not pd.isna(sbp) and sbp >= 130)
            or (dbp is not None and not pd.isna(dbp) and dbp >= 85))
        if not (pd.isna(sbp) and pd.isna(dbp)) else np.nan
    )
    components.append(int(trig >= 150) if trig is not None and not pd.isna(trig) else np.nan)
    components.append(int(hdl < 40) if hdl is not None and not pd.isna(hdl) else np.nan)
    components.append(int(fbg >= 100) if fbg is not None and not pd.isna(fbg) else np.nan)
    obs = [c for c in components if not (isinstance(c, float) and pd.isna(c))]
    out["met_syn_score"] = float(sum(obs)) if obs else np.nan
    return out


def build_raw_input_row(user_inputs: dict, baseline: dict) -> pd.DataFrame:
    """Compose a 1-row engineered DataFrame ready for ``predict_from_raw``.

    Steps:
        baseline overlay → user overrides
          → simple derived (pulse_pressure / chol_hdl_ratio / met_syn_score)
          → winsorize using train bounds
          → engineer_deterministic (bins / logs / interactions / lifestyle)
          → add_metabolic_burden (z-score using train stats)
    """
    row = {**baseline, **user_inputs}
    row = _recompute_simple_derived(row)

    df = pd.DataFrame([row])

    stats = cached_train_fit_stats()
    winsor_bounds = {k: tuple(v) for k, v in stats["winsor_bounds"].items()}
    df = apply_winsor_bounds(df, winsor_bounds)

    df = engineer_deterministic(df)

    z_stats = {k: tuple(v) for k, v in stats["metabolic_burden_zscore_stats"].items()}
    df = add_metabolic_burden(df, z_stats)
    return df


def predict_engineered_row(engineered_row: pd.DataFrame) -> float:
    proba = predict_from_raw(engineered_row)
    return float(proba[0])


# ---------------------------------------------------------------------------
# Local SHAP via LogReg coefficients
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_logreg_coefs() -> tuple[pd.Series, pd.Series]:
    """Pull the linear logistic regression coefficients out of the calibrated
    pipeline, plus the training feature means used for the local contribution
    calculation. Returns (coef_series, train_mean_series)."""
    _, model, card = cached_artifacts()
    cols = card["feature_columns_used"]

    # CalibratedClassifierCV stores the underlying estimator in
    # calibrated_classifiers_[i].estimator (sklearn >= 1.5). Average across
    # CV-fold estimators if more than one.
    coefs = []
    for cc in getattr(model, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is not None and hasattr(est, "coef_"):
            coefs.append(est.coef_.ravel())
    if not coefs and hasattr(model, "coef_"):
        coefs.append(model.coef_.ravel())

    if not coefs:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    coef = pd.Series(np.mean(coefs, axis=0), index=cols)

    X_train = pd.read_csv(ROOT / "outputs/features/X_train.csv")
    means = X_train.reindex(columns=cols).mean()
    return coef, means


def local_shap_logreg(x_row: pd.Series) -> pd.Series:
    """Return per-feature contribution = coef * (x - mean), in log-odds units.

    Cheap exact local explanation for an L2 LogReg before calibration. After
    Platt scaling these are still the most-influential features for the row,
    even though they no longer exactly sum to the calibrated probability.
    """
    coef, means = get_logreg_coefs()
    if coef.empty:
        return pd.Series(dtype=float)
    aligned = x_row.reindex(coef.index).astype(float)
    return (aligned - means) * coef


# ---------------------------------------------------------------------------
# Plotly factories — single colorway, consistent layout
# ---------------------------------------------------------------------------

BASE_LAYOUT = dict(
    template="simple_white",
    colorway=PLOTLY_COLORWAY,
    font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=12),
    legend=dict(
        bgcolor="rgba(255,255,255,0)",
        borderwidth=0,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
    ),
    plot_bgcolor="white",
    paper_bgcolor="rgba(0,0,0,0)",
)


def _styled(fig: go.Figure, **overrides) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT, **overrides)

    # Title position depends on whether the child set a title at all.
    # Plotly renders a literal "undefined" string when title.text is missing
    # but other title attrs are set, so we only enforce position when there
    # is real title text.
    title_text = getattr(getattr(fig.layout, "title", None), "text", None)
    # Plotly renders the literal string "undefined" as a title sometimes
    # when it sees that template. Treat it like an empty title.
    if title_text and "undefined" not in str(title_text):
        fig.update_layout(
            title=dict(x=0.0, xanchor="left", y=0.97, yanchor="top",
                       font=dict(size=14, color="#0F172A")),
            margin=dict(l=10, r=10, t=70, b=10),
        )
    else:
        # subplot-only figures: clear any auto-injected title text and
        # don't reserve top margin for a title.
        fig.update_layout(
            title_text="",
            margin=dict(l=10, r=10, t=50, b=10),
        )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, linecolor="#CBD5E1")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, linecolor="#CBD5E1")
    return fig


def plot_donut_class_balance(positive_rate: float, n: int) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["CVD positive", "CVD negative"],
            values=[positive_rate, 1 - positive_rate],
            hole=0.62,
            marker=dict(colors=[DANGER, PRIMARY]),
            sort=False,
            textinfo="label+percent",
            textposition="outside",
        )
    )
    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>{n:,}</b><br><span style='color:{SLATE_500};font-size:11px'>labelled rows</span>",
                showarrow=False,
                font=dict(size=18),
            )
        ],
        showlegend=False,
    )
    return _styled(fig, height=320)


def plot_effect_sizes(df: pd.DataFrame, top_k: int = 10) -> go.Figure:
    """Horizontal bar of |Cohen's d| (or similar) for top features.

    Expects df with columns like ``feature`` and ``cohens_d`` / ``effect``.
    """
    candidates = [c for c in ["cohens_d", "effect_size", "abs_effect", "d"] if c in df.columns]
    if not candidates:
        # fall back to first numeric column other than feature
        numerics = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numerics:
            return go.Figure()
        candidates = [numerics[0]]
    metric_col = candidates[0]
    name_col = "feature" if "feature" in df.columns else df.columns[0]

    work = df.assign(_abs=df[metric_col].abs()).sort_values("_abs", ascending=False).head(top_k)
    work = work.sort_values("_abs")
    colors = [DANGER if v > 0 else PRIMARY for v in work[metric_col]]
    fig = go.Figure(
        go.Bar(
            y=work[name_col],
            x=work[metric_col],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {top_k} features by |{metric_col}|",
        xaxis_title=metric_col,
        yaxis_title=None,
        height=380,
        showlegend=False,
    )
    return _styled(fig)


def plot_cv_results(cv_df: pd.DataFrame, metric: str = "roc_auc") -> go.Figure:
    """Grouped bar with error bars across the 4×3 = 12 CV runs.

    Zooms the y-axis to a tight window around the observed values so the
    differences between models / feature sets are actually visible — the
    raw 0–1 range collapses everything onto a single horizontal line.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in cv_df.columns:
        raise ValueError(f"{mean_col} not in cv results")
    df = cv_df.copy()
    df["feature_set"] = df["feature_set"].astype(str)
    fig = go.Figure()
    palette = {"A": PRIMARY, "B": ACCENT, "C": "#F59E0B"}
    for fs in ["A", "B", "C"]:
        sub = df[df["feature_set"] == fs]
        fig.add_trace(
            go.Bar(
                name=f"Set {fs}",
                x=sub["model"],
                y=sub[mean_col],
                error_y=dict(type="data", array=sub[std_col], visible=True, color="#94A3B8"),
                marker=dict(color=palette.get(fs, PRIMARY)),
                hovertemplate="%{x} · Set " + fs + "<br>%{y:.4f}<extra></extra>",
            )
        )

    means = df[mean_col]
    stds = df[std_col]
    lo = float((means - stds).min())
    hi = float((means + stds).max())
    pad = max((hi - lo) * 0.25, 0.005)
    y_min = max(0.0, lo - pad)
    y_max = min(1.0, hi + pad) if metric != "brier" else hi + pad

    fig.update_layout(
        barmode="group",
        title=f"5-fold CV {metric.replace('_', ' ').upper()} (mean ± std)",
        yaxis_title=metric.replace("_", " ").upper(),
        xaxis_title=None,
        height=420,
        bargap=0.18,
        bargroupgap=0.05,
    )
    fig.update_yaxes(range=[y_min, y_max])
    return _styled(fig)


def plot_roc_pr(y: np.ndarray, proba: np.ndarray) -> go.Figure:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)

    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"ROC (AUC={auc:.3f})", f"Precision-Recall (AP={ap:.3f})"))
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=PRIMARY, width=2.5), name="ROC"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#CBD5E1", dash="dash"),
                   name="random", showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=rec, y=prec, mode="lines", line=dict(color=DANGER, width=2.5),
                   name="PR", showlegend=False),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="FPR", row=1, col=1)
    fig.update_yaxes(title_text="TPR", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_layout(height=360, showlegend=False)
    return _styled(fig)


def plot_calibration(y: np.ndarray, proba_calibrated: np.ndarray, n_bins: int = 10) -> go.Figure:
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y, proba_calibrated, n_bins=n_bins, strategy="quantile")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#CBD5E1", dash="dash"), name="perfect", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mean_pred, y=frac_pos,
            mode="lines+markers", line=dict(color=PRIMARY, width=2.5),
            marker=dict(size=8, color=PRIMARY),
            name="Calibrated (Platt)", hovertemplate="pred=%{x:.2f} · obs=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Calibration curve (test set, 10 quantile bins)",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction positive",
        height=360,
    )
    return _styled(fig)


def plot_shap_global(top_k: int = 15) -> go.Figure:
    coef, _ = get_logreg_coefs()
    if coef.empty:
        return go.Figure()
    sorted_coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(top_k)
    sorted_coef = sorted_coef[::-1]  # for horizontal bar
    colors = [DANGER if v > 0 else PRIMARY for v in sorted_coef.values]
    short_names = [c.split("__", 1)[-1] for c in sorted_coef.index]
    fig = go.Figure(
        go.Bar(
            y=short_names,
            x=sorted_coef.values,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {top_k} LogReg coefficients (positive → higher CVD risk)",
        xaxis_title="Coefficient (log-odds)",
        yaxis_title=None,
        height=480,
        showlegend=False,
    )
    return _styled(fig)


def plot_shap_local(contribs: pd.Series, top_k: int = 12) -> go.Figure:
    if contribs.empty:
        return go.Figure()
    work = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(top_k)
    work = work[::-1]
    colors = [DANGER if v > 0 else PRIMARY for v in work.values]
    short_names = [c.split("__", 1)[-1] for c in work.index]
    fig = go.Figure(
        go.Bar(
            y=short_names,
            x=work.values,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Local feature contributions (top {top_k} by magnitude)",
        xaxis_title="Contribution to log-odds (red = pushes risk up)",
        yaxis_title=None,
        height=440,
        showlegend=False,
    )
    return _styled(fig)


def plot_gauge(probability: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#0F172A"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": PRIMARY, "thickness": 0.25},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#DCFCE7"},
                    {"range": [threshold * 100, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": DANGER, "width": 4},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
                "borderwidth": 0,
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_threshold_sweep(thr_df: pd.DataFrame, current_threshold: float) -> go.Figure:
    """Recall / precision / F1 vs threshold; current threshold marker."""
    df = thr_df.sort_values("threshold")
    fig = go.Figure()
    for col, color, name in [
        ("recall", DANGER, "Recall"),
        ("precision", PRIMARY, "Precision"),
        ("f1", ACCENT, "F1"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["threshold"], y=df[col], mode="lines",
                    line=dict(color=color, width=2.5), name=name,
                    hovertemplate=f"thr=%{{x:.2f}} · {name}=%{{y:.3f}}<extra></extra>",
                )
            )
    fig.add_vline(x=current_threshold, line=dict(color="#0F172A", dash="dash", width=1.5),
                  annotation_text=f"thr={current_threshold:.2f}", annotation_position="top")
    fig.update_layout(
        title="Operating curves (Part 3 threshold sweep)",
        xaxis_title="Decision threshold",
        yaxis_title="Score",
        height=360,
    )
    return _styled(fig)


def plot_whatif_curve(
    feature_label: str,
    grid: Sequence[float],
    probas: Sequence[float],
    current_value: float,
    current_proba: float,
    threshold: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(grid), y=list(probas), mode="lines",
            line=dict(color=PRIMARY, width=2.5),
            hovertemplate=f"{feature_label}=%{{x:.2f}} · proba=%{{y:.3f}}<extra></extra>",
            name="Predicted prob.",
        )
    )
    fig.add_hline(y=threshold, line=dict(color=DANGER, dash="dash", width=1.5),
                  annotation_text=f"threshold={threshold:.2f}", annotation_position="top right")
    fig.add_trace(
        go.Scatter(
            x=[current_value], y=[current_proba],
            mode="markers", marker=dict(size=12, color=DANGER, line=dict(color="white", width=2)),
            name="Current value", hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=f"What-if: vary {feature_label}",
        xaxis_title=feature_label,
        yaxis_title="Calibrated probability",
        height=320,
        showlegend=False,
    )
    return _styled(fig)


def plot_pca_scatter(pca_df: pd.DataFrame) -> go.Figure:
    """Plotly scatter of PCA-projected samples colored by cluster.

    Expects df with at least PC1, PC2, cluster columns. Falls back gracefully.
    """
    pc1_col = next((c for c in ["PC1", "pc1", "x"] if c in pca_df.columns), None)
    pc2_col = next((c for c in ["PC2", "pc2", "y"] if c in pca_df.columns), None)
    cl_col = next((c for c in ["cluster", "Cluster"] if c in pca_df.columns), None)
    if pc1_col is None or pc2_col is None or cl_col is None:
        return go.Figure()
    fig = go.Figure()
    palette = PLOTLY_COLORWAY
    for i, (k, sub) in enumerate(pca_df.groupby(cl_col)):
        fig.add_trace(
            go.Scatter(
                x=sub[pc1_col], y=sub[pc2_col],
                mode="markers",
                marker=dict(color=palette[i % len(palette)], size=6, opacity=0.65,
                            line=dict(width=0)),
                name=f"Cluster {k}",
                hovertemplate=f"Cluster {k}<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(title="K-Means clusters in PCA space",
                      xaxis_title="PC1", yaxis_title="PC2", height=420)
    return _styled(fig)


def plot_cluster_cvd_rate(profiles: pd.DataFrame) -> go.Figure:
    """Bar of CVD positive rate by cluster from cluster_profiles.csv."""
    if profiles.empty:
        return go.Figure()
    cl = next((c for c in ["cluster", "Cluster"] if c in profiles.columns), profiles.columns[0])
    rate = next(
        (c for c in ["cvd_rate", "cvd_positive_rate", "cvd"] if c in profiles.columns),
        None,
    )
    if rate is None:
        return go.Figure()
    df = profiles.sort_values(cl)
    fig = go.Figure(
        go.Bar(
            x=[f"Cluster {c}" for c in df[cl]],
            y=df[rate],
            marker=dict(color=PRIMARY),
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(title="CVD positive rate by cluster",
                      yaxis_title="CVD positive rate", xaxis_title=None, height=320)
    fig.update_yaxes(tickformat=".0%")
    return _styled(fig)
