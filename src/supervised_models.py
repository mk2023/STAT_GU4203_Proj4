"""Part 3 — Supervised modelling.

4 models (LogReg / RandomForest / XGBoost / CatBoost) x 3 feature subsets
(A=full, B=no diagnosis history, C=raw measurements only) = 12 experiments.
Picks the best by mean CV ROC-AUC, calibrates and selects an operating
threshold, and writes diagnostics to outputs/models/.

Run from repo root:
    python3 src/supervised_models.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import sys
sys.stdout.reconfigure(line_buffering=True)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "models"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ITER = 20
CV_SPLITS = 5

CV = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "brier": "neg_brier_score",
}

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
X_train = pd.read_csv(ROOT / "outputs/features/X_train.csv")
X_test = pd.read_csv(ROOT / "outputs/features/X_test.csv")
y_train = pd.read_csv(ROOT / "outputs/features/y_train.csv")["cvd"].astype(int)
y_test = pd.read_csv(ROOT / "outputs/features/y_test.csv")["cvd"].astype(int)
EXCLUSIONS = json.load(
    open(ROOT / "outputs/features/sensitivity_feature_exclusions.json")
)

print(f"  X_train: {X_train.shape}, positives {y_train.sum()}/{len(y_train)} "
      f"({y_train.mean():.1%})")
print(f"  X_test:  {X_test.shape}, positives {y_test.sum()}/{len(y_test)} "
      f"({y_test.mean():.1%})")

NO_DX_COL = "num__lifestyle_risk_score_no_dx"


def get_feature_subset(X: pd.DataFrame, feat_set: str) -> pd.DataFrame:
    """A: full minus no_dx (80). B: drop diagnosis_history_columns (79). C: drop columns_to_drop (72)."""
    if feat_set == "A":
        return X.drop(columns=[NO_DX_COL])
    if feat_set == "B":
        return X.drop(columns=EXCLUSIONS["diagnosis_history_columns"])
    if feat_set == "C":
        return X.drop(columns=EXCLUSIONS["columns_to_drop"])
    raise ValueError(f"Unknown feature set {feat_set}")


for s in "ABC":
    sub = get_feature_subset(X_train, s)
    print(f"  Feature set {s}: {sub.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Model factory
# ---------------------------------------------------------------------------
SCALE_POS_WEIGHT = float((y_train == 0).sum()) / float((y_train == 1).sum())
print(f"  scale_pos_weight = {SCALE_POS_WEIGHT:.3f}")


def make_model_specs():
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    return {
        "LogReg": {
            "estimator": LogisticRegression(
                class_weight="balanced",
                penalty="l2",
                solver="liblinear",
                random_state=RANDOM_STATE,
                max_iter=2000,
            ),
            "params": {"C": [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
            "search": "grid",
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,  # outer search uses n_jobs=-1
            ),
            "params": {
                "n_estimators": randint(200, 800),
                "max_depth": [None, 6, 10, 14, 20],
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", 0.5],
            },
            "search": "random",
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                scale_pos_weight=SCALE_POS_WEIGHT,
                tree_method="hist",
                eval_metric="auc",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            "params": {
                "n_estimators": randint(200, 800),
                "max_depth": randint(3, 9),
                "learning_rate": loguniform(0.01, 0.3),
                "reg_lambda": loguniform(0.1, 10.0),
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
            "search": "random",
        },
        "CatBoost": {
            "estimator": CatBoostClassifier(
                auto_class_weights="Balanced",
                eval_metric="AUC",
                random_seed=RANDOM_STATE,
                thread_count=1,
                task_type="CPU",
                verbose=0,
                allow_writing_files=False,
            ),
            "params": {
                "iterations": [400, 600, 800, 1000],
                "depth": randint(4, 9),
                "learning_rate": loguniform(0.01, 0.3),
                "l2_leaf_reg": loguniform(1.0, 10.0),
            },
            "search": "random",
        },
    }


# ---------------------------------------------------------------------------
# 3. Run 12 experiments
# ---------------------------------------------------------------------------
records: list[dict] = []
fitted_models: dict[tuple[str, str], object] = {}

print("\nRunning 12 experiments (4 models x 3 feature sets)...")
for feat_set in ["A", "B", "C"]:
    Xs_tr = get_feature_subset(X_train, feat_set)
    print(f"\n[Feature set {feat_set}] {Xs_tr.shape[1]} columns")
    specs = make_model_specs()
    for name, spec in specs.items():
        t0 = time.time()
        if spec["search"] == "grid":
            search = GridSearchCV(
                spec["estimator"],
                spec["params"],
                cv=CV,
                scoring=SCORING,
                refit="roc_auc",
                n_jobs=-1,
            )
        else:
            search = RandomizedSearchCV(
                spec["estimator"],
                spec["params"],
                n_iter=N_ITER,
                cv=CV,
                scoring=SCORING,
                refit="roc_auc",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        search.fit(Xs_tr, y_train)
        bi = search.best_index_
        elapsed = time.time() - t0
        rec = {
            "model": name,
            "feature_set": feat_set,
            "n_features": Xs_tr.shape[1],
            "best_params": dict(search.best_params_),
            "roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][bi]),
            "roc_auc_std": float(search.cv_results_["std_test_roc_auc"][bi]),
            "pr_auc_mean": float(search.cv_results_["mean_test_pr_auc"][bi]),
            "pr_auc_std": float(search.cv_results_["std_test_pr_auc"][bi]),
            "f1_mean": float(search.cv_results_["mean_test_f1"][bi]),
            "f1_std": float(search.cv_results_["std_test_f1"][bi]),
            "brier_mean": float(-search.cv_results_["mean_test_brier"][bi]),
            "brier_std": float(search.cv_results_["std_test_brier"][bi]),
            "fit_seconds": round(elapsed, 1),
        }
        records.append(rec)
        fitted_models[(name, feat_set)] = search.best_estimator_
        print(
            f"  {name:13s} ROC-AUC={rec['roc_auc_mean']:.4f}±{rec['roc_auc_std']:.4f}  "
            f"PR-AUC={rec['pr_auc_mean']:.4f}  F1={rec['f1_mean']:.3f}  "
            f"Brier={rec['brier_mean']:.4f}  ({elapsed:.0f}s)"
        )

cv_df = pd.DataFrame(records)
cv_df.to_csv(OUT / "cv_results.csv", index=False)
print(f"\nWrote {OUT / 'cv_results.csv'}")

# ---------------------------------------------------------------------------
# 4. Pick winner — mean CV ROC-AUC, tie-break by simpler model
# ---------------------------------------------------------------------------
SIMPLICITY = {"LogReg": 0, "RandomForest": 1, "XGBoost": 2, "CatBoost": 3}
cv_df["_simplicity"] = cv_df["model"].map(SIMPLICITY)
ranked = cv_df.sort_values(
    ["roc_auc_mean", "pr_auc_mean", "_simplicity"],
    ascending=[False, False, True],
).reset_index(drop=True)
winner = ranked.iloc[0]
print(
    f"\n=== Winner: {winner['model']} on feature set {winner['feature_set']} "
    f"(CV ROC-AUC={winner['roc_auc_mean']:.4f}) ==="
)

best_estimator = fitted_models[(winner["model"], winner["feature_set"])]
Xs_tr_w = get_feature_subset(X_train, winner["feature_set"])
Xs_te_w = get_feature_subset(X_test, winner["feature_set"])

# ---------------------------------------------------------------------------
# 5. Test set evaluation + calibration
# ---------------------------------------------------------------------------
proba_raw = best_estimator.predict_proba(Xs_te_w)[:, 1]
test_metrics = {
    "model": winner["model"],
    "feature_set": winner["feature_set"],
    "n_features": int(winner["n_features"]),
    "roc_auc": float(roc_auc_score(y_test, proba_raw)),
    "pr_auc": float(average_precision_score(y_test, proba_raw)),
    "brier_raw": float(brier_score_loss(y_test, proba_raw)),
    "f1_at_0.5": float(f1_score(y_test, (proba_raw >= 0.5).astype(int))),
}

print("Calibrating (Platt + isotonic) via 5-fold CV on train...")
calibrated_platt = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=5)
calibrated_iso = CalibratedClassifierCV(best_estimator, method="isotonic", cv=5)
calibrated_platt.fit(Xs_tr_w, y_train)
calibrated_iso.fit(Xs_tr_w, y_train)
proba_platt = calibrated_platt.predict_proba(Xs_te_w)[:, 1]
proba_iso = calibrated_iso.predict_proba(Xs_te_w)[:, 1]

calib_options = {"raw": proba_raw, "platt": proba_platt, "isotonic": proba_iso}
calib_briers = {k: brier_score_loss(y_test, v) for k, v in calib_options.items()}
chosen_calib = min(calib_briers, key=calib_briers.get)
test_metrics["brier_platt"] = float(calib_briers["platt"])
test_metrics["brier_isotonic"] = float(calib_briers["isotonic"])
test_metrics["chosen_calibration"] = chosen_calib
print(f"  Best calibration: {chosen_calib} (Brier={calib_briers[chosen_calib]:.4f})")

final_proba = calib_options[chosen_calib]
if chosen_calib == "isotonic":
    final_estimator = calibrated_iso
elif chosen_calib == "platt":
    final_estimator = calibrated_platt
else:
    final_estimator = best_estimator

# ---------------------------------------------------------------------------
# 6. Threshold analysis (OOF on train) + recommended threshold
# ---------------------------------------------------------------------------
print("Threshold analysis on OOF train probabilities (using calibrated model)...")
# Use final_estimator (calibrated or raw, whichever wins on Brier) so the
# threshold is selected on the same probability distribution it will be
# applied to on the test set.
oof_proba = cross_val_predict(
    final_estimator, Xs_tr_w, y_train, cv=CV, method="predict_proba", n_jobs=-1
)[:, 1]

rows = []
for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
    preds = (oof_proba >= t).astype(int)
    tn = int(((preds == 0) & (y_train == 0)).sum())
    fp = int(((preds == 1) & (y_train == 0)).sum())
    rows.append(
        {
            "threshold": float(t),
            "precision": float(precision_score(y_train, preds, zero_division=0)),
            "recall": float(recall_score(y_train, preds, zero_division=0)),
            "f1": float(f1_score(y_train, preds, zero_division=0)),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "n_positive_predicted": int(preds.sum()),
        }
    )
thresh_df = pd.DataFrame(rows)
thresh_df.to_csv(OUT / "threshold_analysis.csv", index=False)

# Recommended threshold: highest specificity at recall >= 0.80 (clinical screening)
candidates = thresh_df[thresh_df["recall"] >= 0.80]
if candidates.empty:
    rec_threshold = 0.50
else:
    rec_threshold = float(
        candidates.sort_values("specificity", ascending=False).iloc[0]["threshold"]
    )
test_metrics["recommended_threshold"] = rec_threshold
print(f"  Recommended threshold: {rec_threshold:.2f} (highest specificity at recall>=0.80)")

# Apply to test
preds_rec = (final_proba >= rec_threshold).astype(int)
test_metrics["test_recall_at_recommended"] = float(recall_score(y_test, preds_rec))
test_metrics["test_precision_at_recommended"] = float(
    precision_score(y_test, preds_rec, zero_division=0)
)
test_metrics["test_f1_at_recommended"] = float(
    f1_score(y_test, preds_rec, zero_division=0)
)

# ---------------------------------------------------------------------------
# 7. Persist winner + card
# ---------------------------------------------------------------------------
joblib.dump(final_estimator, OUT / "final_model.joblib")
model_card = {
    **test_metrics,
    "best_params": winner["best_params"],
    "training_timestamp": pd.Timestamp.now().isoformat(),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "feature_columns_used": list(Xs_tr_w.columns),
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "cv_splits": CV_SPLITS,
    "n_iter_random_search": N_ITER,
    "random_state": RANDOM_STATE,
}
with open(OUT / "final_model_card.json", "w") as f:
    json.dump(model_card, f, indent=2, default=str)
with open(OUT / "test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2, default=str)
print(f"Wrote {OUT / 'final_model.joblib'}, final_model_card.json, test_metrics.json")

# ---------------------------------------------------------------------------
# 8. Plots — ROC + PR for all 12 models on test set
# ---------------------------------------------------------------------------
print("Plotting ROC + PR curves...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
COLOR_MAP = {"LogReg": "C0", "RandomForest": "C1", "XGBoost": "C2", "CatBoost": "C3"}
LINESTYLE = {"A": "-", "B": "--", "C": ":"}
for (name, fset), est in fitted_models.items():
    Xte = get_feature_subset(X_test, fset)
    p = est.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    pre_, rec_, _ = precision_recall_curve(y_test, p)
    auc = roc_auc_score(y_test, p)
    pra = average_precision_score(y_test, p)
    is_winner = name == winner["model"] and fset == winner["feature_set"]
    lw = 3.0 if is_winner else 1.2
    alpha = 1.0 if is_winner else 0.7
    label = f"{name}-{fset} (AUC={auc:.3f}, PR={pra:.3f})"
    if is_winner:
        label += " ★"
    axes[0].plot(fpr, tpr, color=COLOR_MAP[name], linestyle=LINESTYLE[fset],
                 lw=lw, alpha=alpha, label=label)
    axes[1].plot(rec_, pre_, color=COLOR_MAP[name], linestyle=LINESTYLE[fset],
                 lw=lw, alpha=alpha, label=label)
axes[0].plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curves (test set)")
axes[1].axhline(y_test.mean(), ls="--", c="k", lw=0.7, alpha=0.4,
                label=f"prevalence={y_test.mean():.1%}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall curves (test set)")
axes[0].legend(fontsize=7, loc="lower right")
axes[1].legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig(OUT / "roc_pr_curves.png", dpi=140)
plt.close()

# Calibration
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="perfect")
for label_, p in calib_options.items():
    pt, pp = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    ax.plot(pp, pt, marker="o",
            label=f"{label_} (Brier={calib_briers[label_]:.4f})")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title(f"Calibration — {winner['model']} on feature set {winner['feature_set']}")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "calibration.png", dpi=140)
plt.close()

# ---------------------------------------------------------------------------
# 9. SHAP — global feature attribution on test set
# ---------------------------------------------------------------------------
print("Computing SHAP values...")
try:
    import shap

    if winner["model"] == "LogReg":
        explainer = shap.LinearExplainer(best_estimator, Xs_tr_w)
        shap_values = explainer.shap_values(Xs_te_w)
    else:
        explainer = shap.TreeExplainer(best_estimator)
        sv = explainer.shap_values(Xs_te_w)
        shap_values = sv[1] if isinstance(sv, list) else sv

    plt.figure()
    shap.summary_plot(shap_values, Xs_te_w, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values, Xs_te_w, plot_type="bar", show=False, max_display=20
    )
    plt.tight_layout()
    plt.savefig(OUT / "shap_bar.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Wrote SHAP plots ({Xs_te_w.shape[0]} rows, {Xs_te_w.shape[1]} features)")
except Exception as exc:  # pragma: no cover
    print(f"  SHAP failed: {exc!r}")

# ---------------------------------------------------------------------------
# 10. Summary print
# ---------------------------------------------------------------------------
print("\n=== Final summary ===")
print(json.dumps(test_metrics, indent=2, default=str))
print(f"\nAll artifacts under {OUT}")
