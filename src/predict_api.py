"""Thin inference wrapper for the Part 4 dashboard.

Usage from the dashboard process:

    from src.predict_api import predict_with_threshold, load_artifacts

    proba = predict_proba(X_test_already_preprocessed)
    preds, proba = predict_with_threshold(X_test_already_preprocessed)

The model matrices stored in ``outputs/features/X_*.csv`` are already
preprocessed (median imputation, scaling, one-hot encoding) so the dashboard
can pass them directly. To score *raw* NHANES rows, transform them through
``outputs/features/preprocessor.joblib`` first; that pipeline is the same
one Part 2 fit on the training fold and is the only safe way to encode new
participants without leaking test statistics.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "outputs" / "models" / "final_model.joblib"
PREPROCESSOR_PATH = ROOT / "outputs" / "features" / "preprocessor.joblib"
CARD_PATH = ROOT / "outputs" / "models" / "final_model_card.json"


@lru_cache(maxsize=1)
def load_artifacts():
    """Load preprocessor, final model, and model card. Cached after first call."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found at {MODEL_PATH}. "
            "Run `python src/supervised_models.py` first."
        )
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    card = json.load(open(CARD_PATH))
    return preprocessor, model, card


def _select_features(X: pd.DataFrame, card: dict) -> pd.DataFrame:
    """Subset / reorder columns to match the feature set the winner was trained on."""
    cols = card["feature_columns_used"]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required columns "
            f"(first few: {missing[:5]})."
        )
    return X[cols]


def predict_proba(X: pd.DataFrame) -> np.ndarray:
    """Probability of CVD positive class for already-preprocessed input.

    Pass a DataFrame whose columns are the 81 ``num__`` / ``cat__`` features
    produced by ``outputs/features/preprocessor.joblib`` — the function will
    drop any columns the winner doesn't use and reorder the rest.
    """
    _, model, card = load_artifacts()
    Xs = _select_features(X, card)
    return model.predict_proba(Xs)[:, 1]


def predict_with_threshold(
    X: pd.DataFrame, threshold: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary_predictions, probabilities).

    Uses the model card's ``recommended_threshold`` when ``threshold`` is None.
    The card's threshold targets sensitivity >= 80% on out-of-fold training
    predictions (clinical screening criterion); pass a different value to
    move to a higher-precision operating point.
    """
    _, _, card = load_artifacts()
    proba = predict_proba(X)
    t = float(threshold) if threshold is not None else float(card["recommended_threshold"])
    return (proba >= t).astype(int), proba


def predict_from_raw(X_raw: pd.DataFrame) -> np.ndarray:
    """Score *unpreprocessed* engineered features by running them through the saved preprocessor first.

    ``X_raw`` should have the columns expected by Part 2's preprocessor —
    i.e. the same shape as ``data/processed/analytic_dataset_engineered_train.csv``
    *before* the ColumnTransformer runs. The function applies
    ``preprocessor.transform`` and then ``predict_proba``.
    """
    preprocessor, _, card = load_artifacts()
    arr = preprocessor.transform(X_raw)
    full_cols = json.load(open(ROOT / "outputs/features/feature_names.json"))
    df = pd.DataFrame(arr, columns=full_cols, index=X_raw.index)
    return predict_proba(df)


if __name__ == "__main__":  # smoke test
    pre, model, card = load_artifacts()
    print("Loaded model:", card.get("model"), "/", card.get("feature_set"))
    print(f"  recommended_threshold = {card['recommended_threshold']}")
    print(f"  test ROC-AUC = {card['roc_auc']:.4f}")
    X = pd.read_csv(ROOT / "outputs/features/X_test.csv").head(5)
    preds, proba = predict_with_threshold(X)
    for i, (p, pr) in enumerate(zip(preds, proba)):
        print(f"  row {i}: pred={p}, proba={pr:.4f}")
