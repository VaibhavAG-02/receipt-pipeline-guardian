"""Train and honestly evaluate the anomaly scorer.

Metric choice matters more than model choice here. The positive class is ~6%,
so plain accuracy is useless and ROC-AUC is flattering. The headline numbers
are **PR-AUC** and **precision@k**, because the operational question is "of the
N receipts we can actually afford to review today, how many are real defects?"

Per-anomaly-type recall is reported too, so a model that nails arithmetic
errors while missing every duplicate can't hide behind a good average.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .config import FEATURE_LIST_PATH, METRICS_PATH, MODEL_PATH, ensure_dirs
from .features import FEATURE_COLUMNS, time_split


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    """Precision within the top k_frac of scores -- the review-queue metric."""
    n = max(1, int(len(scores) * k_frac))
    idx = np.argsort(-scores)[:n]
    return float(np.mean(y_true[idx]))


def train(features: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    ensure_dirs()
    train_df, test_df = time_split(features)

    X_tr = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_tr = train_df["is_anomaly"].to_numpy(dtype=int)
    X_te = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_te = test_df["is_anomaly"].to_numpy(dtype=int)

    pos = max(1, int(y_tr.sum()))
    neg = max(1, int((1 - y_tr).sum()))

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        min_child_weight=2,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(X_tr, y_tr)

    scores = model.predict_proba(X_te)[:, 1]

    # Operating threshold picked on the training split, not the test split.
    tr_scores = model.predict_proba(X_tr)[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, tr_scores)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    threshold = float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5

    flagged = scores >= threshold
    tp = int(((flagged == 1) & (y_te == 1)).sum())
    fp = int(((flagged == 1) & (y_te == 0)).sum())
    fn = int(((flagged == 0) & (y_te == 1)).sum())

    by_type: dict[str, dict[str, Any]] = {}
    if "anomaly_type" in test_df.columns:
        types = test_df["anomaly_type"].to_numpy()
        for t in sorted({t for t in types if isinstance(t, str)}):
            mask = types == t
            by_type[t] = {
                "n": int(mask.sum()),
                "recall_at_threshold": round(float(flagged[mask].mean()), 4),
            }

    metrics = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "positive_rate_train": round(float(y_tr.mean()), 4),
        "positive_rate_test": round(float(y_te.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_te, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_te, scores)), 4),
        "precision_at_1pct": round(precision_at_k(y_te, scores, 0.01), 4),
        "precision_at_5pct": round(precision_at_k(y_te, scores, 0.05), 4),
        "precision_at_10pct": round(precision_at_k(y_te, scores, 0.10), 4),
        "threshold": round(threshold, 4),
        "precision_at_threshold": round(tp / max(1, tp + fp), 4),
        "recall_at_threshold": round(tp / max(1, tp + fn), 4),
        "recall_by_anomaly_type": by_type,
        "feature_importance": {
            c: round(float(v), 5)
            for c, v in sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_, strict=True),
                key=lambda kv: -kv[1],
            )
        },
    }

    model.save_model(MODEL_PATH)
    FEATURE_LIST_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def score(features: pd.DataFrame) -> pd.DataFrame:
    """Attach anomaly_score to a feature frame using the persisted model."""
    model = load_model()
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    threshold = float(metrics.get("threshold", 0.5))
    X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    out = features[["receipt_id", "submitted_at"]].copy()
    out["anomaly_score"] = model.predict_proba(X)[:, 1]
    out["is_flagged"] = (out["anomaly_score"] >= threshold).astype(int)
    return out
