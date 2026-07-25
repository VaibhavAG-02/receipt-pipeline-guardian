"""Per-receipt SHAP explanations.

`feature_importances_` tells you which features mattered *on average across the
training set*. That is close to useless to a reviewer holding one receipt: the
question is why **this** receipt scored 0.94, and the global answer is the same
for every row.

TreeSHAP gives the per-row decomposition, is exact for tree ensembles (not a
sampled approximation), and is fast enough to run over the whole flagged set.
The top contributing features are written into the review queue so a reviewer
sees the reason next to the score rather than in a separate notebook.

Signed contributions are kept, not absolute values. "arithmetic residual pushed
this *up*" and "prior user history pushed it *down*" are different facts and
collapsing them loses the direction a reviewer needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


def explain(model, features: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Return per-receipt top-k signed SHAP contributions.

    Falls back to an empty explanation frame if shap is unavailable, so the
    pipeline degrades rather than breaking on an optional dependency.
    """
    try:
        import shap
    except ImportError:  # pragma: no cover - optional dependency
        return pd.DataFrame(
            {
                "receipt_id": features["receipt_id"],
                "top_factors": ["shap not installed"] * len(features),
                "shap_base_value": np.nan,
            }
        )

    X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    base = float(np.ravel(explainer.expected_value)[0])

    # Rank by magnitude, then report with sign preserved.
    order = np.argsort(-np.abs(values), axis=1)[:, :top_k]

    factors = []
    for row_i, cols in enumerate(order):
        parts = []
        for c in cols:
            direction = "+" if values[row_i, c] >= 0 else "-"
            parts.append(f"{direction}{FEATURE_COLUMNS[c]} ({values[row_i, c]:+.3f})")
        factors.append("; ".join(parts))

    out = pd.DataFrame(
        {
            "receipt_id": features["receipt_id"].to_numpy(),
            "top_factors": factors,
            "shap_base_value": base,
        }
    )
    for rank in range(top_k):
        out[f"factor_{rank + 1}"] = [
            FEATURE_COLUMNS[order[i, rank]] for i in range(len(order))
        ]
        out[f"factor_{rank + 1}_value"] = [
            round(float(values[i, order[i, rank]]), 5) for i in range(len(order))
        ]
    return out


def global_importance(model, features: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature -- the honest global ranking.

    Kept separate from the per-row view and reported alongside it, because gain
    based `feature_importances_` and mean |SHAP| routinely disagree, and the
    disagreement is itself informative.
    """
    try:
        import shap
    except ImportError:  # pragma: no cover
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    values = shap.TreeExplainer(model).shap_values(X)
    return (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "mean_abs_shap": np.abs(values).mean(axis=0).round(5),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
