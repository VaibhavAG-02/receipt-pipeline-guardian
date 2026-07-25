"""Drift detection and fairness slicing for the retraining loop.

Three distinct questions, deliberately separated because they fail
independently and have different remedies:

  1. **Feature drift** -- has the input distribution moved? Population Stability
     Index per feature against the training reference. PSI is used rather than a
     KS test because it is bucketed and therefore stable on the mixed
     continuous/discrete features here, and because its conventional thresholds
     (0.1 / 0.25) give operators a shared vocabulary.

  2. **Prediction drift** -- has the *score* distribution moved? This can shift
     while inputs look stable (the model is extrapolating) and is the earlier
     warning of the two.

  3. **Fairness / slice degradation** -- is performance uniform across
     populations? An aggregate PR-AUC can improve while a segment collapses.
     Slices here are operational (scanner version, payment method, store
     format, image-quality band) rather than protected attributes, because
     receipts carry no demographics -- but the mechanism is identical and this
     is where a protected attribute would plug in.

Retraining is gated on all three. A model that improves on average while
degrading for a slice is blocked, not shipped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Conventional PSI bands. Widely used in credit risk; stated explicitly rather
# than buried so the thresholds can be argued with.
PSI_NO_DRIFT = 0.10
PSI_MODERATE = 0.25

# A slice may not fall more than this far below the overall recall before the
# candidate model is blocked.
MAX_SLICE_RECALL_GAP = 0.20
MIN_SLICE_SIZE = 50


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between two samples using quantile bins from the reference.

    Bins come from the reference so the metric answers "where did the new data
    land relative to what we trained on", not "how do two arbitrary binnings
    compare". Empty buckets are floored rather than dropped; dropping them
    silently understates drift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:  # near-constant feature: PSI is not meaningful
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    cur_pct = np.histogram(current, bins=edges)[0] / current.size

    floor = 1e-6
    ref_pct = np.clip(ref_pct, floor, None)
    cur_pct = np.clip(cur_pct, floor, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify(psi: float) -> str:
    if psi < PSI_NO_DRIFT:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate"
    return "significant"


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    drifted_features: list[str]
    prediction_psi: float
    prediction_status: str
    retrain_recommended: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str],
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
) -> DriftReport:
    psi = {
        c: round(population_stability_index(reference[c].to_numpy(), current[c].to_numpy()), 4)
        for c in feature_columns
        if c in reference.columns and c in current.columns
    }
    drifted = sorted(
        [c for c, v in psi.items() if v >= PSI_MODERATE], key=lambda c: -psi[c]
    )

    pred_psi = 0.0
    if reference_scores is not None and current_scores is not None:
        pred_psi = round(
            population_stability_index(reference_scores, current_scores), 4
        )

    reasons: list[str] = []
    if drifted:
        reasons.append(
            f"{len(drifted)} feature(s) past PSI {PSI_MODERATE}: {', '.join(drifted[:5])}"
        )
    if pred_psi >= PSI_MODERATE:
        reasons.append(f"prediction distribution PSI {pred_psi} >= {PSI_MODERATE}")

    return DriftReport(
        feature_psi=psi,
        drifted_features=drifted,
        prediction_psi=pred_psi,
        prediction_status=classify(pred_psi),
        retrain_recommended=bool(reasons),
        reasons=reasons,
    )


# ------------------------------------------------------------ fairness ----
def _image_quality_band(v: float) -> str:
    if v < 0.5:
        return "poor"
    if v < 0.75:
        return "fair"
    return "good"


SLICE_SPECS = {
    "scanner_version": lambda d: d["scanner_version"],
    "payment_method": lambda d: d["payment_method"],
    "store_format": lambda d: d.get("store_format", pd.Series("unknown", index=d.index)),
    "image_quality_band": lambda d: d["image_quality"].map(_image_quality_band),
}


def slice_metrics(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_flagged: np.ndarray,
) -> pd.DataFrame:
    """Recall and flag rate per slice value, with sample sizes attached.

    Sample size is reported alongside every number on purpose: a slice of 11
    receipts producing 0.0 recall is noise, and a table that hides n invites
    someone to act on it.
    """
    rows = []
    overall_recall = float(y_flagged[y_true == 1].mean()) if (y_true == 1).any() else 0.0

    for slice_name, getter in SLICE_SPECS.items():
        if slice_name not in frame.columns and slice_name != "image_quality_band":
            continue
        try:
            values = getter(frame)
        except KeyError:
            continue
        for val in sorted(pd.unique(values.dropna())):
            mask = (values == val).to_numpy()
            n = int(mask.sum())
            pos = int(y_true[mask].sum())
            recall = float(y_flagged[mask & (y_true == 1)].mean()) if pos else float("nan")
            rows.append(
                {
                    "slice": slice_name,
                    "value": str(val),
                    "n": n,
                    "n_positive": pos,
                    "recall": None if pos == 0 else round(recall, 4),
                    "flag_rate": round(float(y_flagged[mask].mean()), 4) if n else 0.0,
                    "recall_gap_vs_overall": (
                        None if pos == 0 else round(overall_recall - recall, 4)
                    ),
                    "reliable": n >= MIN_SLICE_SIZE and pos >= 10,
                }
            )
    return pd.DataFrame(rows)


def fairness_gate(slices: pd.DataFrame) -> tuple[bool, list[str]]:
    """Block promotion if any adequately-sized slice lags the overall recall."""
    if slices.empty:
        return True, []
    violations = []
    for _, r in slices.iterrows():
        if not r["reliable"] or r["recall_gap_vs_overall"] is None:
            continue
        if r["recall_gap_vs_overall"] > MAX_SLICE_RECALL_GAP:
            violations.append(
                f"{r['slice']}={r['value']} recall {r['recall']} "
                f"lags overall by {r['recall_gap_vs_overall']} (n={r['n']})"
            )
    return (not violations), violations


def write_report(path, report: DriftReport, slices: pd.DataFrame) -> None:
    payload = report.as_dict()
    payload["slices"] = json.loads(slices.to_json(orient="records"))
    ok, violations = fairness_gate(slices)
    payload["fairness_passed"] = ok
    payload["fairness_violations"] = violations
    path.write_text(json.dumps(payload, indent=2))
