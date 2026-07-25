"""End-to-end local run: generate -> gate -> features -> train -> score -> load.

This is the path that needs no services at all. The Kafka path (stream.py)
replaces only the first step; everything downstream is identical, which is the
point -- the streaming and batch paths converge on the same landing zone.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
import pandas as pd

from . import drift as drift_mod
from . import quality
from .config import (
    ARTIFACTS,
    METRICS_PATH,
    MODEL_PATH,
    QUARANTINE_DIR,
    RAW_DIR,
    WAREHOUSE,
    ensure_dirs,
)
from .explain import explain, global_importance
from .features import FEATURE_COLUMNS, build_features, time_split
from .generate import GenConfig, generate, generate_store_master
from .train import load_model, train
from .train import score as score_features


def _flatten_items(df: pd.DataFrame) -> pd.DataFrame:
    """Explode nested items into a tidy line-item table for the warehouse."""
    rows = []
    for rid, items in zip(df["receipt_id"], df["items"], strict=True):
        for pos, it in enumerate(items or []):
            rows.append(
                {
                    "receipt_id": rid,
                    "line_no": pos + 1,
                    "sku": it["sku"],
                    "item_name": it["name"],
                    "qty": int(it["qty"]),
                    "unit_price": float(it["unit_price"]),
                }
            )
    return pd.DataFrame(rows)


def run(
    n_receipts: int = 20_000,
    seed: int = 42,
    retrain: bool = True,
) -> dict[str, Any]:
    """Build the whole warehouse from scratch. Returns a run summary."""
    ensure_dirs()

    gen_cfg = GenConfig(n_receipts=n_receipts, seed=seed)
    raw = generate(gen_cfg)

    # The batch's business date. Anything stamped after this is device clock
    # skew, not data. Passing it explicitly keeps replays deterministic.
    as_of = pd.Timestamp(gen_cfg.end)
    clean, quarantined = quality.split_quarantine(raw, as_of=as_of)
    gate = quality.evaluate_gates(clean, quarantined)

    # Quarantined rows are persisted, never dropped.
    q_out = quarantined.drop(columns=["items"], errors="ignore")
    q_out.to_parquet(QUARANTINE_DIR / "quarantine.parquet", index=False)

    receipts = clean.drop(columns=["items"]).copy()
    items = _flatten_items(clean)
    receipts.to_parquet(RAW_DIR / "receipts.parquet", index=False)
    items.to_parquet(RAW_DIR / "receipt_items.parquet", index=False)

    # Effective-dated store attributes: the source for the Type 2 dimension.
    store_master = generate_store_master(seed=seed, days=gen_cfg.days, end=gen_cfg.end)
    store_master.to_parquet(RAW_DIR / "store_master.parquet", index=False)

    feats = build_features(clean)

    if retrain or not MODEL_PATH.exists():
        metrics = train(feats)
    else:
        metrics = json.loads(METRICS_PATH.read_text())

    scored = score_features(feats)
    scored.to_parquet(RAW_DIR / "scores.parquet", index=False)

    # ---- drift + fairness on the hold-out, the retraining gate -----------
    model = load_model()
    train_f, test_f = time_split(feats)
    ref_scores = model.predict_proba(train_f[FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]
    cur_scores = model.predict_proba(test_f[FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]
    drift_report = drift_mod.detect_drift(
        train_f, test_f, FEATURE_COLUMNS, ref_scores, cur_scores
    )

    # Slice metrics need the categorical context, which lives on the raw rows.
    ctx = clean.set_index("receipt_id").loc[test_f["receipt_id"]].reset_index()
    ctx = ctx.merge(
        store_master.sort_values("effective_from")
        .groupby("store_id")
        .last()
        .reset_index()[["store_id", "store_format"]],
        on="store_id",
        how="left",
    )
    threshold = float(metrics["threshold"])
    y_true = test_f["is_anomaly"].to_numpy(dtype=int)
    y_flag = (cur_scores >= threshold).astype(int)
    slices = drift_mod.slice_metrics(ctx, y_true, y_flag)
    fairness_ok, fairness_violations = drift_mod.fairness_gate(slices)
    drift_mod.write_report(ARTIFACTS / "drift_report.json", drift_report, slices)
    slices.to_parquet(RAW_DIR / "slice_metrics.parquet", index=False)

    # ---- per-receipt explanations ---------------------------------------
    explanations = explain(model, feats)
    explanations.to_parquet(RAW_DIR / "explanations.parquet", index=False)
    shap_global = global_importance(model, test_f)
    shap_global.to_parquet(RAW_DIR / "shap_global.parquet", index=False)

    # Land everything in DuckDB. dbt builds its models on top of these.
    con = duckdb.connect(str(WAREHOUSE))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        for name, frame in (
            ("receipts", receipts),
            ("receipt_items", items),
            ("scores", scored),
            ("quarantine", q_out),
            ("store_master", store_master),
            ("explanations", explanations),
            ("slice_metrics", slices),
            ("shap_global", shap_global),
        ):
            con.register("tmp_df", frame)
            con.execute(f"CREATE OR REPLACE TABLE raw.{name} AS SELECT * FROM tmp_df")
            con.unregister("tmp_df")
    finally:
        con.close()

    summary = {
        "gate": gate.as_dict(),
        "metrics": metrics,
        "drift": drift_report.as_dict(),
        "fairness_passed": fairness_ok,
        "fairness_violations": fairness_violations,
        "n_receipts_clean": int(len(receipts)),
        "n_receipts_quarantined": int(len(q_out)),
        "n_line_items": int(len(items)),
    }
    (ARTIFACTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run the Receipt Pipeline Guardian")
    ap.add_argument("--receipts", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-retrain", action="store_true")
    a = ap.parse_args()

    s = run(n_receipts=a.receipts, seed=a.seed, retrain=not a.no_retrain)
    g = s["gate"]
    m = s["metrics"]
    print(f"gate passed={g['passed']}  quarantined={g['n_quarantined']} "
          f"({g['quarantine_rate']:.2%})")
    if g["failures"]:
        print("  failures:", "; ".join(g["failures"]))
    print(f"PR-AUC={m['pr_auc']}  ROC-AUC={m['roc_auc']}  "
          f"P@5%={m['precision_at_5pct']}")
    d = s["drift"]
    print(f"drift: retrain_recommended={d['retrain_recommended']} "
          f"pred_psi={d['prediction_psi']} drifted={len(d['drifted_features'])}")
    print(f"fairness gate passed={s['fairness_passed']}")
    for v in s["fairness_violations"]:
        print("  !", v)
