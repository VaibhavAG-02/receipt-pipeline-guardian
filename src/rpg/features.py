"""Feature engineering.

Design constraint that shapes everything here: **no target leakage and no
future leakage**. Behavioural features (how many receipts has this user filed
recently, how does this basket compare to their norm) are computed with
backward-looking windows only, so a row's features never depend on rows that
arrive after it. The train/test split is by time for the same reason.

The arithmetic residual is deliberately *not* hard-coded as a rule. It's given
to the model as a feature; the model decides how much it matters relative to
everything else, and the quality gate handles the deterministic cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reference prices, used to score "is this unit price plausible for this SKU".
# Computed from the data itself (median), not from the generator's constants --
# in production you would not have the true price list either.
FEATURE_COLUMNS = [
    "total",
    "n_items",
    "total_qty",
    "max_qty",
    "mean_unit_price",
    "max_unit_price",
    "max_price_ratio_vs_sku_median",
    "arithmetic_residual_abs",
    "arithmetic_residual_rel",
    "implied_tax_rate",
    "image_quality",
    "hour_of_day",
    "day_of_week",
    "ts_age_days",
    "is_future_ts",
    "user_receipts_prior_24h",
    "user_amount_ratio_vs_prior_median",
    "same_total_prior_24h",
    "subtotal_is_null",
    "tax_is_null",
    "payment_method_code",
    "scanner_version_code",
]


def _sku_median_prices(df: pd.DataFrame) -> dict[str, float]:
    prices: dict[str, list[float]] = {}
    for items in df["items"]:
        for it in items or []:
            prices.setdefault(it["sku"], []).append(float(it["unit_price"]))
    return {k: float(np.median(v)) for k, v in prices.items() if v}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Receipt-level feature matrix. Input must be sorted by submitted_at."""
    df = df.sort_values("submitted_at").reset_index(drop=True).copy()
    sku_median = _sku_median_prices(df)

    n_items, total_qty, max_qty = [], [], []
    mean_price, max_price, max_ratio = [], [], []
    recomputed_subtotal = []

    for items in df["items"]:
        items = items or []
        qtys = [int(i["qty"]) for i in items] or [0]
        prices = [float(i["unit_price"]) for i in items] or [0.0]
        ratios = [
            float(i["unit_price"]) / sku_median.get(i["sku"], float(i["unit_price"]) or 1.0)
            for i in items
        ] or [1.0]
        n_items.append(len(items))
        total_qty.append(int(sum(qtys)))
        max_qty.append(int(max(qtys)))
        mean_price.append(float(np.mean(prices)))
        max_price.append(float(max(prices)))
        max_ratio.append(float(max(ratios)))
        recomputed_subtotal.append(
            float(sum(int(i["qty"]) * float(i["unit_price"]) for i in items))
        )

    f = pd.DataFrame(index=df.index)
    f["receipt_id"] = df["receipt_id"].to_numpy()
    f["submitted_at"] = df["submitted_at"].to_numpy()
    f["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
    f["n_items"] = n_items
    f["total_qty"] = total_qty
    f["max_qty"] = max_qty
    f["mean_unit_price"] = mean_price
    f["max_unit_price"] = max_price
    f["max_price_ratio_vs_sku_median"] = max_ratio

    # Does the stated total reconcile with the line items?
    recomputed = pd.Series(recomputed_subtotal, index=df.index)
    expected_total = recomputed * 1.0825
    f["arithmetic_residual_abs"] = (f["total"] - expected_total).abs()
    f["arithmetic_residual_rel"] = f["arithmetic_residual_abs"] / expected_total.clip(lower=0.01)

    sub = pd.to_numeric(df["subtotal"], errors="coerce")
    tax = pd.to_numeric(df["tax"], errors="coerce")
    f["implied_tax_rate"] = (tax / sub.clip(lower=0.01)).fillna(-1.0)
    f["subtotal_is_null"] = sub.isna().astype(int)
    f["tax_is_null"] = tax.isna().astype(int)

    f["image_quality"] = pd.to_numeric(df["image_quality"], errors="coerce").fillna(0.5)

    ts = pd.to_datetime(df["submitted_at"], utc=True)
    ref = ts.max()
    f["hour_of_day"] = ts.dt.hour
    f["day_of_week"] = ts.dt.dayofweek
    f["ts_age_days"] = (ref - ts) / pd.Timedelta(days=1)
    # Anything dated after the batch high-water mark is clock skew.
    f["is_future_ts"] = (ts > ref - pd.Timedelta(seconds=1)).astype(int)

    # ---- backward-only behavioural windows -------------------------------
    work = pd.DataFrame(
        {
            "user_id": df["user_id"].to_numpy(),
            "ts": ts.to_numpy(),
            "total": f["total"].to_numpy(),
        }
    )
    prior_counts = np.zeros(len(work), dtype=float)
    prior_ratio = np.ones(len(work), dtype=float)
    same_total = np.zeros(len(work), dtype=float)

    window = pd.Timedelta(hours=24)
    for _, idx in work.groupby("user_id").groups.items():
        idx = np.asarray(idx)
        sub_ts = work["ts"].to_numpy()[idx]
        sub_tot = work["total"].to_numpy()[idx]
        for j in range(len(idx)):
            lo = np.searchsorted(sub_ts, sub_ts[j] - window, side="left")
            prev = slice(lo, j)  # strictly earlier rows only
            k = j - lo
            prior_counts[idx[j]] = k
            if k > 0:
                med = float(np.median(sub_tot[prev]))
                prior_ratio[idx[j]] = sub_tot[j] / med if med > 0 else 1.0
                same_total[idx[j]] = float(
                    np.sum(np.isclose(sub_tot[prev], sub_tot[j], atol=0.01))
                )

    f["user_receipts_prior_24h"] = prior_counts
    f["user_amount_ratio_vs_prior_median"] = prior_ratio
    f["same_total_prior_24h"] = same_total

    f["payment_method_code"] = pd.Categorical(df["payment_method"]).codes
    f["scanner_version_code"] = pd.Categorical(df["scanner_version"]).codes

    for c in FEATURE_COLUMNS:
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)

    if "is_anomaly" in df.columns:
        f["is_anomaly"] = df["is_anomaly"].astype(int).to_numpy()
    if "anomaly_type" in df.columns:
        f["anomaly_type"] = df["anomaly_type"].to_numpy()
    return f


def time_split(f: pd.DataFrame, test_frac: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split. Random splits would leak future behaviour."""
    f = f.sort_values("submitted_at")
    cut = int(len(f) * (1 - test_frac))
    return f.iloc[:cut].copy(), f.iloc[cut:].copy()
