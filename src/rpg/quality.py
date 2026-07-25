"""The Guardian: contract checks that run *before* anything reaches the marts.

Two distinct jobs that are easy to conflate:

  1. Row-level contract checks. A row that violates one is quarantined, not
     dropped. Quarantine is a table you can query, because silently discarding
     records is how you end up with metrics nobody can reconcile.

  2. Batch-level gates. Aggregate properties (volume, null rate, freshness,
     quarantine share). Failing one fails the pipeline run, because a batch
     that is 40% garbage should not partially land.

Deliberately *not* using the ML model here. Quality gates must be explainable
to whoever gets paged at 3am, and must keep working when the model is stale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .config import (
    FRESHNESS_MAX_AGE_DAYS,
    MAX_NULL_RATE,
    MAX_QUARANTINE_RATE,
    MIN_ROWS_PER_BATCH,
)
from .generate import STORES

REQUIRED_COLUMNS = [
    "receipt_id",
    "user_id",
    "store_id",
    "submitted_at",
    "currency",
    "total",
    "items",
]

# Rows failing any of these are quarantined with the reason recorded.
CONTRACT_RULES = [
    "missing_required_field",
    "non_positive_total",
    "unknown_store",
    "empty_items",
    "future_timestamp",
    "bad_currency",
]


@dataclass
class GateResult:
    passed: bool
    n_rows: int
    n_quarantined: int
    quarantine_rate: float
    null_rate: float
    max_age_days: float
    failures: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reason(row: pd.Series, as_of: pd.Timestamp) -> str | None:
    for col in REQUIRED_COLUMNS:
        if col not in row or pd.isna(row.get(col)) if col != "items" else False:
            return "missing_required_field"
    if row.get("receipt_id") in (None, "") or row.get("user_id") in (None, ""):
        return "missing_required_field"
    total = row.get("total")
    if total is None or pd.isna(total) or float(total) <= 0:
        return "non_positive_total"
    if row.get("store_id") not in STORES:
        return "unknown_store"
    items = row.get("items")
    if not isinstance(items, (list, tuple)) or len(items) == 0:
        return "empty_items"
    if row.get("currency") != "USD":
        return "bad_currency"
    ts = row.get("submitted_at")
    if ts is not None and not pd.isna(ts) and ts > as_of:
        return "future_timestamp"
    return None


def split_quarantine(
    df: pd.DataFrame, as_of: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean, quarantined). Quarantined rows gain `quarantine_reason`.

    `as_of` is the batch's business date -- the point after which a timestamp
    is clock skew rather than data. It is passed in rather than read from the
    wall clock so that replaying a historical batch gives identical results.
    """
    if df.empty:
        return df.copy(), df.copy().assign(quarantine_reason=pd.Series(dtype=str))

    if as_of is None:
        as_of = pd.Timestamp.utcnow().tz_convert("UTC")
    reasons = df.apply(lambda r: _reason(r, as_of), axis=1)
    bad = reasons.notna()

    quarantined = df[bad].copy()
    quarantined["quarantine_reason"] = reasons[bad].to_numpy()
    clean = df[~bad].copy()
    return clean, quarantined


def evaluate_gates(clean: pd.DataFrame, quarantined: pd.DataFrame) -> GateResult:
    n_total = len(clean) + len(quarantined)
    failures: list[str] = []

    q_rate = (len(quarantined) / n_total) if n_total else 0.0

    # Null rate is measured on the columns that feed the marts. `subtotal` and
    # `tax` are legitimately nullable coming out of OCR, so they're excluded --
    # the marts recompute them from items.
    checked = [c for c in ("receipt_id", "user_id", "store_id", "submitted_at", "total")
               if c in clean.columns]
    null_rate = float(clean[checked].isna().mean().mean()) if len(clean) else 0.0

    if n_total < MIN_ROWS_PER_BATCH:
        failures.append(f"volume: {n_total} rows < {MIN_ROWS_PER_BATCH}")
    if q_rate > MAX_QUARANTINE_RATE:
        failures.append(f"quarantine_rate: {q_rate:.3f} > {MAX_QUARANTINE_RATE}")
    if null_rate > MAX_NULL_RATE:
        failures.append(f"null_rate: {null_rate:.3f} > {MAX_NULL_RATE}")

    if len(clean):
        span = clean["submitted_at"].max() - clean["submitted_at"].min()
        max_age = float(span / pd.Timedelta(days=1))
    else:
        max_age = 0.0

    # Freshness: the newest record in a batch shouldn't be older than the SLA.
    if len(clean):
        newest = clean["submitted_at"].max()
        lag_days = float((clean["submitted_at"].max() - newest) / pd.Timedelta(days=1))
        if lag_days > FRESHNESS_MAX_AGE_DAYS:
            failures.append(f"freshness: {lag_days:.2f}d > {FRESHNESS_MAX_AGE_DAYS}d")

    return GateResult(
        passed=not failures,
        n_rows=n_total,
        n_quarantined=len(quarantined),
        quarantine_rate=round(q_rate, 4),
        null_rate=round(null_rate, 4),
        max_age_days=round(max_age, 2),
        failures=failures,
    )
