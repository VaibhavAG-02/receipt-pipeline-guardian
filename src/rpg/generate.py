"""Synthetic receipt stream with *labelled* defects.

Why synthetic: a public repo can't ship real receipt data, and a demo whose
labels come from the same heuristics the model learns is circular. Here the
generator decides ground truth independently, then corrupts the record. The
model never sees the label-generating rule, only the resulting fields, so the
evaluation is meaningful even though the data is fake.

Every anomaly type mirrors a failure that actually happens in receipt-scanning
pipelines: OCR misreads, double submissions, unit confusion, clock skew.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from .config import ANOMALY_RATE, DEFAULT_N_RECEIPTS, DEFAULT_SEED

STORES = [f"ST{str(i).zfill(3)}" for i in range(1, 41)]
PAYMENT_METHODS = ["card", "cash", "wallet", "ebt"]
SCANNER_VERSIONS = ["v3.1", "v3.2", "v4.0", "v4.1"]

CATALOG: list[tuple[str, str, float]] = [
    ("SKU1001", "Whole Milk 1gal", 4.29),
    ("SKU1002", "Large Eggs 12ct", 3.89),
    ("SKU1003", "Sourdough Loaf", 5.49),
    ("SKU1004", "Bananas lb", 0.62),
    ("SKU1005", "Chicken Breast lb", 7.15),
    ("SKU1006", "Cheddar Block 8oz", 4.75),
    ("SKU1007", "Ground Coffee 12oz", 9.99),
    ("SKU1008", "Paper Towels 6pk", 12.49),
    ("SKU1009", "Olive Oil 500ml", 11.25),
    ("SKU1010", "Pasta 16oz", 1.99),
    ("SKU1011", "Tomato Sauce 24oz", 2.65),
    ("SKU1012", "Orange Juice 52oz", 4.99),
    ("SKU1013", "Greek Yogurt 32oz", 6.29),
    ("SKU1014", "Frozen Peas 16oz", 2.19),
    ("SKU1015", "Dish Soap 25oz", 3.99),
]

ANOMALY_TYPES = [
    "arithmetic_mismatch",
    "duplicate_submission",
    "impossible_quantity",
    "price_outlier",
    "timestamp_skew",
    "ocr_dropout",
]

# Hard schema violations from a misbehaving client build. These are handled by
# the deterministic contract gate, never by the model -- you don't need ML to
# know a total of -12.00 is wrong.
MALFORMED_RATE = 0.015

TAX_RATE = 0.0825


@dataclass
class GenConfig:
    n_receipts: int = DEFAULT_N_RECEIPTS
    seed: int = DEFAULT_SEED
    anomaly_rate: float = ANOMALY_RATE
    n_users: int = 2_500
    malformed_rate: float = MALFORMED_RATE
    days: int = 45
    end: datetime = field(
        default_factory=lambda: datetime(2026, 6, 30, tzinfo=timezone.utc)
    )


def _round2(x: float) -> float:
    return float(round(x + 1e-9, 2))


def _base_receipt(rng: random.Random, cfg: GenConfig) -> dict[str, Any]:
    n_items = rng.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[6, 10, 14, 16, 14, 10, 6, 4])[0]
    picks = rng.sample(CATALOG, k=min(n_items, len(CATALOG)))
    items = []
    for sku, name, base_price in picks:
        qty = rng.choices([1, 2, 3, 4], weights=[62, 24, 10, 4])[0]
        # small honest price drift: promos, regional pricing
        unit_price = _round2(base_price * rng.uniform(0.92, 1.10))
        items.append(
            {"sku": sku, "name": name, "qty": qty, "unit_price": unit_price}
        )

    subtotal = _round2(sum(i["qty"] * i["unit_price"] for i in items))
    tax = _round2(subtotal * TAX_RATE)
    total = _round2(subtotal + tax)

    offset_days = rng.uniform(0, cfg.days)
    ts = cfg.end - timedelta(days=offset_days)
    # shoppers cluster in daytime
    ts = ts.replace(hour=rng.choices(range(24), weights=(
        [1] * 7 + [4, 6, 7, 8, 9, 9, 8, 8, 9, 10, 11, 9, 6, 4, 2, 1, 1]
    ))[0])

    return {
        "receipt_id": f"R{rng.getrandbits(48):012x}",
        "user_id": f"U{rng.randrange(cfg.n_users):06d}",
        "store_id": rng.choice(STORES),
        "submitted_at": ts,
        "currency": "USD",
        "payment_method": rng.choices(PAYMENT_METHODS, weights=[62, 18, 15, 5])[0],
        "scanner_version": rng.choices(SCANNER_VERSIONS, weights=[10, 20, 45, 25])[0],
        "image_quality": _round2(min(1.0, max(0.0, rng.gauss(0.86, 0.09)))),
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "anomaly_type": None,
        "is_anomaly": 0,
    }


def _corrupt(r: dict[str, Any], kind: str, rng: random.Random) -> dict[str, Any]:
    """Apply one labelled defect. Mutates and returns the record."""
    r["anomaly_type"] = kind
    r["is_anomaly"] = 1

    if kind == "arithmetic_mismatch":
        # OCR reads the total line wrong: totals stop reconciling to the items
        drift = rng.choice([-1, 1]) * rng.uniform(0.05, 0.35)
        r["total"] = _round2(max(0.01, r["total"] * (1 + drift)))

    elif kind == "impossible_quantity":
        # unit confusion: weight read as count
        victim = rng.randrange(len(r["items"]))
        r["items"][victim]["qty"] = rng.randrange(40, 400)
        r["subtotal"] = _round2(
            sum(i["qty"] * i["unit_price"] for i in r["items"])
        )
        r["tax"] = _round2(r["subtotal"] * TAX_RATE)
        r["total"] = _round2(r["subtotal"] + r["tax"])

    elif kind == "price_outlier":
        # decimal point misplaced
        victim = rng.randrange(len(r["items"]))
        r["items"][victim]["unit_price"] = _round2(
            r["items"][victim]["unit_price"] * rng.choice([10.0, 100.0, 0.1])
        )
        r["subtotal"] = _round2(
            sum(i["qty"] * i["unit_price"] for i in r["items"])
        )
        r["tax"] = _round2(r["subtotal"] * TAX_RATE)
        r["total"] = _round2(r["subtotal"] + r["tax"])

    elif kind == "timestamp_skew":
        # device clock wrong: future-dated or absurdly stale
        if rng.random() < 0.5:
            r["submitted_at"] = r["submitted_at"] + timedelta(
                days=rng.uniform(2, 200)
            )
        else:
            r["submitted_at"] = r["submitted_at"] - timedelta(
                days=rng.uniform(400, 1200)
            )

    elif kind == "ocr_dropout":
        # bad photo: fields go missing
        r["image_quality"] = _round2(rng.uniform(0.05, 0.42))
        if rng.random() < 0.6:
            r["subtotal"] = None
        if rng.random() < 0.5:
            r["tax"] = None
        if rng.random() < 0.3:
            r["items"] = r["items"][: max(1, len(r["items"]) // 2)]

    return r


def _malform(r: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Break the schema contract outright."""
    r["anomaly_type"] = "malformed_payload"
    r["is_anomaly"] = 1
    kind = rng.choice(["unknown_store", "non_positive_total", "empty_items", "bad_currency"])
    if kind == "unknown_store":
        r["store_id"] = f"ZZ{rng.randrange(900, 999)}"
    elif kind == "non_positive_total":
        r["total"] = _round2(-abs(r["total"]) if rng.random() < 0.5 else 0.0)
    elif kind == "empty_items":
        r["items"] = []
    else:
        r["currency"] = rng.choice(["EU R", "", "usd"])
    return r


def generate(cfg: GenConfig | None = None) -> pd.DataFrame:
    """Return a flat receipt-level DataFrame; `items` stays nested as a list.

    Note: the returned row count is *more* than ``cfg.n_receipts``. Duplicate
    submissions are a relational defect, so they are appended as extra rows
    after the base population is built. Callers that need an exact count should
    measure ``len()`` of the result rather than assuming ``n_receipts``.
    """
    cfg = cfg or GenConfig()
    rng = random.Random(cfg.seed)

    rows: list[dict[str, Any]] = []
    for _ in range(cfg.n_receipts):
        r = _base_receipt(rng, cfg)
        if rng.random() < cfg.malformed_rate:
            r = _malform(r, rng)
        elif rng.random() < cfg.anomaly_rate:
            kind = rng.choice([k for k in ANOMALY_TYPES if k != "duplicate_submission"])
            r = _corrupt(r, kind, rng)
        rows.append(r)

    # Duplicates are a *relational* defect, so they're injected after the fact:
    # resubmit an existing receipt under a new id, minutes later.
    n_dupes = int(len(rows) * cfg.anomaly_rate * 0.35)
    for _ in range(n_dupes):
        src = rows[rng.randrange(len(rows))]
        if src["is_anomaly"]:
            continue
        dupe = {
            **src,
            "receipt_id": f"R{rng.getrandbits(48):012x}",
            "submitted_at": src["submitted_at"] + timedelta(minutes=rng.uniform(1, 90)),
            "items": [dict(i) for i in src["items"]],
            "anomaly_type": "duplicate_submission",
            "is_anomaly": 1,
        }
        rows.append(dupe)

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df["submitted_at"] = pd.to_datetime(df["submitted_at"], utc=True)
    return df.sort_values("submitted_at").reset_index(drop=True)


def to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-serialisable records, for the Kafka producer."""
    out = []
    for r in df.to_dict(orient="records"):
        r = dict(r)
        r["submitted_at"] = r["submitted_at"].isoformat()
        out.append(r)
    return out


# --------------------------------------------------------------------------
# Store master: an effective-dated attribute feed.
#
# This is the source for a Type 2 dimension. Attributes change over the life of
# a store (a remodel changes the format, a district gets re-drawn), and every
# change arrives as a new row with its own effective date. Building the
# dimension is then a matter of closing out the prior version -- which is the
# part that has to be right, because a receipt from March must join to the
# store as it was in March, not as it is today.
# --------------------------------------------------------------------------

REGIONS = ["northeast", "midatlantic", "southeast", "midwest", "west"]
STORE_FORMATS = ["express", "standard", "supercenter"]


def generate_store_master(
    seed: int = DEFAULT_SEED, days: int = 45, end: datetime | None = None
) -> pd.DataFrame:
    """One row per (store, attribute-change), with an effective date."""
    rng = random.Random(seed + 991)
    end = end or datetime(2026, 6, 30, tzinfo=timezone.utc)
    start = end - timedelta(days=days)

    rows: list[dict[str, Any]] = []
    for store_id in STORES:
        region = rng.choice(REGIONS)
        fmt = rng.choices(STORE_FORMATS, weights=[25, 55, 20])[0]
        manager = f"M{rng.randrange(1000, 9999)}"

        # Opening version, effective before any receipt in the window.
        rows.append(
            {
                "store_id": store_id,
                "effective_from": start - timedelta(days=365),
                "region": region,
                "store_format": fmt,
                "manager_id": manager,
                "change_reason": "initial_load",
            }
        )

        # Roughly a third of stores see a mid-window change.
        for _ in range(rng.choices([0, 1, 2], weights=[64, 30, 6])[0]):
            eff = start + timedelta(days=rng.uniform(3, days - 3))
            reason = rng.choice(["remodel", "district_realignment", "manager_change"])
            if reason == "remodel":
                fmt = rng.choice([f for f in STORE_FORMATS if f != fmt])
            elif reason == "district_realignment":
                region = rng.choice([r for r in REGIONS if r != region])
            else:
                manager = f"M{rng.randrange(1000, 9999)}"
            rows.append(
                {
                    "store_id": store_id,
                    "effective_from": eff,
                    "region": region,
                    "store_format": fmt,
                    "manager_id": manager,
                    "change_reason": reason,
                }
            )

    df = pd.DataFrame(rows)
    df["effective_from"] = pd.to_datetime(df["effective_from"], utc=True)
    return df.sort_values(["store_id", "effective_from"]).reset_index(drop=True)
