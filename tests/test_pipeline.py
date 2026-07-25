from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rpg import quality
from rpg.features import FEATURE_COLUMNS, build_features, time_split
from rpg.generate import GenConfig, generate
from rpg.train import precision_at_k


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate(GenConfig(n_receipts=3_000, seed=1))


@pytest.fixture(scope="module")
def as_of(raw) -> pd.Timestamp:
    return pd.Timestamp(GenConfig().end)


# --------------------------------------------------------------- generator --
def test_generator_is_deterministic():
    a = generate(GenConfig(n_receipts=500, seed=5))
    b = generate(GenConfig(n_receipts=500, seed=5))
    pd.testing.assert_frame_equal(
        a.drop(columns=["items"]), b.drop(columns=["items"])
    )


def test_generator_produces_both_classes(raw):
    assert raw["is_anomaly"].nunique() == 2
    rate = raw["is_anomaly"].mean()
    assert 0.02 < rate < 0.20, f"implausible anomaly rate {rate}"


def test_receipt_ids_unique(raw):
    assert raw["receipt_id"].is_unique


def test_sorted_by_time(raw):
    assert raw["submitted_at"].is_monotonic_increasing


# ------------------------------------------------------------------- gate ---
def test_gate_quarantines_malformed_rows(raw, as_of):
    clean, quarantined = quality.split_quarantine(raw, as_of=as_of)
    assert len(quarantined) > 0, "malformed payloads should be caught"
    assert len(clean) + len(quarantined) == len(raw), "rows must not vanish"
    assert set(quarantined["quarantine_reason"]).issubset(set(quality.CONTRACT_RULES))


def test_clean_rows_satisfy_the_contract(raw, as_of):
    clean, _ = quality.split_quarantine(raw, as_of=as_of)
    assert (clean["total"] > 0).all()
    assert (clean["currency"] == "USD").all()
    assert clean["items"].map(len).gt(0).all()
    assert (clean["submitted_at"] <= as_of).all()


def test_gate_is_deterministic_under_replay(raw, as_of):
    """Same batch, same as_of -> identical verdict. No wall-clock dependence."""
    c1, q1 = quality.split_quarantine(raw, as_of=as_of)
    c2, q2 = quality.split_quarantine(raw, as_of=as_of)
    assert len(c1) == len(c2) and len(q1) == len(q2)


def test_gate_fails_on_tiny_batch():
    tiny = generate(GenConfig(n_receipts=10, seed=3))
    clean, quarantined = quality.split_quarantine(tiny, as_of=pd.Timestamp(GenConfig().end))
    result = quality.evaluate_gates(clean, quarantined)
    assert not result.passed
    assert any("volume" in f for f in result.failures)


# --------------------------------------------------------------- features ---
def test_all_declared_features_exist(raw, as_of):
    clean, _ = quality.split_quarantine(raw, as_of=as_of)
    f = build_features(clean)
    missing = set(FEATURE_COLUMNS) - set(f.columns)
    assert not missing, f"missing features: {missing}"


def test_features_have_no_nulls_or_infinities(raw, as_of):
    clean, _ = quality.split_quarantine(raw, as_of=as_of)
    f = build_features(clean)
    X = f[FEATURE_COLUMNS].to_numpy(dtype=float)
    assert not np.isnan(X).any()
    assert np.isfinite(X).all()


def test_behavioural_features_do_not_see_the_future():
    """The 24h user window must count only strictly-earlier receipts.

    Constructed directly: three receipts for one user, one hour apart. The
    first must see zero priors, the last must see two.
    """
    ts = pd.to_datetime(
        ["2026-06-01T10:00:00Z", "2026-06-01T11:00:00Z", "2026-06-01T12:00:00Z"]
    )
    items = [{"sku": "SKU1001", "name": "x", "qty": 1, "unit_price": 5.0}]
    df = pd.DataFrame(
        {
            "receipt_id": ["a", "b", "c"],
            "user_id": ["U1", "U1", "U1"],
            "store_id": ["ST001"] * 3,
            "submitted_at": ts,
            "currency": ["USD"] * 3,
            "payment_method": ["card"] * 3,
            "scanner_version": ["v4.0"] * 3,
            "image_quality": [0.9] * 3,
            "items": [items, items, items],
            "subtotal": [5.0] * 3,
            "tax": [0.41] * 3,
            "total": [5.41] * 3,
        }
    )
    f = build_features(df)
    assert list(f["user_receipts_prior_24h"]) == [0.0, 1.0, 2.0]
    # identical totals -> the duplicate signal must also be backward-only
    assert list(f["same_total_prior_24h"]) == [0.0, 1.0, 2.0]


def test_time_split_is_chronological(raw, as_of):
    clean, _ = quality.split_quarantine(raw, as_of=as_of)
    f = build_features(clean)
    tr, te = time_split(f, test_frac=0.25)
    assert tr["submitted_at"].max() <= te["submitted_at"].min()
    assert len(tr) + len(te) == len(f)


def test_arithmetic_residual_flags_mismatches(raw, as_of):
    """Receipts whose totals were corrupted should show a larger residual."""
    clean, _ = quality.split_quarantine(raw, as_of=as_of)
    f = build_features(clean)
    mism = f["anomaly_type"] == "arithmetic_mismatch"
    if mism.sum() >= 5:
        assert (
            f.loc[mism, "arithmetic_residual_rel"].median()
            > f.loc[f["anomaly_type"].isna(), "arithmetic_residual_rel"].median()
        )


# ---------------------------------------------------------------- metrics ---
def test_precision_at_k_ranks_correctly():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.9, 0.95])
    assert precision_at_k(y, perfect, 0.5) == 1.0
    inverted = np.array([0.95, 0.9, 0.2, 0.1])
    assert precision_at_k(y, inverted, 0.5) == 0.0


def test_precision_at_k_handles_tiny_k():
    y = np.array([0, 1])
    s = np.array([0.1, 0.9])
    assert precision_at_k(y, s, 0.0001) == 1.0
