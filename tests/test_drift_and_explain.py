"""Tests for the retraining gate: drift detection, fairness slicing, SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rpg import drift
from rpg.drift import population_stability_index as psi


# ------------------------------------------------------------------ PSI ----
def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5_000)
    assert psi(x, x) < 1e-6


def test_psi_is_small_for_samples_from_the_same_distribution():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=5_000), rng.normal(size=5_000)
    assert psi(a, b) < drift.PSI_NO_DRIFT


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=5_000)
    small = psi(ref, rng.normal(loc=0.3, size=5_000))
    large = psi(ref, rng.normal(loc=2.0, size=5_000))
    assert small < large
    assert large > drift.PSI_MODERATE


def test_psi_detects_a_variance_change_not_just_a_mean_shift():
    """A mean-only test would miss this; PSI must not."""
    rng = np.random.default_rng(3)
    ref = rng.normal(scale=1.0, size=5_000)
    cur = rng.normal(scale=3.0, size=5_000)
    assert psi(ref, cur) > drift.PSI_MODERATE


def test_psi_handles_constant_and_empty_input_without_crashing():
    const = np.ones(100)
    assert psi(const, const) == 0.0
    assert psi(np.array([]), np.ones(10)) == 0.0


def test_classify_bands():
    assert drift.classify(0.05) == "stable"
    assert drift.classify(0.15) == "moderate"
    assert drift.classify(0.40) == "significant"


# ---------------------------------------------------------------- drift ----
def _frame(rng, n, loc=0.0):
    return pd.DataFrame({"a": rng.normal(loc, size=n), "b": rng.normal(size=n)})


def test_detect_drift_reports_stable_when_nothing_moved():
    rng = np.random.default_rng(4)
    ref, cur = _frame(rng, 4_000), _frame(rng, 4_000)
    rep = drift.detect_drift(ref, cur, ["a", "b"])
    assert rep.drifted_features == []
    assert not rep.retrain_recommended


def test_detect_drift_identifies_the_moved_feature_only():
    rng = np.random.default_rng(5)
    ref = _frame(rng, 4_000)
    cur = _frame(rng, 4_000, loc=2.5)  # only 'a' shifts
    rep = drift.detect_drift(ref, cur, ["a", "b"])
    assert "a" in rep.drifted_features
    assert "b" not in rep.drifted_features
    assert rep.retrain_recommended
    assert rep.reasons


def test_prediction_drift_alone_triggers_retraining():
    """Scores can move while inputs look stable -- the earlier warning."""
    rng = np.random.default_rng(6)
    ref = cur = _frame(rng, 3_000)
    rep = drift.detect_drift(
        ref, cur, ["a", "b"],
        reference_scores=rng.beta(2, 8, size=3_000),
        current_scores=rng.beta(8, 2, size=3_000),
    )
    assert rep.prediction_psi > drift.PSI_MODERATE
    assert rep.retrain_recommended


# ------------------------------------------------------------- fairness ----
def _slice_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "scanner_version": rng.choice(["v3.1", "v4.0"], size=n),
            "payment_method": rng.choice(["card", "cash"], size=n),
            "image_quality": rng.uniform(0.2, 1.0, size=n),
        }
    )


def test_slice_metrics_reports_sample_sizes_and_reliability():
    f = _slice_frame()
    y = np.random.default_rng(1).integers(0, 2, size=len(f))
    flagged = y.copy()
    s = drift.slice_metrics(f, y, flagged)
    assert {"slice", "value", "n", "n_positive", "recall", "reliable"} <= set(s.columns)
    assert (s["n"] > 0).all()


def test_fairness_gate_passes_when_performance_is_uniform():
    f = _slice_frame()
    y = np.ones(len(f), dtype=int)
    flagged = np.ones(len(f), dtype=int)  # perfect recall everywhere
    ok, violations = drift.fairness_gate(drift.slice_metrics(f, y, flagged))
    assert ok and violations == []


def test_fairness_gate_blocks_a_collapsed_slice():
    """Aggregate recall stays high while one scanner version collapses."""
    n = 800
    f = pd.DataFrame(
        {
            "scanner_version": ["v3.1"] * 200 + ["v4.0"] * (n - 200),
            "payment_method": ["card"] * n,
            "image_quality": [0.9] * n,
        }
    )
    y = np.ones(n, dtype=int)
    flagged = np.ones(n, dtype=int)
    flagged[:200] = 0  # v3.1 is never caught
    s = drift.slice_metrics(f, y, flagged)
    ok, violations = drift.fairness_gate(s)
    assert not ok
    assert any("v3.1" in v for v in violations)


def test_fairness_gate_ignores_slices_too_small_to_judge():
    """A 5-row slice at 0.0 recall is noise and must not block a release."""
    f = pd.DataFrame(
        {
            "scanner_version": ["rare"] * 5 + ["common"] * 500,
            "payment_method": ["card"] * 505,
            "image_quality": [0.9] * 505,
        }
    )
    y = np.ones(505, dtype=int)
    flagged = np.ones(505, dtype=int)
    flagged[:5] = 0
    ok, _ = drift.fairness_gate(drift.slice_metrics(f, y, flagged))
    assert ok


# ------------------------------------------------------------ SHAP --------
def test_shap_explanations_are_per_receipt_and_signed():
    from rpg.explain import explain, global_importance
    from rpg.features import FEATURE_COLUMNS, build_features
    from rpg.generate import GenConfig, generate
    from rpg.quality import split_quarantine
    from rpg.train import train

    raw = generate(GenConfig(n_receipts=1_500, seed=21))
    clean, _ = split_quarantine(raw, as_of=pd.Timestamp(GenConfig().end))
    feats = build_features(clean)
    train(feats)

    from rpg.train import load_model

    model = load_model()
    ex = explain(model, feats.head(200))
    assert len(ex) == 200
    assert ex["receipt_id"].is_unique
    # Different receipts must get different explanations -- otherwise this is
    # just global importance wearing a disguise.
    assert ex["top_factors"].nunique() > 1
    # Signs must be present, not absolute values.
    assert ex["top_factors"].str.contains(r"[+-]").all()

    gi = global_importance(model, feats.head(300))
    assert set(gi["feature"]) == set(FEATURE_COLUMNS)
    assert (gi["mean_abs_shap"] >= 0).all()
    assert gi["mean_abs_shap"].is_monotonic_decreasing
