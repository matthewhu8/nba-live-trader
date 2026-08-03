"""
Tests for the config-driven garbage/blowout trade GATE.

The gate flags (is_garbage_time / is_blowout) moved from hardcoded Python into
trading.yaml, forwarded per-possession. These tests pin the boundary behavior
and guard the decoupling from the frozen `garbage_time_risk` model feature.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "live-trader"))

import inference.features as feat
from inference.main import compute_gate_flags


def test_blowout_boundary_is_strict_greater_than():
    # margin 30: exactly 30 is NOT a blowout, 31 is.
    _, is_blowout = compute_gate_flags(30, 4, 100, 30, 4, 360)
    assert is_blowout is False
    _, is_blowout = compute_gate_flags(31, 4, 100, 30, 4, 360)
    assert is_blowout is True
    # sign-independent
    _, is_blowout = compute_gate_flags(-31, 4, 100, 30, 4, 360)
    assert is_blowout is True


def test_garbage_time_requires_blowout_period_and_clock():
    # Blowout but wrong period → not garbage time.
    gt, _ = compute_gate_flags(40, 3, 100, 30, 4, 360)
    assert gt is False
    # Blowout, right period, but clock not yet under threshold.
    gt, _ = compute_gate_flags(40, 4, 400, 30, 4, 360)
    assert gt is False
    # All three satisfied.
    gt, _ = compute_gate_flags(40, 4, 359, 30, 4, 360)
    assert gt is True


def test_thresholds_are_actually_used():
    # Lowering the margin flips a previously-safe score into a blowout.
    _, tight = compute_gate_flags(26, 4, 100, 30, 4, 360)
    _, loose = compute_gate_flags(26, 4, 100, 25, 4, 360)
    assert tight is False and loose is True


def test_overtime_counts_as_garbage_with_period_floor():
    # period >= garbage_time_period, so OT (period 5) qualifies — harmless since
    # the Go agent skips OT independently, but documents the >= semantics.
    gt, _ = compute_gate_flags(40, 5, 100, 30, 4, 360)
    assert gt is True


def test_model_input_garbage_risk_is_continuous_and_shared():
    """
    The model input and the trade gate are still separate concerns, but the model
    input is no longer a local binary.

    This test previously asserted `_TRAIN_BLOWOUT_MARGIN_PTS == 30`, describing
    those constants as "the training definition". They never were: training used a
    sigmoid centred on a 15-point margin with a continuous time factor, so the
    assertion was pinning a train/serve divergence in place. The model input now
    comes from transforms.garbage_time_risk(), which both paths call.
    """
    from models.features import transforms as T

    # The live module must no longer carry its own copy of these constants.
    assert not hasattr(feat, "_TRAIN_BLOWOUT_MARGIN_PTS")
    assert not hasattr(feat, "_TRAIN_GARBAGE_PERIOD")

    # Continuous, and nonzero well before Q4 — the old binary returned 0.0 here.
    assert 0.0 < float(T.garbage_time_risk(25, 2, 360.0)) < 1.0
    # Monotone in margin at fixed time.
    risks = [float(T.garbage_time_risk(d, 4, 300.0)) for d in (5, 15, 25, 35)]
    assert risks == sorted(risks)


def test_trade_gate_stays_independent_of_the_model_input():
    """The tunable gate is driven by trading.yaml and must not follow the feature."""
    gt_tight, _ = compute_gate_flags(26, 4, 100, 25, 4, 360)
    gt_loose, _ = compute_gate_flags(26, 4, 100, 30, 4, 360)
    assert gt_tight != gt_loose, "gate thresholds must remain configurable"
