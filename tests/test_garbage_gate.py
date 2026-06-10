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


def test_model_feature_constants_frozen():
    # The model-input garbage_time_risk must stay at the training definition.
    # If these change, retrain the model — they are NOT the tunable gate.
    assert feat._TRAIN_BLOWOUT_MARGIN_PTS == 30
    assert feat._TRAIN_GARBAGE_PERIOD == 4
    assert feat._TRAIN_GARBAGE_CLOCK_SECS == 360
