"""
Fee model — asserted against Kalshi's PUBLISHED schedule, not against our implementation.

    taker: ceil(0.07   x C x P x (1-P) x 100) / 100
    maker: ceil(0.0175 x C x P x (1-P) x 100) / 100

The expected values in `KALSHI_PUBLISHED_100C` are copied from Kalshi's own table
(effective 2026-02-05). If a test here fails, the presumption is that our code is wrong
and the table is right.

Why this file exists: the naive float64 implementation disagreed with the published table
on three of five taker rows, because `rate * C * P * (1-P) * 100` lands a few ulp above an
exact cent and `ceil` then rounds a whole cent up. `_fee_one_leg` computes in Decimal for
exactly this reason. See the $0.20 and $0.50 rows below.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.mmoe_backtest import (  # noqa: E402
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    _MAKER_EXIT_REASONS,
    _compute_fees,
    _fee_one_leg,
    pnl_dollars,
)

# price_cents -> (taker, maker) for 100 contracts, from Kalshi's published table.
KALSHI_PUBLISHED_100C = {
    10: (0.63, 0.16),
    20: (1.12, 0.28),
    50: (1.75, 0.44),
    85: (0.90, 0.23),
    90: (0.63, 0.16),
}


@pytest.mark.parametrize("price_cents,expected", sorted(KALSHI_PUBLISHED_100C.items()))
def test_taker_matches_published_table(price_cents, expected):
    taker_expected, _ = expected
    assert _fee_one_leg(TAKER_FEE_RATE, price_cents, 100) == pytest.approx(taker_expected)


@pytest.mark.parametrize("price_cents,expected", sorted(KALSHI_PUBLISHED_100C.items()))
def test_maker_matches_published_table(price_cents, expected):
    _, maker_expected = expected
    assert _fee_one_leg(MAKER_FEE_RATE, price_cents, 100) == pytest.approx(maker_expected)


def test_float_ceil_trap_is_actually_present():
    """Guard the guard: prove the naive float form really does disagree.

    If this ever stops failing, the Decimal implementation is no longer load-bearing and
    the tests above would pass for the wrong reason.
    """
    def naive(rate, price_cents, contracts):
        p = price_cents / 100.0
        return math.ceil(rate * contracts * p * (1.0 - p) * 100.0) / 100.0

    # Three taker rows where naive float disagrees with Kalshi.
    assert naive(TAKER_FEE_RATE, 50, 100) == 1.76   # published: 1.75
    assert naive(TAKER_FEE_RATE, 20, 100) == 1.13   # published: 1.12
    assert naive(TAKER_FEE_RATE, 10, 100) == 0.64   # published: 0.63
    assert naive(MAKER_FEE_RATE, 20, 100) == 0.29   # published: 0.28

    # ...and the shipped implementation does not.
    assert _fee_one_leg(TAKER_FEE_RATE, 50, 100) == 0.0 + 1.75
    assert _fee_one_leg(TAKER_FEE_RATE, 20, 100) == 0.0 + 1.12
    assert _fee_one_leg(TAKER_FEE_RATE, 10, 100) == 0.0 + 0.63
    assert _fee_one_leg(MAKER_FEE_RATE, 20, 100) == 0.0 + 0.28


def test_legitimate_ceil_still_rounds_up():
    """The fix must not turn `ceil` into `round` — 43.75c really does bill as 44c."""
    # 0.0175 * 100 * 0.5 * 0.5 * 100 = 43.75 exactly -> 44
    assert _fee_one_leg(MAKER_FEE_RATE, 50, 100) == pytest.approx(0.44)
    # 0.0175 * 100 * 0.85 * 0.15 * 100 = 22.3125 exactly -> 23
    assert _fee_one_leg(MAKER_FEE_RATE, 85, 100) == pytest.approx(0.23)
    # 0.07 * 100 * 0.85 * 0.15 * 100 = 89.25 exactly -> 90
    assert _fee_one_leg(TAKER_FEE_RATE, 85, 100) == pytest.approx(0.90)


def test_fee_is_symmetric_about_the_midpoint():
    """P x (1-P) is symmetric, so 30c and 70c must cost the same."""
    for rate in (MAKER_FEE_RATE, TAKER_FEE_RATE):
        for low in range(1, 50):
            assert _fee_one_leg(rate, low, 100) == _fee_one_leg(rate, 100 - low, 100)


def test_taker_is_four_times_maker_rate():
    assert TAKER_FEE_RATE == pytest.approx(4.0 * MAKER_FEE_RATE)


# ── Which leg pays what ──────────────────────────────────────────────────────

def test_take_profit_exit_pays_maker():
    """PR #50 rests the TP as a limit order, so it earns the maker rate."""
    fees = _compute_fees(entry_price=50, exit_price=55, contracts=100, exit_reason="take_profit")
    expected = _fee_one_leg(MAKER_FEE_RATE, 50, 100) + _fee_one_leg(MAKER_FEE_RATE, 55, 100)
    assert fees == pytest.approx(expected)


def test_widened_take_profit_also_pays_maker():
    """`take_profit_widened` (dynamic_exit's streak rule) re-posts the limit further out —
    still a resting order, so still maker.

    It was missing from `_MAKER_EXIT_REASONS`, so `tools/sweep_dynamic_exit.py` charged the
    4x taker rate on exactly the exits the streak rule exists to produce. Fixed 2026-08-10
    alongside that module's unclamped take-profit.
    """
    fees = _compute_fees(entry_price=50, exit_price=58, contracts=100,
                         exit_reason="take_profit_widened")
    expected = _fee_one_leg(MAKER_FEE_RATE, 50, 100) + _fee_one_leg(MAKER_FEE_RATE, 58, 100)
    assert fees == pytest.approx(expected)
    assert "take_profit_widened" in _MAKER_EXIT_REASONS


@pytest.mark.parametrize("reason", ["stop_loss", "momentum_flip", "garbage_time", "time_gate"])
def test_crossing_exits_pay_taker(reason):
    """Everything that is not a resting TP crosses the book at 4x the rate.

    This is the property PR #54's `_compute_maker_fees` did not have: it charged maker on
    both legs, and its own docstring conceded the result was a floor rather than a cost.
    """
    fees = _compute_fees(entry_price=50, exit_price=47, contracts=100, exit_reason=reason)
    expected = _fee_one_leg(MAKER_FEE_RATE, 50, 100) + _fee_one_leg(TAKER_FEE_RATE, 47, 100)
    assert fees == pytest.approx(expected)
    assert reason not in _MAKER_EXIT_REASONS


def test_entry_leg_is_always_maker():
    """Entry is a post-only limit in every path, so the entry leg never pays taker."""
    maker_entry = _fee_one_leg(MAKER_FEE_RATE, 50, 100)
    for reason in ("take_profit", "stop_loss", "momentum_flip", "garbage_time", "time_gate"):
        fees = _compute_fees(50, 50, 100, reason)
        exit_rate = MAKER_FEE_RATE if reason in _MAKER_EXIT_REASONS else TAKER_FEE_RATE
        assert fees - _fee_one_leg(exit_rate, 50, 100) == pytest.approx(maker_entry)


# ── Round-trip economics at a 50c entry, TP=5 / SL=3 ─────────────────────────

def test_round_trip_win_nets_4_12():
    """Maker in at 50c, maker out at 55c: $0.44 + $0.44 = $0.88, gross +$5.00."""
    fees = _compute_fees(50, 55, 100, "take_profit")
    assert fees == pytest.approx(0.88)
    assert pnl_dollars(5, 100) - fees == pytest.approx(4.12)


def test_round_trip_loss_nets_minus_5_19():
    """Maker in at 50c, taker out at 47c: $0.44 + $1.75 = $2.19, gross -$3.00.

    The $1.75 is the row the float bug broke — it billed $1.76 and made this -$5.20.
    """
    fees = _compute_fees(50, 47, 100, "stop_loss")
    assert fees == pytest.approx(2.19)
    assert pnl_dollars(-3, 100) - fees == pytest.approx(-5.19)


def test_break_even_win_rate_is_55_7_percent():
    win = pnl_dollars(5, 100) - _compute_fees(50, 55, 100, "take_profit")
    loss = pnl_dollars(-3, 100) - _compute_fees(50, 47, 100, "stop_loss")
    break_even = -loss / (win - loss)
    assert break_even == pytest.approx(0.557, abs=0.0005)


# ── Units: cents vs dollars (defect 4) ───────────────────────────────────────

def test_pnl_dollars_converts_cents_to_dollars():
    """100 contracts x 5c = $5.00, NOT $500."""
    assert pnl_dollars(5, 100) == pytest.approx(5.00)
    assert pnl_dollars(1, 1) == pytest.approx(0.01)
    assert pnl_dollars(-3, 100) == pytest.approx(-3.00)


def test_position_cannot_swing_more_than_contract_value():
    """A contract settles between $0 and $1, so 100 contracts cannot move more than $100."""
    assert abs(pnl_dollars(99, 100)) <= 100.0
    assert abs(pnl_dollars(-99, 100)) <= 100.0


def test_fees_are_not_zero():
    """Defect 3: `_compute_maker_fees` once returned 0.0 unconditionally."""
    for reason in ("take_profit", "stop_loss", "momentum_flip", "garbage_time", "time_gate"):
        assert _compute_fees(50, 50, 100, reason) > 0.0
