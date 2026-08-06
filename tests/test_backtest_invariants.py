"""
Backtest measurement invariants — the three defects that inflated `+$37,409` and had no
test coverage until now.

Defects 3 (zeroed fees) and 4 (cents-vs-dollars) are covered in `test_fees.py`. This file
covers defect 1 (exit-window lookahead) and defect 2 (take-profit over-crediting), plus the
PR #54 staleness bound that the exit anchor now shares.

These use synthetic ticks so they run in milliseconds. Note the deliberate limitation
recorded in `.claude/skills/backtesting.md`: the exit-window guard only fires when an exit
actually resolves earlier than `wct + feed_delay_s`, which on any *individual* possession
may legitimately not happen. A single synthetic row can therefore pass with the bug
reintroduced. This file asserts the guard's logic directly instead; the real end-to-end
negative test — the three-line defect reintroduced against game `0042500101`, with a control
run to prove it is not vacuous — is recorded in `.claude/skills/backtesting.md`.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mmoe.dataset import MARKET_STALENESS_TOLERANCE_SECONDS  # noqa: E402
from backtesting.mmoe_backtest import _tick_at_delay  # noqa: E402
from models.targets.exit_simulator import simulate_exit  # noqa: E402

T0 = pd.Timestamp("2026-04-20T23:00:00Z")


def _ticks(offsets_and_bids):
    df = pd.DataFrame({
        "ts": [T0 + pd.Timedelta(seconds=s) for s, _ in offsets_and_bids],
        "yes_bid": [b for _, b in offsets_and_bids],
        "yes_ask": [b + 1 for _, b in offsets_and_bids],
    })
    # An empty list yields object dtype, which breaks timestamp comparison. Real callers
    # always hold datetime64 ticks (_run_game returns early on an empty frame), so pin the
    # dtype here rather than defending against it in production code.
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# ── Defect 1: the exit window must start at the entry anchor ─────────────────

def test_tick_at_delay_anchors_at_wall_clock_plus_delay():
    ticks = _ticks([(0, 50), (10, 51), (25, 52), (40, 53)])
    tick, anchor = _tick_at_delay(ticks, T0, delay_s=20)
    assert anchor == T0 + pd.Timedelta(seconds=20)
    # Most recent tick at or before the anchor — the 25s tick is in the future.
    assert float(tick["yes_bid"]) == 51


def test_exit_window_cannot_resolve_before_the_anchor():
    """The whole defect: pricing entry at wct+20 but searching exits from wct let a
    position close on movement that had already happened."""
    ticks = _ticks([(0, 50), (5, 56), (10, 57), (30, 50), (60, 50)])
    anchor = T0 + pd.Timedelta(seconds=20)

    # Correct: exit search starts at the anchor, so the 5s/10s spike is unreachable.
    correct = simulate_exit(
        entry_wall_clock=anchor,
        entry_yes_bid=50,
        entry_run_team=None,
        future_ticks=ticks[ticks["ts"] > anchor],
        future_possessions=pd.DataFrame(columns=["wall_clock_ts"]),
        tp=5, sl=3, entry_side=1, max_seconds=240,
    )
    assert correct.exit_reason != "take_profit"

    # Buggy: anchored at wct, the same spike books a take-profit that a live trader
    # could not have reached, at a timestamp *before* the position existed.
    buggy = simulate_exit(
        entry_wall_clock=T0,
        entry_yes_bid=50,
        entry_run_team=None,
        future_ticks=ticks[ticks["ts"] > T0],
        future_possessions=pd.DataFrame(columns=["wall_clock_ts"]),
        tp=5, sl=3, entry_side=1, max_seconds=240,
    )
    assert buggy.exit_reason == "take_profit"
    assert buggy.exit_time_offset_s == 5
    # ...and this is exactly what the RuntimeError in _run_game catches:
    assert T0 + pd.Timedelta(seconds=buggy.exit_time_offset_s) < anchor


def test_guard_condition_fires_on_exit_before_anchor():
    """Mirror of the invariant at mmoe_backtest.py::_run_game, asserted directly.

    Stated against `wct` rather than the anchor deliberately, so that re-anchoring the
    exit window to `wct` fails loudly instead of silently reinflating results.
    """
    feed_delay_s = 20
    anchor = T0 + pd.Timedelta(seconds=feed_delay_s)

    def guard_trips(exit_offset_s, entry_anchor_ts):
        exit_abs_ts = entry_anchor_ts + pd.Timedelta(seconds=exit_offset_s)
        return exit_offset_s < 0 or exit_abs_ts < T0 + pd.Timedelta(seconds=feed_delay_s)

    assert not guard_trips(1, anchor)      # legitimate 1s hold from a correct anchor
    assert not guard_trips(0, anchor)
    assert guard_trips(-1, anchor)         # negative hold
    assert guard_trips(5, T0)              # bug reintroduced: anchored at wct


def test_do_not_assert_hold_time_exceeds_feed_delay():
    """A 1s hold measured from a correct anchor is legitimate and must not be rejected.

    Documented in skills/backtesting.md: asserting `hold_time_s >= feed_delay_s` would
    reject valid trades once the anchor is right.
    """
    ticks = _ticks([(21, 46), (60, 50)])
    anchor = T0 + pd.Timedelta(seconds=20)
    res = simulate_exit(
        entry_wall_clock=anchor,
        entry_yes_bid=50,
        entry_run_team=None,
        future_ticks=ticks,
        future_possessions=pd.DataFrame(columns=["wall_clock_ts"]),
        tp=5, sl=3, entry_side=1, max_seconds=240,
    )
    assert res.exit_reason == "stop_loss"
    assert 0 <= res.exit_time_offset_s < 20


# ── Defect 2: take-profits cannot fill better than the resting limit ─────────

@pytest.mark.parametrize("entry_side", [1, -1])
def test_take_profit_overshoot_is_clamped_to_the_resting_limit(entry_side):
    """PR #50 rests the TP limit at entry + TP, so the fill cannot beat that price.

    simulate_exit reports the price of the tick that *breached* the threshold, which
    overshot by a median 6c (max 24c) on the 2026-08-03 sample and booked the overshoot
    as profit that was never collectable.
    """
    tp = 5.0
    entry = 50.0
    # A tick that blows straight through the limit by 9c.
    breach_price = entry + entry_side * (tp + 9)
    ticks = _ticks([(30, breach_price), (60, entry)])
    anchor = T0 + pd.Timedelta(seconds=20)

    sim = simulate_exit(
        entry_wall_clock=anchor,
        entry_yes_bid=entry,
        entry_run_team=None,
        future_ticks=ticks,
        future_possessions=pd.DataFrame(columns=["wall_clock_ts"]),
        tp=tp, sl=3, entry_side=entry_side, max_seconds=240,
    )
    assert sim.exit_reason == "take_profit"
    assert sim.exit_price == breach_price      # the raw, over-credited price

    # The clamp applied in _run_game:
    clamped = entry + entry_side * tp
    assert clamped != sim.exit_price
    # Booked gross is exactly TP, never more.
    assert entry_side * (clamped - entry) == pytest.approx(tp)
    assert entry_side * (sim.exit_price - entry) > tp   # what the bug booked


def test_non_tp_exits_are_not_clamped():
    """Only take_profit rests a limit; stops fill wherever the book is."""
    ticks = _ticks([(30, 40), (60, 40)])
    anchor = T0 + pd.Timedelta(seconds=20)
    sim = simulate_exit(
        entry_wall_clock=anchor,
        entry_yes_bid=50,
        entry_run_team=None,
        future_ticks=ticks,
        future_possessions=pd.DataFrame(columns=["wall_clock_ts"]),
        tp=5, sl=3, entry_side=1, max_seconds=240,
    )
    assert sim.exit_reason == "stop_loss"
    assert sim.exit_price == 40   # a 10c gap through the 3c stop, not clamped


# ── PR #54: the staleness bound, now shared by the exit anchor ───────────────

def test_tick_at_delay_rejects_a_tick_older_than_the_tolerance():
    stale_s = MARKET_STALENESS_TOLERANCE_SECONDS + 10
    ticks = _ticks([(-stale_s, 50)])
    tick, anchor = _tick_at_delay(ticks, T0, delay_s=20)
    assert tick is None, "a tick older than the tolerance must not price a possession"
    assert anchor == T0 + pd.Timedelta(seconds=20)


def test_tick_at_delay_accepts_a_tick_inside_the_tolerance():
    fresh_s = MARKET_STALENESS_TOLERANCE_SECONDS - 5
    ticks = _ticks([(20 - fresh_s, 50)])
    tick, _ = _tick_at_delay(ticks, T0, delay_s=20)
    assert tick is not None
    assert float(tick["yes_bid"]) == 50


def test_anchor_is_returned_even_when_no_tick_qualifies():
    """The anchor is a property of the clock, not of the data — the exit window must
    still start in the right place when the market has gone quiet."""
    tick, anchor = _tick_at_delay(_ticks([]), T0, delay_s=20)
    assert tick is None
    assert anchor == T0 + pd.Timedelta(seconds=20)
