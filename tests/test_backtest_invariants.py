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
from backtesting.dynamic_exit import simulate_exit_dynamic  # noqa: E402
from backtesting.mmoe_backtest import _tick_at_delay  # noqa: E402
from models.targets.exit_simulator import (  # noqa: E402
    _logit_delta,
    build_trajectory_targets,
    simulate_exit,
)

T0 = pd.Timestamp("2026-04-20T23:00:00Z")
GAME = "0042500101"


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

    The tick loop finds the tick that *breached* the threshold, which overshot by a median
    6c (max 24c) on the 2026-08-03 sample. The clamp lived in `_run_game` until Level 2
    moved it inside `simulate_exit`, so that the Head B trajectory — which freezes at
    `exit_price` after the exit — is capped too. Before that move the backtest's P&L was
    right while the labels it trained on still carried the uncollectable overshoot.
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

    # Clamped to the resting limit, not the 9c overshoot the breaching tick printed.
    clamped = entry + entry_side * tp
    assert sim.exit_price == clamped
    assert sim.exit_price != breach_price
    # Booked gross is exactly TP, never more.
    assert entry_side * (sim.exit_price - entry) == pytest.approx(tp)
    assert sim.simulated_pnl == pytest.approx(tp)

    # The exit fires 10s in and every checkpoint lands after it, so the whole trajectory
    # is the frozen exit price — the label a clamp regression would silently inflate.
    assert sim.exit_time_offset_s == 10
    for value in sim.trajectory:
        assert value == pytest.approx(_logit_delta(entry, clamped))


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


# ── Level 2: the training-label path must use the same anchor as the backtest ─
#
# `build_trajectory_targets` produced Head B's targets while anchored at `wall_clock_ts`,
# even though `yes_bid` on the same row was read at `wall_clock_ts + feed_delay` by
# dataset._join_ticks_to_possessions. Each test below carries its own control: the same
# ticks run at `feed_delay_seconds=0` reproduce the defect, so none of these can pass
# vacuously — the failure mode that made the original exit-window guard a tautology.


def _game_ticks(offsets_and_bids):
    df = _ticks(offsets_and_bids)
    df["game_id"] = GAME
    return df


def _entry_rows(bid=50.0, wall_clock_offset=0):
    return pd.DataFrame({
        "game_id": [GAME],
        "wall_clock_ts": [T0 + pd.Timedelta(seconds=wall_clock_offset)],
        "yes_bid": [bid],
        "current_run_team": ["home"],
        "current_run_team_encoded": [1],
    })


def _no_possessions():
    """A correctly-typed empty possession frame, so momentum-flip and garbage-time never
    fire and the tick-driven exits are what is under test."""
    return pd.DataFrame({
        "game_id":          pd.Series([], dtype="object"),
        "wall_clock_ts":    pd.Series([], dtype="datetime64[ns, UTC]"),
        "current_run_team": pd.Series([], dtype="object"),
        "is_blowout":       pd.Series([], dtype="bool"),
        "is_garbage_time":  pd.Series([], dtype="bool"),
    })


def _labels(ticks, feed_delay_seconds, tp=5.0, sl=3.0, horizon_seconds=120):
    return build_trajectory_targets(
        entry_rows=_entry_rows(),
        all_ticks=ticks,
        all_possessions=_no_possessions(),
        feed_delay_seconds=feed_delay_seconds,
        tp=tp, sl=sl, horizon_seconds=horizon_seconds,
    )


def test_label_exit_window_opens_at_the_feed_delay_anchor():
    """A spike inside the delay window is unreachable: it happened before the position
    could exist, and before the price it entered at was even observed."""
    ticks = _game_ticks([(0, 50), (5, 57), (10, 58), (30, 50), (130, 50), (260, 50)])

    delayed = _labels(ticks, feed_delay_seconds=20)
    assert delayed.at[0, "exit_reason"] == "time_gate"

    # Control — the defect restored. The 5s spike books a take-profit the trader could
    # never have reached, at a moment preceding entry.
    undelayed = _labels(ticks, feed_delay_seconds=0)
    assert undelayed.at[0, "exit_reason"] == "take_profit"
    assert undelayed.at[0, "exit_time_offset_s"] == 5


def test_label_checkpoint_grid_starts_at_the_anchor():
    """traj_0 sat at wall_clock_ts + 12s — eight seconds *before* entry at a 20s delay.

    TP/SL are disabled here so nothing exits early and the grid itself is isolated: the
    tick at +12s is visible only to the buggy anchoring.
    """
    ticks = _game_ticks([(12, 90), (32, 60), (140, 60), (260, 60)])
    off = 1000.0  # disable TP/SL

    delayed = _labels(ticks, feed_delay_seconds=20, tp=off, sl=off)
    assert delayed.at[0, "exit_reason"] == "time_gate"
    assert delayed.at[0, "traj_0"] == pytest.approx(_logit_delta(50, 60))

    # Control: anchored at wall_clock_ts, traj_0 reads the pre-entry 90c print.
    undelayed = _labels(ticks, feed_delay_seconds=0, tp=off, sl=off)
    assert undelayed.at[0, "traj_0"] == pytest.approx(_logit_delta(50, 90))


def test_label_hold_window_is_a_true_horizon_from_entry():
    """The deadline was `wall_clock_ts + horizon`, leaving a 100s effective hold at a 20s
    delay. Anchoring at entry makes it a real 120s — and picks up stops in the last 20s
    that the old window could not see."""
    ticks = _game_ticks([(0, 50), (130, 40), (260, 40)])

    delayed = _labels(ticks, feed_delay_seconds=20)
    assert delayed.at[0, "exit_reason"] == "stop_loss"
    assert delayed.at[0, "exit_time_offset_s"] == pytest.approx(110)

    # Control: the same stop falls outside the truncated window and is labelled a
    # flat time_gate exit, understating the downside Head B is trained on.
    undelayed = _labels(ticks, feed_delay_seconds=0)
    assert undelayed.at[0, "exit_reason"] == "time_gate"


def test_label_take_profit_trajectory_freezes_at_the_clamped_price():
    """The label-side half of defect 2: post-exit checkpoints freeze at `exit_price`, so
    an unclamped take-profit taught Head B to expect a fill it could not get."""
    ticks = _game_ticks([(0, 50), (30, 64), (260, 64)])

    res = _labels(ticks, feed_delay_seconds=20, tp=5.0)
    assert res.at[0, "exit_reason"] == "take_profit"
    assert res.at[0, "exit_time_offset_s"] == pytest.approx(10)
    assert res.at[0, "exit_price"] == 55        # entry + tp, not the 64c print
    for i in range(10):
        assert res.at[0, f"traj_{i}"] == pytest.approx(_logit_delta(50, 55))


def test_dynamic_exit_clamps_take_profit_to_the_resting_limit():
    """`dynamic_exit.py` carried the same unclamped take-profit, in a module Level 2 did not
    touch at first. Only `tools/sweep_dynamic_exit.py` consumes it, so no label or baseline
    depended on it — but the streak rule *widens* the TP, so the overshoot grew with the very
    knob those sweeps exist to tune.

    An empty possession frame means no re-inference happens, so `predictor` is never touched
    and the tick-driven TP path is isolated. The widened variant clamps on the same line, to
    `effective_tp` rather than `tp`.
    """
    ticks = _game_ticks([(30, 64), (60, 50)])
    anchor = T0 + pd.Timedelta(seconds=20)

    sim = simulate_exit_dynamic(
        entry_wall_clock=anchor,
        entry_yes_bid=50,
        entry_run_team=None,
        future_ticks=ticks,
        future_possessions=_no_possessions(),
        enriched_ticks_for_game=ticks,
        predictor=None,
        entry_traj_for_compare=0.10,
        tp=5.0, sl=3.0, entry_side=1, max_seconds=240,
    )
    assert sim.exit_reason == "take_profit"
    assert sim.exit_price == 55        # entry + tp, not the 64c print
    assert sim.simulated_pnl == pytest.approx(5.0)
    assert not sim.tp_widened
