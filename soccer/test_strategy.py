"""
Unit tests for the break-fade strategy and match clock. Pure logic, no network.

  ./venv/bin/python -m unittest soccer.test_strategy -v
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from soccer.espn_clock import parse_espn_minute
from soccer.match_clock import MatchClock
from soccer.strategy import BreakFadeConfig, BreakFadeStrategy, PriceSnapshot

T0 = datetime(2026, 6, 17, 19, 0, 0, tzinfo=timezone.utc)


def _feed(strategy: BreakFadeStrategy, points: list[tuple[float, int, int]]):
    """Feed (match_minute, yes_bid, yes_ask) points; return the last Decision."""
    decision = None
    for minute, bid, ask in points:
        snap = PriceSnapshot(
            ts=T0 + timedelta(minutes=minute), match_minute=minute, yes_bid=bid, yes_ask=ask
        )
        decision = strategy.on_snapshot(snap)
    return decision


class TestBreakFade(unittest.TestCase):

    def _strategy(self, **overrides) -> BreakFadeStrategy:
        cfg = BreakFadeConfig(**overrides)
        return BreakFadeStrategy(cfg)

    def test_no_trade_outside_break_window(self):
        s = self._strategy()
        d = _feed(s, [(m, 50, 52) for m in range(0, 18, 2)])
        self.assertEqual(d.action, "WAIT")
        self.assertIsNone(s.position)

    def test_fade_riser_takes_no(self):
        # YES drifts 48 -> 60 into the 22' break -> fade up move -> take NO.
        s = self._strategy()
        points = [
            (16, 47, 49), (18, 50, 52), (20, 54, 56), (22, 59, 61),
        ]
        d = _feed(s, points)
        self.assertEqual(d.action, "ENTER")
        self.assertEqual(d.side, "NO")
        self.assertGreater(d.drift_cents, 0)

    def test_fade_faller_takes_yes(self):
        # YES drifts 58 -> 44 into the break -> fade down move -> take YES.
        s = self._strategy()
        points = [
            (16, 57, 59), (18, 54, 56), (20, 49, 51), (22, 43, 45),
        ]
        d = _feed(s, points)
        self.assertEqual(d.action, "ENTER")
        self.assertEqual(d.side, "YES")
        self.assertLess(d.drift_cents, 0)

    def test_drift_below_threshold_skips(self):
        s = self._strategy(min_drift_cents=6.0)
        points = [(16, 49, 51), (18, 49, 51), (20, 50, 52), (22, 51, 53)]
        d = _feed(s, points)
        self.assertEqual(d.action, "WAIT")
        self.assertIsNone(s.position)

    def test_trades_even_when_far_from_fifty(self):
        # Lopsided line at ~80c still fades the drift — no price-band veto.
        s = self._strategy()
        points = [(16, 74, 76), (18, 77, 79), (20, 80, 82), (22, 84, 86)]
        d = _feed(s, points)
        self.assertEqual(d.action, "ENTER")
        self.assertEqual(d.side, "NO")
        self.assertGreater(d.drift_cents, 0)

    def test_only_one_entry_per_break(self):
        s = self._strategy()
        points = [(16, 47, 49), (18, 50, 52), (20, 54, 56), (22, 59, 61)]
        first = _feed(s, points)
        self.assertEqual(first.action, "ENTER")
        # Another poll still inside the window should not open a second position.
        d = s.on_snapshot(PriceSnapshot(T0, 22.5, 60, 62))
        self.assertIn(d.action, ("WAIT",))

    def test_exit_after_hold(self):
        s = self._strategy(hold_min=8.0)
        points = [(16, 47, 49), (18, 50, 52), (20, 54, 56), (22, 59, 61)]
        enter = _feed(s, points)
        self.assertEqual(enter.action, "ENTER")
        # Still holding before the window elapses.
        mid = s.on_snapshot(PriceSnapshot(T0, 27, 58, 60))
        self.assertEqual(mid.action, "WAIT")
        # Past the hold window -> exit.
        out = s.on_snapshot(PriceSnapshot(T0, 31, 55, 57))
        self.assertEqual(out.action, "EXIT")
        self.assertEqual(out.side, "NO")

    def test_second_break_fires_independently(self):
        s = self._strategy()
        # First break entry + exit.
        _feed(s, [(16, 47, 49), (18, 50, 52), (20, 54, 56), (22, 59, 61)])
        s.on_snapshot(PriceSnapshot(T0, 31, 55, 57))  # exit first
        # Drift into the 67' break, opposite direction.
        d = _feed(s, [(61, 56, 58), (63, 53, 55), (65, 49, 51), (67, 44, 46)])
        self.assertEqual(d.action, "ENTER")
        self.assertEqual(d.side, "YES")


class TestMatchClock(unittest.TestCase):

    def test_first_half_tracks_real_time(self):
        clock = MatchClock(kickoff=T0)
        self.assertAlmostEqual(clock.match_minute(T0 + timedelta(minutes=22)), 22.0)

    def test_halftime_freezes_at_45(self):
        clock = MatchClock(kickoff=T0)
        self.assertAlmostEqual(clock.match_minute(T0 + timedelta(minutes=52)), 45.0)

    def test_second_half_resumes(self):
        clock = MatchClock(kickoff=T0)
        # 45 real + 15 halftime + 22 -> match minute 67
        self.assertAlmostEqual(clock.match_minute(T0 + timedelta(minutes=82)), 67.0)


class TestEspnMinute(unittest.TestCase):

    def _status(self, state: str, clock: str) -> dict:
        return {"type": {"state": state}, "displayClock": clock}

    def test_regular_minute(self):
        self.assertEqual(parse_espn_minute(self._status("in", "67'")), 67.0)

    def test_stoppage_time_added(self):
        self.assertEqual(parse_espn_minute(self._status("in", "90'+14'")), 104.0)
        self.assertEqual(parse_espn_minute(self._status("in", "45'+2'")), 47.0)

    def test_pre_and_post_return_none(self):
        self.assertIsNone(parse_espn_minute(self._status("pre", "0'")))
        self.assertIsNone(parse_espn_minute(self._status("post", "90'+5'")))


if __name__ == "__main__":
    unittest.main()
