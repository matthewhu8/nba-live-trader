"""
Hydration-break fade strategy — pure decision logic, no network.

Thesis (unproven, fun-money): a mandatory cooling break (~22' and ~67' in every
World Cup 2026 match) interrupts whichever team has been building momentum. The
market's own price drift INTO the break is our momentum proxy — we have no
external momentum feed. So at each break we *fade the recent mover*: if the YES
side has been drifting up, we bet it reverts (take NO), and vice versa.

We trade the market the runner selects — the spread line whose YES mid is closest
to 50¢ — whatever that price turns out to be. There is no price-band veto: even on
a lopsided game we fade the drift on the nearest-to-even line available.

This module is intentionally side-effect free and fully unit-testable: feed it
price snapshots via `on_snapshot`, it returns a `Decision`. The runner owns all
I/O, ordering, and P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PriceSnapshot:
    """One poll of the target market's top-of-book, in integer cents."""

    ts: datetime
    match_minute: float
    yes_bid: int
    yes_ask: int

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2.0


@dataclass
class Decision:
    action: str                 # "ENTER" | "EXIT" | "WAIT"
    side: str | None = None     # "YES" | "NO" (on ENTER)
    reason: str = ""
    drift_cents: float = 0.0
    match_minute: float = 0.0


@dataclass
class BreakFadeConfig:
    # Estimated match-minute of each mandatory cooling break.
    break_minutes: tuple[float, ...] = (22.0, 67.0)
    # How long after a break mark we will still open the fade.
    entry_window_min: float = 3.0
    # Window of price history (match-minutes) used to measure drift into the break.
    lookback_min: float = 6.0
    # Minimum absolute drift to consider there is momentum worth fading.
    min_drift_cents: float = 4.0
    # Hold the fade this many match-minutes, then exit.
    hold_min: float = 8.0


@dataclass
class _Position:
    side: str
    entry_minute: float
    entry_snapshot: PriceSnapshot
    break_minute: float


@dataclass
class BreakFadeStrategy:
    """
    Single-threaded, single-position. The runner calls `on_snapshot` once per poll
    (every ~15s) with the latest top-of-book and estimated match minute.
    """

    config: BreakFadeConfig = field(default_factory=BreakFadeConfig)
    _snaps: list[PriceSnapshot] = field(default_factory=list)
    _position: _Position | None = None
    _fired_breaks: set[float] = field(default_factory=set)

    @property
    def position(self) -> _Position | None:
        return self._position

    def on_snapshot(self, snap: PriceSnapshot) -> Decision:
        self._snaps.append(snap)
        self._prune(snap.match_minute)

        if self._position is not None:
            return self._manage_open_position(snap)

        return self._maybe_enter(snap)

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _manage_open_position(self, snap: PriceSnapshot) -> Decision:
        pos = self._position
        assert pos is not None
        held = snap.match_minute - pos.entry_minute
        if held < self.config.hold_min:
            return Decision("WAIT", reason="holding fade", match_minute=snap.match_minute)

        self._position = None
        return Decision(
            "EXIT",
            side=pos.side,
            reason=f"hold window elapsed ({held:.1f}m)",
            match_minute=snap.match_minute,
        )

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _maybe_enter(self, snap: PriceSnapshot) -> Decision:
        break_min = self._active_break(snap.match_minute)
        if break_min is None:
            return Decision("WAIT", reason="not in a break window", match_minute=snap.match_minute)

        drift = self._drift(snap.match_minute)
        if drift is None:
            # Not enough history yet — leave the break un-fired so a later poll
            # inside the window can still take the trade.
            return Decision("WAIT", reason="insufficient price history", match_minute=snap.match_minute)

        self._fired_breaks.add(break_min)

        if abs(drift) < self.config.min_drift_cents:
            return Decision(
                "WAIT",
                reason=f"drift {drift:+.1f}c below {self.config.min_drift_cents:.1f}c threshold",
                drift_cents=drift,
                match_minute=snap.match_minute,
            )

        # Fade the mover: YES drifted up -> take NO; YES drifted down -> take YES.
        side = "NO" if drift > 0 else "YES"
        self._position = _Position(
            side=side,
            entry_minute=snap.match_minute,
            entry_snapshot=snap,
            break_minute=break_min,
        )
        return Decision(
            "ENTER",
            side=side,
            reason=f"fade {drift:+.1f}c drift into {break_min:.0f}' break",
            drift_cents=drift,
            match_minute=snap.match_minute,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_break(self, minute: float) -> float | None:
        window = self.config.entry_window_min
        for bm in self.config.break_minutes:
            if bm in self._fired_breaks:
                continue
            if bm <= minute < bm + window:
                return bm
        return None

    def _drift(self, now_min: float) -> float | None:
        """YES-mid change over the lookback window, or None if history is too short."""
        if len(self._snaps) < 2:
            return None

        target = now_min - self.config.lookback_min
        baseline: PriceSnapshot | None = None
        for s in self._snaps:
            if s.match_minute >= target:
                baseline = s
                break
        if baseline is None:
            return None

        current = self._snaps[-1]
        span = current.match_minute - baseline.match_minute
        if span < self.config.lookback_min * 0.5:
            return None  # window not yet covered — avoid a noisy short-baseline read

        return current.yes_mid - baseline.yes_mid

    def _prune(self, now_min: float) -> None:
        keep_after = now_min - self.config.lookback_min * 2.0
        self._snaps = [s for s in self._snaps if s.match_minute >= keep_after]
