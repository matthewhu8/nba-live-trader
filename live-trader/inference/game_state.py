"""
GameState — per-game streaming state accumulator.

This is the cache that makes live X vector computation possible.
All rolling windows, run state, foul counts, and lineup history live here.

Two categories of state:
  1. Rolling caches   — deques used to compute window features (last 5/10 possessions)
  2. Accumulators     — counters/dicts that grow through the game (fouls, timeouts)

The GameState is updated by state.advance() AFTER features are extracted each possession.
This preserves the shift(1) invariant from training: features for possession N reflect
state strictly before possession N is processed.

Prediction history cache (prediction_history) stores the last 20 full
(features, mmoe_output, action) records. Used for:
  - Dashboard: "show last 10 decisions"
  - Thompson Sampling: reward signal on position close
  - Future model upgrade: feed last N feature vectors as context to an attention layer
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ScoredPossession:
    """Compact record of a scoring possession — used for xPPP rolling window."""
    team:        str    # "home" | "away"
    points:      int
    shot_value:  int    # 2 or 3
    shot_distance: float
    shot_area:   str
    was_paint:   bool
    xppp:        float  # expected PPP from shot selection


@dataclass
class PredictionRecord:
    """
    Full record of one possession's model output.
    maxlen=20 gives enough lookback for a future attention-based model upgrade
    without wasting memory.
    """
    possession_id: int
    wall_clock_ts: datetime
    features:      dict[str, float]  # full 83-feature dict
    run_prob:      float
    trajectory:    list[float]       # 10 checkpoints
    hazard:        list[float]       # 10 horizons
    action:        str
    yes_bid:       int
    yes_ask:       int


@dataclass
class GameState:
    game_id: str

    # ── Rolling caches (feature computation) ──────────────────────────────────
    # All maxlen values match the rolling windows in training (momentum_features.py)

    recent_possessions:   deque = field(default_factory=lambda: deque(maxlen=20))
    # last 20 possession rows — general rolling window source

    home_scored_poss:     deque = field(default_factory=lambda: deque(maxlen=5))
    away_scored_poss:     deque = field(default_factory=lambda: deque(maxlen=5))
    # last 5 scoring possessions per team — for xPPP and sustainability features

    possession_durations: deque = field(default_factory=lambda: deque(maxlen=10))
    # seconds between consecutive possessions — for pace_last_10_possessions

    # ── Prediction history cache ──────────────────────────────────────────────
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=20))
    # PredictionRecord objects — dashboard + bandit reward + future model context

    # ── Run state ────────────────────────────────────────────────────────────
    run_team:       str = ""     # "home" | "away" | "" (no run)
    run_length:     int = 0      # consecutive scoring possessions by run_team
    run_points:     int = 0
    run_3pt_count:  int = 0
    run_paint_pts:  int = 0

    # ── Foul tracking (full game — never truncated) ───────────────────────────
    player_fouls:    dict[int, int] = field(default_factory=dict)  # player_id → count
    home_team_fouls: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    away_team_fouls: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    current_period:  int = 1

    # ── Timeout tracking ─────────────────────────────────────────────────────
    timeouts:          list[dict[str, Any]] = field(default_factory=list)  # {period, clock, team}
    home_timeouts_used: int = 0
    away_timeouts_used: int = 0

    # ── Lineup state ─────────────────────────────────────────────────────────
    home_lineup:      list[int] = field(default_factory=list)  # current 5 player IDs
    away_lineup:      list[int] = field(default_factory=list)
    prev_home_lineup: list[int] = field(default_factory=list)  # for lineup_just_changed
    prev_away_lineup: list[int] = field(default_factory=list)

    # ── Star / cumulative tracking ────────────────────────────────────────────
    home_star_points: int = 0   # points scored by on-court Tier-1 star this game
    away_star_points: int = 0
    home_star_fouls:  int = 0   # fouls by on-court highest-tier star
    away_star_fouls:  int = 0

    # ── Pre-loaded at game start (static for entire game) ─────────────────────
    lineup_ratings:  dict[str, float] = field(default_factory=dict)  # lineup_id → net_rating
    player_apm:      dict[int, float] = field(default_factory=dict)  # player_id → APM
    star_players:    dict[int, int]   = field(default_factory=dict)  # player_id → tier (1/2/3)
    pregame:         dict[str, float] = field(default_factory=dict)  # 11 pregame features
    home_b2b:        bool = False
    away_b2b:        bool = False
    pace_baseline:   float = 14.0   # season avg seconds/possession, loaded at game start

    # ── Internal ─────────────────────────────────────────────────────────────
    last_possession_clock_secs: float = 720.0  # for pace computation
    last_possession_period:     int   = 1
    possession_count:           int   = 0

    def update_from_event(self, raw_event: dict) -> None:
        """
        Handle non-possession-completing events (fouls, subs, timeouts mid-possession).
        Called when PossessionBuilder.parse() returns None.
        Updates side-effect state without triggering inference.
        """
        action_type = raw_event.get("actionType", "")

        if action_type == "substitution":
            self._handle_substitution(raw_event)
        elif action_type in ("foul", "personal", "shooting"):
            self._handle_foul(raw_event)
        elif action_type == "timeout":
            self._handle_timeout(raw_event)
        elif raw_event.get("period", self.current_period) != self.current_period:
            self._handle_period_change(raw_event.get("period"))

    def advance(self, possession: Any, raw_event: dict) -> None:
        """
        Update all rolling state AFTER features have been extracted for this possession.
        This preserves the shift(1) invariant: possession N's features see state
        strictly before possession N.
        """
        self._update_run_state(possession)
        self._update_points_buffers(possession)
        self._update_pace(possession)
        self.recent_possessions.append(possession)
        self.possession_count += 1

    # ── Private helpers ──────────────────────────────────────────────────────

    def _handle_substitution(self, event: dict) -> None:
        # TODO: update home_lineup / away_lineup; set prev_*_lineup before update
        pass

    def _handle_foul(self, event: dict) -> None:
        # TODO: increment player_fouls[player_id] and team_fouls[period]
        pass

    def _handle_timeout(self, event: dict) -> None:
        # TODO: append to timeouts list; increment home/away_timeouts_used
        pass

    def _handle_period_change(self, new_period: int) -> None:
        # Reset quarter foul counts; update current_period
        self.current_period = new_period
        self.home_team_fouls[new_period] = 0
        self.away_team_fouls[new_period] = 0

    def _update_run_state(self, possession: Any) -> None:
        # TODO: mirror the sequential run state machine from momentum_features.py
        # Update run_team, run_length, run_points, run_3pt_count, run_paint_pts
        pass

    def _update_points_buffers(self, possession: Any) -> None:
        # TODO: if possession scored: append to home_scored_poss or away_scored_poss
        pass

    def _update_pace(self, possession: Any) -> None:
        # TODO: compute duration from current possession's clock vs last_possession_clock_secs
        # Append to possession_durations
        pass
