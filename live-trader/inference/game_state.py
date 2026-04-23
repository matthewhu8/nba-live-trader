"""
GameState — per-game streaming state accumulator.

All rolling windows, run state, foul counts, lineup history, and possession
state machine variables live here.

Two categories of state:
  1. Possession SM state — mutated by PossessionBuilder.parse() as events arrive.
     Tracks which team has the ball, pending missed shots, free throw sequences.
  2. Feature state — updated by state.advance() AFTER features are extracted.
     This preserves the shift(1) invariant: features for possession N reflect
     state strictly before possession N.

Prediction history (prediction_history) stores the last 20 full
(features, mmoe_output, action) records for dashboard + Thompson Sampling reward.
"""

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _parse_clock(clock_str: str) -> float:
    """Parse NBA CDN clock format 'PT06M23.00S' → seconds remaining."""
    m = re.match(r"PT(\d+)M([\d.]+)S", clock_str or "")
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


@dataclass
class ScoredPossession:
    """Compact record of a scoring possession — used for xPPP rolling window."""
    team:         str
    points:       int
    shot_value:   int
    shot_distance: float
    shot_area:    str
    was_paint:    bool   # shot_distance < 5 ft (sustainability threshold)
    xppp:         float


@dataclass
class PredictionRecord:
    """
    Full record of one possession's model output.
    maxlen=20 gives enough lookback for a future attention-based upgrade.
    """
    possession_id: int
    wall_clock_ts: datetime
    features:      dict[str, float]
    run_prob:      float
    trajectory:    list[float]
    hazard:        list[float]
    action:        str
    yes_bid:       int
    yes_ask:       int


@dataclass
class GameState:
    game_id: str

    # ── Game identity (loaded at start) ──────────────────────────────────────
    home_team_id: int = 0   # NBA teamId integer — used to map CDN events to home/away
    away_team_id: int = 0

    # ── Possession SM state (mutated by PossessionBuilder) ────────────────────
    # These track real-time basketball possession boundaries.

    possessing_team:  str = ""   # "home" | "away" — current ball holder
    missed_shot_team: str = ""   # set after missed shot; cleared on rebound
    pending_home_score: int = 0  # last known score (updated with each scoring event)
    pending_away_score: int = 0

    # Free throw sequence accumulator
    ft_in_seq:    bool  = False
    ft_total:     int   = 0     # total FTs in current sequence (e.g. 2)
    ft_made:      int   = 0     # FTs made so far
    ft_player_id: int   = 0
    ft_clock_secs: float = 0.0
    ft_period:    int   = 1
    ft_team:      str   = ""    # "home" | "away"

    # Per-possession event flags (read by PossessionBuilder, reset after each possession)
    poss_had_foul:           bool = False
    poss_had_shooting_foul:  bool = False
    poss_had_personal_foul:  bool = False
    poss_had_sub:            bool = False

    # ── Rolling caches (feature computation) ─────────────────────────────────
    recent_possessions:   deque = field(default_factory=lambda: deque(maxlen=20))
    home_scored_poss:     deque = field(default_factory=lambda: deque(maxlen=5))
    away_scored_poss:     deque = field(default_factory=lambda: deque(maxlen=5))
    possession_durations: deque = field(default_factory=lambda: deque(maxlen=10))

    # ── Prediction history cache ──────────────────────────────────────────────
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # ── Run state ────────────────────────────────────────────────────────────
    run_team:       str = ""
    run_length:     int = 0
    run_points:     int = 0
    run_3pt_count:  int = 0
    run_paint_pts:  int = 0

    # ── Foul tracking (full game — never truncated) ───────────────────────────
    player_fouls:    dict[int, int] = field(default_factory=dict)
    home_team_fouls: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    away_team_fouls: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    current_period:  int = 1

    # ── Timeout tracking ─────────────────────────────────────────────────────
    timeouts:            list[dict[str, Any]] = field(default_factory=list)
    home_timeouts_used:  int = 0
    away_timeouts_used:  int = 0

    # ── Lineup state ─────────────────────────────────────────────────────────
    home_lineup:      list[int] = field(default_factory=list)
    away_lineup:      list[int] = field(default_factory=list)
    prev_home_lineup: list[int] = field(default_factory=list)
    prev_away_lineup: list[int] = field(default_factory=list)
    home_sub_count:   int = 0
    away_sub_count:   int = 0

    # ── Pre-loaded at game start (static for entire game) ─────────────────────
    lineup_ratings:  dict[str, float] = field(default_factory=dict)
    player_apm:      dict[int, float] = field(default_factory=dict)
    star_players:    dict[int, int]   = field(default_factory=dict)
    pregame:         dict[str, float] = field(default_factory=dict)
    home_b2b:        bool  = False
    away_b2b:        bool  = False
    pace_baseline:   float = 14.0   # season avg seconds/possession

    # ── Internal ─────────────────────────────────────────────────────────────
    last_possession_clock_secs: float = 720.0
    last_possession_period:     int   = 1
    possession_count:           int   = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def update_from_event(self, raw_event: dict) -> None:
        """
        Handle side-effect events that don't complete a possession.
        Called by main.py when PossessionBuilder.parse() returns None.
        """
        action_type = raw_event.get("actionType", "")
        new_period  = raw_event.get("period", self.current_period)

        if action_type == "substitution":
            self._handle_substitution(raw_event)
        elif action_type == "foul":
            self._handle_foul(raw_event)
        elif action_type == "timeout":
            self._handle_timeout(raw_event)

        # Period change: any event with a new period triggers a reset
        if new_period != self.current_period:
            self._handle_period_change(new_period)

    def advance(self, possession: Any) -> None:
        """
        Update all rolling feature state AFTER features are extracted.
        Preserves the shift(1) invariant from training.
        """
        self._update_run_state(possession)
        self._update_points_buffers(possession)
        self._update_pace(possession)
        self.recent_possessions.append(possession)
        self.possession_count += 1

        # Reset "just changed" flags: next possession starts fresh
        self.prev_home_lineup = list(self.home_lineup)
        self.prev_away_lineup = list(self.away_lineup)

    def _team_side(self, team_id: int) -> str:
        """Map a CDN teamId integer to 'home' or 'away'. Returns '' if unknown."""
        if team_id and team_id == self.home_team_id:
            return "home"
        if team_id and team_id == self.away_team_id:
            return "away"
        return ""

    def _reset_poss_flags(self) -> None:
        """Clear per-possession event flags after a possession row is built."""
        self.poss_had_foul           = False
        self.poss_had_shooting_foul  = False
        self.poss_had_personal_foul  = False
        self.poss_had_sub            = False

    # ── Private state updaters ────────────────────────────────────────────────

    def _handle_substitution(self, event: dict) -> None:
        player_id = event.get("personId", 0)
        team_id   = event.get("teamId", 0)
        sub_type  = (event.get("subType") or "").lower()
        if not player_id:
            return

        team_side = self._team_side(team_id)
        lineup = self.home_lineup if team_side == "home" else self.away_lineup

        # Save prev lineup before modifying (for home_lineup_just_changed feature)
        if team_side == "home":
            self.prev_home_lineup = list(self.home_lineup)
            self.home_sub_count  += 1
        elif team_side == "away":
            self.prev_away_lineup = list(self.away_lineup)
            self.away_sub_count  += 1

        # CDN sends two events per substitution: subType "out" then "in"
        if sub_type == "out":
            if player_id in lineup:
                lineup.remove(player_id)
        elif sub_type == "in":
            if player_id not in lineup:
                lineup.append(player_id)
        else:
            # Fallback: single-event substitution (personId = player out)
            if player_id in lineup:
                lineup.remove(player_id)

        self.poss_had_sub = True

    def _handle_foul(self, event: dict) -> None:
        player_id = event.get("personId", 0)
        team_id   = event.get("teamId", 0)
        period    = event.get("period", self.current_period)
        sub_type  = (event.get("subType") or "").lower()

        if player_id:
            self.player_fouls[player_id] = self.player_fouls.get(player_id, 0) + 1

        team_side = self._team_side(team_id)
        if team_side == "home":
            self.home_team_fouls[period] = self.home_team_fouls.get(period, 0) + 1
        elif team_side == "away":
            self.away_team_fouls[period] = self.away_team_fouls.get(period, 0) + 1

        self.poss_had_foul = True
        if "shooting" in sub_type:
            self.poss_had_shooting_foul = True
        elif "personal" in sub_type:
            self.poss_had_personal_foul = True

    def _handle_timeout(self, event: dict) -> None:
        team_id   = event.get("teamId", 0)
        period    = event.get("period", self.current_period)
        clock_str = event.get("clock", "")
        team_side = self._team_side(team_id)

        self.timeouts.append({
            "period":       period,
            "clock_secs":   _parse_clock(clock_str),
            "team":         team_side,
            "possession_id": self.possession_count,
        })
        if team_side == "home":
            self.home_timeouts_used += 1
        elif team_side == "away":
            self.away_timeouts_used += 1

    def _handle_period_change(self, new_period: int) -> None:
        self.current_period = new_period
        # Reset quarter foul counts for new period (OT periods get their own entry)
        self.home_team_fouls[new_period] = 0
        self.away_team_fouls[new_period] = 0

    def _update_run_state(self, possession: Any) -> None:
        """
        Mirror the sequential run state machine from momentum_features.py.
        Only scoring possessions change run state — stops/turnovers are ignored.
        """
        pts = possession.points
        if pts <= 0:
            return

        team     = possession.team_scored
        is3      = possession.shot_value == 3
        is_paint = possession.shot_distance < 8.0 and possession.shot_value != 3

        if team == self.run_team:
            self.run_length   += 1
            self.run_points   += pts
            if is3:
                self.run_3pt_count += 1
            if is_paint:
                self.run_paint_pts += pts
        else:
            self.run_team      = team
            self.run_length    = 1
            self.run_points    = pts
            self.run_3pt_count = 1 if is3 else 0
            self.run_paint_pts = pts if is_paint else 0

    def _update_points_buffers(self, possession: Any) -> None:
        if possession.points <= 0:
            return
        sp = ScoredPossession(
            team          = possession.team_scored,
            points        = possession.points,
            shot_value    = possession.shot_value,
            shot_distance = possession.shot_distance,
            shot_area     = possession.shot_area,
            was_paint     = possession.is_paint,
            xppp          = possession.xppp,
        )
        if possession.team_scored == "home":
            self.home_scored_poss.append(sp)
        else:
            self.away_scored_poss.append(sp)

    def _update_pace(self, possession: Any) -> None:
        """
        Compute seconds elapsed since previous possession (same quarter only).
        Matches the pace computation in momentum_features.py.
        """
        if possession.period == self.last_possession_period:
            gap = self.last_possession_clock_secs - possession.game_clock_secs
            if 0 < gap < 60:
                self.possession_durations.append(gap)

        self.last_possession_clock_secs = possession.game_clock_secs
        self.last_possession_period     = possession.period
