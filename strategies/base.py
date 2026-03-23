"""
Base strategy interface.

All strategies implement this interface. The simulator calls these hooks
in strict possession order, enforcing latency and blowout rules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class FeatureRow:
    # Identity
    game_id: str
    possession_id: int
    period: int
    game_clock_secs: float
    minutes_into_game: float

    # Scores
    home_score: int
    away_score: int
    score_diff: int                    # home - away

    # Lineup state
    home_lineup_id: str
    away_lineup_id: str
    home_lineup_net_rating: float
    away_lineup_net_rating: float
    lineup_net_rating_delta: float     # home - away
    home_lineup_sample_size: int
    away_lineup_sample_size: int
    home_lineup_just_changed: bool
    away_lineup_just_changed: bool

    # Momentum
    home_points_last_5_poss: int
    away_points_last_5_poss: int
    home_points_last_10_poss: int
    away_points_last_10_poss: int
    current_run_team: str | None       # "home" | "away" | None
    current_run_length: int
    current_run_points: int

    # Shot quality
    home_scoring_sustainable: bool
    away_scoring_sustainable: bool
    shot_distance: float
    shot_value: int

    # Pace
    pace_last_10_possessions: float
    pace_season_baseline: float

    # Game context
    is_blowout: bool
    is_garbage_time: bool
    home_back_to_back: bool
    away_back_to_back: bool
    home_key_foul_count: int
    away_key_foul_count: int

    # Synthetic price (injected by simulator)
    synthetic_yes_price: int           # cents, 1-99

    # ── New features (all have defaults for backward compatibility) ────────

    # Run shot composition
    current_run_3pt_count: int   = field(default=0)
    current_run_3pt_pct: float   = field(default=0.0)   # high = volatile, likely to revert
    current_run_paint_pct: float = field(default=0.0)   # high = sustainable

    # xPPP shot quality (continuous — replaces/augments home_scoring_sustainable)
    home_xPPP_last_5: float           = field(default=1.0)
    away_xPPP_last_5: float           = field(default=1.0)
    home_actual_vs_expected_PPP: float = field(default=0.0)  # positive = overperforming → revert
    away_actual_vs_expected_PPP: float = field(default=0.0)
    home_shot_quality_trend: float    = field(default=0.0)   # +1 improving, -1 declining
    away_shot_quality_trend: float    = field(default=0.0)

    # Bonus state (≥5 team fouls in current quarter → free throws on contact)
    home_in_bonus: bool          = field(default=False)
    away_in_bonus: bool          = field(default=False)
    home_fouls_until_bonus: int  = field(default=4)
    away_fouls_until_bonus: int  = field(default=4)
    both_teams_in_bonus: bool    = field(default=False)

    # Score × time interactions
    trailing_team_urgency: float       = field(default=0.0)  # |score_diff| / min_remaining
    comeback_probability_proxy: float  = field(default=0.0)  # nonlinear urgency
    q4_close_game: bool                = field(default=False) # Q4 AND |score_diff| ≤ 5
    garbage_time_risk: float           = field(default=0.0)  # 0–1 continuous

    # Timeout signals
    possessions_since_last_timeout: int            = field(default=999)
    home_called_timeout_in_last_3_poss: bool       = field(default=False)
    away_called_timeout_in_last_3_poss: bool       = field(default=False)
    timeout_on_opponent_run: bool                  = field(default=False)
    home_full_timeouts_remaining: int              = field(default=4)
    away_full_timeouts_remaining: int              = field(default=4)

    # Player APM on court (from rolling adjusted plus/minus)
    home_best_player_apm: float   = field(default=0.0)
    away_best_player_apm: float   = field(default=0.0)
    home_worst_player_apm: float  = field(default=0.0)
    away_worst_player_apm: float  = field(default=0.0)
    home_apm_spread: float        = field(default=0.0)  # max - min; star-dependence
    away_apm_spread: float        = field(default=0.0)
    home_lineup_apm_sum: float    = field(default=0.0)
    away_lineup_apm_sum: float    = field(default=0.0)
    home_off_court_best_apm: float = field(default=0.0)  # bench quality
    away_off_court_best_apm: float = field(default=0.0)
    apm_delta: float              = field(default=0.0)   # home_sum - away_sum


@dataclass
class Signal:
    direction: Literal["YES", "NO"]
    limit_price: int                   # cents; must be a limit order
    size: int                          # number of contracts
    strategy_name: str
    confidence: float                  # 0.0–1.0


@dataclass
class ExitSignal:
    reason: str                        # "target_hit" | "time_limit" | "lineup_changed" | "game_end"


@dataclass
class OrderFill:
    direction: Literal["YES", "NO"]
    fill_price: int
    size: int
    possession_id: int


@dataclass
class Position:
    direction: Literal["YES", "NO"]
    entry_price: int
    size: int
    entry_possession: int
    current_price: int

    @property
    def unrealized_pnl(self) -> float:
        """Gross PnL before fees."""
        if self.direction == "YES":
            return (self.current_price - self.entry_price) * self.size / 100.0
        else:
            return (self.entry_price - self.current_price) * self.size / 100.0


class BaseStrategy(ABC):
    @abstractmethod
    def on_game_start(self, game_id: str, pregame_features: dict) -> None:
        """Called once before first possession. Initialize per-game state here."""

    @abstractmethod
    def on_possession(self, row: FeatureRow) -> Signal | None:
        """
        Called each possession. Return a Signal to request an order, or None to pass.
        Never enter during is_garbage_time or is_blowout — enforce this here.
        """

    @abstractmethod
    def on_fill(self, fill: OrderFill) -> None:
        """Called when a limit order fills. Update internal position tracking."""

    @abstractmethod
    def on_position_update(self, position: Position, row: FeatureRow) -> ExitSignal | None:
        """Called each possession while a position is open. Return ExitSignal to close."""

    def on_position_closed(self) -> None:
        """
        Called by simulator immediately after a position closes (any reason).
        Override to reset per-trade internal state. Default is a no-op since
        most state is managed by the simulator; only needed if strategy tracks
        internal position flags.
        """

    @abstractmethod
    def on_game_end(self, game_id: str) -> None:
        """Called at end of game. Simulator will force-close any open positions."""
