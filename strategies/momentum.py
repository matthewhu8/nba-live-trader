"""
MomentumStrategy — ride sustainable scoring runs.

Thesis: When a team is on a sustained run built on high-quality shots (open looks,
paint scoring), the run tends to continue longer than the market expects. Retail
bettors on Kalshi underreact to the quality difference between runs. Fade MeanReversion
by taking the same run in the opposite direction — but only when shot quality is high.

Entry:
  - current_run_points >= run_threshold (default 6)
  - The team currently running has scoring_sustainable == True
  - Not garbage time or blowout

Direction:
  - YES if home team is running (home outscoring → home WP goes up)
  - NO if away team is running

Exit:
  - Run ends (run_points drops back to 0 for the running team)
  - score_diff becomes blowout (>20)
  - max_hold_possessions reached
  - Game ends (force-close)
"""

import logging
from typing import Literal

from strategies.base import (
    BaseStrategy,
    ExitSignal,
    FeatureRow,
    OrderFill,
    Position,
    Signal,
)

logger = logging.getLogger(__name__)

MAKER_FEE_RATE = 0.0175


def maker_fee(size: int, price_cents: int) -> float:
    return MAKER_FEE_RATE * size * price_cents / 100.0


class MomentumStrategy(BaseStrategy):
    def __init__(
        self,
        run_threshold: int = 6,
        min_edge_cents: int = 2,
        max_hold_possessions: int = 8,
        default_size: int = 10,
    ) -> None:
        self.run_threshold = run_threshold
        self.min_edge_cents = min_edge_cents
        self.max_hold_possessions = max_hold_possessions
        self.default_size = default_size

        self._in_position = False
        self._entry_possession: int = 0
        self._running_team: str | None = None  # "home" or "away"

    @property
    def name(self) -> str:
        return "MomentumStrategy"

    def on_game_start(self, game_id: str, pregame_features: dict) -> None:
        self._in_position = False
        self._entry_possession = 0
        self._running_team = None

    def on_possession(self, row: FeatureRow) -> Signal | None:
        if row.is_garbage_time or row.is_blowout:
            return None

        if self._in_position:
            return None

        if row.current_run_points < self.run_threshold:
            return None

        # Determine which team is running and whether the run is sustainable
        direction: Literal["YES", "NO"]
        if row.current_run_team == "home" and row.home_scoring_sustainable:
            direction = "YES"
            running_team = "home"
        elif row.current_run_team == "away" and row.away_scoring_sustainable:
            direction = "NO"
            running_team = "away"
        else:
            # Either no clear running team, or run is not sustainable → skip
            return None

        # Price the limit order: buy slightly below current price (maker only)
        if direction == "YES":
            limit_price = row.synthetic_yes_price - self.min_edge_cents
        else:
            no_price = 100 - row.synthetic_yes_price
            limit_price = no_price - self.min_edge_cents

        if limit_price < 1 or limit_price > 99:
            return None

        # Fee check: edge must clear both-side maker fees
        entry_fee = maker_fee(self.default_size, limit_price)
        exit_fee = maker_fee(self.default_size, limit_price + self.min_edge_cents)
        gross_edge = self.min_edge_cents * self.default_size / 100.0

        if gross_edge <= entry_fee + exit_fee:
            logger.debug(
                "Signal rejected: edge $%.4f does not clear fees $%.4f",
                gross_edge,
                entry_fee + exit_fee,
            )
            return None

        self._running_team = running_team
        confidence = min(1.0, row.current_run_points / 12.0)

        return Signal(
            direction=direction,
            limit_price=limit_price,
            size=self.default_size,
            strategy_name=self.name,
            confidence=confidence,
        )

    def on_fill(self, fill: OrderFill) -> None:
        self._in_position = True
        self._entry_possession = fill.possession_id

    def on_position_closed(self) -> None:
        self._in_position = False
        self._entry_possession = 0
        self._running_team = None

    def on_position_update(self, position: Position, row: FeatureRow) -> ExitSignal | None:
        if row.is_blowout or row.is_garbage_time:
            return ExitSignal(reason="blowout_or_garbage")

        possessions_held = row.possession_id - self._entry_possession
        if possessions_held >= self.max_hold_possessions:
            return ExitSignal(reason="time_limit")

        # Exit when the run we entered on ends — run_points dropped back
        if self._running_team == "home":
            run_over = row.current_run_team != "home" or row.current_run_points < 2
        else:
            run_over = row.current_run_team != "away" or row.current_run_points < 2

        if run_over:
            return ExitSignal(reason="run_ended")

        return None

    def on_game_end(self, game_id: str) -> None:
        self._in_position = False
        self._entry_possession = 0
        self._running_team = None
