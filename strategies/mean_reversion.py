"""
MeanReversionStrategy — fade unsustainable scoring runs.

Thesis: retail bettors on Kalshi overreact to scoring runs built on contested
shots and lucky bounces. When a team is on a run but shot quality is low
(unsustainable), the price should snap back. We bet the other side.

Entry:
  - current_run_points >= run_threshold (default 8)
  - scoring_sustainable == False for the team ON the run
  - Not garbage time or blowout

Direction: NO on the team currently on the run (fade them).

Exit:
  - Run reverses (opponent takes the lead in recent possessions)
  - max_hold_possessions reached
  - Blowout condition triggers
  - Game ends (simulator force-closes)
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

MAKER_FEE_RATE = 0.0175  # 1.75% of (contracts × price_cents / 100)


def maker_fee(size: int, price_cents: int) -> float:
    return MAKER_FEE_RATE * size * price_cents / 100.0


class MeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        run_threshold: int = 8,
        min_edge_cents: int = 3,
        max_hold_possessions: int = 8,
        default_size: int = 10,
    ) -> None:
        self.run_threshold = run_threshold
        self.min_edge_cents = min_edge_cents
        self.max_hold_possessions = max_hold_possessions
        self.default_size = default_size

        self._in_position = False
        self._entry_possession: int = 0
        self._entry_run_team: str | None = None

    @property
    def name(self) -> str:
        return "MeanReversionStrategy"

    def on_game_start(self, game_id: str, pregame_features: dict) -> None:
        self._in_position = False
        self._entry_possession = 0
        self._entry_run_team = None

    def on_possession(self, row: FeatureRow) -> Signal | None:
        if row.is_garbage_time or row.is_blowout:
            return None

        if self._in_position:
            return None  # already holding; managed via on_position_update

        # Need a clear run in progress
        if row.current_run_team is None:
            return None
        if row.current_run_points < self.run_threshold:
            return None

        # Check shot quality — only fade unsustainable runs
        if row.current_run_team == "home":
            if row.home_scoring_sustainable:
                return None  # run is sustainable; don't fade
            # Fade home run → bet NO on home (i.e., we think home wins less)
            direction: Literal["YES", "NO"] = "NO"
            # For NO, limit_price is what we'll pay for NO contract.
            # NO price = 100 - YES price. We want to buy NO cheaply.
            yes_price = row.synthetic_yes_price
            no_price = 100 - yes_price
            limit_price = no_price - self.min_edge_cents  # bid below current NO

        else:  # away run
            if row.away_scoring_sustainable:
                return None
            direction = "YES"
            # Fade away run → bet YES on home
            yes_price = row.synthetic_yes_price
            limit_price = yes_price - self.min_edge_cents  # bid below current YES

        # Validate limit price
        if limit_price < 1 or limit_price > 99:
            return None

        # Verify edge clears fees (both entry and expected exit fees)
        entry_fee = maker_fee(self.default_size, limit_price)
        exit_fee = maker_fee(self.default_size, limit_price + self.min_edge_cents)
        total_fees = entry_fee + exit_fee
        gross_edge = self.min_edge_cents * self.default_size / 100.0

        if gross_edge <= total_fees:
            logger.debug(
                "Signal rejected: edge $%.4f does not clear fees $%.4f",
                gross_edge,
                total_fees,
            )
            return None

        return Signal(
            direction=direction,
            limit_price=limit_price,
            size=self.default_size,
            strategy_name=self.name,
            confidence=0.6,
        )

    def on_fill(self, fill: OrderFill) -> None:
        self._in_position = True
        self._entry_possession = fill.possession_id

    def on_position_closed(self) -> None:
        self._in_position = False
        self._entry_possession = 0
        self._entry_run_team = None

    def on_position_update(self, position: Position, row: FeatureRow) -> ExitSignal | None:
        if row.is_blowout or row.is_garbage_time:
            return ExitSignal(reason="blowout_or_garbage")

        possessions_held = row.possession_id - self._entry_possession

        if possessions_held >= self.max_hold_possessions:
            return ExitSignal(reason="time_limit")

        # Exit if run reversed: opponent is now scoring
        if position.direction == "NO":
            # We bet against home run; exit if home stops running
            if row.current_run_team != "home":
                return ExitSignal(reason="run_reversed")
        else:
            # We bet YES against away run; exit if away stops running
            if row.current_run_team != "away":
                return ExitSignal(reason="run_reversed")

        return None

    def on_game_end(self, game_id: str) -> None:
        self._in_position = False
        self._entry_possession = 0
        self._entry_run_team = None
