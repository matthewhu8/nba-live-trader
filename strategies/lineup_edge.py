"""
LineupEdgeStrategy — exploit lineup quality delta after a substitution.

Thesis: Kalshi doesn't immediately reprice when lineup matchup quality shifts.
If the home team just subbed in a significantly better lineup (net_rating_delta
jumps), there's a window before the market catches up.

Entry:
  - home_lineup_just_changed OR away_lineup_just_changed
  - abs(lineup_net_rating_delta) >= delta_threshold
  - Not garbage time or blowout

Direction: YES if delta > 0 (home lineup now superior), NO if delta < 0.

Exit:
  - Either team makes another substitution (matchup invalidated)
  - max_hold_possessions reached
  - Blowout triggers
  - Game ends
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


class LineupEdgeStrategy(BaseStrategy):
    def __init__(
        self,
        delta_threshold: float = 5.0,
        min_edge_cents: int = 2,
        max_hold_possessions: int = 10,
        default_size: int = 10,
        min_lineup_sample: int = 20,
    ) -> None:
        self.delta_threshold = delta_threshold
        self.min_edge_cents = min_edge_cents
        self.max_hold_possessions = max_hold_possessions
        self.default_size = default_size
        self.min_lineup_sample = min_lineup_sample

        self._in_position = False
        self._entry_possession: int = 0
        self._entry_home_lineup: str = ""
        self._entry_away_lineup: str = ""

    @property
    def name(self) -> str:
        return "LineupEdgeStrategy"

    def on_game_start(self, game_id: str, pregame_features: dict) -> None:
        self._in_position = False
        self._entry_possession = 0
        self._entry_home_lineup = ""
        self._entry_away_lineup = ""

    def on_possession(self, row: FeatureRow) -> Signal | None:
        if row.is_garbage_time or row.is_blowout:
            return None

        if self._in_position:
            return None

        # Require a fresh substitution
        if not row.home_lineup_just_changed and not row.away_lineup_just_changed:
            return None

        delta = row.lineup_net_rating_delta
        if abs(delta) < self.delta_threshold:
            return None

        # Require some data confidence on at least one lineup
        min_sample = min(row.home_lineup_sample_size, row.away_lineup_sample_size)
        if min_sample < self.min_lineup_sample:
            return None

        direction: Literal["YES", "NO"]
        limit_price: int

        if delta > 0:
            # Home lineup now better → expect home to outscore → YES
            direction = "YES"
            limit_price = row.synthetic_yes_price - self.min_edge_cents
        else:
            # Away lineup now better → NO on home
            direction = "NO"
            no_price = 100 - row.synthetic_yes_price
            limit_price = no_price - self.min_edge_cents

        if limit_price < 1 or limit_price > 99:
            return None

        # Fee check
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

        return Signal(
            direction=direction,
            limit_price=limit_price,
            size=self.default_size,
            strategy_name=self.name,
            confidence=min(1.0, abs(delta) / 10.0),
        )

    def on_fill(self, fill: OrderFill) -> None:
        self._in_position = True
        self._entry_possession = fill.possession_id

    def on_position_closed(self) -> None:
        self._in_position = False
        self._entry_possession = 0
        self._entry_home_lineup = ""
        self._entry_away_lineup = ""

    def on_position_update(self, position: Position, row: FeatureRow) -> ExitSignal | None:
        if row.is_blowout or row.is_garbage_time:
            return ExitSignal(reason="blowout_or_garbage")

        possessions_held = row.possession_id - self._entry_possession

        if possessions_held >= self.max_hold_possessions:
            return ExitSignal(reason="time_limit")

        # Exit if either lineup changed — the matchup we entered on is gone
        if row.home_lineup_just_changed or row.away_lineup_just_changed:
            return ExitSignal(reason="lineup_changed")

        return None

    def on_game_end(self, game_id: str) -> None:
        self._in_position = False
        self._entry_possession = 0
        self._entry_home_lineup = ""
        self._entry_away_lineup = ""
