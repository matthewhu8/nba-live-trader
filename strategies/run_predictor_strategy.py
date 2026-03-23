"""
RunPredictorStrategy — replaces hand-coded thresholds with model probabilities.

Uses the trained XGBoost run predictor to estimate the probability of
a meaningful scoring run in the next 5 possessions, then enters a
position when probability exceeds the configured threshold.

Entry logic:
  - run_prob_home > threshold → buy YES (home team about to go on a run)
  - run_prob_away > threshold → buy NO  (away team about to go on a run,
                                         home YES price will fall)
  - Both signals: take the higher-confidence direction

Still enforces all hard rules:
  - Maker-only limit orders
  - No blowout, no garbage time
  - Fee check: edge must clear both-side maker fees
  - One position at a time
"""

import logging
import pickle
from pathlib import Path

from models.run_predictor import FEATURE_COLS, RunPredictor, build_away_features, prepare_features
from strategies.base import (
    BaseStrategy,
    ExitSignal,
    FeatureRow,
    OrderFill,
    Position,
    Signal,
)

import pandas as pd

logger = logging.getLogger(__name__)

MAKER_FEE_RATE = 0.0175


def _maker_fee(size: int, price_cents: int) -> float:
    return MAKER_FEE_RATE * size * price_cents / 100.0


def _feature_row_to_dict(row: FeatureRow) -> dict:
    """Convert FeatureRow dataclass to a dict for RunPredictor."""
    return {
        "lineup_net_rating_delta":  row.lineup_net_rating_delta,
        "home_lineup_net_rating":   row.home_lineup_net_rating,
        "away_lineup_net_rating":   row.away_lineup_net_rating,
        "home_lineup_sample_size":  row.home_lineup_sample_size,
        "away_lineup_sample_size":  row.away_lineup_sample_size,
        "home_lineup_just_changed": int(row.home_lineup_just_changed),
        "away_lineup_just_changed": int(row.away_lineup_just_changed),
        "home_points_last_5_poss":  row.home_points_last_5_poss,
        "away_points_last_5_poss":  row.away_points_last_5_poss,
        "home_points_last_10_poss": row.home_points_last_10_poss,
        "away_points_last_10_poss": row.away_points_last_10_poss,
        "current_run_team":         row.current_run_team,
        "current_run_length":       row.current_run_length,
        "current_run_points":       row.current_run_points,
        "home_scoring_sustainable": int(row.home_scoring_sustainable),
        "away_scoring_sustainable": int(row.away_scoring_sustainable),
        "home_key_foul_count":      row.home_key_foul_count,
        "away_key_foul_count":      row.away_key_foul_count,
        "score_diff":               row.score_diff,
        "period":                   row.period,
        "minutes_into_game":        row.minutes_into_game,
        "pace_last_10_possessions": row.pace_last_10_possessions,
        "pace_season_baseline":     row.pace_season_baseline,
        "home_back_to_back":        int(row.home_back_to_back),
        "away_back_to_back":        int(row.away_back_to_back),
        "shot_distance":            row.shot_distance,
        "shot_value":               float(row.shot_value),
    }


class RunPredictorStrategy(BaseStrategy):
    """Model-driven strategy using XGBoost run probability predictions."""

    name = "RunPredictorStrategy"

    def __init__(
        self,
        model_path: str = "models/saved/run_predictor.pkl",
        entry_prob_threshold: float = 0.15,   # ~2× the 7.6% base rate
        min_edge_cents: int = 2,
        max_hold_possessions: int = 8,
        default_size: int = 10,
    ) -> None:
        self.predictor = RunPredictor.from_file(Path(model_path))
        self.entry_prob_threshold = entry_prob_threshold
        self.min_edge_cents = min_edge_cents
        self.max_hold_possessions = max_hold_possessions
        self.default_size = default_size

        self._in_position = False
        self._entry_possession: int = 0

    def on_game_start(self, game_id: str, pregame_features: dict) -> None:
        self._in_position = False
        self._entry_possession = 0

    def on_possession(self, row: FeatureRow) -> Signal | None:
        if row.is_blowout or row.is_garbage_time or self._in_position:
            return None

        feat_dict = _feature_row_to_dict(row)
        run_prob_home = self.predictor.predict_home_run(feat_dict)
        run_prob_away = self.predictor.predict_away_run(feat_dict)

        # Determine direction: take the higher confidence signal if above threshold
        if run_prob_home >= run_prob_away and run_prob_home > self.entry_prob_threshold:
            direction = "YES"
            confidence = run_prob_home
        elif run_prob_away > run_prob_home and run_prob_away > self.entry_prob_threshold:
            direction = "NO"
            confidence = run_prob_away
        else:
            return None

        # Use synthetic price as limit — maker order, so we post at market
        limit_price = row.synthetic_yes_price if direction == "YES" else (100 - row.synthetic_yes_price)

        # Fee check: limit price must be valid and leave room for entry + exit fees.
        # Using min_edge_cents × size as the required gross edge (same as other strategies).
        entry_fee = _maker_fee(self.default_size, limit_price)
        exit_fee = _maker_fee(self.default_size, limit_price)
        total_fees = entry_fee + exit_fee
        gross_edge = self.min_edge_cents * self.default_size / 100.0

        if gross_edge <= total_fees:
            return None

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

    def on_position_update(self, position: Position, row: FeatureRow) -> ExitSignal | None:
        if row.is_blowout or row.is_garbage_time:
            return ExitSignal(reason="blowout_garbage_time")

        if abs(position.entry_possession - row.possession_id) >= self.max_hold_possessions:
            return ExitSignal(reason="time_limit")

        return None

    def on_position_closed(self) -> None:
        self._in_position = False
        self._entry_possession = 0

    def on_game_end(self, game_id: str) -> None:
        self._in_position = False
        self._entry_possession = 0
