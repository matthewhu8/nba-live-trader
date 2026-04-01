"""
BacktestSimulator — replays historical games against a strategy.

Enforces:
  - Strict chronological order (by game, then by possession_id)
  - Latency: strategy sees features from latency_possessions ago
  - Blowout/garbage-time skip (no signals entered)
  - Maker-only limit orders
  - Fill simulation: fills when limit price reaches synthetic price
  - Maker fees on both entry and exit
  - Force-close at game end
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow.parquet as pq

from strategies.base import (
    BaseStrategy,
    ExitSignal,
    FeatureRow,
    OrderFill,
    Position,
    Signal,
)

if TYPE_CHECKING:
    from models.dk_price_validator import DKPriceValidator

logger = logging.getLogger(__name__)

MAKER_FEE_RATE = 0.0175


def _maker_fee(size: int, price_cents: int) -> float:
    return MAKER_FEE_RATE * size * price_cents / 100.0


@dataclass
class TradeRecord:
    game_id: str
    entry_possession: int
    exit_possession: int
    direction: str
    entry_price: int
    exit_price: int
    size: int
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str
    period: int
    score_diff_at_entry: int
    lineup_delta_at_entry: float
    run_points_at_entry: int
    shot_sustainable_at_entry: bool
    minutes_into_game_at_entry: float
    # Model + DK validation fields (populated when using RunPredictorStrategy + DKPriceValidator)
    run_prob_predicted: float = 0.0
    run_actually_occurred: bool = False
    dk_wp_entry: float | None = None
    dk_wp_exit: float | None = None
    dk_net_pnl: float | None = None


@dataclass
class SimulationResult:
    trades: list[TradeRecord] = field(default_factory=list)
    pnl_by_possession: list[float] = field(default_factory=list)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def net_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def gross_pnl(self) -> float:
        return sum(t.gross_pnl for t in self.trades)

    @property
    def total_fees(self) -> float:
        return sum(t.fees for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        return wins / len(self.trades)


def _row_to_feature_row(row: pd.Series, synthetic_yes_price: int) -> FeatureRow:
    """Convert a feature_rows DataFrame row + synthetic price into a FeatureRow."""
    def _bool(val: Any) -> bool:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return False
        return bool(val)

    def _int(val: Any, default: int = 0) -> int:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return int(val)

    def _float(val: Any, default: float = 0.0) -> float:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return float(val)

    def _str(val: Any, default: str = "") -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return str(val)

    def _str_or_none(val: Any) -> str | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val)
        return s if s not in ("", "nan", "None") else None

    return FeatureRow(
        game_id=str(row["game_id"]),
        possession_id=_int(row["possession_id"]),
        period=_int(row["period"]),
        game_clock_secs=_float(row["game_clock_secs"]),
        minutes_into_game=_float(row["minutes_into_game"]),
        home_score=_int(row["home_score"]),
        away_score=_int(row["away_score"]),
        score_diff=_int(row["score_diff"]),
        home_lineup_id=_str(row.get("home_lineup_id")),
        away_lineup_id=_str(row.get("away_lineup_id")),
        home_lineup_net_rating=_float(row.get("home_lineup_net_rating")),
        away_lineup_net_rating=_float(row.get("away_lineup_net_rating")),
        lineup_net_rating_delta=_float(row.get("lineup_net_rating_delta")),
        home_lineup_sample_size=_int(row.get("home_lineup_sample_size")),
        away_lineup_sample_size=_int(row.get("away_lineup_sample_size")),
        home_lineup_just_changed=_bool(row.get("home_lineup_just_changed")),
        away_lineup_just_changed=_bool(row.get("away_lineup_just_changed")),
        home_points_last_5_poss=_int(row.get("home_points_last_5_poss")),
        away_points_last_5_poss=_int(row.get("away_points_last_5_poss")),
        home_points_last_10_poss=_int(row.get("home_points_last_10_poss")),
        away_points_last_10_poss=_int(row.get("away_points_last_10_poss")),
        current_run_team=_str_or_none(row.get("current_run_team")),
        current_run_length=_int(row.get("current_run_length")),
        current_run_points=_int(row.get("current_run_points")),
        home_scoring_sustainable=_bool(row.get("home_scoring_sustainable")),
        away_scoring_sustainable=_bool(row.get("away_scoring_sustainable")),
        shot_distance=_float(row.get("shot_distance")),
        shot_value=_int(row.get("shot_value")),
        pace_last_10_possessions=_float(row.get("pace_last_10_possessions")),
        pace_season_baseline=_float(row.get("pace_season_baseline")),
        is_blowout=_bool(row.get("is_blowout")),
        is_garbage_time=_bool(row.get("is_garbage_time")),
        home_back_to_back=_bool(row.get("home_back_to_back")),
        away_back_to_back=_bool(row.get("away_back_to_back")),
        home_key_foul_count=_int(row.get("home_key_foul_count")),
        away_key_foul_count=_int(row.get("away_key_foul_count")),
        synthetic_yes_price=synthetic_yes_price,
        # Run shot composition
        current_run_3pt_count=_int(row.get("current_run_3pt_count")),
        current_run_3pt_pct=_float(row.get("current_run_3pt_pct")),
        current_run_paint_pct=_float(row.get("current_run_paint_pct")),
        # xPPP
        home_xPPP_last_5=_float(row.get("home_xPPP_last_5"), 1.0),
        away_xPPP_last_5=_float(row.get("away_xPPP_last_5"), 1.0),
        home_actual_vs_expected_PPP=_float(row.get("home_actual_vs_expected_PPP")),
        away_actual_vs_expected_PPP=_float(row.get("away_actual_vs_expected_PPP")),
        home_shot_quality_trend=_float(row.get("home_shot_quality_trend")),
        away_shot_quality_trend=_float(row.get("away_shot_quality_trend")),
        # Bonus state
        home_in_bonus=_bool(row.get("home_in_bonus")),
        away_in_bonus=_bool(row.get("away_in_bonus")),
        home_fouls_until_bonus=_int(row.get("home_fouls_until_bonus"), 4),
        away_fouls_until_bonus=_int(row.get("away_fouls_until_bonus"), 4),
        both_teams_in_bonus=_bool(row.get("both_teams_in_bonus")),
        # Score × time
        trailing_team_urgency=_float(row.get("trailing_team_urgency")),
        comeback_probability_proxy=_float(row.get("comeback_probability_proxy")),
        q4_close_game=_bool(row.get("q4_close_game")),
        garbage_time_risk=_float(row.get("garbage_time_risk")),
        # Timeout
        possessions_since_last_timeout=_int(row.get("possessions_since_last_timeout"), 999),
        home_called_timeout_in_last_3_poss=_bool(row.get("home_called_timeout_in_last_3_poss")),
        away_called_timeout_in_last_3_poss=_bool(row.get("away_called_timeout_in_last_3_poss")),
        timeout_on_opponent_run=_bool(row.get("timeout_on_opponent_run")),
        home_full_timeouts_remaining=_int(row.get("home_full_timeouts_remaining"), 4),
        away_full_timeouts_remaining=_int(row.get("away_full_timeouts_remaining"), 4),
        # Player APM
        home_best_player_apm=_float(row.get("home_best_player_apm")),
        away_best_player_apm=_float(row.get("away_best_player_apm")),
        home_worst_player_apm=_float(row.get("home_worst_player_apm")),
        away_worst_player_apm=_float(row.get("away_worst_player_apm")),
        home_apm_spread=_float(row.get("home_apm_spread")),
        away_apm_spread=_float(row.get("away_apm_spread")),
        home_lineup_apm_sum=_float(row.get("home_lineup_apm_sum")),
        away_lineup_apm_sum=_float(row.get("away_lineup_apm_sum")),
        home_off_court_best_apm=_float(row.get("home_off_court_best_apm")),
        away_off_court_best_apm=_float(row.get("away_off_court_best_apm")),
        apm_delta=_float(row.get("apm_delta")),
    )


class BacktestSimulator:
    def __init__(
        self,
        strategy: BaseStrategy,
        feature_rows: pd.DataFrame,
        synthetic_prices: pd.DataFrame,
        latency_possessions: int = 1,
        dk_validator: "DKPriceValidator | None" = None,
    ) -> None:
        self.strategy = strategy
        self.feature_rows = feature_rows.sort_values(["game_id", "possession_id"]).reset_index(drop=True)
        self.latency_possessions = latency_possessions
        self.dk_validator = dk_validator

        # Build synthetic price lookup: (game_id, possession_id) → synthetic_yes_price
        self._price_lookup: dict[tuple[str, int], int] = {
            (str(r["game_id"]), int(r["possession_id"])): int(r["synthetic_yes_price"])
            for _, r in synthetic_prices.iterrows()
        }

        # Build target lookup for run_actually_occurred validation
        self._target_lookup: dict[tuple[str, int], bool] = {}
        if "target_meaningful_run_5_scoring" in feature_rows.columns:
            for _, r in feature_rows.iterrows():
                key = (str(r["game_id"]), int(r["possession_id"]))
                self._target_lookup[key] = bool(r["target_meaningful_run_5_scoring"])

    def run(self, game_ids: list[str] | None = None) -> SimulationResult:
        result = SimulationResult()

        games = self.feature_rows["game_id"].unique()
        if game_ids is not None:
            games = [g for g in games if g in set(game_ids)]

        for game_id in games:
            game_rows = self.feature_rows[self.feature_rows["game_id"] == game_id].reset_index(drop=True)
            game_result = self._run_game(game_id, game_rows)
            result.trades.extend(game_result.trades)
            result.pnl_by_possession.extend(game_result.pnl_by_possession)

        logger.info(
            "Backtest complete: %d trades, net PnL $%.2f, win rate %.1f%%",
            result.num_trades,
            result.net_pnl,
            result.win_rate * 100,
        )
        return result

    def _run_game(self, game_id: str, game_rows: pd.DataFrame) -> SimulationResult:
        result = SimulationResult()
        self.strategy.on_game_start(game_id, {})

        n = len(game_rows)
        open_position: Position | None = None
        open_signal: Signal | None = None
        open_entry_row: pd.Series | None = None
        pending_order: Signal | None = None  # limit order waiting to fill

        for i in range(n):
            # Apply latency: strategy sees data from `latency_possessions` ago
            strategy_idx = max(0, i - self.latency_possessions)
            stale_row = game_rows.iloc[strategy_idx]
            current_row = game_rows.iloc[i]

            current_poss_id = int(current_row["possession_id"])
            synthetic_price = self._price_lookup.get(
                (game_id, current_poss_id), 50  # default to 50 if missing
            )

            # Build FeatureRow from stale data but with current synthetic price
            stale_price = self._price_lookup.get(
                (game_id, int(stale_row["possession_id"])), 50
            )
            feature_row = _row_to_feature_row(stale_row, stale_price)

            # --- Check for pending order fill ---
            if pending_order is not None and open_position is None:
                filled = self._try_fill(pending_order, synthetic_price)
                if filled:
                    fill = OrderFill(
                        direction=pending_order.direction,
                        fill_price=pending_order.limit_price,
                        size=pending_order.size,
                        possession_id=current_poss_id,
                    )
                    self.strategy.on_fill(fill)
                    open_position = Position(
                        direction=pending_order.direction,
                        entry_price=pending_order.limit_price,
                        size=pending_order.size,
                        entry_possession=current_poss_id,
                        current_price=synthetic_price,
                    )
                    open_signal = pending_order
                    open_entry_row = current_row
                    pending_order = None

            # --- Update open position ---
            if open_position is not None:
                open_position.current_price = synthetic_price
                exit_signal = self.strategy.on_position_update(open_position, feature_row)

                if exit_signal is not None:
                    trade = self._close_position(
                        open_position,
                        open_entry_row,
                        current_row,
                        synthetic_price,
                        exit_signal.reason,
                        open_signal,
                    )
                    result.trades.append(trade)
                    result.pnl_by_possession.append(trade.net_pnl)
                    open_position = None
                    open_signal = None
                    open_entry_row = None
                    self.strategy.on_position_closed()
                    continue

            # --- Request new signal (only if no open position and no pending order) ---
            if open_position is None and pending_order is None:
                signal = self.strategy.on_possession(feature_row)
                if signal is not None:
                    pending_order = signal

            result.pnl_by_possession.append(0.0)

        # Force-close any open position at game end
        if open_position is not None and open_entry_row is not None:
            last_row = game_rows.iloc[-1]
            last_poss_id = int(last_row["possession_id"])
            last_price = self._price_lookup.get((game_id, last_poss_id), 50)
            open_position.current_price = last_price

            trade = self._close_position(
                open_position, open_entry_row, last_row, last_price, "game_end", open_signal
            )
            result.trades.append(trade)

        self.strategy.on_game_end(game_id)
        return result

    def _try_fill(self, signal: Signal, current_synthetic_price: int) -> bool:
        """
        Simplified fill model: limit order fills when the market price is
        at or through the limit price.

        YES buy: fills if synthetic_yes_price <= limit_price (we're willing to pay more)
        NO buy:  NO price = 100 - synthetic_yes_price; fills if that <= limit_price
        """
        if signal.direction == "YES":
            return current_synthetic_price <= signal.limit_price
        else:
            no_price = 100 - current_synthetic_price
            return no_price <= signal.limit_price

    def _close_position(
        self,
        position: Position,
        entry_row: pd.Series,
        exit_row: pd.Series,
        exit_price: int,
        reason: str,
        signal: Signal | None = None,
    ) -> TradeRecord:
        # exit_price is always synthetic_yes_price.
        # entry_price for YES is a YES price; for NO it is a NO price (100 - yes_at_entry).
        # We must work in the same space for each direction.
        if position.direction == "YES":
            gross_pnl = (exit_price - position.entry_price) * position.size / 100.0
            exit_fee = _maker_fee(position.size, exit_price)
        else:
            no_exit_price = 100 - exit_price
            gross_pnl = (no_exit_price - position.entry_price) * position.size / 100.0
            exit_fee = _maker_fee(position.size, no_exit_price)

        entry_fee = _maker_fee(position.size, position.entry_price)
        total_fees = entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees

        game_id = str(entry_row["game_id"])
        entry_poss_id = int(entry_row["possession_id"])

        # --- Run probability and ground-truth target ---
        run_prob = signal.confidence if signal is not None else 0.0
        run_occurred = self._target_lookup.get((game_id, entry_poss_id), False)

        # --- DK line validation (optional) ---
        dk_wp_entry: float | None = None
        dk_wp_exit: float | None = None
        dk_net_pnl_val: float | None = None

        if self.dk_validator is not None:
            dk_result = self.dk_validator.get_dk_move(
                game_id,
                entry_poss_id,
                position.direction,
                lookahead_minutes=10,
                size=position.size,
            )
            if dk_result is not None:
                dk_wp_entry = dk_result.dk_wp_entry
                dk_wp_exit = dk_result.dk_wp_exit
                dk_net_pnl_val = dk_result.dk_net_pnl

        return TradeRecord(
            game_id=game_id,
            entry_possession=entry_poss_id,
            exit_possession=int(exit_row["possession_id"]),
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            gross_pnl=gross_pnl,
            fees=total_fees,
            net_pnl=net_pnl,
            exit_reason=reason,
            period=int(entry_row.get("period", 0)),
            score_diff_at_entry=int(entry_row.get("score_diff", 0)),
            lineup_delta_at_entry=float(entry_row.get("lineup_net_rating_delta", 0.0)),
            run_points_at_entry=int(entry_row.get("current_run_points", 0)),
            shot_sustainable_at_entry=bool(entry_row.get("home_scoring_sustainable", False)),
            minutes_into_game_at_entry=float(entry_row.get("minutes_into_game", 0.0)),
            run_prob_predicted=run_prob,
            run_actually_occurred=run_occurred,
            dk_wp_entry=dk_wp_entry,
            dk_wp_exit=dk_wp_exit,
            dk_net_pnl=dk_net_pnl_val,
        )


def run_backtest(
    strategy: BaseStrategy,
    game_ids: list[str] | None = None,
    feature_rows_path: str = "data/feature_store/feature_rows.parquet",
    synthetic_prices_path: str = "data/feature_store/synthetic_prices.parquet",
    dk_validator: "DKPriceValidator | None" = None,
) -> SimulationResult:
    """Convenience wrapper: load data and run a backtest."""
    feature_rows = pq.read_table(feature_rows_path).to_pandas()
    synthetic_prices = pq.read_table(synthetic_prices_path).to_pandas()

    sim = BacktestSimulator(strategy, feature_rows, synthetic_prices, dk_validator=dk_validator)
    return sim.run(game_ids)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from strategies.mean_reversion import MeanReversionStrategy

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_backtest(MeanReversionStrategy())
    print(f"Trades:    {result.num_trades}")
    print(f"Net PnL:   ${result.net_pnl:.2f}")
    print(f"Gross PnL: ${result.gross_pnl:.2f}")
    print(f"Fees:      ${result.total_fees:.2f}")
    print(f"Win rate:  {result.win_rate:.1%}")
