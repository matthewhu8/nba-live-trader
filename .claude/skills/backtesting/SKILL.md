---
name: backtesting
description: Working on the backtesting simulator, strategy implementations, evaluator, backtest results analysis, or adding new strategies
---

# Backtesting Skill

## Core Principle: Enforce Reality
The simulator's only job is enforcing constraints that exist in live trading.
Every shortcut taken here = a gap between backtest and live performance.
If a backtest result looks suspiciously good, assume lookahead bias first.

## Data Latency is Mandatory
Live Sportradar feed = 15-20s behind real action.
In simulation, all features derived from live events must be delayed:
```python
LIVE_FEED_LATENCY_SECONDS = 17  # use middle of 15-20s range

def apply_latency(wall_clock_time: datetime, feature_type: str) -> datetime:
    if feature_type == "live_feed":
        return wall_clock_time + timedelta(seconds=LIVE_FEED_LATENCY_SECONDS)
    return wall_clock_time  # pre-game computed features have no latency
```
Pre-game computed features (lineup ratings, rotation tendencies) have no latency.
They were computed before the game started.

## Strategy Interface (every strategy must implement this)
```python
class BaseStrategy:
    def on_game_start(self, pregame_context: PreGameContext) -> None: ...
    def on_possession(self, features: FeatureRow) -> Signal | None: ...
    def on_fill(self, fill: OrderFill) -> None: ...
    def on_position_update(self, position: Position) -> ExitSignal | None: ...
    def on_game_end(self, final_state: GameState) -> None: ...
```
Strategies never compute features themselves. They only read from FeatureRow.
Strategies never call the Kalshi API. They return Signal objects.

## Strategies in `strategies/`
- `mean_reversion.py` — fade unsustainable scoring runs
- `lineup_edge.py` — enter on substitution that creates favorable lineup delta
- `rotation_anticipation.py` — enter before expected sub that will create edge
- `momentum.py` — ride sustainable scoring runs
- `composite.py` — weighted combination of above

## Fee Application
Apply maker fees to every simulated fill. Never forget fees.
```python
net_pnl = gross_pnl - (0.0175 * contracts * fill_price / 100)
```
If fee_drag / gross_pnl > 40% — something is wrong with the strategy.
Too many small trades that barely clear fees.

## Blowout Guard
```python
if features.is_blowout or features.is_garbage_time:
    return None  # never trade garbage time
```
This must be the first check in every strategy's on_possession().

## Synthetic Kalshi Prices (until real data available)
Location: `models/synthetic_kalshi.py`
Parameters to tune once real data is collected:
- `lag_seconds`: how far behind sharp book Kalshi reprices (15-45s range)
- `overreaction_coeff`: how much Kalshi overshoots on scoring runs
- `reversion_speed`: how fast synthetic price mean-reverts to sharp book
Large discrepancy between synthetic and real backtest = wrong assumptions. Investigate.

## Train/Test Split — Non-Negotiable
```
2021-22, 2022-23  →  training
2023-24           →  validation (tune here)
2024-25           →  TEST SET — touch only once, final evaluation only
Current season    →  paper trading observation
```
Never evaluate on test set during development. Never shuffle across seasons.

## Evaluator Output (standard metrics every run)
Required fields in every results object:
- sharpe_ratio, sortino_ratio
- total_pnl_gross, total_pnl_net, fee_drag_pct
- win_rate, n_trades, avg_hold_time_min
- max_drawdown, max_drawdown_recovery_possessions
- pnl_by_quarter, pnl_by_score_diff_bucket, pnl_by_run_length_at_entry
- signal_calibration_curve, edge_predicted_vs_captured

Context breakdown (by_quarter, by_score_diff_bucket etc.) is where improvement signals live.
A weak overall strategy may be strong in specific contexts — always look for this.

## Simulator Replay Order
Games must replay in strict chronological order. Never shuffle.
Within a game, possessions in strict time order. Never shuffle.
The model must not see future seasons during training on past seasons.