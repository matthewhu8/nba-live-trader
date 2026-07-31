# Backtesting & Strategy Evaluation

## MMoE Backtest (Validated)
```bash
python -m backtesting.mmoe_backtest \
  --use-traj-for-side \
  --min-abs-traj 0.08 \
  --min-run-length 2 \
  --hold-seconds 240 \
  --threshold 0.0
```

**Result:** 207 trades, 47.8% win rate, +$37,409 net (110 val games)

## Entry Gates (Current Config)
| Gate | Status |
|------|--------|
| `\|traj_final\| ≥ 0.08` | required |
| `run_length ≥ 2` | required |
| Price 30–70¢ | required |
| No garbage time/blowout | required |
| `run_prob ≥ 0.10` | not used — reduces P&L |

Direction from `traj_final` sign. Edit `trading.yaml` to tune.

## MMoE Performance
- **Head A (Run Classifier):** AUCPR 0.1593 vs baseline 0.0840 (+90%)
- **Head B (Price Trajectory):** Dir Acc 61.1% on Kalshi moves
- **Head C (Survival Hazard):** Brier 0.0860 across 10 horizons

83 features (58 basketball + 10 pregame + 14 market + 1 flag).
37K params. Retrain every ~10 games.

## Strategy Interface
```python
class BaseStrategy:
    def on_game_start(self, pregame_context) -> None
    def on_possession(self, features) -> Signal | None
    def on_fill(self, fill) -> None
    def on_position_update(self, position) -> ExitSignal | None
    def on_game_end(self, final_state) -> None
```

## Metrics to Always Report
- Sharpe, Sortino, Max Drawdown
- Win rate, Trade count, Avg hold time
- Fee drag as % of gross
- **Never cherry-pick.** Report all. Context breakdown reveals real insights.

## Train/Test Split (Sacred)
- Train: 2021-22, 2022-23
- Validation: 2023-24 (tune hyperparams here)
- Test: 2024-25 (evaluate once at end, never during dev)
