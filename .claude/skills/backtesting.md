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

> **VOID as of 2026-08-02.** Measured on the pre-consolidation 83-feature model. The feature
> set is now 58 columns *and* training excludes overtime/blowouts, so this is not a valid
> comparison baseline. Re-run before drawing any conclusion. Expect trade count to shift on
> its own: `min_abs_traj` and `min_run_length` were tuned on feature distributions that no
> longer exist, so re-tune them before judging P&L. See `docs/FEATURE_CONSOLIDATION.md`.

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

| Head | Prior (83-feat) | Physics-only | + market encoder |
|---|---|---|---|
| A — run classifier (AUCPR) | 0.1593 | **0.1675** | 0.1590 |
| B — price trajectory (dir acc) | 61.1% | 61.3% | **62.5%** |
| C — survival hazard (Brier) | 0.0860 | **0.0848** | 0.0860 |
| val loss | — | **0.4185** | 0.4241 |

58 features (33 physics + 11 pregame + 14 market). 30K params. Retrain every ~10 games.

**Select on Head B, not val loss.** Val loss is a 3-head weighted sum, but entries are driven
almost entirely by Head B's trajectory sign (`agent.go`). The physics-only run wins on val
loss and trades worse.

Deltas are single-run with no confidence intervals and may be seed noise. Both runs peaked at
epoch 5 of 20 then early-stopped — an unhealthy curve worth investigating separately.

`--no-market-encoder` on `train_mmoe.py` reproduces the physics-only column.

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
