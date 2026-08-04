# Backtesting & Strategy Evaluation

## MMoE Backtest (⚠️ baseline VOID — see below)
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
>
> **And it was never valid in the first place** — not merely stale. Four measurement defects
> inflated it; the corrected figure on the same config is **negative**. Read the next section
> before re-running anything.

## ⚠️ Four defects corrected 2026-08-03 — branch `fix/backtest-exit-window`

The `+$37,409` and `+$16,100` figures were not merely stale, they were **wrong in four
compounding ways**, all inflating in the same direction. Every one is fixed on that branch and
none are fixed on `main`.

| # | defect | effect |
|---|---|---|
| 1 | **Exit-window lookahead.** Entry price read at `wct + feed_delay_s`, exit search started at `wct` | positions could exit up to 20s *before* entering. 25 of 81 trades held <20s and won **25 for 25** |
| 2 | **TP over-crediting.** TP exits booked at the tick that breached the threshold, not the resting limit at `entry + TP` (PR #50) | 18 of 36 TP exits overshot — median 6c past the limit, max 24c |
| 3 | **Fees hardcoded to 0.0.** `_compute_maker_fees()` returned `0.0` unconditionally | zeroed the taker cost on every stop-out, i.e. on every loser |
| 4 | **P&L in cents, labelled dollars.** `net = gross_cents * contracts - fees_dollars` | every figure **100x too large**, and dollar fees against a cents quantity made fees look 100x too small — which is why defect 3 looked immaterial |

Corrected, same config, 69-game cache:

| | trades | win rate | gross | fees | net |
|---|---|---|---|---|---|
| as reported | 81 | 51.9% | — | $0 | **+$16,100** |
| corrected | 80 | 30.0% | +$2.00 | $138.42 | **-$136.42** |
| production-equivalent (`--prod-features`) | 63 | 34.9% | +$21.00 | $103.53 | **-$82.53** |

Game-clustered bootstrap: both per-trade means are significantly negative (backtest 95% CI
[-$2.70, -$0.79]; production [-$2.33, -$0.24]). The difference between them is **not**
significant (+$0.41/trade, CI [-$0.95, +$1.84]).

**Fixing the pregame pipeline would not create an edge** — it adds ~21% more trades to a
strategy with negative per-trade EV. The broken pipeline has been reducing losses.

### Fee model — the table in CLAUDE.md is missing a term

```
maker: ceil(0.0175 x contracts x P x (1-P) x 100) / 100
taker: ceil(0.07   x contracts x P x (1-P) x 100) / 100
       P = price in dollars (50c -> 0.50)
```

`rate x contracts x price` overstates the fee ~4x at the 50c midpoint. Canonical
implementations: `mmoe_backtest.py::_fee_one_leg`, `inference/dashboard.py::kalshiMakerFee`.

**Which leg pays what:** entry is always maker. Exits pay maker only on `take_profit`
(PR #50 rests the TP limit); `stop_loss`, `momentum_flip`, `garbage_time` and `time_gate`
cross the book and pay taker.

**Break-even at TP=5/SL=3 is 55.7%**: a win nets $5.00 - $0.88 = +$4.12, a loss
-$3.00 - $2.19 = -$5.19, so 5.19/9.31. The model currently delivers ~37% on gross moves.
No threshold tuning closes an 18-point gap.

### The invariant

`mmoe_backtest.py` raises on any exit earlier than `wct + feed_delay_s`. It is stated against
`wct` rather than the entry anchor so that re-anchoring fails loudly instead of silently
reinflating results. Verified to fire when the bug is reintroduced.

Do **not** assert `hold_time_s >= feed_delay_s` — once the anchor is correct, hold time is
measured *from* the anchor and a legitimate 1s hold exists.

### Never estimate a fix by filtering

Dropping sub-20s trades from the biased run suggested -$2,600. Actually re-running gave +$62
(pre-units-fix). A corrected run re-simulates those positions from the right anchor, and the
overlap guard then admits a different trade set. Filtering a biased output is not a fix.

### Event triggers: no benefit, and the test was compromised

118 trades, 36.4% win, **-$200.91**, versus its own possession-close baseline of 208 trades /
33.2% / -$342.75. **Per trade the two are indistinguishable: -$1.70 vs -$1.65.** Mean earliness
is 2.4s against a 20s feed delay — there was never room for a timing edge.

Caveat: the cache has `was_sub` and `was_foul` all-NaN, so the model could not see that a
substitution had occurred. Re-export before treating this as final. See `data-integrity.md`.

### Sample size

80 trades came from only 30 of 69 games. Per `model-provenance.md`, demonstrating a
+$1.00/trade edge needs ~106 games. Size the validation set before running, not after.

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
