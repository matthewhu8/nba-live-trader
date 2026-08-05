# Backtesting & Strategy Evaluation

## MMoE Backtest — current baseline

**The config is part of the result. Always quote both.**

```bash
python -m backtesting.mmoe_backtest \
  --use-traj-for-side \
  --min-abs-traj 0.08 \
  --min-run-length 2 \
  --hold-seconds 240 \
  --threshold 0.15 \
  --traj-aggregator mean
```

**Baseline @ `fix/backtest-exit-window` merged with `origin/main` (2026-08-05):**

| trades | win rate | gross | fees | net | games |
|---|---|---|---|---|---|
| 181 | 26.0% | −$6.00 | $318.35 | **−$324.35** | 44 of 69 |

Per-trade edge **−$1.79**, game-clustered bootstrap 95% CI **[−$2.40, −$1.17]**, P(edge ≥ 0)
< 0.0001 over 10,000 resamples. 100 contracts, TP=5 / SL=3, 20s feed delay, local parquet
cache (71 games / 14,239 possessions / 770,054 ticks, Apr 15 – May 17 2026).

> ⚠️ **This measures a model trained on defective labels.** `exit_simulator.py` still
> generates Head B/C training targets with no feed delay (see `data-integrity.md`), so the
> number describes the current pipeline honestly but says nothing about whether the strategy
> could work once the labels are fixed. Do not read an improvement or a regression into it.

### Superseded numbers — do not quote

| figure | why it is dead |
|---|---|
| `207 trades / 47.8% / +$37,409` | four measurement defects, 83-feature model, and a different config from the one printed beside it |
| `81 trades / 51.9% / +$16,100` | same four defects |
| `80 trades / 30.0% / −$136.42` | **defects fixed, but measured on the 83-feature model** at `85d8a66` (2026-08-03 16:01), hours before this branch merged PR #51 and swapped in the 58-feature model. Also measured with the pre-Decimal fee function. Reproduces exactly at `85d8a66`; does not describe current code. |
| `63 trades / 34.9% / −$82.53` (`--prod-features`) | same 83-feature provenance |

**Two lessons, both cheap to repeat and expensive to catch:**

1. **The command in this file was wrong for two days.** The `--threshold 0.0` block above the
   corrected table belonged to the old 207-trade run; the corrected run actually used
   `--threshold 0.15 --traj-aggregator mean` (which is what `trading.yaml` ships). Re-running
   the documented command gave 466 trades / −$781.40 and looked like non-determinism. The real
   config was recovered by reading `traj_aggregator` and `min(run_prob)` back out of the saved
   position CSVs. **Record the command next to the number, every time.**
2. **A model swap silently invalidates a baseline.** Nothing failed, nothing warned; the
   branch merged PR #51 and the recorded baseline quietly stopped describing the code. Check
   `feature_config.py` and the checkpoint alongside any figure you are about to compare.

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
implementation: `mmoe_backtest.py::_fee_one_leg`. The only other one is
`inference/dashboard.py::kalshiMakerFee` (JavaScript). **There is no Go fee function** —
`orders.go::calcNetPnL` returns gross P&L with no fee deduction, so the
`orders.go::kalshiFee` cross-reference that appeared in `CLAUDE.md` pointed at nothing.

**Compute it in `Decimal`, not float.** Verified against Kalshi's published table
(effective 2026-02-05) in `tests/test_fees.py`:

| price | taker | maker | naive float64 gave |
|---|---|---|---|
| $0.10 | $0.63 | $0.16 | taker **$0.64** |
| $0.20 | $1.12 | $0.28 | taker **$1.13**, maker **$0.29** |
| $0.50 | $1.75 | $0.44 | taker **$1.76** |
| $0.85 | $0.90 | $0.23 | correct |
| $0.90 | $0.63 | $0.16 | correct |

`0.07 * 100 * 0.5 * 0.5 * 100` evaluates to `175.00000000000003`, so `ceil` promoted a whole
cent. Three of five taker rows were wrong. Both the branch's `_fee_one_leg` and the
`kalshi_fee` added by PR #54 had it, and `.claude/CLAUDE.md` on `main` recorded the buggy
**$1.76** as the correct figure — a bug that had been written down as ground truth. Assert
against the published table, never against the implementation.

**Which leg pays what:** entry is always maker. Exits pay maker only on `take_profit`
(PR #50 rests the TP limit); `stop_loss`, `momentum_flip`, `garbage_time` and `time_gate`
cross the book and pay taker.

**Break-even at TP=5/SL=3 is 55.7%**: a win nets $5.00 - $0.88 = +$4.12, a loss
-$3.00 - $2.19 = -$5.19, so 5.19/9.31. The model currently delivers ~37% on gross moves.
No threshold tuning closes an 18-point gap.

### The invariant — the original one was a tautology

The guard used to read:

```python
exit_abs_ts = entry_anchor_ts + pd.Timedelta(seconds=sim.exit_time_offset_s)
if sim.exit_time_offset_s < 0 or exit_abs_ts < wct + pd.Timedelta(seconds=feed_delay_s):
```

`entry_anchor_ts` **is** `wct + feed_delay_s`, so the second clause reduces to
`exit_time_offset_s < 0` — an exact duplicate of the first. This file previously claimed it
was "verified to fire when the bug is reintroduced." **It was not.** Measured 2026-08-05:
with the original three-line defect restored (`future_ticks > wct`, `future_poss > wct`,
`entry_wall_clock=wct`) the backtest ran to completion — 178 trades, 30.9% win rate, 0.03s
minimum hold, net −$258.04 against the correct −$324.35 — and the guard never fired. It had
been protecting nothing for two days while being cited as evidence that it was.

The replacement asserts the **inputs** to `simulate_exit`, not its output, because the offset
it returns is measured from whatever anchor it was handed:

- `exit_search_start` is the single name the tick window, the possession window and the call
  all read, so there is one place to get it wrong;
- the window may not open before `wct + feed_delay_s`;
- nothing reachable by the exit search may predate the anchor (this is the clause that
  catches the actual historical defect);
- a non-`time_gate` exit must land within 1 ms of a real tick, which catches an anchor and a
  tick window that have drifted apart.

Confirmed to raise on game `0042500101` when `exit_search_start` is pointed back at `wct`,
and confirmed not to fire on any of the 181 valid trades.

Do **not** assert `hold_time_s >= feed_delay_s` — once the anchor is correct, hold time is
measured *from* the anchor and a legitimate 1s hold exists.

**General rule this cost us:** an invariant written in terms of quantities that are equal by
construction is not an invariant. Before trusting a guard, reintroduce the bug and watch it
fail — over the full population, not one synthetic row.

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

## Entry Gates

`live-trader/config/trading.yaml` is the source of truth. This table had drifted from it on
two of five rows; check both before quoting either.

| gate | `trading.yaml` (live) | backtest baseline flag |
|---|---|---|
| trajectory magnitude | `min_abs_traj_entry: 0.12` ⚠️ **contaminated** | `--min-abs-traj 0.08` |
| Head A run prob | `min_run_prob_entry: 0.0` | `--threshold 0.15` |
| run length | `min_run_length_entry: 2` | `--min-run-length 2` |
| aggregator | `traj_aggregator: "mean"` | `--traj-aggregator mean` |
| price band | 30–70¢ | 30–70¢ |
| regime | `blowout_margin_pts: 30` | `_filter_to_traded_regime` |

Two divergences are deliberate and two are not:

- **`min_run_prob_entry: 0.0` is intentional**, not a bug — Head A's gating collapsed and
  routes 95%+ to the conservative expert, so the gate carries no information. The backtest's
  `--threshold 0.15` is a leftover default and means the two are **not** measuring the same
  strategy. Reconcile before reading across them.
- **`min_abs_traj_entry: 0.12` vs the backtest's 0.08** is unresolved. The 0.12 was tuned on
  inflated figures (see `trading.yaml:40`) and has no validated basis. Re-derive after Level 2.

Direction comes from the sign of the aggregated trajectory (`traj_used`), not `traj_final`,
whenever `--use-traj-for-side` is set — which the baseline and `agent.go` both do.

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
