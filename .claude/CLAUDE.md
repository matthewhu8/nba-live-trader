# Kalshi Basketball Trading System

## End Goal
Make money live trading NBA games on Kalshi prediction markets via profitable maker orders.

## Core Thesis
Predict short-term scoring runs (3-5 min windows) using lineup data. Enter before runs show on scoreboard, before market reprices.
- Retail overreacts to runs they watched. Underreacts to lineup changes we notice first.
- Kalshi's emotionally-reactive market reprices slowly — that's our edge.

## Non-Negotiable Rules
- **Maker orders only** (4x cheaper fees). Taker only if edge clears higher fee.
- **Fees in every signal** — if edge doesn't clear fees, don't trade
- **Paper trade before real money** — min 4 weeks
- **Position limits enforced before every order** (no exceptions)
- **One execution touchpoint** — all Kalshi calls via `live-trader/go/`
- **Kill switch required** before any live order
- **No hardcoded credentials** — env vars only

## Credentials
```
KALSHI_KEY_ID=<key>
KALSHI_PEM_PATH=~/.kalshi/private_key.pem  # outside repo
```

## Fee Structure
```
fee = ceil(rate × contracts × P × (1-P) × 100) / 100     P = price/100, result in $
  Maker rate 0.0175   →  $0.44 per 100 @ 50¢
  Taker rate 0.07     →  $1.75 per 100 @ 50¢
```
The `P × (1-P)` term is required. Fees peak at 50¢ and fall toward both ends of
the book. The old form here (`0.0175 × contracts × price`) omitted it and gave
$0.88 at 50¢, contradicting its own $0.44 example.

**Compute this in `Decimal`, never float.** In float64 the product lands a few ulp above
an exact cent and `ceil` rounds a whole cent up, disagreeing with Kalshi's published table
on three of five taker rows — 50¢ returns $1.76, 20¢ returns $1.13, 10¢ returns $0.64. An
earlier revision of this table recorded the buggy **$1.76** as if it were correct. Assert
against the published table, not against the code: `tests/test_fees.py`.

Canonical implementation: `backtesting/mmoe_backtest.py::_fee_one_leg`. The only other one
is `inference/dashboard.py::kalshiMakerFee` (JavaScript). **There is no Go fee function** —
`orders.go::calcNetPnL` returns gross P&L with no fee deduction, so a `orders.go::kalshiFee`
cross-reference is a dangling pointer.

Entry is always maker. Exits pay maker only on `take_profit` (PR #50 rests the TP limit);
`stop_loss`, `momentum_flip`, `garbage_time` and `time_gate` cross the book and pay taker.
**Break-even at TP=5/SL=3 is 55.7%** (win +$4.12, loss −$5.19). See `skills/backtesting.md`.

All prices in cents (1–99), never floats. YES @ 60¢ = 60% implied probability.
**P&L is reported in DOLLARS.** A price delta in cents × contracts is cents —
divide by 100. A 100-contract position cannot swing more than $100 total.
100 contracts × 5¢ = **$5.00, not $500** — mixing cents and dollars understated fees 100×.

## Current Status
- [x] Phase 1-5: Data foundation, feature store, backtesting, MMoE model, execution engine
- [ ] **Phase 6: Paper trading** ← CURRENT
  - MMoE (58-feat, 2026-08-03): Head A AUCPR 0.1590, Head B dir acc 62.5%, Head C Brier 0.0860
  - Entry gates: `|traj_final| ≥ 0.08`, `run_length ≥ 2`, price 30-70¢, no garbage time

> 🔴 **The exit-window fix covers the BACKTEST ONLY.**
> `models/targets/exit_simulator.py:237` still reads `entry_ts = pd.Timestamp(row["wall_clock_ts"])`
> with no feed delay applied. That file generates **Head B's and Head C's training labels**, so
> every label the deployed model learned from — and every Head B metric scored against them,
> including the **62.5% directional accuracy that selected the deployed checkpoint** — was
> produced by simulating exits ~20s before a live trader could have acted.
> Fixing it invalidates the current checkpoints and requires a retrain, so it is scoped
> separately (Level 2). **Do not retrain until it lands.** See `skills/data-integrity.md`.

> ⚠️ **`207 trades / 47.8% / +$37,409` is VOID — do not quote it.** That backtest had four
> compounding defects (exit-window lookahead, take-profits credited above the resting limit,
> fees hardcoded to 0.0, and P&L reported in cents but labelled dollars — so 100× too large).
> Full accounting in `skills/backtesting.md`.
>
> ⚠️ **`80 trades / 30.0% / −$136.42` is ALSO superseded.** It was measured on the
> **83-feature** model on 2026-08-03 at 16:01, hours before this branch merged PR #51 and
> swapped in the 58-feature model. It reproduces exactly at `85d8a66`, but it does not
> describe the current code. It was additionally measured with a fee function that
> overcharged up to a cent per leg.
>
> **Current baseline (2026-08-05, post-merge with PR #54):**
> **181 trades, 26.0% win rate, gross −$6.00, fees $318.35, net −$324.35** across 44 games.
> Per-trade **−$1.79**, game-clustered bootstrap 95% CI **[−$2.40, −$1.17]**. Config:
> `--use-traj-for-side --min-abs-traj 0.08 --min-run-length 2 --hold-seconds 240
> --threshold 0.15 --traj-aggregator mean`. Full provenance in `skills/backtesting.md`.
> The strategy still loses money with well over 99% confidence — but see the Level 2 note
> above before treating that as a verdict on the idea.
>
> **Quote the command, not just the number.** The corrected run used
> `--traj-aggregator mean --threshold 0.15`, *not* the `--threshold 0.0` command block that
> sat above it in `skills/backtesting.md` for two days. Config is part of a result.
>
> Head B's `62.5%` dir acc is also provisional: every checkpoint to date was trained before
> the 2026-08-03 `wall_clock_ts` repair, and **55% of Head B's validation rows carried
> settled prices**, so early stopping and metric reporting both ran on corrupt data.
> See `skills/model-provenance.md`.
>
> `min_abs_traj_entry: 0.12` in `trading.yaml` was tuned on these figures and has **no validated
> basis in either direction** — the 0.08-vs-0.12 comparison was between two inflated buckets.
> Re-derive before the next live session.

**Test set (2024-26 season):** Sacred. Never touch during dev.

---

## Key Reminders
- End goal: live profitable trading. Every decision serves this.
- Lookahead bias: backtests look great, live trading terrible. Be paranoid.
- Paper mode default. Confirm before changing execution code.
- Only use taker if edge clears higher fee. Maker is default.
- Never trade garbage time or blowouts. Model off.
- Suspicious backtest results? Assume lookahead bias first.
- Minimum 4 weeks paper trading before real money.
