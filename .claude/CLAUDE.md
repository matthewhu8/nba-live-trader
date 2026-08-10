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

**Compute in `Decimal`, never float** — float64 `ceil` promotes a whole cent on 3 of 5
taker rows. Assert against Kalshi's published table, not the code: `tests/test_fees.py`.

Implemented once per language: `mmoe_backtest.py::_fee_one_leg` and
`inference/dashboard.py::kalshiMakerFee` (JS). **No Go fee function exists** —
`orders.go::calcNetPnL` is gross of fees.

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

> 🟡 **Level 2 landed 2026-08-10: the exit simulator now anchors labels at
> `wall_clock_ts + feed_delay`.** Retraining is unblocked, but **every existing Head B number
> was measured on anti-causal labels** — including the 62.5% dir acc that picked the deployed
> checkpoint. Head C was never affected (its hazards come from `kalshi_targets.py`, not the
> exit simulator; the old "Head B/C" wording here was wrong). Head B's loss mask is a
> threshold on the labels, so pre/post `loss_b` is **not comparable** — score the old
> checkpoint on the new labels instead. See `skills/data-integrity.md`.
>
> ⚠️ **Baseline: 181 trades / 25.4% / −$346.92** (2026-08-10, possession-event feed delay).
> `+$37,409`, `+$16,100`, `−$136.42` and now **−$324.35** are all superseded — **do not quote
> them.** −$324.35 was the same code minus the possession delay; the CI [−$2.40, −$1.17] was
> computed on it and has not been recomputed.
> **Always quote the config with the number**; a wrong command sat beside `−$136.42` for two
> days. Command, provenance and the four defects: `skills/backtesting.md`.
>
> ⚠️ Head B's `62.5%` and every checkpoint predate the `wall_clock_ts` repair; 55% of Head B's
> val rows carried settled prices. See `skills/model-provenance.md`.
>
> ⚠️ `min_abs_traj_entry: 0.12` in `trading.yaml` was tuned on inflated figures — no validated
> basis in either direction. Re-derive before the next live session.

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
