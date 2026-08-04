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
Maker:  ceil(0.0175 × contracts × P × (1-P))  (~$0.44 per 100 @ 50¢)
Taker:  ceil(0.07   × contracts × P × (1-P))  (~$1.75 per 100 @ 50¢)
        P = price in dollars (50¢ → 0.50), rounded up to the cent
```
The `P × (1-P)` term is required. Without it the formula overstates fees ~4× at the 50¢
midpoint — the worked examples above only hold with it. Canonical implementations:
`mmoe_backtest.py::_fee_one_leg`, `inference/dashboard.py::kalshiMakerFee`.

Entry is always maker. Exits pay maker only on `take_profit` (PR #50 rests the TP limit);
`stop_loss`, `momentum_flip`, `garbage_time` and `time_gate` cross the book and pay taker.
**Break-even at TP=5/SL=3 is 55.7%** (win +$4.12, loss −$5.19). See `skills/backtesting.md`.

All prices in cents (1–99), never floats. YES @ 60¢ = 60% implied probability.
100 contracts × 5¢ = **$5.00, not $500** — mixing cents and dollars understated fees 100×.

## Current Status
- [x] Phase 1-5: Data foundation, feature store, backtesting, MMoE model, execution engine
- [ ] **Phase 6: Paper trading** ← CURRENT
  - MMoE (58-feat, 2026-08-03): Head A AUCPR 0.1590, Head B dir acc 62.5%, Head C Brier 0.0860
  - Entry gates: `|traj_final| ≥ 0.08`, `run_length ≥ 2`, price 30-70¢, no garbage time

> ⚠️ **`207 trades / 47.8% / +$37,409` is VOID — do not quote it.** That backtest had four
> compounding defects (exit-window lookahead, take-profits credited above the resting limit,
> fees hardcoded to 0.0, and P&L reported in cents but labelled dollars — so 100× too large).
> Corrected on the same config: **80 trades, 30.0% win rate, −$136.42**, and −$82.53 in
> production-equivalent form. The measured strategy loses money with 99% confidence.
> Full accounting in `skills/backtesting.md`.
>
> Head B's `62.5%` dir acc is also provisional: every checkpoint to date was trained before
> the 2026-08-03 `wall_clock_ts` repair, and **55% of Head B's validation rows carried
> settled prices**, so early stopping and metric reporting both ran on corrupt data.
> See `skills/model-provenance.md`.
>
> `min_abs_traj: 0.12` in `trading.yaml` was tuned on these figures and has **no validated
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
