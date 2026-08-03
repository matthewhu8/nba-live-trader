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
  Taker rate 0.07     →  $1.76 per 100 @ 50¢
```
The `P × (1-P)` term is required. Fees peak at 50¢ and fall toward both ends of
the book. The old form here (`0.0175 × contracts × price`) omitted it and gave
$0.88 at 50¢, contradicting its own $0.44 example.

Implemented once per language: `backtesting/mmoe_backtest.py::kalshi_fee` and
`live-trader/go/orders.go::kalshiFee`. Entry is always maker. TP exits rest as
maker; SL / TRAIL / TIME exits cross as taker at 4× the rate.

All prices in cents (1–99), never floats. YES @ 60¢ = 60% implied probability.
**P&L is reported in DOLLARS.** A price delta in cents × contracts is cents —
divide by 100. A 100-contract position cannot swing more than $100 total.

## Current Status
- [x] Phase 1-5: Data foundation, feature store, backtesting, MMoE model, execution engine
- [ ] **Phase 6: Paper trading** ← CURRENT
  - MMoE: Head A AUCPR 0.1593, Head B dir acc 61.1%, Head C Brier 0.0860
  - Backtest: 207 trades, 47.8% win rate, +$37,409 net
  - Entry gates: `|traj_final| ≥ 0.08`, `run_length ≥ 2`, price 30-70¢, no garbage time

**Test set (2024-25 season):** Sacred. Never touch during dev.

---

## Key Reminders
- End goal: live profitable trading. Every decision serves this.
- Lookahead bias: backtests look great, live trading terrible. Be paranoid.
- Paper mode default. Confirm before changing execution code.
- Only use taker if edge clears higher fee. Maker is default.
- Never trade garbage time or blowouts. Model off.
- Suspicious backtest results? Assume lookahead bias first.
- Minimum 4 weeks paper trading before real money.
