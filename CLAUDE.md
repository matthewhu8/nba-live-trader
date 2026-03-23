# Kalshi Basketball Trading System

## What This Is
Swing trading system for live NBA games on Kalshi prediction markets.
We predict short-term scoring runs (3-5 min windows) and enter positions
BEFORE the run appears on the scoreboard and BEFORE Kalshi reprices.

We are NOT predicting game winners. We are NOT arbitraging sportsbook lines.
We ARE exploiting Kalshi's slower, retail-driven repricing of lineup changes
and scoring run overreactions — using basketball context most bots don't have.

This is a learning project. I'm new to algo trading but comfortable with Python.
Explain tradeoffs before implementing. Flag anything that could cause financial loss.

---

## Hard Rules
- **Maker orders only** — limit orders always, never market orders (4x fee difference)
- **Paper mode is always default** — live requires explicit opt-in
- **All Kalshi API calls** go through `execution/kalshi_client.py` only
- **`risk/position_limits.py`** called before every order, no exceptions
- **Kill switch** must exist before any live order is placed
- **No credentials hardcoded** — environment variables only
- **Blowout / garbage time** (score_diff > 20, Q4 < 6min) — model off, no trades

---

## Fee Structure
```
Maker: 0.0175 × contracts × price_cents / 100
Taker: 0.07   × contracts × price_cents / 100
```
Always calculate edge after fees before signaling a trade.
All prices are integers in cents (1–99). Never floats.

---

## Stack
- **Data**: `nba_api` (historical), Kalshi WebSocket (record from day one)
- **Core signal**: lineup net rating delta between active 5-man units
- **Supporting signals**: shot quality, run length, rotation timing, foul state
- **Model**: XGBoost run predictor → RL agent for entry/exit decisions
- **Backtest**: point-in-time feature store, synthetic Kalshi prices until real data builds
- **Execution**: maker-only, paper mode default, position limits enforced

---

## Current Status
**Phase: 1 — Data Foundation**

- [ ] Kalshi price recorder deployed ← START HERE (data lost forever if not running)
- [ ] nba_api historical pull (3 seasons)
- [ ] Raw parquet tables built
- [ ] Feature store (Phase 2)
- [ ] Backtesting + strategies (Phase 3)
- [ ] Run predictor + RL agent (Phase 4)
- [ ] Execution layer + paper trading (Phase 5)
- [ ] Live trading (Phase 6)

Update this status as phases complete.

---

## Conventions
- Async (asyncio) for all ingestion and live feed handling
- Type hints on all functions
- `logging` everywhere — never `print()`
- Parquet for all stored data
- Never commit `.env`, `data/`, `logs/`