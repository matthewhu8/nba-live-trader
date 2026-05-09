---
name: kalshi-execution
description: Working on Kalshi API integration, order placement, WebSocket feed, order management, paper trading, or the execution layer
---

# Kalshi Execution Skill

## The Golden Rule
ALL Kalshi API calls go through `execution/kalshi_client.py` only.
Nothing else in the codebase touches the Kalshi API directly. Ever.
This is what allows the entire system to run in paper mode by changing one flag.

## Paper Mode is Always Default
```python
PAPER_MODE = True  # default in all execution code
```
Paper mode logs what would have been placed. Nothing hits the real API.
Live mode requires explicit override. Never make live mode the default.
Confirm paper mode is active before running any execution code unless explicitly told otherwise.

## Price Conventions
- All prices are integers in cents: range 1–99
- 60¢ = 60% implied win probability for YES
- Never use floats for prices internally
- Never use decimals in order parameters

## Fee Math (apply before every signal approval)
```python
maker_fee = 0.0175 * contracts * price_in_cents / 100
taker_fee = 0.07   * contracts * price_in_cents / 100
```
We are always makers. Taker orders are never placed under any circumstance.
A limit order that would immediately cross the spread = cancel, reprice, resubmit.

## Order Types
- Limit orders ONLY. Always.
- Place slightly inside the spread to guarantee maker status
- If spread is 1¢ wide, wait — don't cross it

## Kalshi API Reference
- REST docs: https://docs.kalshi.com
- Auth: API key from env var `KALSHI_API_KEY` — never hardcoded
- Order placement endpoint: POST /trade-api/v2/portfolio/orders
- Order book: GET /trade-api/v2/markets/{ticker}/orderbook
- WebSocket: wss://api.kalshi.com/trade-api/ws/v2

## Market ID Convention
Basketball market tickers on Kalshi follow pattern: `KXNBA-[DATE]-[TEAM]-[TYPE]`
Always fetch current market list before a game — don't hardcode tickers.

## WebSocket Feed (kalshi_recorder.py and live system)
- Subscribe to: orderbook_delta, trade, ticker
- Reconnect logic required — connection drops during games
- Buffer events during reconnection — never drop ticks
- All ticks written to `data/raw/kalshi_price_ticks.parquet`

## Order Manager
`execution/order_manager.py` responsibilities:
- Track all open orders and their status
- Cancel stale orders when signal reverses
- Detect partial fills and handle remainder
- Enforce: never hold more than one position per market simultaneously

## Kill Switch (must exist before any live order)
```python
async def cancel_all_orders() -> None:
    """Cancel every open order across all markets. Called on shutdown or emergency."""
```
This function must be implemented and tested in paper mode before going live.
Map it to a keyboard interrupt handler.

## Risk Check Integration
```python
from risk.position_limits import check_position_allowed

if not check_position_allowed(market_id, contracts, current_positions):
    logger.warning("Position limit blocked order: %s", market_id)
    return None
```
This call is mandatory before every order. No exceptions.