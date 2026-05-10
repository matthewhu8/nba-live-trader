---
name: "source-command-check-risk"
description: "Audit the entire codebase for risk and safety issues before any live trading session. Run this every time before switching from paper to live mode."
---

# source-command-check-risk

Use this skill when the user asks to run the migrated source command `check-risk`.

## Command Template

# Pre-Live Risk Audit

Perform a safety audit of the codebase. This must pass before any real money is placed.

## Step 1: Credential Safety
```
grep -r "api_key" . --include="*.py" | grep -v ".env" | grep -v "os.getenv" | grep -v "os.environ"
grep -r "KALSHI" . --include="*.py" | grep -v "os.getenv" | grep -v "os.environ" | grep -v "Codex"
grep -r "password\|secret\|token" . --include="*.py" | grep -v "os.getenv" | grep -v "os.environ"
```
Any hardcoded credentials = CRITICAL BLOCKER. Do not proceed.

## Step 2: Paper Mode Default
Read `execution/kalshi_client.py` and `execution/paper_trader.py`.
Verify:
- `PAPER_MODE = True` is the default
- Live mode requires explicit opt-in (env var or explicit flag)
- Paper mode logs would-be orders without hitting the API
Report the exact line where PAPER_MODE is set.

## Step 3: Kill Switch Exists
Read `execution/kalshi_client.py` and `execution/order_manager.py`.
Verify `cancel_all_orders()` function exists and:
- Cancels ALL open orders across ALL markets
- Has no dependencies that could fail silently
- Is mapped to keyboard interrupt or explicit shutdown hook
If missing = CRITICAL BLOCKER.

## Step 4: Position Limits Enforcement
```
grep -r "place_order\|submit_order\|create_order" execution/ --include="*.py"
```
For each order placement call found, verify `position_limits.check_position_allowed()`
is called immediately before it. Any order placement without this check = CRITICAL BLOCKER.

## Step 5: Maker-Only Enforcement
Read `execution/order_manager.py`.
Verify there is no code path that can place a market order.
Verify there is no code path that places a limit order that crosses the current spread.
Search for any `order_type = "market"` strings in execution code.

## Step 6: Blowout Guard in Live Signal Path
Read `strategy/signal_generator.py` or the live strategy entry point.
Verify `is_blowout` and `is_garbage_time` checks kill signal generation.
These must be the FIRST checks, not buried in logic.

## Step 7: Error Handling in Execution
Read `execution/kalshi_client.py`.
Verify:
- All API calls wrapped in try/except with logging
- Network errors do NOT silently fail — they must be logged and trigger order cancellation
- Retry logic has a maximum (not infinite retries)
- On repeated failure, system falls back to cancel-all + halt

## Step 8: Daily Loss Limit
Read `risk/position_limits.py`.
Verify a daily loss limit exists:
- Tracks cumulative PnL for the day
- Halts all new orders if loss exceeds threshold
- Threshold should be defined as env var, not hardcoded

## Final Report
Classify every finding:
- 🔴 CRITICAL BLOCKER — do not go live until fixed
- 🟡 WARNING — should fix soon, proceed with caution
- 🟢 PASS — verified safe
- ⚪ NOT YET IMPLEMENTED — expected for current phase

If any 🔴 items exist: do not switch from paper to live mode.
