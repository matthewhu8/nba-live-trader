---
name: "source-command-phase-status"
description: "Check current project status — what's built, what's next, and what's blocking progress. Run at the start of any session to orient quickly."
---

# source-command-phase-status

Use this skill when the user asks to run the migrated source command `phase-status`.

## Command Template

# Project Phase Status Check

## Step 1: Read AGENTS.md
Read AGENTS.md and extract the Current Status checklist.
Report which items are checked off and which are not.

## Step 2: Verify Completed Items
For each checked-off item, verify the claimed file actually exists and has meaningful content:

```bash
# Check key files exist
ls data/ingestion/kalshi_recorder.py 2>/dev/null && echo "✓ kalshi_recorder" || echo "✗ kalshi_recorder MISSING"
ls data/ingestion/nba_api_client.py 2>/dev/null && echo "✓ nba_api_client" || echo "✗ nba_api_client MISSING"
ls data/raw/possessions.parquet 2>/dev/null && echo "✓ possessions data" || echo "✗ possessions data MISSING"
ls data/feature_store/feature_rows.parquet 2>/dev/null && echo "✓ feature store" || echo "✗ feature store MISSING"
ls models/run_predictor.py 2>/dev/null && echo "✓ run_predictor" || echo "✗ run_predictor MISSING"
ls backtesting/simulator.py 2>/dev/null && echo "✓ simulator" || echo "✗ simulator MISSING"
ls execution/kalshi_client.py 2>/dev/null && echo "✓ kalshi_client" || echo "✗ kalshi_client MISSING"
ls risk/position_limits.py 2>/dev/null && echo "✓ position_limits" || echo "✗ position_limits MISSING"
```

## Step 3: Identify Current Phase
Based on what's built, identify which phase we're in:
- Phase 1 (Data Foundation): kalshi_recorder + nba_api_client + raw tables
- Phase 2 (Feature Store): player ratings + lineup ratings + feature rows
- Phase 3 (Backtesting): simulator + first strategies + evaluator
- Phase 4 (Models): run_predictor + rl_agent + pregame module
- Phase 5 (Execution): kalshi_client + paper_trader + risk module
- Phase 6 (Live): paper trading → live trading

## Step 4: Identify Next Action
State the single most important next thing to build.
Be specific — not "work on features" but "implement lineup_features.py:
compute lineup_net_rating_delta using the rolling ratings in player_ratings.parquet"

## Step 5: Check for Blockers
Are there any blockers to the next action?
- Missing dependencies (package not installed)
- Data not yet collected (raw tables empty)
- Upstream module not yet implemented

## Step 6: Kalshi Recorder Health (if deployed)
If `data/raw/kalshi_price_ticks.parquet` exists:
- Report approximate row count
- Report date range of collected data
- Report any gaps > 30 minutes (missed games)
This is the most time-sensitive piece — flag if it's not running.

## Output Format
```
CURRENT PHASE: [phase name]
COMPLETED: [n] / 16 items

NEXT ACTION: [specific next thing to build]

BLOCKERS: [none / list]

KALSHI DATA: [n rows / date range / gaps]
```
