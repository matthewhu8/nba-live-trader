---
name: "source-command-pregame-check"
description: "Run before a live or paper trading session. Verifies the system is ready, pulls tonight's game context, and identifies watch flags for tradeable situations."
---

# source-command-pregame-check

Use this skill when the user asks to run the migrated source command `pregame-check`.

## Command Template

# Pre-Game Session Check

## Step 1: System Readiness
Verify the system is ready to run:

```bash
# Check Kalshi recorder is running
pgrep -f "kalshi_recorder" && echo "✓ Recorder running" || echo "✗ RECORDER NOT RUNNING — start it now"

# Check paper mode is active
grep -n "PAPER_MODE" execution/kalshi_client.py
grep -n "PAPER_MODE" execution/paper_trader.py
```

If recorder is not running — start it before anything else. Every missed tick is lost data.
If paper mode is not confirmed True — do not proceed until verified.

## Step 2: Tonight's Games
Run the pregame analyzer if implemented:
```bash
python pregame/pregame_analyzer.py --date today 2>/dev/null || echo "Pregame analyzer not yet built"
```

If not built yet, report which NBA games are scheduled tonight
and note that the pregame analyzer is not yet available.

## Step 3: Watch Flags
Based on pregame context (if available), report:
- Games with highest predicted run-trading opportunity
- Specific lineup matchups to watch (high net rating delta setups)
- Any back-to-back situations (fatigue effects)
- Key players in foul trouble from recent games

## Step 4: Strategy Configuration
Report which strategies are currently active and their current parameters.
Flag any strategy with poor recent paper performance that should be disabled tonight.

## Step 5: Risk Limits Check
Read `risk/position_limits.py` and report current configured limits:
- Max contracts per position
- Max total open exposure
- Daily loss limit threshold
Confirm these are set to appropriate values for paper trading (conservative).

## Step 6: Session Go/No-Go
Report final go/no-go:
✅ GO if: recorder running + paper mode confirmed + risk limits set
❌ NO-GO if: any of the above missing

Report the game(s) to watch and what to look for.
