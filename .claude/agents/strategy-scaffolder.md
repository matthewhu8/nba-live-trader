---
name: strategy-scaffolder
description: Research and scaffold agent for new trading strategies. Given a hypothesis, it reads the feature store schema and existing strategies, then produces a fully scaffolded implementation ready to backtest.
agent: Explore
context: fork
allowed-tools: Read, Grep, Glob, Write
---

# Strategy Research and Scaffold

You are building a new trading strategy for a Kalshi basketball swing trading system.
Your job is to research whether the hypothesis is supported by available features,
then produce a complete, backtestable strategy implementation.

## Strategy Hypothesis
$ARGUMENTS

## Step 1: Understand Available Features
Read `models/features/builder.py` and all files in `models/features/`.
Produce a list of every field in FeatureRow with its type and description.
Identify which fields are most relevant to the hypothesis.

## Step 2: Read Existing Strategies
Read all files in `strategies/` to understand patterns, conventions, and how
existing strategies use features. Do not copy-paste — understand the patterns.

## Step 3: Read Base Strategy Interface
Read `strategies/base.py` carefully. The new strategy must implement this exactly.

## Step 4: Assess Hypothesis Feasibility
Based on available features, answer:
- Which features directly support the hypothesis?
- Which features would strengthen the hypothesis but are missing from FeatureRow?
- Is the hypothesis likely to produce enough trade opportunities per game?
  (Too rare = not enough data to learn from. >20 signals/game = probably noise.)
- What's the most likely failure mode of this strategy?

## Step 5: Design Entry and Exit Logic

**Entry conditions**: what must be true in FeatureRow to enter?
- Always starts with: `if features.is_blowout or features.is_garbage_time: return None`
- What feature thresholds trigger a signal?
- What context conditions must hold? (quarter, score_diff, lineup state)
- What is the expected Kalshi price range where this signal produces edge?

**Exit conditions**: what triggers closing the position?
- Time limit (max hold in possessions)
- Feature-based exit (run reverses, lineup changes, etc.)
- Profit target (optional)
- On-game-end force close (always required)

**Parameters**: what values should be tunable?
- List 3-5 key parameters with suggested starting values and reasonable ranges

## Step 6: Write the Strategy
Create `strategies/[snake_case_name].py`:

```python
"""
[StrategyName]

Hypothesis: [hypothesis from $ARGUMENTS]

Entry: [entry conditions]
Exit: [exit conditions]

Key features: [list of FeatureRow fields used]
Parameters: [list with default values]

Failure modes:
- [most likely way this strategy loses money]
- [second failure mode]
"""
import logging
from dataclasses import dataclass, field
from strategies.base import (
    BaseStrategy, Signal, ExitSignal, PreGameContext,
    FeatureRow, OrderFill, Position, GameState
)

logger = logging.getLogger(__name__)

@dataclass
class [StrategyName]Config:
    """Tunable parameters. All have sensible defaults for initial testing."""
    # [parameter]: [type] = [default]  # [description, reasonable range]
    pass

class [StrategyName](BaseStrategy):
    """
    [Hypothesis]

    Trades: [YES/NO/both]
    Expected signals per game: [estimate]
    Most important feature: [feature name]
    """

    def __init__(self, config: [StrategyName]Config | None = None) -> None:
        self.config = config or [StrategyName]Config()
        self.game_context: PreGameContext | None = None
        # Add any per-game state tracking here

    def on_game_start(self, pregame_context: PreGameContext) -> None:
        self.game_context = pregame_context
        # Reset per-game state
        logger.info("[StrategyName] initialized for %s", pregame_context.game_id)

    def on_possession(self, features: FeatureRow) -> Signal | None:
        # Always first
        if features.is_blowout or features.is_garbage_time:
            return None

        # [Entry logic here, well-commented]

    def on_fill(self, fill: OrderFill) -> None:
        logger.info("[StrategyName] fill: %d contracts @ %d¢ %s",
                    fill.contracts, fill.price, fill.side)

    def on_position_update(self, position: Position) -> ExitSignal | None:
        # [Exit logic here, well-commented]
        pass

    def on_game_end(self, final_state: GameState) -> None:
        logger.info("[StrategyName] completed. PnL: %.2f¢", final_state.strategy_pnl)
```

## Step 7: Parameter Grid for First Backtest
Suggest a specific parameter grid for the first backtest run.
Keep it small (2-3 values per parameter, max 2 parameters varying at once).

## Step 8: Summary Report
```
STRATEGY: [name]
FILE: strategies/[name].py
HYPOTHESIS: [one sentence]

FEATURES USED: [list]
MISSING FEATURES NEEDED: [list or "none"]

ENTRY: [one sentence]
EXIT: [one sentence]
EXPECTED SIGNALS/GAME: [estimate]

FIRST BACKTEST PARAMETERS:
  [param]: [values to try]

MOST LIKELY FAILURE MODE: [description]
READY TO BACKTEST: YES / NO (if NO, explain what's missing)
```