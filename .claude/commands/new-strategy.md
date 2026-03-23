---
description: Scaffold a new strategy. Provide the strategy name and hypothesis as arguments. Example: /new-strategy FoulTroubleStrategy "Enter YES when key defender picks up 2nd foul in first half"
allowed-tools: Read, Write, Glob
---

# Scaffold New Strategy

Arguments: $ARGUMENTS

## Step 1: Parse Arguments
Extract from $ARGUMENTS:
- Strategy class name (e.g. FoulTroubleStrategy)
- Hypothesis (the human-readable thesis)
- Entry condition (if specified)
- Exit condition (if specified)

## Step 2: Read Base Strategy Interface
Read `strategies/base.py` to understand the exact interface to implement.

## Step 3: Check Relevant Features
Read `models/features/builder.py` and the relevant feature files to identify
which existing features support this strategy's hypothesis.
List the specific FeatureRow fields this strategy will use.

## Step 4: Create Strategy File
Create `strategies/[snake_case_name].py` with:

```python
"""
[StrategyName]: [Hypothesis from arguments]

Entry condition: [entry logic]
Exit condition: [exit logic]
Key features used: [list from step 3]
"""
import logging
from dataclasses import dataclass
from strategies.base import BaseStrategy, Signal, ExitSignal, PreGameContext, FeatureRow, OrderFill, Position, GameState

logger = logging.getLogger(__name__)

@dataclass
class [StrategyName]Config:
    # Tunable parameters — all should have sensible defaults
    # Example:
    # edge_threshold: float = 0.05
    # min_run_length: int = 3
    pass

class [StrategyName](BaseStrategy):
    """
    [Hypothesis]
    """

    def __init__(self, config: [StrategyName]Config | None = None):
        self.config = config or [StrategyName]Config()
        self.game_context = None

    def on_game_start(self, pregame_context: PreGameContext) -> None:
        self.game_context = pregame_context
        logger.info("[StrategyName] initialized for game %s", pregame_context.game_id)

    def on_possession(self, features: FeatureRow) -> Signal | None:
        # Always check these first
        if features.is_blowout or features.is_garbage_time:
            return None

        # TODO: implement entry logic
        # Entry condition: [entry condition from arguments]
        raise NotImplementedError("Implement entry logic")

    def on_fill(self, fill: OrderFill) -> None:
        logger.info("[StrategyName] fill: %s contracts at %s¢, side=%s",
                    fill.contracts, fill.price, fill.side)

    def on_position_update(self, position: Position) -> ExitSignal | None:
        # TODO: implement exit logic
        # Exit condition: [exit condition from arguments]
        raise NotImplementedError("Implement exit logic")

    def on_game_end(self, final_state: GameState) -> None:
        logger.info("[StrategyName] game ended. Final PnL: %s¢", final_state.strategy_pnl)
```

## Step 5: Register in Backtesting
Check if `backtesting/simulator.py` has a strategy registry.
If so, add the new strategy to it.

## Step 6: Summary
Report:
- File created at: strategies/[name].py
- Features it uses from FeatureRow: [list]
- Methods that need implementation: on_possession, on_position_update
- Suggested first parameter values to test
- Any features this strategy needs that don't exist yet in FeatureRow (flag these)