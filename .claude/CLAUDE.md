# Kalshi Basketball Trading System

## The End Goal
Make money live trading NBA games on Kalshi prediction markets.
Everything in this project — the data engineering, the models, the backtesting —
exists to serve one outcome: a live system that places profitable maker orders
during NBA games with a validated, understood edge.

This is a learning project. I am new to algo trading but comfortable with Python.
- Prioritize readable, well-commented code — explain *why*, not just *what*
- When multiple approaches exist, explain tradeoffs before implementing
- Flag anything that could cause real financial loss
- Always ask: "does this get us closer to live trading with a real edge?"

---

## The Core Thesis

We are NOT predicting game winners.
We are NOT arbitraging sportsbook vs Kalshi lines (bots close those gaps in seconds).

We ARE predicting short-term scoring runs — 3 to 5 minute windows — using lineup
matchup data and basketball context, and entering Kalshi positions BEFORE the run
shows up on the scoreboard and BEFORE the market reprices.

```
Detect run conditions early
  → Enter Kalshi limit order (maker, always)
    → Scoring run occurs
      → Kalshi price moves
        → Exit position
          → Capture the spread, minus fees
```

Sharp sportsbooks (DraftKings, FanDuel, Pinnacle) price the game outcome with
enormous infrastructure. We don't compete with them on that. We exploit Kalshi's
slower, retail-driven, emotionally-reactive repricing of micro-events within the game.

Retail bettors on Kalshi overreact to scoring runs they just watched on TV.
They underreact to lineup changes they didn't notice.
We notice. We get there first.

---

## Why This Edge Is Defensible

1. **Lineup context requires basketball knowledge** — a bot that closes moneyline gaps
   doesn't model "Jokic just sat, their second unit is now matched against a zone defense
   they historically struggle against." That's domain knowledge most bots don't have.

2. **Kalshi is retail-dominated** — unlike the main sportsbook moneyline, Kalshi's live
   markets have meaningful retail participation. Retail overreacts to runs, doesn't
   account for lineup changes, trades emotionally. That's our counterparty.

3. **We're predictive, not reactive** — pure arbitrage bots react to price gaps after
   they open. We enter before the gap opens by predicting the event that will cause it.

4. **Thin markets lag more** — bots prioritize the highest-liquidity markets. Granular
   Kalshi markets (Q2 team totals, live spreads) reprice slower than the main moneyline.

---

## Non-Negotiable Strategy Rules
- **ALWAYS maker orders** — limit orders only, always. Never market orders. 4x fee difference.
- **Fees are part of every signal calculation** — if the edge doesn't clear fees, don't trade
- **Paper trade before real money** — weeks minimum, not days
- **Position limits enforced before every order** — via `risk/position_limits.py`, no exceptions
- **One execution module** — all Kalshi API calls go through `execution/kalshi_client.py` only
- **Kill switch must exist** before any live order is ever placed
- **Never hardcode credentials** — all keys via environment variables

---

## Fee Structure (apply to every edge calculation)
```
Maker fee:  0.0175 × contracts × price   (~$0.44 per 100 contracts at 50¢)
Taker fee:  0.07   × contracts × price   (~$1.75 per 100 contracts at 50¢)
```
- Taker fees are 4x higher — being a taker destroys edge on small price moves
- All prices are integers in cents (1–99). Never floats. Never decimals internally.
- A YES contract at 60¢ = market implies 60% win probability

---

## Data Architecture

Data operates at four time scales simultaneously. Every feature must be correctly
computed using only information available at the exact decision moment.

### Raw Tables (source of truth, never modified after ingestion)

**`games`**
```
game_id, date, home_team, away_team, season, game_number_in_season,
home_back_to_back, away_back_to_back
```

**`possessions`** — one row per possession end (natural decision unit)
```
game_id, possession_id, wall_clock_time, game_clock, quarter,
home_score, away_score, team_scored, points,
shot_zone, shot_contested, play_type,
home_lineup_id, away_lineup_id
```

**`substitution_events`**
```
game_id, wall_clock_time, game_clock, quarter, team,
player_in, player_out,
resulting_home_lineup_id, resulting_away_lineup_id
```

**`kalshi_price_ticks`** — recorded from day one, grows over time
```
market_id, game_id, wall_clock_time, yes_bid, yes_ask,
yes_last, order_book_depth_bid, order_book_depth_ask
```

Times stored in two forms always: wall_clock (aligns with Kalshi) and game_clock
(basketball context). Both matter. Never confuse them.

---

## Feature Engineering (Point-in-Time, Leak-Free)

The most important engineering constraint in this entire project:
**Every feature must be computable using only data from before the decision moment.**

This is solved via a pre-computed **feature store** — a wide table of feature rows,
one per possession, all features guaranteed leak-free. Strategies never compute
features themselves. They only read from the feature store.

### Rolling Player Ratings
Raw plus/minus is useless (context-dependent, small samples).
Use rolling regularized plus/minus — for each game, ratings computed from prior games only.

```
player_id, as_of_game_id, offensive_rating, defensive_rating,
on_ball_defense_rating, adjusted_plus_minus
```

V1 shortcut: prior season RAPM as starting estimate, exponentially weighted updates
with current season games (recent games weighted more). 80% accuracy, 10% compute.

### Lineup Net Ratings
From player ratings, compute every historical 5-man combination:
```
lineup_id (hash of 5 player ids), as_of_game_id,
offensive_rating, defensive_rating, net_rating,
possessions_together, last_seen_date
```
Small sample lineups (<50 possessions): regress toward average of 5 individual ratings.

### Rotation Tendency Model
Per coach: probability distribution over substitution timing.
```
coach_id, quarter, time_remaining_bucket (0-2, 2-4, 4-6, 6-8 min),
score_differential_bucket, probability_of_sub,
most_likely_players_out, computed_through_game_id
```
Used by pre-game module and rotation_signal feature.

### The Feature Row (one per possession, the core of the feature store)

```python
# --- Identity ---
game_id, possession_id, wall_clock_time, game_clock, quarter,
score_diff, home_score, away_score

# --- Lineup State ---
home_lineup_id, away_lineup_id,
home_lineup_net_rating,           # point-in-time adjusted
away_lineup_net_rating,
lineup_net_rating_delta,          # THE core signal — delta between matchups
home_lineup_sample_size,          # possessions together (confidence weight)
away_lineup_sample_size,
home_star_player_on_court,        # bool
away_star_player_on_court,
home_primary_ballhandler_on_court,
away_primary_ballhandler_on_court,
home_lineup_just_changed,         # bool — substitution just occurred
away_lineup_just_changed,

# --- Foul State ---
home_key_defender_foul_count,
away_key_defender_foul_count,
home_key_defender_fouled_out,
away_key_defender_fouled_out,

# --- Momentum (backward-looking window) ---
home_points_last_5_poss,
away_points_last_5_poss,
home_points_last_10_poss,
away_points_last_10_poss,
current_run_team,                 # which team is on a run (or None)
current_run_length_possessions,
current_run_points,
pace_last_10_possessions,
pace_season_average,
pace_ratio,                       # current vs average

# --- Shot Quality (last 5 possessions per team) ---
home_shot_quality_score,          # weighted FG% above expected
away_shot_quality_score,
home_scoring_sustainable,         # bool: open 3s + paint = True; contested mid = False
away_scoring_sustainable,

# --- Fatigue / Context ---
home_starters_minutes_today,      # fatigue proxy
away_starters_minutes_today,
home_back_to_back,
away_back_to_back,
minutes_into_game,
rotation_signal,                  # sub expected soon per tendency model (0-1 prob)
is_blowout,                       # score_diff > 20 — model should shut off
is_garbage_time,                  # late game, large margin, bench players

# --- Target Variables (forward-looking — used for training ONLY, never as features) ---
home_next_10_poss_margin,         # did home team outscore in next 10 possessions?
home_next_20_poss_margin,
kalshi_price_change_3min,         # null until Kalshi data collected
kalshi_price_change_5min,
meaningful_run_occurred,          # bool: X+ point swing in next N possessions
```

**Target variables are clearly separated and flagged. Feature pipeline is backward-looking.
Training pipeline uses both. Live system uses features only. This is enforced in code.**

---

## Synthetic Kalshi Price Model (bridge until real data exists)

No historical Kalshi prices exist yet. To backtest the trading layer before
collecting weeks of data, generate synthetic prices from sharp book line movement:

```
synthetic_kalshi_price = sharp_book_wp
  + lag_offset(15-45s, random)          # Kalshi reprices slower
  + overreaction_term(run_length)        # retail overreacts to runs
  - mean_reversion_term(time_since_run)  # prices drift back
  + noise(σ=0.5¢)
```

Parameters (lag_amount, overreaction_coefficient, reversion_speed) are tuned
once real Kalshi data is collected. Synthetic model lets you validate strategy
architecture now, swap real prices in later with zero other changes.

Large discrepancy between synthetic and real backtest results = synthetic model
had wrong assumptions. This is useful information about how Kalshi actually behaves.

---

## Strategy Abstraction Layer

Every strategy implements the same interface. The simulator calls these hooks.
Multiple strategies can run against the same historical game in one pass.

```python
class BaseStrategy:
    def on_game_start(self, pregame_context: PreGameContext) -> None:
        """Called 30min before tip. Load priors, set watch flags."""

    def on_possession(self, features: FeatureRow) -> Signal | None:
        """Called each possession. Return trade signal or None."""

    def on_fill(self, fill: OrderFill) -> None:
        """Called when limit order fills. Update internal state."""

    def on_position_update(self, position: Position) -> ExitSignal | None:
        """Called each possession while holding. Return exit signal or None."""

    def on_game_end(self, final_state: GameState) -> None:
        """Force-close open positions. Log game summary."""
```

### Strategies to Implement (in order)

**`MeanReversionStrategy`**
Fade scoring runs above threshold when shot quality is low (unsustainable scoring).
Entry: run_length > X AND home_scoring_sustainable == False.
Exit: run reverses OR time limit reached.
Hypothesis: retail overreacts to runs built on contested shots. Price snaps back.

**`LineupEdgeStrategy`**
Enter when lineup_net_rating_delta exceeds threshold after a substitution.
Entry: home_lineup_just_changed AND lineup_net_rating_delta > X.
Exit: next substitution OR time limit.
Hypothesis: Kalshi doesn't immediately price in lineup quality changes.

**`RotationAnticipationStrategy`**
Enter just before expected substitution that will create favorable lineup delta.
Entry: rotation_signal > 0.7 AND historical sub creates favorable matchup.
Exit: substitution occurs and position filled.
Hypothesis: we know the sub is coming before it happens. Enter before the move.

**`MomentumStrategy`**
Enter in direction of current run when shot quality is high (sustainable scoring).
Entry: run_length > X AND scoring_sustainable == True.
Exit: run ends OR score_diff becomes blowout.
Hypothesis: sustainable scoring runs continue longer than market expects.

**`CompositeStrategy`**
Combines signals from multiple sub-strategies with weighted confidence scores.
This is what the RL agent eventually becomes — a learned version of this weighting.

---

## Simulation Engine

Replays historical games, enforces reality, calls strategy hooks.

```
For each game (strict chronological order — never shuffle):
  → Fetch pregame context → strategy.on_game_start()

  For each possession (strict time order):
    → Load feature row from feature store
    → Enforce data latency: live feed features delayed 15-20s
      (pre-game computed features have no latency)
    → Check is_garbage_time / is_blowout → skip if true
    → strategy.on_possession() → get signal
    → If signal:
        → risk/position_limits.py check
        → Place simulated limit order at signal price
        → Simulate fill probability based on order book depth
        → Track unfilled orders (may not fill if price moves away)
    → For open positions: strategy.on_position_update()
    → Apply maker fees to all fills
    → Update PnL tracker
    → Record full decision log (used for analysis)

  → strategy.on_game_end()
  → Force-close open positions at last available price
  → Write game summary to results
```

**Latency is not optional.** Every feature from a live feed (substitutions, scores,
play-by-play) must be delayed by 15-20s in simulation. Skipping this makes
backtests fantasy. Pre-game computed features (lineup ratings, tendencies) have
no latency — those were computed before the game started.

---

## Backtesting Evaluation Framework

Every strategy run produces a standardized results object. Never cherry-pick metrics.

**Overall performance:**
- Sharpe ratio, Sortino ratio
- Total PnL (gross and net of fees)
- Win rate, number of trades
- Avg hold time (possessions and real minutes)
- Max drawdown, recovery time
- Fee drag as % of gross PnL (if this is > 40%, something is wrong)

**Performance by context (where the real insights live):**
- By quarter
- By score differential bucket (±5, ±6-12, ±13-20, blowout)
- By lineup delta magnitude (small/medium/large)
- By run length at entry (1-3, 4-6, 7+ possessions)
- By shot quality (sustainable vs unsustainable scoring)
- By game type (back-to-back, rivalry, playoff)
- By time of season (early season ratings are noisy)

**Signal quality:**
- Calibration curve: predicted run probability vs actual run frequency
- Edge predicted vs edge captured
- False positive rate by signal type and context
- P&L attribution: how much came from each signal type

Context breakdown is where strategy improvements come from. A mediocre overall
strategy might be excellent in Q2 close games and terrible in blowouts. Add a
context filter → significant improvement. Look for this every backtest run.

---

## Train / Test / Live Split

**Never random split. Always time-based.**

```
2021-22, 2022-23 seasons  →  feature engineering + model training
2023-24 season            →  validation (tune hyperparameters here)
2024-25 season            →  TEST SET (touch only once, final evaluation)
Current season            →  live observation + paper trading
```

The test set is sacred. Do not look at it during development. Evaluate once,
at the end, to get an unbiased estimate of live performance. If you keep using
test results to make decisions, it becomes validation data and the estimate is biased.

---

## Pre-Game Module

Runs ~30 minutes before tip-off. Outputs `game_context.json` for the live system.

- Pull projected starting lineups (rotowire, ESPN, or manual input)
- Load historical matchup data for tonight's likely rotation patterns
- Calculate expected lineup net rating delta for first rotation
- Generate watch flags: situations likely to produce tradeable runs
  Example: "Bam Adebayo expected on bench Q2 min 4-7 — opponent second unit
  has +6.2 net rating vs Miami's bench. Watch for MIA YES entry at that moment."
- Set strategy priors for tonight (which strategies are on/off for this game)

---

## Live System Architecture

Once backtesting validates an edge, this is the live system:

```
[Pre-Game Module]
  → game_context.json (priors, watch flags, rotation map)
        ↓
[Live Feed] (Sportradar WebSocket, 15-20s latency)
  → play-by-play events, substitutions, scores
        ↓
[Feature Computer] (real-time, uses feature store patterns)
  → builds feature row for each possession
        ↓
[Run Prediction Model]
  → probability of run + direction
        ↓
[RL Agent / Signal Generator]
  → trade / wait / exit decision
        ↓
[Risk Module] (position_limits.py — always called)
  → approve or block
        ↓
[Execution Layer] (kalshi_client.py — only API touchpoint)
  → paper_trader.py OR live orders (single flag)
        ↓
[Order Manager]
  → track fills, manage cancellations, enforce maker-only
```

---

## RL Agent

Sits on top of run prediction model. Handles the sequential decision problem:
not just "is a run coming" but "should I enter NOW, how big, and when do I exit."

```
State:  run_probability, kalshi_price, game_context,
        current_position, time_in_position, recent_pnl
Action: buy_yes, buy_no, exit, wait
Reward: realized PnL after maker fees
```

**Start simple:** contextual bandit or Q-learning before deep RL.
Understand what it's learning before making it opaque.
Train entirely through the backtesting simulator.

What the agent learns that rules can't capture:
- Enter early in run signal (Kalshi reprices slowly — time it right)
- Don't enter when score_diff is large (blowout kills price movement)
- This lineup signal is reliable in Q2 but not Q4
- Shot quality matters more than run length for entry timing
- When to cut a losing position vs. hold through noise

---

## Directory Structure

```
kalshi-trader/
├── CLAUDE.md
├── .env                              # credentials — never commit
├── .gitignore                        # .env, data/, logs/, __pycache__
│
├── .claude/
│   ├── skills/
│   │   ├── feature-engineering/SKILL.md
│   │   ├── kalshi-execution/SKILL.md
│   │   ├── backtesting/SKILL.md
│   │   └── rl-agent/SKILL.md
│   └── commands/
│       ├── check-risk.md
│       ├── review-strategy.md
│       └── backtest-summary.md
│
├── data/
│   ├── ingestion/
│   │   ├── nba_api_client.py         # historical pull — play-by-play, lineups
│   │   ├── kalshi_recorder.py        # !! RUNS FROM DAY ONE !! records price ticks
│   │   └── live_feed.py              # future: Sportradar live connector
│   ├── raw/
│   │   ├── games.parquet
│   │   ├── possessions.parquet
│   │   ├── substitution_events.parquet
│   │   └── kalshi_price_ticks.parquet
│   └── feature_store/
│       ├── player_ratings.parquet    # rolling point-in-time ratings
│       ├── lineup_ratings.parquet    # per lineup, per as_of_game
│       ├── rotation_tendencies.parquet
│       └── feature_rows.parquet      # THE feature store — one row per possession
│
├── models/
│   ├── features/
│   │   ├── builder.py                # orchestrates full feature row construction
│   │   ├── lineup_features.py
│   │   ├── momentum_features.py
│   │   ├── context_features.py
│   │   └── targets.py                # forward-looking targets (training only)
│   ├── ratings/
│   │   ├── player_rapm.py            # rolling regularized plus/minus
│   │   └── lineup_net_rating.py
│   ├── rotation_tendency.py
│   ├── stint_segmenter.py            # detects lineup changes from event stream
│   ├── run_predictor.py              # XGBoost run prediction model
│   ├── synthetic_kalshi.py           # synthetic price model (pre-real-data)
│   └── rl_agent.py                   # sequential trade decision agent
│
├── strategies/
│   ├── base.py                       # BaseStrategy interface
│   ├── mean_reversion.py
│   ├── lineup_edge.py
│   ├── rotation_anticipation.py
│   ├── momentum.py
│   └── composite.py
│
├── backtesting/
│   ├── simulator.py                  # core replay engine
│   ├── evaluator.py                  # metrics, context breakdown, attribution
│   └── results/                      # strategy run outputs
│
├── pregame/
│   └── pregame_analyzer.py
│
├── execution/
│   ├── kalshi_client.py              # ONLY Kalshi API touchpoint
│   ├── order_manager.py
│   └── paper_trader.py
│
├── risk/
│   └── position_limits.py            # called before every order, no exceptions
│
└── logs/
    ├── live_trades/
    └── paper_trades/
```

---

## Data Sources

| Data | Source | Cost | Status |
|------|--------|------|--------|
| Historical play-by-play | `nba_api` Python package | Free | Not set up yet |
| Historical lineup stats | `nba_api` / PBPStats | Free | Not set up yet |
| Historical sharp line movement | OddsAPI / TheOddsAPI | Low | Not set up yet |
| Live play-by-play | Sportradar / Genius Sports | Paid | Not yet contracted |
| Live Kalshi prices | Kalshi WebSocket | Free | **START RECORDING NOW** |
| Historical Kalshi prices | Self-recorded | Free | Building from day one |

**The Kalshi recorder is time-critical.** Historical Kalshi price data is not available
from any vendor. Every NBA game that passes without the recorder running is training
data permanently lost. This is the first thing to build and the first thing to deploy.

---

## Python Conventions
- Async (asyncio) for all ingestion, WebSocket, and live feed handling
- Type hints on all functions, always
- `logging` module everywhere — never `print()` for runtime output
- Explicit error handling — log what failed and why, never bare `except:`
- Prefer early returns over deeply nested conditionals
- All prices as integers in cents — never floats
- Parquet format for all stored data (fast, typed, columnar)
- Never commit `.env`, `data/`, or `logs/` to git

---

## Build Order

### Phase 1 — Data Foundation (start here, nothing else matters yet)
1. `data/ingestion/kalshi_recorder.py` — deploy immediately, record every game
2. `data/ingestion/nba_api_client.py` — pull 3 seasons of historical play-by-play
3. Raw table normalization → `possessions.parquet`, `substitution_events.parquet`

### Phase 2 — Feature Store
4. `models/ratings/player_rapm.py` — rolling point-in-time player ratings
5. `models/ratings/lineup_net_rating.py` — lineup ratings from player ratings
6. `models/rotation_tendency.py` — coach rotation tendency model
7. `models/stint_segmenter.py` — lineup change detection
8. `models/features/` — all feature modules + `builder.py`
9. Validate: manually spot-check 50 feature rows for lookahead bias

### Phase 3 — Backtesting Infrastructure
10. `models/synthetic_kalshi.py` — synthetic price model
11. `strategies/base.py` + first two strategies (mean_reversion, lineup_edge)
12. `backtesting/simulator.py` — replay engine with latency enforcement
13. `backtesting/evaluator.py` — metrics and context breakdown
14. First backtest run + analysis

### Phase 4 — Model Layer
15. `models/run_predictor.py` — XGBoost on feature store
16. Remaining strategies (rotation_anticipation, momentum, composite)
17. `models/rl_agent.py` — trained through simulator
18. `pregame/pregame_analyzer.py`

### Phase 5 — Execution
19. `execution/kalshi_client.py` — Kalshi REST + WebSocket
20. `execution/paper_trader.py` — paper mode (default)
21. `execution/order_manager.py`
22. `risk/position_limits.py` + kill switch

### Phase 6 — Live Validation
23. Paper trading — minimum 4 weeks, multiple games per week
24. Analyze paper results vs backtest — explain discrepancies
25. Real money — small size, single game at a time, monitor every trade

---

## Current Status
- [ ] Phase 1: Kalshi recorder deployed
- [ ] Phase 1: nba_api ingestion complete
- [ ] Phase 1: Raw tables normalized
- [ ] Phase 2: Player ratings (rolling RAPM)
- [ ] Phase 2: Lineup ratings
- [ ] Phase 2: Rotation tendency model
- [ ] Phase 2: Full feature store built + validated
- [ ] Phase 3: Backtesting simulator
- [ ] Phase 3: First two strategies running
- [ ] Phase 3: First backtest + analysis complete
- [ ] Phase 4: Run predictor model trained
- [ ] Phase 4: RL agent trained
- [ ] Phase 5: Execution layer (paper mode)
- [ ] Phase 5: Risk module + kill switch
- [ ] Phase 6: Paper trading (4+ weeks)
- [ ] Phase 6: Live trading

**Current phase: Phase 1 — build the Kalshi recorder first.**

---

## Key Reminders for Claude
- The end goal is live profitable trading. Every decision should serve that.
- Kalshi recorder is the highest priority — it is time-sensitive, data is lost forever if not running
- Always verify features are point-in-time before adding them to the feature store
- Lookahead bias makes backtests look great and live trading look terrible — be paranoid about it
- When touching execution code: paper mode must be default, confirm before changing
- Never suggest a taker order under any circumstance
- Blowout / garbage time = model off. Never trade garbage time.
- If a backtest result looks suspiciously good, assume lookahead bias first
- We are in Phase 1. Don't build Phase 5 code while Phase 1 is incomplete.