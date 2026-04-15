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

## Kalshi Price Data (real tick data — synthetic model scrapped)

Real intra-game Kalshi tick data is being recorded continuously and uploaded to
MotherDuck (`kalshi_ticks` table). The synthetic price model has been scrapped —
we have the real thing.

The tick data feeds directly into Layer 2 (price movement model). The goal is to
join run_predictor signals with actual Kalshi bid/ask movements and measure whether
high-confidence run predictions preceded profitable price moves, net of maker fees.

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
- [x] Phase 1: Kalshi recorder deployed
- [x] Phase 1: nba_api ingestion complete
- [x] Phase 1: Raw tables normalized → `features.possession_flat` (400K rows, 86 cols)
- [x] Phase 2: Player ratings (rolling RAPM)
- [x] Phase 2: Lineup ratings (47M rows, 1,065 games)
- [ ] Phase 2: Rotation tendency model
- [x] Phase 2: Feature store built + validated (58 features, lineup signals included)
- [x] Phase 2: **Nightly post-game pipeline** — runs at 3 AM ET on Fly.io, updates all tables automatically
- [ ] Phase 3: Backtesting simulator (built but strategies losing — paused)
- [x] Phase 4: Run predictor trained — AUCPR 0.1075 vs 0.0840 baseline (28% lift)
- [x] Phase 4: **MMoE model trained (2026-04-15)** — replaces L1 + L2 with a single PyTorch multi-task network ← **COMPLETED**
  - **Head A (run classifier):** AUCPR 0.1533 vs 0.0840 baseline (+82%) and vs 0.1075 XGBoost (+43%)
  - **Head B (price trajectory):** RMSE 0.5525 log-odds delta; Dir Acc 59.2% on meaningful-exit rows
  - **Head C (run survival hazard):** Brier 0.0939 across 10 horizons
  - Architecture: 83 input features (58 physics + 10 pregame + 14 market + 1 market flag), 3 experts (64-dim MLP), 3 gating networks, 3 heads. ~37K params.
  - Data: 433K basketball rows (Heads A/C) + 24K joint rows with Kalshi ticks (Head B), 148 games
  - Train/val split: basketball time-based (Jan 2026 cutoff); Head B: Mar 23–Apr 6 train / Apr 7–12 val
  - Model artifacts: `models/saved/mmoe.pt`, `models/saved/mmoe_scaler.pkl`
  - Key fix: zero-inflated trajectory targets (42% of traj_9 == 0) caused 13.1% dir acc bug; fixed via signal-filtered Huber loss mask (`abs mean > 0.02`) + directional accuracy threshold (`abs final chkpt > 0.05`)
- [ ] Phase 4: RL agent ← **CURRENT FOCUS**
- [ ] Phase 5: Execution layer (paper mode)
- [ ] Phase 5: Risk module + kill switch
- [ ] Phase 6: Paper trading (4+ weeks)
- [ ] Phase 6: Live trading

---

## Nightly Post-Game Pipeline (completed 2026-04-02)

### What it does
Runs automatically at 3 AM ET on the existing Fly.io recorder machine after all games finish.
No extra infra — same machine, same Docker image, triggered from `recorder_daemon.py`.

### Files
- **NEW:** `data/ingestion/post_game_pipeline.py` — main pipeline logic
- **MODIFIED:** `data/ingestion/recorder_daemon.py` — 3 AM trigger after `_schedule_games` returns
- **MODIFIED:** `models/ratings/lineup_net_rating.py` — added `games_together: int32` to schema

### Pipeline phases (must run in order)
```
Phase 0  Upsert dim_games                  (all other tables FK on game_id)
Phase 1  Fetch + parse PBP via nba_api     (stateless, no disk cache)
Phase 2  Build possession_flat rows        (uses pre-tonight ASOF ratings — point-in-time correct)
Phase 3  Update player_ratings + lineup_ratings  (Ridge regression + EWMA, for tomorrow)
```

### Key design decisions
- **Stateless**: all reads/writes go directly to MotherDuck — no local DuckDB, no disk
- **ASOF ratings**: `MAX(as_of_game_id) WHERE as_of_game_id <= game_id` — works even if ratings are one game stale
- **Player ratings**: full Ridge regression (same as `player_rapm.py`) run once nightly for ONE new as_of point (~5s, ~40MB peak)
- **Lineup ratings**: EWMA with adaptive α = `max(0.05, 1/n)` for observed component, then Bayesian shrinkage blend
- **`games_together`**: new int32 column added to `features.lineup_ratings` in MotherDuck. Historical rows are NULL; pipeline falls back to `possessions_together // 25`
- **Idempotent**: re-running skips Phase 2 if game_id already in possession_flat; skips Phase 3 inserts if as_of_game_id already exists
- **INSERT OR IGNORE not available** in DuckDB 1.5.1 without a PRIMARY KEY — pipeline uses plain INSERT with pre-check guards instead

### MotherDuck access
- `kalshi_trading` database owned by the account whose token is in `.env` / Fly.io secrets
- The old token was a read-only share token — replaced with owner token on 2026-04-02
- **IMPORTANT**: Fly.io `MOTHERDUCK_TOKEN` secret also needs updating before deploy:
  ```bash
  fly secrets set MOTHERDUCK_TOKEN="<new-token>" -a kalshi-recorder
  fly deploy -a kalshi-recorder
  ```

### Verified results (2026-04-02 test run)
- March 31 games: 1,411 possession_flat rows inserted (7 games)
- player_ratings: 697 players at `as_of=0022501104`
- lineup_ratings: 209 new rows at `as_of=0022501104` with `games_together` populated
- Idempotency confirmed: second run skipped Phase 2 entirely
- Tonight's games (not yet played): Phase 0 inserted 6 dim_games rows, Phase 1 exited cleanly

### Manual trigger (for testing or backfill)
```bash
python -m data.ingestion.post_game_pipeline 2026-03-31
```

### Known issues / gotchas
- `duckdb_loader.py` has a broken `INSERT OR IGNORE` on `dim_teams` in DuckDB 1.5.1 — the `--pull-motherduck` and `--pull-motherduck-full` flags don't work locally. Use `--pull-possession-flat` and `--pull-features-schema` as standalone flags instead.
- Local DuckDB (`kalshi_trading.duckdb`) may be stale — trust MotherDuck as source of truth, work directly against cloud when verifying pipeline results.
- `possession_flat` in MotherDuck uses `wall_clock_ts` but local feature builder produces `period_wall_clock` — the pipeline handles this by aligning DataFrame columns to the target table schema before inserting.

---

## Common Commands

All commands run from the project root with `source venv/bin/activate` first.

### Data Pipeline

```bash
# Fetch PBP for games missing from possessions parquet, then update local DuckDB + sync to cloud
python -m data.ingestion.nba_api_client                        # all missing games
python -m data.ingestion.nba_api_client --since 2026-03-13     # only games on/after this date
python -m data.ingestion.nba_api_client --game 0022501039      # single game

# Build/update local DuckDB only
python -m data.ingestion.duckdb_loader

# Build local DuckDB, then push new rows to MotherDuck (additive — never deletes remote rows)
python -m data.ingestion.duckdb_loader --sync-motherduck

# Pull new rows from MotherDuck into local DB (additive — never deletes local rows)
python -m data.ingestion.duckdb_loader --pull-motherduck

# DESTRUCTIVE: replace entire local DB with MotherDuck copy (use for fresh setup or reset)
python -m data.ingestion.duckdb_loader --pull-motherduck-full
```

### Sync behavior
Both `--sync-motherduck` and `--pull-motherduck` are additive merges keyed on each table's
natural key (e.g. `game_id + event_id` for possession_feed). Running either direction twice
is safe — the second run sees everything already exists and does nothing. Remote-only rows
are never deleted by a push; local-only rows are never deleted by a merge pull.

Use `--pull-motherduck-full` only when you want a clean slate (new machine, corrupted local DB).

### Recorder

```bash
# Record live Kalshi ticks during NBA games (run continuously on game days)
python data/ingestion/kalshi_recorder.py

# Check today's game schedule + recommended recorder start time
python data/ingestion/game_schedule.py
python data/ingestion/game_schedule.py --date 2026-03-25
```

---

## ML Model Stack

Three distinct modeling layers. Don't conflate them.

### Layer 1 — Run Predictor (XGBoost Classifier)
- Input: 58 FEATURE_COLS from `features.possession_flat` — ALL possessions (scoring + non-scoring)
- Output: calibrated P(`target_meaningful_run_5_scoring`) — home outscores by 6+ in next 5 **scoring** possessions (~3-4 min window, pace-independent)
- **Current: AUCPR 0.1075 vs baseline 0.0840 (~28% lift)** — retrained 2026-03-31
  - Train: Oct 21, 2025–Jan 31, 2026 (296,257 rows); Val: Feb 1–Mar 5, 2026 (37,293 rows)
  - Test: Mar 6, 2026–present — untouched
  - Top features: current_run_team_encoded, minutes_into_game, away_star_on_court, trailing_team_urgency, away_in_bonus
- **Use isotonic regression calibration** — raw XGBoost scores cluster near base rate (8.4%) and are not trustworthy as probabilities without calibration. This matters for trading.
- Do NOT use `scale_pos_weight` — shifts probs toward 0.5, destroying calibration
- Do NOT use focal loss — harder to calibrate post-hoc
- Optimize decision threshold against Sharpe on simulator, not against accuracy or F1
- Stay with XGBoost. Trees beat deep learning on tabular data under 1M rows. LSTMs/Transformers add nothing — our features already encode temporal context (momentum windows, run state).
- Retrain on rolling 60-game window every ~10 games during live season to handle concept drift

### Layer 2 — Trade Outcome Model (XGBoost Regressor)
- **Training set: L1-entry rows only** — rows where L1 run_prob exceeds the entry threshold. L2 is never called on rows we wouldn't trade, so it should never train on them. Training distribution must match inference distribution.
- **Trade direction from L1** — L1 determines which side to enter (home run → buy YES, away run → buy NO). The PnL label sign flips accordingly. Raw Δ(yes_bid) is always adjusted for direction before labeling.
- **Target: simulated PnL under hybrid exit strategy** — NOT Δ(yes_bid) at fixed t+120s. For each training row, simulate forward through tick + possession data and exit at whichever fires first:
  1. Stop Loss triggered (price moves X¢ against position)
  2. Take Profit triggered (price moves Y¢ in favor)
  3. `current_team_run` flips to opposing team or neutral
  4. N scoring possessions elapsed (primary time stop — pace-independent)
  5. `is_blowout` or `is_garbage_time` becomes true
- **Target units: log-odds change**, not raw cents. `logit(p_exit/100) - logit(p_entry/100)` normalizes for price level — a 10¢ move at 50¢ is a different probability shift than at 80¢. This prevents high-price buckets from dominating training.
- **Trade only in 30–70¢ band** — restrict entry rows to `30 ≤ yes_bid ≤ 70`. Outside this range, market certainty is too high for basketball signal to move price meaningfully.
- Real intra-game Kalshi tick data recorded continuously to `kalshi_ticks` in MotherDuck.
- **Synthetic price model is scrapped** — we have real tick data.
- Current model (`models/saved/kalshi_price_movement_predictor.pkl`) was trained with fixed 120s target on all rows — do not use as-is. Retrain once exit simulation is built.

### Layer 3 — Entry/Exit Agent
- **Start with contextual bandit (Thompson Sampling)** — treats each possession decision as independent. Trains in 100-200 games. Easy to debug. Good fit for thin, illiquid markets.
- State: run_prob, L2_expected_pnl, yes_bid, quarter, score_diff, lineup_delta, position_state
- Actions: buy_yes / buy_no / exit / wait
- Reward: realized PnL after maker fees (must match hybrid exit simulation used to train L2)
- **Upgrade to PPO** only if bandit plateaus — PPO can learn multi-step planning but needs 10x more data and is much harder to debug
- Never use: DQN (sparse rewards cause Q-value instability), SAC (continuous action space mismatch)
- If going full offline RL: use IQL (simple, stable, trains on simulator rollouts)

---

## Live System — Information Flow

Two phases: pre-game (30 min before tip) and in-game (possession by possession).

### Pre-Game (~30 min before tip)
Runs once. No trades happen here.
```
Projected lineups + historical feature store
    → pregame_analyzer.py
    → game_context.json (loaded into memory at tip-off)
```
Contains: lineup net ratings, player APM lookup, coaching tendencies, watch flags, strategy priors.
Compute everything slow-changing before the game starts. Nothing in game_context.json is recomputed mid-game.

### In-Game Loop
```
Sportradar WebSocket (raw events, ~15-20s latency)
    → Event Parser (possession parser — see critical note below)
    → Feature Computer (merges live state + game_context.json)
    → Run Predictor (XGBoost) → P(run)
    → Price Movement Model → expected Δ(yes_bid)
    → RL Agent → BUY YES / BUY NO / EXIT / WAIT
    → Risk Module (position_limits.py — always)
    → Execution Layer (kalshi_client.py)
```

### Latency Reality
~15-25 seconds from real-world event to order placement. Still faster than retail because:
- Retail doesn't notice lineup changes
- Retail reacts to what they saw on TV, not what's coming
- Retail doesn't have a model
This is the core of our edge. Don't try to compete on speed.

---

## Live Feature Data — Availability Audit

All 88 features are available in real-time. Nothing requires future data.
Two engineering problems exist:

### 1. Shot Coordinate System Mismatch (Low Risk)
- nba_api historical data: shot_x/shot_y in NBA's court coordinate system
- Sportradar live data: different coordinate system
- Fix: write a one-time coordinate converter calibrated against a game where both sources exist
- Affected features: `shot_distance`, and everything derived from it (xPPP, sustainability, run_paint_pct)
- These features matter for shot quality signals — don't skip the calibration

### 2. Possession Parser (HIGH RISK — most critical piece of live system)
- Sportradar sends individual events (fouls, shots, free throws, etc.)
- Our training data is one row per possession
- The live feature computer needs a state machine that groups events into possessions using the **exact same boundary definition as nba_api**
- If the parser diverges from training, features will be systematically wrong even though raw data is correct
- **Validation requirement**: run the parser on a historical game and compare output possession-by-possession against nba_api's output. Must match exactly before going live.

### Feature availability by category:
| Category | Status |
|----------|--------|
| Score, clock, game state | Direct from feed |
| Momentum/run features | Computed (accumulated from possession history) |
| Shot quality (xPPP, sustainability) | Computed — requires shot_distance coord conversion |
| Foul/bonus/timeout features | From foul + timeout event streams |
| Lineup/APM features | Pre-game loaded (game_context.json) + live lineup tracking |
| Foul trouble flags | Live foul tracking + pre-game key player list |
| back_to_back, schedule context | Pre-game (never changes mid-game) |

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