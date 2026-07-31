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

The tick data is a direct input to the MMoE model (Head B — price trajectory).
The goal is to join basketball possession features with Kalshi bid/ask movements
and train the model end-to-end to predict run probability, price direction, and
run survival — all from one unified network, net of maker fees.

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
[MMoE Model] (models/saved/mmoe.pt)
  → Head A: P(run) | Head B: Δ(yes_bid) | Head C: survival hazard
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

Sits on top of the MMoE. Handles the sequential decision problem:
not just "what does the model predict" but "should I enter NOW, how big, and when do I exit."

```
State:  mmoe_run_prob (Head A), mmoe_price_delta (Head B), mmoe_survival (Head C),
        yes_bid, quarter, score_diff, lineup_delta,
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
- When Head C survival drops fast → cut early; when it stays high → hold
- When to cut a losing position vs. hold through noise

---

## Directory Structure

```
kalshi-trader/
├── AGENTS.md                         # Architecture and principles
├── .env                              # credentials — never commit
├── .gitignore
│
├── data/
│   ├── ingestion/
│   │   ├── nba_api_client.py         # Historical play-by-play + lineups
│   │   ├── kalshi_recorder.py        # Live tick recording
│   │   └── post_game_pipeline.py     # Nightly 3 AM pipeline (Fly.io)
│   ├── raw/
│   │   ├── games.parquet
│   │   ├── possessions.parquet
│   │   ├── substitution_events.parquet
│   │   └── kalshi_price_ticks.parquet
│   └── feature_store/
│       ├── player_ratings.parquet
│       ├── lineup_ratings.parquet
│       └── feature_rows.parquet
│
├── models/
│   ├── mmoe/                         # Multi-task Mixture-of-Experts model
│   │   ├── model.py
│   │   └── train.py
│   ├── features/
│   │   ├── builder.py
│   │   ├── lineup_features.py
│   │   ├── momentum_features.py
│   │   ├── context_features.py
│   │   └── targets.py
│   ├── ratings/
│   │   ├── player_rapm.py
│   │   └── lineup_net_rating.py
│   └── saved/
│       ├── mmoe_delay20.pt
│       └── mmoe_scaler_delay20.pkl
│
├── backtesting/
│   ├── simulator.py                  # Core replay engine
│   ├── evaluator.py                  # Metrics and context breakdown
│   ├── mmoe_backtest.py
│   └── results/
│
├── live-trader/
│   ├── go/                           # Go execution engine
│   │   ├── main.go
│   │   ├── coordinator.go
│   │   ├── agent.go
│   │   ├── risk.go
│   │   └── jsonlog.go
│   ├── inference/                    # Python inference (FastAPI)
│   │   ├── main.py
│   │   ├── features.py
│   │   └── dashboard.py
│   └── config/
│       └── trading.yaml
│
├── pregame/
│   └── pregame_analyzer.py
│
└── logs/
    └── runs/{date}/{run_id}/
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

## Current Status

- [x] Phase 1: Data foundation (recorder deployed, nba_api ingestion, raw tables normalized)
- [x] Phase 2: Feature store (58 features, player ratings, lineup ratings, nightly pipeline)
- [x] Phase 3: Backtesting infrastructure (simulator, evaluator, MMoE validation)
- [x] Phase 4: MMoE model trained and validated (Head A, B, C all deployed)
- [x] Phase 5: Go execution engine + Python inference service (paper mode)
- [ ] **Phase 6: Paper trading** ← CURRENT FOCUS

**See `/data-ingestion`, `/backtesting`, and `/kalshi-execution` skills for operational commands.**

---

## ML Model Stack

### MMoE (Multi-task Mixture-of-Experts, PyTorch) ← PRIMARY MODEL
A single unified network that replaces both the XGBoost run predictor and the price movement regressor.

- **Architecture:** 83 input features (58 basketball + 10 pregame + 14 market + 1 market flag),
  3 expert networks (64-dim MLP), 3 gating networks, 3 task heads. ~37K params.
- **Head A — Run Classifier:** P(meaningful run in next 5 scoring possessions)
  - AUCPR 0.1533 vs 0.0840 baseline (+82%) and vs 0.1075 XGBoost (+43%)
- **Head B — Price Trajectory:** expected Δ(yes_bid) in log-odds over the hold window
  - RMSE 0.5525 log-odds delta; Dir Acc 59.2% on meaningful-exit rows
  - Trained only on joint rows where Kalshi ticks exist (24K rows, 148 games)
- **Head C — Run Survival Hazard:** probability run is still ongoing at each of 10 time horizons
  - Brier score 0.0939 across all horizons — used for dynamic exit timing
- **Model artifacts:** `models/saved/mmoe.pt`, `models/saved/mmoe_scaler.pkl`
- **Data split:** basketball time-based (Jan 2026 cutoff); Head B: Mar 23–Apr 6 train / Apr 7–12 val
- **Trade only in 30–70¢ band** — restrict entry rows to `30 ≤ yes_bid ≤ 70`. Outside this range,
  market certainty is too high for basketball signal to move price meaningfully.
- **Target units for Head B: log-odds change** — `logit(p_exit/100) - logit(p_entry/100)`.
  Normalizes for price level so a 10¢ move at 50¢ ≠ 10¢ move at 80¢.
- **Retrain cadence:** rolling 60-game window every ~10 games during live season to handle concept drift.
  All three heads retrain together — they share the expert networks.
- **Key fix on record:** zero-inflated trajectory targets (42% of traj_9 == 0) caused a 13.1% dir acc
  bug; fixed via signal-filtered Huber loss mask (`abs mean > 0.02`) + directional accuracy threshold
  (`abs final checkpoint > 0.05`).

**Why MMoE over separate XGBoost models:**
- Shared experts capture basketball context that matters for both run prediction AND price movement
- Head B (price) can only train on the ~5% of rows where Kalshi ticks exist — the shared experts
  transfer knowledge from 433K basketball rows to inform that thin Head B training set
- Single inference call at decision time instead of two separate model calls
- Multi-task regularization reduces overfitting on each individual head

### Entry/Exit Agent (RL)
Sits on top of MMoE. Handles the sequential decision problem: not just "is a run coming and will
price move" but "should I enter NOW, how big, and when do I exit."

- **Start with contextual bandit (Thompson Sampling)** — treats each possession as independent.
  Trains in 100-200 games. Easy to debug. Good fit for thin, illiquid markets.
- **State:** mmoe_run_prob (Head A), mmoe_price_delta (Head B), mmoe_survival (Head C),
  yes_bid, quarter, score_diff, lineup_delta, current_position, time_in_position
- **Actions:** buy_yes / buy_no / exit / wait
- **Reward:** realized PnL after maker fees
- **Upgrade to PPO** only if bandit plateaus — PPO can learn multi-step planning but needs 10x
  more data and is much harder to debug
- Never use: DQN (sparse rewards cause Q-value instability), SAC (continuous action space mismatch)
- If going full offline RL: use IQL (simple, stable, trains on simulator rollouts)

**Exit logic informed by Head C (survival):** when the survival hazard drops sharply across horizons,
that's a signal the run is ending — use it to trigger exit before the price reverts.

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
    → MMoE Model → Head A: P(run) | Head B: Δ(yes_bid) | Head C: survival hazard
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

## Key Reminders for Codex
- The end goal is live profitable trading. Every decision should serve that.
- Kalshi recorder is the highest priority — it is time-sensitive, data is lost forever if not running
- Always verify features are point-in-time before adding them to the feature store
- Lookahead bias makes backtests look great and live trading look terrible — be paranoid about it
- When touching execution code: paper mode must be default, confirm before changing
- Never suggest a taker order under any circumstance
- Blowout / garbage time = model off. Never trade garbage time.
- If a backtest result looks suspiciously good, assume lookahead bias first
- We are in Phase 1. Don't build Phase 5 code while Phase 1 is incomplete.