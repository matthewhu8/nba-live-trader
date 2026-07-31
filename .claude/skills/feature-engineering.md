# Feature Engineering & Data Architecture

## Core Constraint (MOST IMPORTANT)
**Every feature must be computable using only data from before the decision moment.**
No lookahead bias. Features stored in pre-computed feature store. Strategies read features only.

## Data Schema

**Raw Tables (never modified after ingestion)**
- `games` — game metadata
- `possessions` — one row per possession end (natural decision unit)
- `substitution_events` — lineup changes
- `kalshi_price_ticks` — recorded tick data

Times in two forms: `wall_clock_time` (aligns with Kalshi) and `game_clock` (basketball context).

## Feature Row (one per possession)
88 total features across categories:
- **Identity:** game_id, possession_id, wall_clock_time, quarter, score
- **Lineup State:** lineup net ratings, deltas, sample sizes, star players on court, subs
- **Foul State:** key defender fouls, fouled-out flags
- **Momentum:** points last 5/10 possessions, current run info, pace
- **Shot Quality:** xPPP above expected, sustainable scoring flags
- **Context:** starters minutes, back-to-back, garbage time flags

**Target Variables (training only, never live):**
- `home_next_10_poss_margin`, `kalshi_price_change_3min`, `meaningful_run_occurred`

Targets separated in code. Feature pipeline backward-looking. Live system uses features only.

## Point-in-Time Ratings (Critical)

**Player Ratings:** Rolling regularized plus/minus per game using only prior games.
```
player_id, as_of_game_id, offensive_rating, defensive_rating
```

**Lineup Ratings:** Historical 5-man combinations with ASOF join.
```
lineup_id, as_of_game_id, net_rating, possessions_together
```
Regress lineups <50 possessions toward 5-player average.

**Rotation Tendency Model:** Coach-specific substitution timing (per quarter, score bucket, time remaining).

## Feature Store
`data/feature_store/feature_rows.parquet` — 400K+ rows, 88 columns.
Pre-computed. Every possession has all features available at decision moment.
Rolling updates via nightly pipeline.

## Live Feature Validation
**Shot Coordinate Mismatch (Low Risk):** nba_api and live feed use different systems. One-time converter needed.

**Possession Parser (HIGH RISK):** Live parser groups events into possessions. Must match nba_api boundary definition exactly. Validate historical game possession-by-possession before going live.
