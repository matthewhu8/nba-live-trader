---
name: feature-engineering
description: Working on feature construction, the feature store, player ratings, lineup ratings, rotation tendencies, stint segmentation, momentum features, or target variables
---

# Feature Engineering Skill

## The Prime Directive: No Lookahead Bias
Every feature must be computable using ONLY data available before the decision moment.
This is the most important constraint in the entire project.
When adding any feature, ask: "Could this value have been known at possession time T?"
If any doubt exists — flag it before implementing.

Common lookahead traps to watch for:
- Season averages that silently include future games
- Lineup ratings computed from the full game when the decision was at halftime
- "Recent form" windows that cross game boundaries incorrectly
- Any target variable leaking into features

## Feature Store Schema
Location: `data/feature_store/feature_rows.parquet`
One row per possession. All features backward-looking. Targets forward-looking and clearly separated.

Key feature groups and their files:
- `models/features/lineup_features.py` — net rating delta, player presence, foul state
- `models/features/momentum_features.py` — run length, pace, shot quality, scoring sustainability
- `models/features/context_features.py` — quarter, score diff, fatigue, blowout flag, rotation signal
- `models/features/targets.py` — forward-looking targets (TRAINING ONLY, never used as features)
- `models/features/builder.py` — orchestrates all modules into one feature row

## Rolling Ratings Convention
- Player ratings: `data/feature_store/player_ratings.parquet`
  - Fields: player_id, as_of_game_id, off_rating, def_rating, adjusted_plus_minus
  - V1: prior season RAPM as base, exponential decay updates with current season
  - Recent games weighted more heavily than early season
- Lineup ratings: `data/feature_store/lineup_ratings.parquet`
  - Fields: lineup_id (hash of 5 player ids), as_of_game_id, net_rating, possessions_together
  - Small sample (<50 possessions): regress toward mean of 5 individual player ratings
  - Never use within-game data to compute a rating used earlier in that same game

## The Core Signal
`lineup_net_rating_delta` = home_lineup_net_rating - away_lineup_net_rating
This is the most predictive single feature. Weight it accordingly.
Large positive delta after a home substitution = likely home team run signal.

## Shot Quality / Scoring Sustainability
`scoring_sustainable` = True when team is scoring via open 3s + paint pressure
`scoring_sustainable` = False when scoring via contested midrange
Sustainability matters more than run length for predicting run continuation.
Source data: shot zone + contest level from nba_api play-by-play.

## Rotation Signal
`rotation_signal` (0.0–1.0): probability that a substitution is imminent
Computed from `data/feature_store/rotation_tendencies.parquet`
High rotation_signal + favorable incoming lineup = RotationAnticipationStrategy trigger

## Blowout / Garbage Time Rules
`is_blowout` = score_diff > 20
`is_garbage_time` = is_blowout AND time_remaining < 6 min AND bench players on court
Both flags = model off. Do not compute signals. Do not place orders.
These situations have different dynamics — training on them corrupts the model.

## Validation Checklist (run after any feature change)
1. Sample 50 random feature rows — manually verify each feature could have been known at that time
2. Check that targets.py is not imported anywhere in the live feature pipeline
3. Confirm rolling ratings use as_of_game_id correctly (never future games)
4. Run correlation check: any feature >0.95 correlated with a target = likely leak