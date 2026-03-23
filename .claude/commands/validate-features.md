---
description: Audit the feature store and feature engineering code for lookahead bias. Run this after any change to models/features/ or the feature store pipeline.
allowed-tools: Read, Grep, Glob
---

# Validate Features for Lookahead Bias

Perform a thorough audit of the feature engineering pipeline for lookahead bias.
Lookahead bias is the #1 failure mode in this project — it makes backtests look
great and live trading look terrible.

## Step 1: Scan for Target Variable Leakage
Search all files in `models/features/` for any import or reference to `targets.py`.
```
grep -r "targets" models/features/
grep -r "next_10_poss" models/features/
grep -r "next_20_poss" models/features/
grep -r "kalshi_price_change" models/features/
```
Report any matches. These are critical violations.

## Step 2: Audit Rolling Rating Computation
Read `models/ratings/player_rapm.py` and `models/ratings/lineup_net_rating.py`.
Verify:
- Ratings are keyed by `as_of_game_id` (not as_of_date, not season-wide)
- No game's rating computation uses data from that same game
- The training set for ratings only uses games with date < current game date
- No future games are accidentally included via season-level aggregations

## Step 3: Audit Momentum Features
Read `models/features/momentum_features.py`.
Verify:
- All rolling windows (last 5 possessions, last 10 possessions) look BACKWARD only
- Pace calculations use only possessions up to and including current possession
- Current run length does not include the current possession's outcome
- No "next possession" data anywhere

## Step 4: Audit Context Features
Read `models/features/context_features.py`.
Verify:
- `rotation_signal` uses rotation tendency model computed from games BEFORE current game
- `home_fatigue_proxy` uses minutes played UP TO current possession only
- `is_blowout` uses current score, not future score

## Step 5: Spot Check 10 Feature Rows
Read a sample of rows from `data/feature_store/feature_rows.parquet` (first 10 rows).
For each row, manually verify 3 features:
- Does lineup_net_rating_delta match expected value given as_of_game context?
- Does current_run_length match what the game log shows up to that possession?
- Is is_blowout consistent with the score at that possession?

## Step 6: Correlation Check
If feature_rows.parquet exists, check correlations:
- Any feature showing >0.8 correlation with a target variable is suspicious
- Report all correlations >0.5 between features and targets

## Final Report
Produce a summary:
- CRITICAL issues (definite lookahead bias) — must fix before any backtesting
- WARNING issues (possible lookahead, needs investigation)
- PASS items (verified clean)
- SKIPPED items (files not yet implemented)