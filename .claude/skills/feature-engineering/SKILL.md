---
name: feature-engineering
description: Feature construction and data architecture: the no-lookahead constraint, the one formula/two callers rule in models/features/transforms.py, point-in-time ratings, the feature store schema, and live feature validation. Use when adding or changing a feature, editing the offline builder or live inference feature path, or checking offline/live parity.
---

# Feature Engineering & Data Architecture

## Core Constraint (MOST IMPORTANT)
**Every feature must be computable using only data from before the decision moment.**
No lookahead bias. Features stored in pre-computed feature store. Strategies read features only.

## Second Constraint: one formula, two callers
Every derived feature lives in **`models/features/transforms.py`** and is imported by both the
offline builder (`models/mmoe/dataset.py`) and live inference
(`live-trader/inference/features.py`). Never inline a formula in either path — five silent
train/live divergences came from exactly that. Functions are pure numpy ufunc chains, so the
same one takes Series offline and floats online.

Verify with `tests/test_feature_parity.py` (asserts offline == live to 1e-6).
Background: `docs/FEATURE_CONSOLIDATION.md`.

## Data Schema

**Raw Tables (never modified after ingestion)**
- `games` — game metadata
- `possessions` — one row per possession end (natural decision unit)
- `substitution_events` — lineup changes
- `kalshi_price_ticks` — recorded tick data

Times in two forms: `wall_clock_time` (aligns with Kalshi) and `game_clock` (basketball context).

## Model Input (one per possession)
**58 features = 33 physics + 11 pregame + 14 market.** Source of truth:
`models/mmoe/feature_config.py`.

`possession_flat` still stores the wider raw column set; the 33 physics features are derived
from it in `dataset.py::_add_derived_features`. Adding a model input means adding it there
*and* to the live path, both via `transforms.py`.

Design rules the 33 follow:
- **Fold direction into magnitude.** One signed column, not a sign column plus a magnitude
  column — an MLP learns sums easily and products badly. `home − away` for every pair.
- **No column that is an algebraic function of another.** Seven such columns were removed.
- **No sentinels, no `np.sign()`.** A `999` sentinel on 9.4% of rows crushed a feature's real
  range to a z-span of 0.24 under StandardScaler; `sign()` throws away magnitude the model
  cannot recover. Use a bounded value plus a separate flag, or the raw difference.

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
`data/feature_store/feature_rows.parquet` — 400K+ rows.
Pre-computed. Every possession has all features available at decision moment.
Rolling updates via nightly pipeline.

MMoE training reads `features.possession_flat` from MotherDuck (449K rows), **not** the local
DuckDB copy — the local copy is stale and has NULL rows that do not exist upstream.

Training is filtered to the regime the agent actually trades: drops overtime (`period ≥ 5`)
and blowouts (`|score_diff| > 30`, read from `trading.yaml`). Do **not** filter on the stored
`is_garbage_time` column — it uses a 20-pt margin vs the gate's 30 and discards 28,627 rows
the system would really trade.

## Live Feature Validation
**Shot Coordinate Mismatch (Low Risk):** nba_api and live feed use different systems. One-time converter needed.

**Possession Parser (HIGH RISK):** Live parser groups events into possessions. Must match nba_api boundary definition exactly. Validate historical game possession-by-possession before going live.
