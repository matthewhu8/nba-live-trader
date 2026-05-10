---
name: data-ingestion
description: Working on nba_api data pulling, Kalshi price recording, raw data normalization, play-by-play processing, or data pipeline infrastructure
---

# Data Ingestion Skill

## Priority Order
1. `kalshi_recorder.py` — time-sensitive, deploy first, data lost forever if not running
2. `nba_api_client.py` — historical pull, can be done anytime
3. Raw table normalization — runs after nba_api pull
4. `live_feed.py` — future Sportradar connector, not needed until Phase 5

## Kalshi Recorder (`data/ingestion/kalshi_recorder.py`)
Records every price tick on every live NBA basketball market.
Must run continuously during all NBA games. Must reconnect automatically on drop.

Key behaviors:
- Subscribe to Kalshi WebSocket: orderbook_delta, trade, ticker channels
- Write every tick to `data/raw/kalshi_price_ticks.parquet` (append mode)
- Reconnect with exponential backoff on disconnect — never silently drop ticks
- Buffer events during reconnection window
- Log gaps in recording (missed seconds) — these affect backtest quality
- Must identify basketball markets automatically — don't hardcode tickers

Schema for ticks:
```
market_id, game_id (nullable until matched), wall_clock_time,
yes_bid, yes_ask, yes_last, bid_depth, ask_depth
```

NBA game schedule: check nba_api or ESPN API for game times.
Recorder should auto-start 15 min before tip-off and stop 15 min after final whistle.

## nba_api Client (`data/ingestion/nba_api_client.py`)
Python package: `pip install nba_api`
Rate limit: ~1 req/sec — always add delays between calls or you get 429s.

Key endpoints to pull:
```python
from nba_api.stats.endpoints import (
    PlayByPlayV2,        # possession-level play-by-play
    BoxScoreTraditionalV2,  # lineups, minutes, stats
    LeagueGameLog,       # all games for a season
    TeamGameLog,
)
```

Pull strategy for historical data:
1. Pull all game IDs for seasons 2021-22, 2022-23, 2023-24, 2024-25
2. For each game: pull play-by-play + box score
3. Normalize into possession and substitution_event tables
4. Store as parquet: `data/raw/possessions.parquet`, `data/raw/substitution_events.parquet`

Rate limiting wrapper (always use this):
```python
import time
def safe_request(endpoint_class, **kwargs):
    time.sleep(0.6)  # stay under rate limit
    return endpoint_class(**kwargs).get_data_frames()
```

## Raw Table Normalization
`possessions.parquet` — one row per possession end:
```
game_id, possession_id, wall_clock_time, game_clock_seconds,
quarter, home_score, away_score, team_scored, points,
shot_zone, shot_contested, play_type,
home_lineup_id, away_lineup_id
```

`substitution_events.parquet` — one row per substitution:
```
game_id, wall_clock_time, game_clock_seconds, quarter, team,
player_in, player_out,
resulting_home_lineup_id, resulting_away_lineup_id
```

`lineup_id` convention: SHA256 hash of sorted tuple of 5 player IDs (as strings).
Same 5 players always produces same lineup_id regardless of insertion order.

## Time Handling
Always store BOTH:
- `wall_clock_time`: real datetime (UTC) — for aligning with Kalshi prices
- `game_clock_seconds`: seconds remaining in quarter — for basketball context
Never confuse these. A timeout at 6:00 game clock is not 6 minutes of wall clock.

## Data Quality Checks (run after every ingestion batch)
- No null game_ids or possession_ids
- wall_clock_time is monotonically increasing within each game
- Substitution events align with lineup changes in possession table
- No duplicate possession_ids per game
- home_score + away_score are monotonically non-decreasing

## Storage Format
All data in parquet (columnar, fast, typed).
Partition by season for large tables: `possessions/season=2023-24/`
Use pyarrow or pandas for read/write. Never CSV for large tables.