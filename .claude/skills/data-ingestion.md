# Data Ingestion Operations

## Recorder (Live Price Recording)
```bash
# Record live Kalshi ticks (run on game days)
python data/ingestion/kalshi_recorder.py

# Check today's schedule
python data/ingestion/game_schedule.py
python data/ingestion/game_schedule.py --date 2026-03-25
```
**CRITICAL:** Recorder runs on Fly.io. Historical Kalshi data not available elsewhere. Every missed game = lost training data.

## NBA Play-by-Play (nba_api)
```bash
# All missing games
python -m data.ingestion.nba_api_client

# Games on/after date
python -m data.ingestion.nba_api_client --since 2026-03-13

# Single game
python -m data.ingestion.nba_api_client --game 0022501039
```

## Local DuckDB & MotherDuck Sync
```bash
# Build/update local DuckDB only
python -m data.ingestion.duckdb_loader

# Push to MotherDuck (additive, safe to run twice)
python -m data.ingestion.duckdb_loader --sync-motherduck

# Pull from MotherDuck (additive, safe to run twice)
python -m data.ingestion.duckdb_loader --pull-motherduck

# DESTRUCTIVE: Replace local with remote (fresh setup only)
python -m data.ingestion.duckdb_loader --pull-motherduck-full
```

## Nightly Pipeline (Fly.io @ 3 AM ET)
Automatic trigger via `recorder_daemon.py`. Manual trigger:
```bash
python -m data.ingestion.post_game_pipeline 2026-03-31
```
Idempotent. Trust MotherDuck as source of truth.

## Environment
- Source venv: `source venv/bin/activate`
- MotherDuck token in `.env` as `MOTHERDUCK_TOKEN`
- Never commit `.env`, `data/`, or `logs/`
