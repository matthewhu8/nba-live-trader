"""
Backfill possession_flat for Group 1 and Group 2 null rows.

Group 1 (207 games, 2025-26, Mar 25 – May 5, 2026):
  event_id + all foul/sub/timeout columns NULL.
  Root cause: nightly post-game pipeline never wrote to raw tables or possession_feed.
  Fix: fetch PBP via nba_api, insert raw events, build possession_feed, re-run features.

Group 2 (29 games, 2024-25, Dec 10-16 2024 + Feb 13 2025):
  All momentum/context features NULL (home_points_last_5_poss, current_run_team, …).
  Root cause: early pipeline insertion before momentum computation was implemented.
  Fix: re-run feature pipeline using existing raw tables and possession_feed.

Usage:
    python -m data.ingestion.backfill_possession_flat
    python -m data.ingestion.backfill_possession_flat --group 1
    python -m data.ingestion.backfill_possession_flat --group 2
    python -m data.ingestion.backfill_possession_flat --dry-run
    python -m data.ingestion.backfill_possession_flat --game 0022501051
"""

import argparse
import logging
import os
import time
from datetime import timedelta

import duckdb
import pandas as pd
from dotenv import load_dotenv

from data.ingestion.nba_api_client import parse_game_to_events
from models.features.builder import process_game

logger = logging.getLogger(__name__)

# nba_api rate-limit courtesy pause between game fetches
_NBA_API_SLEEP = 0.7


# ---------------------------------------------------------------------------
# MotherDuck connection
# ---------------------------------------------------------------------------

def _md_connect() -> duckdb.DuckDBPyConnection:
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set — check .env or environment")
    conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={token}")
    conn.execute("SET memory_limit='512MB'")
    return conn


# ---------------------------------------------------------------------------
# Game discovery
# ---------------------------------------------------------------------------

def _get_group2_game_ids(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """29 games with NULL momentum features (home_points_last_5_poss IS NULL)."""
    rows = conn.execute("""
        SELECT DISTINCT game_id FROM features.possession_flat
        WHERE home_points_last_5_poss IS NULL
        ORDER BY game_id
    """).fetchall()
    return [r[0] for r in rows]


def _get_group1_game_ids(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """207 games with NULL event_id but non-NULL momentum (post-game pipeline rows)."""
    rows = conn.execute("""
        SELECT DISTINCT game_id FROM features.possession_flat
        WHERE event_id IS NULL
          AND home_points_last_5_poss IS NOT NULL
        ORDER BY game_id
    """).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Raw data fetchers
# ---------------------------------------------------------------------------

def _fetch_raw_events(game_id: str, conn: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Pull raw event tables from MotherDuck for an existing game."""
    possessions = conn.execute(
        "SELECT * FROM main.raw_possessions WHERE game_id = ?", [game_id]
    ).df()
    fouls = conn.execute(
        "SELECT * FROM main.raw_fouls WHERE game_id = ?", [game_id]
    ).df()
    substitutions = conn.execute(
        "SELECT * FROM main.raw_substitutions WHERE game_id = ?", [game_id]
    ).df()
    timeouts = conn.execute(
        "SELECT * FROM main.raw_timeouts WHERE game_id = ?", [game_id]
    ).df()

    for col in ["shot_distance", "shot_x", "shot_y", "player_id", "points", "shot_value"]:
        if col in possessions.columns:
            possessions[col] = possessions[col].astype("float32")

    return {
        "possessions": possessions,
        "fouls": fouls,
        "substitutions": substitutions,
        "timeouts": timeouts,
    }


def _fetch_asof_lineup_ratings(game_id: str, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """ASOF lineup_ratings snapshot for this game (same as post_game_pipeline)."""
    return conn.execute(
        """
        SELECT
            lr.lineup_id,
            ? AS as_of_game_id,
            lr.net_rating,
            lr.observed_net_rating,
            lr.predicted_net_rating,
            lr.possessions_together,
            lr.shrinkage_weight,
            lr.player_ids
        FROM features.lineup_ratings lr
        WHERE lr.as_of_game_id = (
            SELECT MAX(as_of_game_id) FROM features.lineup_ratings
            WHERE as_of_game_id <= ?
        )
        """,
        [game_id, game_id],
    ).df()


def _fetch_asof_player_ratings(
    game_id: str, conn: duckdb.DuckDBPyConnection
) -> pd.DataFrame | None:
    df = conn.execute(
        """
        SELECT pr.player_id, pr.player_name, ? AS as_of_game_id, pr.adjusted_plus_minus, pr.games_fitted
        FROM features.player_ratings pr
        WHERE pr.as_of_game_id = (
            SELECT MAX(as_of_game_id) FROM features.player_ratings
            WHERE as_of_game_id <= ?
        )
        """,
        [game_id, game_id],
    ).df()
    return df if not df.empty else None


def _fetch_dim_games_context(game_id: str, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Fetch ±30 days of dim_games context around this game."""
    game_date_row = conn.execute(
        "SELECT game_date FROM dim_games WHERE game_id = ?", [game_id]
    ).fetchone()
    if not game_date_row:
        raise ValueError(f"game_id {game_id!r} not found in dim_games")
    game_date = game_date_row[0]

    # Use string arithmetic since game_date might be a date object
    from datetime import date, timedelta
    if isinstance(game_date, str):
        from datetime import date
        gd = date.fromisoformat(game_date)
    else:
        gd = game_date
    start = (gd - timedelta(days=30)).isoformat()
    end   = (gd + timedelta(days=30)).isoformat()

    return conn.execute(
        """
        SELECT game_id, game_date, home_team, away_team, home_back_to_back, away_back_to_back
        FROM dim_games
        WHERE game_date BETWEEN ? AND ?
        """,
        [start, end],
    ).df()


def _fetch_player_tier_map(conn: duckdb.DuckDBPyConnection) -> dict[int, int]:
    rows = conn.execute(
        "SELECT player_id, star_tier FROM dim_players WHERE star_tier IS NOT NULL"
    ).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


# ---------------------------------------------------------------------------
# possession_feed aggregation helpers
# ---------------------------------------------------------------------------

def _fetch_pf_possession_rows(game_id: str, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Pull possession rows from possession_feed with their running team stats.
    Returns: possession_id, event_id, home_team_fouls_q, away_team_fouls_q,
             home_cum_fouls, away_cum_fouls, home_timeouts_used, away_timeouts_used
    """
    return conn.execute(
        """
        SELECT
            possession_id,
            event_id,
            home_team_fouls_q,
            away_team_fouls_q,
            home_cum_fouls,
            away_cum_fouls,
            home_timeouts_used,
            away_timeouts_used
        FROM main.possession_feed
        WHERE game_id = ? AND event_type = 'possession'
        """,
        [game_id],
    ).df()


def _fetch_pf_foul_aggs(game_id: str, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Aggregate foul events from possession_feed per possession_id."""
    return conn.execute(
        """
        SELECT
            possession_id,
            COUNT(*)                                                   AS foul_count,
            BOOL_OR(foul_type ILIKE '%Shooting%')                      AS had_shooting_foul,
            BOOL_OR(foul_type ILIKE '%Personal%'
                 OR foul_type ILIKE '%Loose Ball%'
                 OR foul_type ILIKE '%Blocking%')                      AS had_personal_foul,
            BOOL_OR(foul_type ILIKE '%Offensive%')                     AS had_offensive_foul,
            BOOL_OR(foul_type ILIKE '%Flagrant%')                      AS had_flagrant_foul,
            BOOL_OR(foul_type ILIKE '%Technical%')                     AS had_technical_foul
        FROM main.possession_feed
        WHERE game_id = ? AND event_type = 'foul' AND possession_id IS NOT NULL
        GROUP BY possession_id
        """,
        [game_id],
    ).df()


def _fetch_pf_sub_aggs(game_id: str, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Aggregate substitution events from possession_feed per possession_id."""
    # Need home_team to split home_sub_count vs away_sub_count
    home_team_row = conn.execute(
        "SELECT home_team FROM dim_games WHERE game_id = ?", [game_id]
    ).fetchone()
    home_team = home_team_row[0] if home_team_row else None

    return conn.execute(
        """
        SELECT
            possession_id,
            COUNT(*)                                                        AS sub_count,
            SUM(CASE WHEN sub_team = ? THEN 1.0 ELSE 0.0 END)              AS home_sub_count,
            SUM(CASE WHEN sub_team != ? AND sub_team IS NOT NULL THEN 1.0 ELSE 0.0 END)
                                                                            AS away_sub_count
        FROM main.possession_feed
        WHERE game_id = ? AND event_type = 'substitution' AND possession_id IS NOT NULL
        GROUP BY possession_id
        """,
        [home_team, home_team, game_id],
    ).df()


def _fetch_pf_timeout_aggs(game_id: str, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Count timeout events per possession_id."""
    return conn.execute(
        """
        SELECT
            possession_id,
            COUNT(*) AS timeout_count
        FROM main.possession_feed
        WHERE game_id = ? AND event_type = 'timeout' AND possession_id IS NOT NULL
        GROUP BY possession_id
        """,
        [game_id],
    ).df()


# ---------------------------------------------------------------------------
# Complete row builder
# ---------------------------------------------------------------------------

def _build_corrected_rows(
    game_id: str,
    events: dict[str, pd.DataFrame],
    conn: duckdb.DuckDBPyConnection,
    games_df: pd.DataFrame,
    lineup_ratings: pd.DataFrame,
    player_ratings: pd.DataFrame | None,
    tier_map: dict[int, int],
) -> pd.DataFrame:
    """
    Build a complete corrected possession_flat DataFrame for one game.

    Runs the feature pipeline then joins possession_feed for event-level columns.
    wall_clock_ts is intentionally left NULL here — it gets joined from existing
    possession_flat rows in the final CTAS write.
    """
    # Feature pipeline
    feature_df = process_game(
        game_id,
        events["possessions"],
        events["fouls"],
        games_df,
        lineup_ratings,
        player_ratings=player_ratings,
        timeout_events=events.get("timeouts"),
        player_tier_map=tier_map,
    )

    # Columns from possession_feed
    pf_poss     = _fetch_pf_possession_rows(game_id, conn)
    pf_fouls    = _fetch_pf_foul_aggs(game_id, conn)
    pf_subs     = _fetch_pf_sub_aggs(game_id, conn)
    pf_timeouts = _fetch_pf_timeout_aggs(game_id, conn)

    result = feature_df.merge(pf_poss,     on="possession_id", how="left")
    result = result.merge(pf_fouls,         on="possession_id", how="left")
    result = result.merge(pf_subs,          on="possession_id", how="left")
    result = result.merge(pf_timeouts,      on="possession_id", how="left")

    # Fill integer counts with 0
    for col in ["foul_count", "sub_count", "timeout_count"]:
        result[col] = result.get(col, pd.Series(0, index=result.index)).fillna(0).astype("int64")
    for col in ["home_sub_count", "away_sub_count"]:
        result[col] = result.get(col, pd.Series(0.0, index=result.index)).fillna(0.0)

    # Fill bool foul flags with False
    for col in ["had_shooting_foul", "had_offensive_foul", "had_personal_foul",
                "had_flagrant_foul", "had_technical_foul"]:
        result[col] = result.get(col, pd.Series(False, index=result.index)).fillna(False)

    # NULL array-type columns (were always NULL in working rows too)
    for col in ["foul_types", "foul_player_ids", "foul_teams", "sub_teams",
                "timeout_teams", "timeout_types", "players_in_ids", "players_out_ids"]:
        result[col] = None

    # NULL boolean-per-row flags that were never populated
    for col in ["was_foul", "was_sub", "was_timeout"]:
        result[col] = None

    # wall_clock_ts filled by the CTAS join — set to None here
    result["wall_clock_ts"] = None

    return result


# ---------------------------------------------------------------------------
# Raw-table insert for Group 1
# ---------------------------------------------------------------------------

def _raw_table_game_exists(
    table: str, game_id: str, conn: duckdb.DuckDBPyConnection
) -> bool:
    count = conn.execute(
        f"SELECT COUNT(*) FROM main.{table} WHERE game_id = ?", [game_id]
    ).fetchone()[0]
    return count > 0


def _insert_raw_events(
    game_id: str,
    events: dict[str, pd.DataFrame],
    conn: duckdb.DuckDBPyConnection,
    dry_run: bool = False,
) -> None:
    """
    Insert parse_game_to_events() output into MotherDuck raw tables.
    Pre-checks idempotency: skips tables where game_id already has rows.
    """
    table_map = {
        "raw_possessions": "possessions",
        "raw_fouls": "fouls",
        "raw_substitutions": "substitutions",
        "raw_timeouts": "timeouts",
    }

    for table, key in table_map.items():
        df = events.get(key, pd.DataFrame())
        if df.empty:
            logger.debug("game %s: %s is empty — skipping", game_id, table)
            continue

        if _raw_table_game_exists(table, game_id, conn):
            logger.debug("game %s: %s already has rows — skipping", game_id, table)
            continue

        # Align to remote schema: keep only columns that exist in the table
        remote_cols = [r[0] for r in conn.execute(f"DESCRIBE main.{table}").fetchall()]
        aligned = pd.DataFrame(index=df.index)
        for col in remote_cols:
            aligned[col] = df[col] if col in df.columns else None

        if dry_run:
            logger.info("DRY RUN: would insert %d rows into %s for game %s", len(aligned), table, game_id)
            continue

        conn.register("_raw_batch", aligned)
        conn.execute(f"INSERT INTO main.{table} SELECT * FROM _raw_batch")
        conn.unregister("_raw_batch")
        logger.debug("game %s: inserted %d rows into %s", game_id, len(aligned), table)


# ---------------------------------------------------------------------------
# possession_feed build for Group 1 games
# ---------------------------------------------------------------------------

_POSSESSION_FEED_SQL = """
INSERT INTO main.possession_feed BY NAME
WITH events AS (
    SELECT
        game_id, period, game_clock_secs, 1 AS event_priority,
        'possession'   AS event_type,
        possession_id,
        COALESCE(points, 0)   AS points_scored,
        home_score, away_score,
        home_lineup_id, away_lineup_id,
        possessing_team, team_scored, outcome,
        points, shot_value, shot_distance, shot_type, play_type,
        player_id AS scorer_player_id,
        NULL::VARCHAR AS foul_team,   NULL::BIGINT AS foul_player_id,  NULL::VARCHAR AS foul_type,
        NULL::VARCHAR AS sub_team,    NULL::BIGINT AS player_in_id,    NULL::BIGINT AS player_out_id,
        NULL::VARCHAR AS timeout_team, NULL::VARCHAR AS timeout_type,
        NULL::VARCHAR AS team_for_foul, NULL::VARCHAR AS team_for_timeout
    FROM main.raw_possessions
    WHERE game_id IN ({placeholders})

    UNION ALL

    SELECT
        game_id, period, game_clock_secs, 2,
        'foul', NULL,
        0, NULL, NULL, NULL, NULL,
        NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        team_tricode, player_id, foul_type,
        NULL, NULL, NULL,
        NULL, NULL,
        team_tricode, NULL
    FROM main.raw_fouls
    WHERE game_id IN ({placeholders})

    UNION ALL

    SELECT
        game_id, period, game_clock_secs, 3,
        'timeout', NULL,
        0, NULL, NULL, NULL, NULL,
        NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        NULL, NULL, NULL,
        NULL, NULL, NULL,
        team_tricode, timeout_type,
        NULL, team_tricode
    FROM main.raw_timeouts
    WHERE game_id IN ({placeholders})

    UNION ALL

    SELECT
        game_id, period, game_clock_secs, 4,
        'substitution', NULL,
        0, NULL, NULL, home_lineup_id, away_lineup_id,
        NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        NULL, NULL, NULL,
        team_tricode, player_in_id, player_out_id,
        NULL, NULL,
        NULL, NULL
    FROM main.raw_substitutions
    WHERE game_id IN ({placeholders})
),
ordered AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id
            ORDER BY period ASC, game_clock_secs DESC, event_priority ASC
        ) AS seq
    FROM events
),
with_state AS (
    SELECT *,
        LAST_VALUE(home_score    IGNORE NULLS) OVER w AS ff_home_score,
        LAST_VALUE(away_score    IGNORE NULLS) OVER w AS ff_away_score,
        LAST_VALUE(home_lineup_id IGNORE NULLS) OVER w AS ff_home_lineup,
        LAST_VALUE(away_lineup_id IGNORE NULLS) OVER w AS ff_away_lineup,

        COALESCE(SUM(CASE WHEN event_type='foul' AND foul_team IN (
            SELECT home_team FROM dim_games g WHERE g.game_id = ordered.game_id
        ) THEN 1 ELSE 0 END) OVER (
            PARTITION BY game_id
            ORDER BY seq
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS home_cum_fouls,

        COALESCE(SUM(CASE WHEN event_type='foul' AND foul_team NOT IN (
            SELECT home_team FROM dim_games g WHERE g.game_id = ordered.game_id
        ) AND foul_team IS NOT NULL THEN 1 ELSE 0 END) OVER (
            PARTITION BY game_id
            ORDER BY seq
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS away_cum_fouls,

        COALESCE(SUM(CASE WHEN event_type='timeout' AND team_for_timeout IN (
            SELECT home_team FROM dim_games g WHERE g.game_id = ordered.game_id
        ) THEN 1 ELSE 0 END) OVER (
            PARTITION BY game_id
            ORDER BY seq
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS home_timeouts_used,

        COALESCE(SUM(CASE WHEN event_type='timeout' AND team_for_timeout NOT IN (
            SELECT home_team FROM dim_games g WHERE g.game_id = ordered.game_id
        ) AND team_for_timeout IS NOT NULL THEN 1 ELSE 0 END) OVER (
            PARTITION BY game_id
            ORDER BY seq
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS away_timeouts_used

    FROM ordered
    WINDOW w AS (
        PARTITION BY game_id
        ORDER BY seq
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
),
with_game AS (
    SELECT ws.*, g.home_team, g.away_team
    FROM with_state ws
    LEFT JOIN dim_games g ON ws.game_id = g.game_id
)
SELECT
    game_id,
    seq                                              AS event_id,
    event_type,
    possession_id,
    period,
    game_clock_secs,
    COALESCE(ff_home_score, 0)                       AS home_score,
    COALESCE(ff_away_score, 0)                       AS away_score,
    COALESCE(ff_home_score, 0) - COALESCE(ff_away_score, 0) AS score_diff,
    ff_home_lineup                                   AS home_lineup_id,
    ff_away_lineup                                   AS away_lineup_id,

    COALESCE(SUM(CASE WHEN event_type='foul' AND foul_team = home_team
                 THEN 1 ELSE 0 END) OVER (
        PARTITION BY game_id, period ORDER BY seq
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS home_team_fouls_q,
    COALESCE(SUM(CASE WHEN event_type='foul' AND foul_team = away_team
                 THEN 1 ELSE 0 END) OVER (
        PARTITION BY game_id, period ORDER BY seq
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS away_team_fouls_q,

    home_cum_fouls,
    away_cum_fouls,
    home_timeouts_used,
    away_timeouts_used,

    possessing_team, team_scored, outcome,
    points, shot_value, shot_distance, shot_type, play_type, scorer_player_id,
    foul_team, foul_player_id, foul_type,
    sub_team, player_in_id, player_out_id,
    timeout_team, timeout_type

FROM with_game
ORDER BY game_id, seq
"""


def _build_possession_feed_for_games(
    game_ids: list[str],
    conn: duckdb.DuckDBPyConnection,
    dry_run: bool = False,
) -> int:
    """
    Build and INSERT possession_feed rows for the given game_ids.
    Skips any game already present in possession_feed.
    Returns count of rows inserted.
    """
    new_ids = []
    for gid in game_ids:
        existing = conn.execute(
            "SELECT COUNT(*) FROM main.possession_feed WHERE game_id = ?", [gid]
        ).fetchone()[0]
        if existing == 0:
            new_ids.append(gid)
        else:
            logger.debug("game %s: already in possession_feed (%d rows) — skipping", gid, existing)

    if not new_ids:
        logger.info("possession_feed: all games already present — skipping")
        return 0

    logger.info("Building possession_feed for %d games: %s…", len(new_ids), new_ids[:3])

    placeholders = ", ".join("?" * len(new_ids))
    sql = _POSSESSION_FEED_SQL.replace("{placeholders}", placeholders)

    if dry_run:
        logger.info("DRY RUN: would insert possession_feed rows for %d games", len(new_ids))
        return 0

    conn.execute(sql, new_ids * 4)  # 4 subqueries each need the list

    inserted = conn.execute(
        f"SELECT COUNT(*) FROM main.possession_feed WHERE game_id IN ({placeholders})",
        new_ids,
    ).fetchone()[0]
    logger.info("possession_feed: inserted rows for %d games (%d total rows)", len(new_ids), inserted)
    return inserted


# ---------------------------------------------------------------------------
# PBP fetch for Group 1
# ---------------------------------------------------------------------------

def _fetch_pbp(game_id: str) -> pd.DataFrame | None:
    """Fetch PlayByPlayV3 from nba_api (uses local PBP cache if available)."""
    from pathlib import Path
    cache_path = Path("data/raw/pbp_cache") / f"{game_id}.parquet"
    if cache_path.exists():
        logger.debug("game %s: loading PBP from cache", game_id)
        return pd.read_parquet(cache_path)

    logger.info("game %s: fetching PBP from nba_api", game_id)
    try:
        from nba_api.stats.endpoints import playbyplayv3
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, end_period=10)
        df = pbp.get_data_frames()[0]
        if df.empty:
            logger.warning("game %s: PBP returned empty DataFrame", game_id)
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df
    except Exception:
        logger.exception("game %s: PBP fetch failed", game_id)
        return None


# ---------------------------------------------------------------------------
# possession_flat schema alignment
# ---------------------------------------------------------------------------

def _align_to_schema(
    df: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    """Align DataFrame to target column order, filling missing columns with None."""
    aligned = pd.DataFrame(index=df.index)
    for col in target_cols:
        aligned[col] = df[col] if col in df.columns else None
    return aligned


# ---------------------------------------------------------------------------
# CTAS write
# ---------------------------------------------------------------------------

def _fetch_wall_clock_ts(
    game_ids: list[str],
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Fetch (game_id, possession_id, wall_clock_ts) for the given games.
    Uses MAX(wall_clock_ts) per (game_id, possession_id) to handle any
    pre-existing duplicate rows in possession_flat.
    """
    placeholders = ", ".join("?" * len(game_ids))
    return conn.execute(
        f"""
        SELECT game_id, possession_id, MAX(wall_clock_ts) AS wall_clock_ts
        FROM features.possession_flat
        WHERE game_id IN ({placeholders})
          AND wall_clock_ts IS NOT NULL
        GROUP BY game_id, possession_id
        """,
        game_ids,
    ).df()


def _write_corrections(
    corrections: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """
    Atomically replace corrected games in possession_flat via CTAS.

    Wall_clock_ts values are pre-fetched from the existing table before the CTAS
    so that the CTAS avoids a JOIN (which would multiply rows when the original
    possession_flat has duplicate possession_ids, as some post-game pipeline
    runs inserted games twice).
    """
    game_ids = corrections["game_id"].unique().tolist()

    # Pre-fetch wall_clock_ts so we can merge in Python (avoids JOIN fan-out)
    wcts_df = _fetch_wall_clock_ts(game_ids, conn)
    if not wcts_df.empty:
        corrections = corrections.merge(
            wcts_df.rename(columns={"wall_clock_ts": "_wcts"}),
            on=["game_id", "possession_id"],
            how="left",
        )
        corrections["wall_clock_ts"] = corrections["_wcts"]
        corrections = corrections.drop(columns=["_wcts"])

    target_cols = [r[0] for r in conn.execute("DESCRIBE features.possession_flat").fetchall()]
    aligned = _align_to_schema(corrections, target_cols)

    conn.register("_corrections", aligned)
    conn.execute("DROP TABLE IF EXISTS features.pf_backfill_staging")
    conn.execute("""
        CREATE TABLE features.pf_backfill_staging AS SELECT * FROM _corrections
    """)
    conn.unregister("_corrections")

    n_games = corrections["game_id"].nunique()
    logger.info(
        "Rebuilding possession_flat via CTAS — replacing %d games (%d rows)",
        n_games, len(corrections),
    )

    conn.execute("""
        CREATE OR REPLACE TABLE features.possession_flat AS
        SELECT * FROM features.pf_backfill_staging

        UNION ALL

        SELECT pf.*
        FROM features.possession_flat pf
        WHERE pf.game_id NOT IN (
            SELECT DISTINCT game_id FROM features.pf_backfill_staging
        )
    """)

    conn.execute("DROP TABLE IF EXISTS features.pf_backfill_staging")
    logger.info("CTAS complete — possession_flat rebuilt")


# ---------------------------------------------------------------------------
# Group 2 fix: re-run feature pipeline for existing data
# ---------------------------------------------------------------------------

def fix_group2(
    game_ids: list[str],
    conn: duckdb.DuckDBPyConnection,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Fix 29 games with NULL momentum features.
    Raw data + possession_feed already exist in MotherDuck; just re-run the pipeline.
    """
    logger.info("Group 2: fixing %d games", len(game_ids))
    tier_map = _fetch_player_tier_map(conn)
    all_frames: list[pd.DataFrame] = []

    for i, game_id in enumerate(game_ids, 1):
        logger.info("Group 2 [%d/%d] game %s", i, len(game_ids), game_id)
        try:
            events       = _fetch_raw_events(game_id, conn)
            lineup_rat   = _fetch_asof_lineup_ratings(game_id, conn)
            player_rat   = _fetch_asof_player_ratings(game_id, conn)
            games_df     = _fetch_dim_games_context(game_id, conn)

            corrected = _build_corrected_rows(
                game_id, events, conn, games_df, lineup_rat, player_rat, tier_map
            )
            all_frames.append(corrected)
            logger.info("  → %d rows built", len(corrected))
        except Exception:
            logger.exception("  game %s: failed — skipping", game_id)

    if not all_frames:
        logger.error("Group 2: no corrected rows produced")
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Group 1 fix: fetch PBP, insert raw, build possession_feed, then features
# ---------------------------------------------------------------------------

def fix_group1(
    game_ids: list[str],
    conn: duckdb.DuckDBPyConnection,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Fix 207 games with NULL event_id (absent from raw tables + possession_feed).
    Fetches PBP, inserts raw events, builds possession_feed, then runs feature pipeline.
    """
    logger.info("Group 1: fixing %d games", len(game_ids))
    tier_map = _fetch_player_tier_map(conn)

    # Fetch dim_games info for all game_ids at once
    placeholders = ", ".join("?" * len(game_ids))
    dim_rows = conn.execute(
        f"SELECT game_id, home_team, away_team FROM dim_games WHERE game_id IN ({placeholders})",
        game_ids,
    ).df()
    dim_lookup: dict[str, tuple[str, str]] = {
        r["game_id"]: (r["home_team"], r["away_team"])
        for _, r in dim_rows.iterrows()
    }

    # Phase 1: Fetch PBP and insert raw events (with rate limiting)
    parsed: dict[str, dict[str, pd.DataFrame]] = {}
    for i, game_id in enumerate(game_ids, 1):
        teams = dim_lookup.get(game_id)
        if not teams:
            logger.warning("game %s not found in dim_games — skipping", game_id)
            continue

        home_tri, away_tri = teams
        logger.info("Group 1 [%d/%d] game %s (%s vs %s)", i, len(game_ids), game_id, home_tri, away_tri)

        pbp_df = _fetch_pbp(game_id)
        if pbp_df is None:
            continue

        try:
            events = parse_game_to_events(game_id, pbp_df, home_tri, away_tri)
        except Exception:
            logger.exception("  game %s: parse_game_to_events failed — skipping", game_id)
            continue

        if events["possessions"].empty:
            logger.warning("  game %s: parsed 0 possessions — skipping", game_id)
            continue

        _insert_raw_events(game_id, events, conn, dry_run=dry_run)
        parsed[game_id] = events

        time.sleep(_NBA_API_SLEEP)

    if not parsed:
        logger.error("Group 1: no games parsed successfully")
        return pd.DataFrame()

    # Phase 2: Build possession_feed for newly-inserted games
    _build_possession_feed_for_games(list(parsed.keys()), conn, dry_run=dry_run)

    if dry_run:
        logger.info("DRY RUN: skipping feature pipeline and CTAS")
        return pd.DataFrame()

    # Phase 3: Run feature pipeline (same as Group 2 from here)
    all_frames: list[pd.DataFrame] = []
    for i, (game_id, events) in enumerate(parsed.items(), 1):
        logger.info("Group 1 features [%d/%d] game %s", i, len(parsed), game_id)
        try:
            lineup_rat = _fetch_asof_lineup_ratings(game_id, conn)
            player_rat = _fetch_asof_player_ratings(game_id, conn)
            games_df   = _fetch_dim_games_context(game_id, conn)

            corrected = _build_corrected_rows(
                game_id, events, conn, games_df, lineup_rat, player_rat, tier_map
            )
            all_frames.append(corrected)
            logger.info("  → %d rows built", len(corrected))
        except Exception:
            logger.exception("  game %s: feature build failed — skipping", game_id)

    if not all_frames:
        logger.error("Group 1: no corrected feature rows produced")
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(conn: duckdb.DuckDBPyConnection) -> None:
    """Print null counts for Group 1 and Group 2 indicators."""
    g1_nulls = conn.execute(
        "SELECT COUNT(*) FROM features.possession_flat WHERE event_id IS NULL"
    ).fetchone()[0]
    g2_nulls = conn.execute(
        "SELECT COUNT(*) FROM features.possession_flat WHERE home_points_last_5_poss IS NULL"
    ).fetchone()[0]
    total = conn.execute(
        "SELECT COUNT(*) FROM features.possession_flat"
    ).fetchone()[0]

    print(f"\n=== possession_flat verification ===")
    print(f"  Total rows:               {total:,}")
    print(f"  Group 1 nulls (event_id): {g1_nulls:,}  {'✓ FIXED' if g1_nulls == 0 else '✗ STILL NULL'}")
    print(f"  Group 2 nulls (momentum): {g2_nulls:,}  {'✓ FIXED' if g2_nulls == 0 else '✗ STILL NULL'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--group", type=int, choices=[1, 2], help="Fix only this group (1 or 2)")
    group.add_argument("--game", help="Fix a single game_id (auto-detects group)")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing to MotherDuck")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    load_dotenv()
    args = _parse_args()

    conn = _md_connect()
    try:
        all_corrections: list[pd.DataFrame] = []

        if args.game:
            # Single-game mode: auto-detect which group
            g1_ids = _get_group1_game_ids(conn)
            g2_ids = _get_group2_game_ids(conn)
            if args.game in g1_ids:
                logger.info("game %s is Group 1 — fixing", args.game)
                df = fix_group1([args.game], conn, dry_run=args.dry_run)
            elif args.game in g2_ids:
                logger.info("game %s is Group 2 — fixing", args.game)
                df = fix_group2([args.game], conn, dry_run=args.dry_run)
            else:
                logger.error("game %s not found in Group 1 or Group 2 null sets", args.game)
                raise SystemExit(1)
            if not df.empty:
                all_corrections.append(df)

        else:
            run_g2 = args.group in (None, 2)
            run_g1 = args.group in (None, 1)

            if run_g2:
                g2_ids = _get_group2_game_ids(conn)
                if g2_ids:
                    df2 = fix_group2(g2_ids, conn, dry_run=args.dry_run)
                    if not df2.empty:
                        all_corrections.append(df2)
                else:
                    logger.info("Group 2: no games to fix")

            if run_g1:
                g1_ids = _get_group1_game_ids(conn)
                if g1_ids:
                    df1 = fix_group1(g1_ids, conn, dry_run=args.dry_run)
                    if not df1.empty:
                        all_corrections.append(df1)
                else:
                    logger.info("Group 1: no games to fix")

        if args.dry_run:
            logger.info("DRY RUN complete — nothing written")
            verify(conn)
            raise SystemExit(0)

        if not all_corrections:
            logger.info("No corrections to write — nothing to do")
            verify(conn)
            raise SystemExit(0)

        corrections = pd.concat(all_corrections, ignore_index=True)
        logger.info(
            "Writing %d corrected rows (%d games) to MotherDuck…",
            len(corrections), corrections["game_id"].nunique(),
        )
        _write_corrections(corrections, conn)
        verify(conn)

    finally:
        try:
            conn.close()
        except Exception:
            pass
