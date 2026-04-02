"""
DuckDB loader — migrates all parquet data into a structured DuckDB database.

Tables created (in dependency order):
  Tier 0 — Dimensions:   dim_teams, dim_players, dim_games
  Tier 1 — Raw events:   raw_possessions, raw_fouls, raw_substitutions, raw_timeouts
  Tier 2 — Kalshi:       kalshi_market_map, kalshi_settled, kalshi_ticks
  Tier 3 — Derived:      possession_feed, player_game_stats

All functions are idempotent (CREATE TABLE IF NOT EXISTS + INSERT OR IGNORE).
Kalshi tick loading is incremental — only appends ticks newer than what's already stored.

Usage:
    python data/ingestion/duckdb_loader.py                          # local only
    python data/ingestion/duckdb_loader.py --sync-motherduck        # push new local rows to MotherDuck
    python data/ingestion/duckdb_loader.py --pull-motherduck        # merge new cloud rows into local
    python data/ingestion/duckdb_loader.py --pull-motherduck-full   # DESTRUCTIVE: replace local with cloud
"""

import argparse
import logging
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.features.star_players import STAR_PLAYERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR        = Path("data/raw")
FEATURE_DIR    = Path("data/feature_store")
DB_PATH        = Path("kalshi_trading.duckdb")

POSSESSIONS_PATH    = RAW_DIR / "possessions_202526.parquet"
GAMES_PATH          = RAW_DIR / "games_202526.parquet"
FOULS_PATH          = RAW_DIR / "foul_events_202526.parquet"
SUBS_PATH           = RAW_DIR / "substitution_events_202526.parquet"
TIMEOUTS_PATH       = RAW_DIR / "timeout_events_202526.parquet"
SETTLED_PATH        = RAW_DIR / "kalshi_markets_settled.parquet"
PLAYER_RATINGS_PATH = FEATURE_DIR / "player_ratings.parquet"
TICKS_DIR           = RAW_DIR / "kalshi_ticks"

TICKER_RE = re.compile(
    r"KXNBASPREAD-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})-([A-Z]+)(\d+)"
)
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# ---------------------------------------------------------------------------
# Tier 0 — Dimension tables
# ---------------------------------------------------------------------------

NBA_TEAMS = [
    (1, "ATL", "Atlanta Hawks",           "East", "Southeast"),
    (2, "BOS", "Boston Celtics",          "East", "Atlantic"),
    (3, "BKN", "Brooklyn Nets",           "East", "Atlantic"),
    (4, "CHA", "Charlotte Hornets",       "East", "Southeast"),
    (5, "CHI", "Chicago Bulls",           "East", "Central"),
    (6, "CLE", "Cleveland Cavaliers",     "East", "Central"),
    (7, "DAL", "Dallas Mavericks",        "West", "Southwest"),
    (8, "DEN", "Denver Nuggets",          "West", "Northwest"),
    (9, "DET", "Detroit Pistons",         "East", "Central"),
    (10, "GSW", "Golden State Warriors",  "West", "Pacific"),
    (11, "HOU", "Houston Rockets",        "West", "Southwest"),
    (12, "IND", "Indiana Pacers",         "East", "Central"),
    (13, "LAC", "LA Clippers",            "West", "Pacific"),
    (14, "LAL", "Los Angeles Lakers",     "West", "Pacific"),
    (15, "MEM", "Memphis Grizzlies",      "West", "Southwest"),
    (16, "MIA", "Miami Heat",             "East", "Southeast"),
    (17, "MIL", "Milwaukee Bucks",        "East", "Central"),
    (18, "MIN", "Minnesota Timberwolves", "West", "Northwest"),
    (19, "NOP", "New Orleans Pelicans",   "West", "Southwest"),
    (20, "NYK", "New York Knicks",        "East", "Atlantic"),
    (21, "OKC", "Oklahoma City Thunder",  "West", "Northwest"),
    (22, "ORL", "Orlando Magic",          "East", "Southeast"),
    (23, "PHI", "Philadelphia 76ers",     "East", "Atlantic"),
    (24, "PHX", "Phoenix Suns",           "West", "Pacific"),
    (25, "POR", "Portland Trail Blazers", "West", "Northwest"),
    (26, "SAC", "Sacramento Kings",       "West", "Pacific"),
    (27, "SAS", "San Antonio Spurs",      "West", "Southwest"),
    (28, "TOR", "Toronto Raptors",        "East", "Atlantic"),
    (29, "UTA", "Utah Jazz",              "West", "Northwest"),
    (30, "WAS", "Washington Wizards",     "East", "Southeast"),
]


def load_dim_teams(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dim_teams (
            team_id    SMALLINT    PRIMARY KEY,
            tricode    VARCHAR(3)  NOT NULL UNIQUE,
            full_name  VARCHAR(64) NOT NULL,
            conference VARCHAR(4)  NOT NULL,
            division   VARCHAR(16) NOT NULL
        )
    """)
    conn.executemany(
        "INSERT OR IGNORE INTO dim_teams VALUES (?, ?, ?, ?, ?)",
        NBA_TEAMS,
    )
    count = conn.execute("SELECT COUNT(*) FROM dim_teams").fetchone()[0]
    logger.info("dim_teams: %d rows", count)


def load_dim_players(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dim_players (
            player_id   BIGINT      PRIMARY KEY,
            player_name VARCHAR(64) NOT NULL,
            star_tier   SMALLINT    -- 1=Tier1, 2=Tier2, NULL=not a star
        )
    """)

    # Build player list from all event tables — deduplicated by player_id
    players: dict[int, str] = {}

    for path, id_col, name_col in [
        (POSSESSIONS_PATH, "player_id",      "player_name"),
        (FOULS_PATH,       "player_id",      "player_name"),
        (SUBS_PATH,        "player_out_id",  "player_out_name"),
        (SUBS_PATH,        "player_in_id",   "player_in_name"),
    ]:
        df = pq.read_table(path, columns=[id_col, name_col]).to_pandas()
        df = df.dropna(subset=[id_col, name_col])
        df = df[df[id_col].astype(str).str.strip() != "0"]
        for pid, name in zip(df[id_col].astype(int), df[name_col]):
            if pid > 0 and name:
                players[pid] = name

    rows = [
        (pid, name, STAR_PLAYERS.get(pid))
        for pid, name in players.items()
    ]
    conn.executemany("INSERT OR IGNORE INTO dim_players VALUES (?, ?, ?)", rows)
    count = conn.execute("SELECT COUNT(*) FROM dim_players").fetchone()[0]
    logger.info("dim_players: %d rows", count)


def load_dim_games(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dim_games (
            game_id           VARCHAR(10) PRIMARY KEY,
            game_date         DATE        NOT NULL,
            season            VARCHAR(7)  NOT NULL,
            home_team         VARCHAR(3)  NOT NULL,
            away_team         VARCHAR(3)  NOT NULL,
            home_final_score  SMALLINT,
            away_final_score  SMALLINT,
            home_back_to_back BOOLEAN     NOT NULL DEFAULT FALSE,
            away_back_to_back BOOLEAN     NOT NULL DEFAULT FALSE
        )
    """)

    # Load 2025-26 games from parquet (covers Oct 2025 – Mar 5 2026)
    games_2526 = pq.read_table(GAMES_PATH).to_pandas()
    for _, row in games_2526.iterrows():
        conn.execute("""
            INSERT OR IGNORE INTO dim_games
            VALUES (?, ?, '2025-26', ?, ?, ?, ?, FALSE, FALSE)
        """, [
            row["game_id"],
            row["game_date"],
            row["home_team"],
            row["away_team"],
            row.get("home_final_score"),
            row.get("away_final_score"),
        ])

    # Fetch both seasons from nba_api to fill gaps and pick up recent games
    _load_games_from_nba_api(conn, "2025-26", "0022500")
    _load_games_from_nba_api(conn, "2024-25", "0022400")

    count = conn.execute("SELECT COUNT(*) FROM dim_games").fetchone()[0]
    logger.info("dim_games: %d rows", count)


def _load_games_from_nba_api(
    conn: duckdb.DuckDBPyConnection,
    season: str,
    game_id_prefix: str,
) -> None:
    """Fetch game metadata for a season from nba_api and upsert into dim_games."""
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        import time

        logger.info("Fetching %s games from nba_api...", season)
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
        )
        time.sleep(0.6)
        df = finder.get_data_frames()[0]

        home_df = df[df["MATCHUP"].str.contains(r" vs\. ")].copy()
        away_df = df[df["MATCHUP"].str.contains(r" @ ")].copy()

        home_map = home_df.set_index("GAME_ID")[["GAME_DATE", "TEAM_ABBREVIATION", "PTS"]].rename(
            columns={"TEAM_ABBREVIATION": "home_team", "PTS": "home_pts", "GAME_DATE": "game_date"}
        )
        away_map = away_df.set_index("GAME_ID")[["TEAM_ABBREVIATION", "PTS"]].rename(
            columns={"TEAM_ABBREVIATION": "away_team", "PTS": "away_pts"}
        )
        merged = home_map.join(away_map, how="inner")

        season_label = "2025-26" if game_id_prefix == "0022500" else "2024-25"
        rows_loaded = 0
        for game_id, row in merged.iterrows():
            gid = str(game_id)
            if not gid.startswith(game_id_prefix):
                continue
            try:
                game_date = pd.to_datetime(row["game_date"]).date()
            except Exception:
                continue
            conn.execute("""
                INSERT OR REPLACE INTO dim_games
                VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, FALSE)
            """, [
                gid, game_date, season_label,
                row["home_team"], row["away_team"],
                int(row["home_pts"]) if pd.notna(row["home_pts"]) else None,
                int(row["away_pts"]) if pd.notna(row["away_pts"]) else None,
            ])
            rows_loaded += 1

        logger.info("Upserted %d %s games from nba_api", rows_loaded, season)

    except Exception as exc:
        logger.warning("nba_api fetch failed for %s (%s)", season, exc)


# ---------------------------------------------------------------------------
# Tier 1 — Raw event tables
# ---------------------------------------------------------------------------

def load_raw_events(conn: duckdb.DuckDBPyConnection) -> None:
    _load_raw_possessions(conn)
    _load_raw_fouls(conn)
    _load_raw_substitutions(conn)
    _load_raw_timeouts(conn)


def _load_raw_possessions(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_possessions (
            game_id         VARCHAR(10) NOT NULL,
            possession_id   INTEGER     NOT NULL,
            period          SMALLINT    NOT NULL,
            game_clock_secs FLOAT       NOT NULL,
            possessing_team VARCHAR(3),
            team_scored     VARCHAR(3),
            outcome         VARCHAR(32),
            home_score      SMALLINT    NOT NULL DEFAULT 0,
            away_score      SMALLINT    NOT NULL DEFAULT 0,
            points          SMALLINT    NOT NULL DEFAULT 0,
            shot_value      SMALLINT    NOT NULL DEFAULT 0,
            shot_x          SMALLINT,
            shot_y          SMALLINT,
            shot_distance   SMALLINT,
            shot_type       VARCHAR(32),
            play_type       VARCHAR(32),
            player_id       BIGINT,
            home_lineup_id  VARCHAR(12) NOT NULL,
            away_lineup_id  VARCHAR(12) NOT NULL,
            PRIMARY KEY (game_id, possession_id)
        )
    """)
    conn.execute(f"""
        INSERT OR IGNORE INTO raw_possessions
        SELECT
            game_id, possession_id, period, game_clock_secs,
            possessing_team, team_scored, outcome,
            COALESCE(home_score, 0), COALESCE(away_score, 0),
            COALESCE(points, 0), COALESCE(shot_value, 0),
            shot_x, shot_y, shot_distance, shot_type, play_type,
            CASE WHEN player_id = 0 THEN NULL ELSE player_id END,
            home_lineup_id, away_lineup_id
        FROM read_parquet('{POSSESSIONS_PATH}')
    """)
    count = conn.execute("SELECT COUNT(*) FROM raw_possessions").fetchone()[0]
    logger.info("raw_possessions: %d rows", count)


def _load_raw_fouls(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_fouls (
            game_id         VARCHAR(10) NOT NULL,
            action_number   INTEGER     NOT NULL,
            period          SMALLINT    NOT NULL,
            game_clock_secs FLOAT       NOT NULL,
            team_tricode    VARCHAR(3)  NOT NULL,
            player_id       BIGINT      NOT NULL,
            foul_type       VARCHAR(64),
            description     VARCHAR(256),
            PRIMARY KEY (game_id, action_number)
        )
    """)
    conn.execute(f"""
        INSERT OR IGNORE INTO raw_fouls
        SELECT game_id, action_number, period, game_clock_secs,
               team_tricode, COALESCE(player_id, 0), foul_type, description
        FROM read_parquet('{FOULS_PATH}')
    """)
    count = conn.execute("SELECT COUNT(*) FROM raw_fouls").fetchone()[0]
    logger.info("raw_fouls: %d rows", count)


def _load_raw_substitutions(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_substitutions (
            game_id         VARCHAR(10) NOT NULL,
            action_number   INTEGER     NOT NULL,
            period          SMALLINT    NOT NULL,
            game_clock_secs FLOAT       NOT NULL,
            team_tricode    VARCHAR(3)  NOT NULL,
            player_out_id   BIGINT      NOT NULL,
            player_in_id    BIGINT      NOT NULL,
            home_lineup_id  VARCHAR(12) NOT NULL,
            away_lineup_id  VARCHAR(12) NOT NULL,
            PRIMARY KEY (game_id, action_number)
        )
    """)
    conn.execute(f"""
        INSERT OR IGNORE INTO raw_substitutions
        SELECT game_id, action_number, period, game_clock_secs,
               team_tricode, COALESCE(player_out_id, 0), COALESCE(player_in_id, 0),
               COALESCE(home_lineup_id, ''), COALESCE(away_lineup_id, '')
        FROM read_parquet('{SUBS_PATH}')
    """)
    count = conn.execute("SELECT COUNT(*) FROM raw_substitutions").fetchone()[0]
    logger.info("raw_substitutions: %d rows", count)


def _load_raw_timeouts(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_timeouts (
            game_id                 VARCHAR(10) NOT NULL,
            action_number           INTEGER     NOT NULL,
            period                  SMALLINT    NOT NULL,
            game_clock_secs         FLOAT       NOT NULL,
            team_tricode            VARCHAR(3)  NOT NULL,
            timeout_type            VARCHAR(32),
            full_timeouts_remaining SMALLINT,
            PRIMARY KEY (game_id, action_number)
        )
    """)
    conn.execute(f"""
        INSERT OR IGNORE INTO raw_timeouts
        SELECT game_id, action_number, period, game_clock_secs,
               team_tricode, timeout_type, full_timeouts_remaining
        FROM read_parquet('{TIMEOUTS_PATH}')
    """)
    count = conn.execute("SELECT COUNT(*) FROM raw_timeouts").fetchone()[0]
    logger.info("raw_timeouts: %d rows", count)


# ---------------------------------------------------------------------------
# Tier 2 — Kalshi tables
# ---------------------------------------------------------------------------

def _parse_ticker(ticker: str) -> dict | None:
    """Parse a Kalshi market ticker into its component parts."""
    m = TICKER_RE.search(ticker)
    if not m:
        return None
    yy, mon, dd, away3, home3, spread_team, spread_pts = m.groups()
    year = 2000 + int(yy)
    month = MONTH_MAP.get(mon.upper())
    if not month:
        return None
    return {
        "game_date":    date(year, month, int(dd)),
        "away_team":    away3,
        "home_team":    home3,
        "spread_team":  spread_team,
        "spread_points": int(spread_pts),
    }


def load_kalshi(conn: duckdb.DuckDBPyConnection) -> None:
    _load_kalshi_settled(conn)
    _load_kalshi_market_map(conn)
    _load_kalshi_ticks(conn)


def _load_kalshi_settled(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_settled (
            market_ticker VARCHAR(64)  PRIMARY KEY,
            event_ticker  VARCHAR(48)  NOT NULL,
            game_date     DATE,
            away_team     VARCHAR(3),
            home_team     VARCHAR(3),
            spread_team   VARCHAR(3),
            spread_points SMALLINT,
            open_time     TIMESTAMPTZ,
            close_time    TIMESTAMPTZ,
            result        VARCHAR(4),
            last_price    SMALLINT,
            yes_bid       SMALLINT,
            yes_ask       SMALLINT,
            volume        BIGINT
        )
    """)
    existing = conn.execute("SELECT COUNT(*) FROM kalshi_settled").fetchone()[0]
    if existing == 0 and SETTLED_PATH.exists():
        conn.execute(f"""
            INSERT OR IGNORE INTO kalshi_settled
            SELECT market_ticker, event_ticker, game_date, away_team, home_team,
                   spread_team, spread_points, open_time, close_time,
                   result, last_price, yes_bid, yes_ask, volume
            FROM read_parquet('{SETTLED_PATH}')
        """)
    count = conn.execute("SELECT COUNT(*) FROM kalshi_settled").fetchone()[0]
    logger.info("kalshi_settled: %d rows", count)


def _load_kalshi_market_map(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_market_map (
            market_ticker VARCHAR(64) PRIMARY KEY,
            event_ticker  VARCHAR(48) NOT NULL,
            game_id       VARCHAR(10),
            game_date     DATE        NOT NULL,
            away_team     VARCHAR(3)  NOT NULL,
            home_team     VARCHAR(3)  NOT NULL,
            spread_team   VARCHAR(3)  NOT NULL,
            spread_points SMALLINT    NOT NULL
        )
    """)

    # Collect all unique tickers from settled + tick files
    tickers: set[str] = set()

    if SETTLED_PATH.exists():
        for t in pq.read_table(SETTLED_PATH, columns=["market_ticker"]).to_pandas()["market_ticker"]:
            tickers.add(t)

    for date_dir in TICKS_DIR.glob("*/"):
        for f in date_dir.glob("*.parquet"):
            tickers.add(f.stem)

    rows_inserted = 0
    for ticker in tickers:
        # Skip if already mapped
        existing = conn.execute(
            "SELECT 1 FROM kalshi_market_map WHERE market_ticker=?", [ticker]
        ).fetchone()
        if existing:
            continue

        parsed = _parse_ticker(ticker)
        if not parsed:
            continue

        # Resolve game_id from dim_games
        result = conn.execute("""
            SELECT game_id FROM dim_games
            WHERE game_date = ?
              AND home_team = ?
              AND away_team = ?
            LIMIT 1
        """, [parsed["game_date"], parsed["home_team"], parsed["away_team"]]).fetchone()

        game_id = result[0] if result else None
        event_ticker = ticker.rsplit("-", 1)[0]

        conn.execute("""
            INSERT OR IGNORE INTO kalshi_market_map VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            ticker, event_ticker, game_id,
            parsed["game_date"], parsed["away_team"], parsed["home_team"],
            parsed["spread_team"], parsed["spread_points"],
        ])
        rows_inserted += 1

    count = conn.execute("SELECT COUNT(*) FROM kalshi_market_map").fetchone()[0]
    unresolved = conn.execute("SELECT COUNT(*) FROM kalshi_market_map WHERE game_id IS NULL").fetchone()[0]
    logger.info("kalshi_market_map: %d rows (%d unresolved game_ids)", count, unresolved)


def _load_kalshi_ticks(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_ticks (
            market_ticker VARCHAR(64) NOT NULL,
            ts            TIMESTAMPTZ NOT NULL,
            game_id       VARCHAR(10),
            yes_bid       SMALLINT    NOT NULL,
            yes_ask       SMALLINT    NOT NULL,
            yes_last      SMALLINT,
            volume        BIGINT      NOT NULL DEFAULT 0,
            open_interest BIGINT      NOT NULL DEFAULT 0,
            PRIMARY KEY (market_ticker, ts)
        )
    """)

    # Build game_id lookup from market_map
    game_id_map: dict[str, str | None] = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT market_ticker, game_id FROM kalshi_market_map"
        ).fetchall()
    }

    total_new = 0
    for date_dir in sorted(TICKS_DIR.glob("*/")):
        for tick_file in sorted(date_dir.glob("*.parquet")):
            ticker = tick_file.stem

            # Incremental: only load ticks newer than what's already stored
            max_ts_row = conn.execute(
                "SELECT MAX(ts) FROM kalshi_ticks WHERE market_ticker=?", [ticker]
            ).fetchone()
            max_ts = max_ts_row[0] if max_ts_row else None

            df = pq.read_table(tick_file).to_pandas()
            if df.empty:
                continue

            if max_ts is not None:
                df = df[df["ts"] > pd.Timestamp(max_ts)]

            if df.empty:
                continue

            # Only load valid live ticks (yes_bid > 0)
            df = df[df["yes_bid"] > 0].copy()
            if df.empty:
                continue

            gid = game_id_map.get(ticker)
            df["game_id"] = gid
            df["market_ticker"] = ticker

            conn.execute("""
                INSERT OR IGNORE INTO kalshi_ticks
                SELECT market_ticker, ts, game_id, yes_bid, yes_ask, yes_last,
                       COALESCE(volume, 0), COALESCE(open_interest, 0)
                FROM df
            """)
            total_new += len(df)

    count = conn.execute("SELECT COUNT(*) FROM kalshi_ticks").fetchone()[0]
    logger.info("kalshi_ticks: %d total rows (%d new this run)", count, total_new)


# ---------------------------------------------------------------------------
# Tier 3 — Derived tables
# ---------------------------------------------------------------------------

def build_possession_feed(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Unified event stream: possessions + fouls + substitutions + timeouts merged,
    one row per event with forward-filled game state and point-in-time running
    team stats (no lookahead bias).
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS possession_feed (
            game_id             VARCHAR(10)  NOT NULL,
            event_id            BIGINT       NOT NULL,
            event_type          VARCHAR(16)  NOT NULL,
            possession_id       INTEGER,
            period              SMALLINT     NOT NULL,
            game_clock_secs     FLOAT        NOT NULL,

            -- Forward-filled game state
            home_score          SMALLINT     NOT NULL DEFAULT 0,
            away_score          SMALLINT     NOT NULL DEFAULT 0,
            score_diff          SMALLINT     NOT NULL DEFAULT 0,
            home_lineup_id      VARCHAR(12),
            away_lineup_id      VARCHAR(12),

            -- Running team stats (BEFORE this event — no lookahead)
            home_cum_points     SMALLINT     NOT NULL DEFAULT 0,
            away_cum_points     SMALLINT     NOT NULL DEFAULT 0,
            home_team_fouls_q   SMALLINT     NOT NULL DEFAULT 0,
            away_team_fouls_q   SMALLINT     NOT NULL DEFAULT 0,
            home_cum_fouls      SMALLINT     NOT NULL DEFAULT 0,
            away_cum_fouls      SMALLINT     NOT NULL DEFAULT 0,
            home_timeouts_used  SMALLINT     NOT NULL DEFAULT 0,
            away_timeouts_used  SMALLINT     NOT NULL DEFAULT 0,

            -- Possession-specific (NULL for non-possession events)
            possessing_team     VARCHAR(3),
            team_scored         VARCHAR(3),
            outcome             VARCHAR(32),
            points              SMALLINT,
            shot_value          SMALLINT,
            shot_distance       SMALLINT,
            shot_type           VARCHAR(32),
            play_type           VARCHAR(32),
            scorer_player_id    BIGINT,

            -- Foul-specific
            foul_team           VARCHAR(3),
            foul_player_id      BIGINT,
            foul_type           VARCHAR(64),

            -- Substitution-specific
            sub_team            VARCHAR(3),
            player_in_id        BIGINT,
            player_out_id       BIGINT,

            -- Timeout-specific
            timeout_team        VARCHAR(3),
            timeout_type        VARCHAR(32),

            PRIMARY KEY (game_id, event_id)
        )
    """)

    raw_poss_count  = conn.execute("SELECT COUNT(*) FROM raw_possessions").fetchone()[0]
    feed_poss_count = conn.execute("SELECT COUNT(*) FROM possession_feed WHERE event_type='possession'").fetchone()[0]
    if feed_poss_count == raw_poss_count and feed_poss_count > 0:
        logger.info("possession_feed up to date (%d possession rows) — skipping", feed_poss_count)
        return
    if feed_poss_count > 0:
        logger.info(
            "possession_feed stale (%d rows vs %d in raw_possessions) — rebuilding...",
            feed_poss_count, raw_poss_count,
        )
        conn.execute("DROP TABLE possession_feed")
        conn.execute("""
            CREATE TABLE possession_feed (
                game_id             VARCHAR(10)  NOT NULL,
                event_id            BIGINT       NOT NULL,
                event_type          VARCHAR(16)  NOT NULL,
                possession_id       INTEGER,
                period              SMALLINT     NOT NULL,
                game_clock_secs     FLOAT        NOT NULL,
                home_score          SMALLINT     NOT NULL DEFAULT 0,
                away_score          SMALLINT     NOT NULL DEFAULT 0,
                score_diff          SMALLINT     NOT NULL DEFAULT 0,
                home_lineup_id      VARCHAR(12),
                away_lineup_id      VARCHAR(12),
                home_cum_points     SMALLINT     NOT NULL DEFAULT 0,
                away_cum_points     SMALLINT     NOT NULL DEFAULT 0,
                home_team_fouls_q   SMALLINT     NOT NULL DEFAULT 0,
                away_team_fouls_q   SMALLINT     NOT NULL DEFAULT 0,
                home_cum_fouls      SMALLINT     NOT NULL DEFAULT 0,
                away_cum_fouls      SMALLINT     NOT NULL DEFAULT 0,
                home_timeouts_used  SMALLINT     NOT NULL DEFAULT 0,
                away_timeouts_used  SMALLINT     NOT NULL DEFAULT 0,
                possessing_team     VARCHAR(3),
                team_scored         VARCHAR(3),
                outcome             VARCHAR(32),
                points              SMALLINT,
                shot_value          SMALLINT,
                shot_distance       SMALLINT,
                shot_type           VARCHAR(32),
                play_type           VARCHAR(32),
                scorer_player_id    BIGINT,
                foul_team           VARCHAR(3),
                foul_player_id      BIGINT,
                foul_type           VARCHAR(64),
                sub_team            VARCHAR(3),
                player_in_id        BIGINT,
                player_out_id       BIGINT,
                timeout_team        VARCHAR(3),
                timeout_type        VARCHAR(32),
                PRIMARY KEY (game_id, event_id)
            )
        """)

    logger.info("Building possession_feed (this may take a few minutes)...")
    conn.execute("""
        INSERT INTO possession_feed
        WITH events AS (
            -- Possessions (priority 1)
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
            FROM raw_possessions

            UNION ALL

            -- Fouls (priority 2)
            SELECT
                game_id, period, game_clock_secs, 2,
                'foul', NULL,
                0, NULL, NULL, NULL, NULL,
                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                team_tricode, player_id, foul_type,
                NULL, NULL, NULL,
                NULL, NULL,
                team_tricode, NULL
            FROM raw_fouls

            UNION ALL

            -- Timeouts (priority 3)
            SELECT
                game_id, period, game_clock_secs, 3,
                'timeout', NULL,
                0, NULL, NULL, NULL, NULL,
                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                NULL, NULL, NULL,
                NULL, NULL, NULL,
                team_tricode, timeout_type,
                NULL, team_tricode
            FROM raw_timeouts

            UNION ALL

            -- Substitutions (priority 4)
            SELECT
                game_id, period, game_clock_secs, 4,
                'substitution', NULL,
                0, NULL, NULL, home_lineup_id, away_lineup_id,
                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                NULL, NULL, NULL,
                team_tricode, player_in_id, player_out_id,
                NULL, NULL,
                NULL, NULL
            FROM raw_substitutions
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
                -- Forward-fill scores and lineups from last known possession
                LAST_VALUE(home_score    IGNORE NULLS) OVER w AS ff_home_score,
                LAST_VALUE(away_score    IGNORE NULLS) OVER w AS ff_away_score,
                LAST_VALUE(home_lineup_id IGNORE NULLS) OVER w AS ff_home_lineup,
                LAST_VALUE(away_lineup_id IGNORE NULLS) OVER w AS ff_away_lineup,

                -- Running team points (BEFORE this event)
                COALESCE(SUM(CASE WHEN team_scored IS NOT NULL
                             THEN points_scored ELSE 0 END) OVER (
                    PARTITION BY game_id
                    ORDER BY seq
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS _home_cum_pts_raw,

                -- Home cumulative fouls (BEFORE this event)
                COALESCE(SUM(CASE WHEN event_type='foul' AND foul_team IN (
                    SELECT home_team FROM dim_games g WHERE g.game_id = ordered.game_id
                ) THEN 1 ELSE 0 END) OVER (
                    PARTITION BY game_id
                    ORDER BY seq
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS home_cum_fouls,

                -- Away cumulative fouls (BEFORE this event)
                COALESCE(SUM(CASE WHEN event_type='foul' AND foul_team NOT IN (
                    SELECT home_team FROM dim_games g WHERE g.game_id = ordered.game_id
                ) AND foul_team IS NOT NULL THEN 1 ELSE 0 END) OVER (
                    PARTITION BY game_id
                    ORDER BY seq
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS away_cum_fouls,

                -- Timeouts used (BEFORE this event)
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

            -- Running team points split by home/away
            COALESCE(SUM(CASE WHEN team_scored = home_team
                         THEN points_scored ELSE 0 END) OVER (
                PARTITION BY game_id ORDER BY seq
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS home_cum_points,
            COALESCE(SUM(CASE WHEN team_scored = away_team
                         THEN points_scored ELSE 0 END) OVER (
                PARTITION BY game_id ORDER BY seq
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) AS away_cum_points,

            -- Team fouls per quarter (resets each period)
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

            -- Event-specific sparse columns
            possessing_team, team_scored, outcome,
            points, shot_value, shot_distance, shot_type, play_type, scorer_player_id,
            foul_team, foul_player_id, foul_type,
            sub_team, player_in_id, player_out_id,
            timeout_team, timeout_type

        FROM with_game
        ORDER BY game_id, seq
    """)

    count = conn.execute("SELECT COUNT(*) FROM possession_feed").fetchone()[0]
    logger.info("possession_feed: %d rows", count)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pf_game ON possession_feed(game_id, period, game_clock_secs DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pf_poss ON possession_feed(game_id, possession_id)")


def build_player_game_stats(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Per-player cumulative in-game stats at each possession.
    One row per (game_id, possession_id, player_id) — state BEFORE this possession.
    Only populated for possessions where player_id IS NOT NULL (actual scorers).
    Includes pre-game season APM from player_ratings.parquet.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_game_stats (
            game_id          VARCHAR(10)  NOT NULL,
            possession_id    INTEGER      NOT NULL,
            player_id        BIGINT       NOT NULL,
            team_tricode     VARCHAR(3),

            -- All stats BEFORE this possession (no lookahead)
            cum_points       SMALLINT     NOT NULL DEFAULT 0,
            cum_fga          SMALLINT     NOT NULL DEFAULT 0,
            cum_fgm          SMALLINT     NOT NULL DEFAULT 0,
            cum_fg_pct       FLOAT,
            cum_3pa          SMALLINT     NOT NULL DEFAULT 0,
            cum_3pm          SMALLINT     NOT NULL DEFAULT 0,
            cum_3p_pct       FLOAT,
            cum_fta          SMALLINT     NOT NULL DEFAULT 0,
            cum_ftm          SMALLINT     NOT NULL DEFAULT 0,
            cum_ft_pct       FLOAT,
            cum_fouls        SMALLINT     NOT NULL DEFAULT 0,

            -- Pre-game season context (static per game)
            season_apm       FLOAT,
            apm_games_fitted INTEGER,

            PRIMARY KEY (game_id, possession_id, player_id)
        )
    """)

    existing = conn.execute("SELECT COUNT(*) FROM player_game_stats").fetchone()[0]
    if existing > 0:
        logger.info("player_game_stats: %d rows (already built — skipping)", existing)
        return

    logger.info("Building player_game_stats (this may take a few minutes)...")
    conn.execute(f"""
        INSERT INTO player_game_stats
        WITH scored_poss AS (
            -- One row per scoring possession (player_id IS NOT NULL)
            SELECT
                game_id, possession_id, player_id,
                possessing_team AS team_tricode,
                period, game_clock_secs,
                COALESCE(points, 0)      AS pts,
                CASE WHEN shot_value >= 1 AND outcome IN ('Made Shot', 'Free Throw')
                     THEN 1 ELSE 0 END  AS is_fgm,
                CASE WHEN shot_value >= 1 THEN 1 ELSE 0 END AS is_fga,
                CASE WHEN shot_value = 3 AND outcome IN ('Made Shot')
                     THEN 1 ELSE 0 END  AS is_3pm,
                CASE WHEN shot_value = 3 THEN 1 ELSE 0 END AS is_3pa,
                CASE WHEN shot_value = 1 AND outcome IN ('Free Throw', 'Made Shot')
                     THEN 1 ELSE 0 END  AS is_ftm,
                CASE WHEN shot_value = 1 THEN 1 ELSE 0 END AS is_fta
            FROM raw_possessions
            WHERE player_id IS NOT NULL
        ),
        player_fouls AS (
            SELECT
                f.game_id, f.player_id,
                f.period, f.game_clock_secs,
                COUNT(*) OVER (
                    PARTITION BY f.game_id, f.player_id
                    ORDER BY f.period ASC, f.game_clock_secs DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS cum_fouls_before
            FROM raw_fouls f
        ),
        player_season_apm AS (
            SELECT player_id, as_of_game_id, adjusted_plus_minus, games_fitted
            FROM read_parquet('{PLAYER_RATINGS_PATH}')
        ),
        with_running AS (
            SELECT
                sp.game_id, sp.possession_id, sp.player_id, sp.team_tricode,

                -- Cumulative stats BEFORE this possession
                COALESCE(SUM(sp.pts) OVER (
                    PARTITION BY sp.game_id, sp.player_id
                    ORDER BY sp.possession_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS cum_points,

                COALESCE(SUM(sp.is_fga) OVER (
                    PARTITION BY sp.game_id, sp.player_id
                    ORDER BY sp.possession_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS cum_fga,

                COALESCE(SUM(sp.is_fgm) OVER (
                    PARTITION BY sp.game_id, sp.player_id
                    ORDER BY sp.possession_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS cum_fgm,

                COALESCE(SUM(sp.is_3pa) OVER (
                    PARTITION BY sp.game_id, sp.player_id
                    ORDER BY sp.possession_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS cum_3pa,

                COALESCE(SUM(sp.is_3pm) OVER (
                    PARTITION BY sp.game_id, sp.player_id
                    ORDER BY sp.possession_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS cum_3pm,

                COALESCE(SUM(sp.is_fta) OVER (
                    PARTITION BY sp.game_id, sp.player_id
                    ORDER BY sp.possession_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS cum_fta,

                COALESCE(SUM(sp.is_ftm) OVER (
                    PARTITION BY sp.game_id, sp.player_id
                    ORDER BY sp.possession_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS cum_ftm

            FROM scored_poss sp
        )
        SELECT
            wr.game_id,
            wr.possession_id,
            wr.player_id,
            wr.team_tricode,
            wr.cum_points,
            wr.cum_fga,
            wr.cum_fgm,
            CASE WHEN wr.cum_fga > 0
                 THEN ROUND(wr.cum_fgm::FLOAT / wr.cum_fga, 3)
                 ELSE NULL END AS cum_fg_pct,
            wr.cum_3pa,
            wr.cum_3pm,
            CASE WHEN wr.cum_3pa > 0
                 THEN ROUND(wr.cum_3pm::FLOAT / wr.cum_3pa, 3)
                 ELSE NULL END AS cum_3p_pct,
            wr.cum_fta,
            wr.cum_ftm,
            CASE WHEN wr.cum_fta > 0
                 THEN ROUND(wr.cum_ftm::FLOAT / wr.cum_fta, 3)
                 ELSE NULL END AS cum_ft_pct,
            -- Foul count: most recent foul event before this possession
            COALESCE((
                SELECT MAX(pf.cum_fouls_before)
                FROM player_fouls pf
                WHERE pf.game_id = wr.game_id
                  AND pf.player_id = wr.player_id
                  AND (pf.period < rp.period
                       OR (pf.period = rp.period
                           AND pf.game_clock_secs > rp.game_clock_secs))
            ), 0) AS cum_fouls,
            apm.adjusted_plus_minus AS season_apm,
            apm.games_fitted        AS apm_games_fitted
        FROM with_running wr
        JOIN raw_possessions rp ON wr.game_id = rp.game_id AND wr.possession_id = rp.possession_id
        LEFT JOIN player_season_apm apm
            ON apm.player_id = wr.player_id
            AND apm.as_of_game_id = wr.game_id
    """)

    count = conn.execute("SELECT COUNT(*) FROM player_game_stats").fetchone()[0]
    logger.info("player_game_stats: %d rows", count)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pgs_game ON player_game_stats(game_id, possession_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pgs_player ON player_game_stats(game_id, player_id)")


# ---------------------------------------------------------------------------
# possession_feed enrichment
# ---------------------------------------------------------------------------

_ENRICH_COLS = [
    ("home_points_last_5_poss",           "SMALLINT"),
    ("away_points_last_5_poss",           "SMALLINT"),
    ("home_points_last_10_poss",          "SMALLINT"),
    ("away_points_last_10_poss",          "SMALLINT"),
    ("current_run_team",                  "VARCHAR"),
    ("current_run_length",                "SMALLINT"),
    ("current_run_points",                "SMALLINT"),
    ("current_run_3pt_pct",               "FLOAT"),
    ("current_run_paint_pct",             "FLOAT"),
    ("pace_last_10_possessions",          "FLOAT"),
    ("pace_season_baseline",              "FLOAT"),
    ("home_scoring_sustainable",          "BOOLEAN"),
    ("away_scoring_sustainable",          "BOOLEAN"),
    ("home_xPPP_last_5",                  "FLOAT"),
    ("away_xPPP_last_5",                  "FLOAT"),
    ("home_actual_vs_expected_PPP",       "FLOAT"),
    ("away_actual_vs_expected_PPP",       "FLOAT"),
    ("home_shot_quality_trend",           "FLOAT"),
    ("away_shot_quality_trend",           "FLOAT"),
    ("is_blowout",                        "BOOLEAN"),
    ("is_garbage_time",                   "BOOLEAN"),
    ("trailing_team_urgency",             "FLOAT"),
    ("comeback_probability_proxy",        "FLOAT"),
    ("q4_close_game",                     "BOOLEAN"),
    ("garbage_time_risk",                 "FLOAT"),
    ("home_player_in_trouble_id",         "BIGINT"),
    ("away_player_in_trouble_id",         "BIGINT"),
    ("home_star_in_foul_trouble",         "BOOLEAN"),
    ("away_star_in_foul_trouble",         "BOOLEAN"),
    ("home_star_on_court",                "BOOLEAN"),
    ("away_star_on_court",                "BOOLEAN"),
    ("possessions_since_last_timeout",    "SMALLINT"),
    ("home_called_timeout_in_last_3_poss", "BOOLEAN"),
    ("away_called_timeout_in_last_3_poss", "BOOLEAN"),
]

_ENRICH_COL_NAMES = [c for c, _ in _ENRICH_COLS]


def enrich_possession_feed(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Adds 34 feature columns to possession_feed by running the Python feature pipeline
    (momentum + context) per game and writing results back.

    For possession events: features are computed directly.
    For foul/sub/timeout events: features are forward-filled from the most recent possession
    using LAST_VALUE IGNORE NULLS window functions.

    Idempotent — skips if columns already exist.
    """
    existing_cols = {r[0] for r in conn.execute("DESCRIBE possession_feed").fetchall()}
    if "current_run_team" in existing_cols:
        # Only skip if data was actually written (not just columns added)
        null_poss = conn.execute(
            "SELECT COUNT(*) FROM possession_feed WHERE current_run_team IS NULL AND event_type='possession'"
        ).fetchone()[0]
        total_poss = conn.execute(
            "SELECT COUNT(*) FROM possession_feed WHERE event_type='possession'"
        ).fetchone()[0]
        if null_poss == 0 and total_poss > 0:
            logger.info("possession_feed already enriched — skipping")
            return
        logger.info("possession_feed columns exist but data incomplete (%d/%d possession rows populated) — re-enriching", total_poss - null_poss, total_poss)

    from models.features.momentum_features import add_momentum_features
    from models.features.context_features import add_context_features
    from models.features.lineup_features import add_lineup_features

    # Step 1: Add new columns (skip any that already exist)
    for col, dtype in _ENRICH_COLS:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE possession_feed ADD COLUMN {col} {dtype}")
    logger.info("Columns ready, loading data...")

    # Step 2: Load raw data
    logger.info("Loading raw data for enrichment pipeline...")
    possessions    = conn.execute("SELECT * FROM raw_possessions").df()
    foul_events    = conn.execute("SELECT * FROM raw_fouls").df()
    timeout_events = conn.execute("SELECT * FROM raw_timeouts").df()
    games          = conn.execute("SELECT * FROM dim_games").df()

    # DuckDB returns nullable integer columns as pandas Int64 with pd.NA.
    # Feature code expects float-compatible NaN — convert numeric columns.
    for col in ["shot_distance", "shot_x", "shot_y", "player_id", "points", "shot_value"]:
        if col in possessions.columns:
            possessions[col] = possessions[col].astype("float32")

    lineup_ratings = pd.read_parquet(FEATURE_DIR / "lineup_ratings.parquet")
    player_ratings = pd.read_parquet(PLAYER_RATINGS_PATH)

    # Pre-group lineup_ratings by game_id (avoids O(20M) scan per game)
    lineup_by_game: dict[str, pd.DataFrame] = {
        gid: grp for gid, grp in lineup_ratings.groupby("as_of_game_id")
    }

    # Step 3: Only process game_ids that have NULL current_run_team in possession rows
    null_game_ids = conn.execute("""
        SELECT DISTINCT game_id FROM possession_feed
        WHERE current_run_team IS NULL AND event_type='possession'
    """).df()["game_id"].tolist()

    game_ids = (
        games[games["game_id"].isin(null_game_ids)]
        .sort_values("game_date")["game_id"]
        .tolist()
    )
    logger.info("Running enrichment pipeline for %d games (with NULL features)...", len(game_ids))

    all_rows: list[pd.DataFrame] = []
    errors = 0
    for i, game_id in enumerate(game_ids):
        game_poss = (
            possessions[possessions["game_id"] == game_id]
            .sort_values("possession_id")
            .reset_index(drop=True)
        )
        if game_poss.empty:
            continue
        try:
            lr = lineup_by_game.get(game_id, pd.DataFrame())
            lineup_player_map: dict[str, list[int]] | None = None
            if not lr.empty and "lineup_id" in lr.columns:
                game_df, lineup_player_map = add_lineup_features(game_poss, lr, game_id, player_ratings)
            else:
                game_df = game_poss.copy()
            game_df = add_momentum_features(game_df)
            game_df = add_context_features(
                game_df, foul_events, games, game_id,
                timeout_events=timeout_events,
                lineup_player_map=lineup_player_map,
            )
            # Keep only identity + the new columns present in output
            available = ["game_id", "possession_id"] + [c for c in _ENRICH_COL_NAMES if c in game_df.columns]
            all_rows.append(game_df[available])
        except Exception as exc:
            logger.warning("Enrichment failed for game %s: %s", game_id, exc)
            errors += 1

        if (i + 1) % 200 == 0:
            logger.info("  Enriched %d / %d games...", i + 1, len(game_ids))

    if errors:
        logger.warning("%d games failed enrichment and will have NULL feature values", errors)

    if not all_rows:
        logger.error("No enrichment rows produced — aborting")
        return

    features_df = pd.concat(all_rows, ignore_index=True)
    logger.info("Enrichment pipeline complete: %d possession rows", len(features_df))

    # Step 4: UPDATE possession_feed for possession events
    logger.info("Writing enrichment data to possession_feed...")
    conn.register("_poss_features", features_df)

    set_clauses = ",\n        ".join(
        f"{col} = f.{col}"
        for col in _ENRICH_COL_NAMES
        if col in features_df.columns
    )
    conn.execute(f"""
        UPDATE possession_feed pf
        SET {set_clauses}
        FROM _poss_features f
        WHERE pf.game_id = f.game_id
          AND pf.possession_id = f.possession_id
          AND pf.event_type = 'possession'
    """)
    conn.unregister("_poss_features")

    # Step 5: Forward-fill enriched columns for non-possession events
    # LAST_VALUE IGNORE NULLS carries the most recent possession's feature state forward
    # to every foul/sub/timeout event that follows it in the same game.
    logger.info("Forward-filling feature columns for non-possession events...")

    ff_cols = ",\n        ".join(
        f"LAST_VALUE({col} IGNORE NULLS) OVER w AS {col}"
        for col in _ENRICH_COL_NAMES
    )
    base_cols = ", ".join(
        r[0] for r in conn.execute("DESCRIBE possession_feed").fetchall()
        if r[0] not in _ENRICH_COL_NAMES
    )
    conn.execute(f"""
        CREATE OR REPLACE TABLE possession_feed AS
        SELECT
            {base_cols},
            {ff_cols}
        FROM possession_feed
        WINDOW w AS (
            PARTITION BY game_id
            ORDER BY period ASC, game_clock_secs DESC, event_id ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
    """)

    enriched_count = conn.execute("SELECT COUNT(*) FROM possession_feed").fetchone()[0]
    null_run = conn.execute(
        "SELECT COUNT(*) FROM possession_feed WHERE current_run_team IS NULL AND event_type='possession'"
    ).fetchone()[0]
    logger.info("possession_feed enriched: %d rows, %d possession rows with NULL current_run_team", enriched_count, null_run)


# ---------------------------------------------------------------------------
# MotherDuck sync
# ---------------------------------------------------------------------------

def sync_to_motherduck(local_db_path: str, db_name: str = "kalshi_trading") -> None:
    """
    Merge local DuckDB into MotherDuck — additive only.

    For each table:
      1. If the table doesn't exist in MotherDuck yet, create it.
      2. If local has new columns, ALTER TABLE ADD COLUMN on the remote.
      3. Insert rows that exist locally but not in MotherDuck (keyed by each
         table's natural key). Remote-only rows are never deleted.

    This means two teammates can each run the loader independently and push
    their own new data without overwriting each other's contributions.
    """
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError(
            "MOTHERDUCK_TOKEN not set. Add it to .env: MOTHERDUCK_TOKEN=eyJ..."
        )

    # Natural keys used to detect duplicate rows per table.
    # Only columns listed here are compared; extra local columns are added via ALTER.
    _TABLE_KEYS: dict[str, list[str]] = {
        "dim_teams":          ["team_id"],
        "dim_players":        ["player_id"],
        "dim_games":          ["game_id"],
        "raw_possessions":    ["game_id", "possession_id"],
        "raw_fouls":          ["game_id", "action_number"],
        "raw_substitutions":  ["game_id", "action_number"],
        "raw_timeouts":       ["game_id", "action_number"],
        "kalshi_settled":     ["market_ticker"],
        "kalshi_market_map":  ["market_ticker"],
        "kalshi_ticks":       ["market_ticker", "ts"],
        "possession_feed":    ["game_id", "event_id"],
        "player_game_stats":  ["game_id", "player_id"],
    }

    logger.info("Connecting to MotherDuck as db '%s'...", db_name)
    md_conn = duckdb.connect(f"md:{db_name}?motherduck_token={token}")

    logger.info("Attaching local database...")
    md_conn.execute(f"ATTACH '{local_db_path}' AS local_db (READ_ONLY)")

    local_conn = duckdb.connect(local_db_path, read_only=True)
    tables = [
        r[0] for r in local_conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    ]
    local_conn.close()

    for table in tables:
        md_tables = {r[0] for r in md_conn.execute("SHOW TABLES").fetchall()}

        # --- Table doesn't exist remotely yet: create it ---
        if table not in md_tables:
            logger.info("%-30s creating in MotherDuck...", table)
            md_conn.execute(f"CREATE TABLE main.{table} AS SELECT * FROM local_db.main.{table}")
            count = md_conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
            logger.info("%-30s created (%d rows)", table, count)
            continue

        # --- Schema migration: add columns present locally but missing remotely ---
        local_schema = {r[0]: r[1] for r in md_conn.execute(f"DESCRIBE local_db.main.{table}").fetchall()}
        md_schema    = {r[0] for r in md_conn.execute(f"DESCRIBE main.{table}").fetchall()}
        for col, dtype in local_schema.items():
            if col not in md_schema:
                logger.info("  %-28s adding column %s (%s)", table, col, dtype)
                md_conn.execute(f"ALTER TABLE main.{table} ADD COLUMN {col} {dtype}")

        # --- Insert new rows: local rows whose key doesn't exist in remote ---
        keys = _TABLE_KEYS.get(table)
        if not keys:
            logger.warning("%-30s no key defined — skipping row sync", table)
            continue

        # Build EXISTS check: WHERE NOT EXISTS (SELECT 1 FROM remote WHERE key matches)
        key_join = " AND ".join(f"r.{k} = l.{k}" for k in keys)
        col_list = ", ".join(local_schema.keys())  # only insert cols local has

        before = md_conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
        md_conn.execute(f"""
            INSERT INTO main.{table} ({col_list})
            SELECT {col_list} FROM local_db.main.{table} l
            WHERE NOT EXISTS (
                SELECT 1 FROM main.{table} r WHERE {key_join}
            )
        """)
        after = md_conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
        added = after - before
        if added > 0:
            logger.info("%-30s +%d new rows → %d total", table, added, after)
        else:
            logger.info("%-30s up to date (%d rows)", table, after)

    md_conn.close()
    logger.info("MotherDuck sync complete.")


def sync_from_motherduck(
    local_db_path: str,
    db_name: str = "kalshi_trading",
    full: bool = False,
) -> None:
    """
    Pull data from MotherDuck into local DuckDB.

    Two modes:
      full=False  (merge, default)
        Additive only — mirrors sync_to_motherduck in reverse.
        Adds columns/rows from MotherDuck that aren't in local.
        Safe to run anytime; your local-only data is never deleted.

      full=True   (DESTRUCTIVE full copy)
        Replaces every local table with the exact MotherDuck copy.
        Use this when you want a clean pull — e.g. first setup on a new
        machine, or to reset a corrupted local DB.
        WARNING: all local-only rows and tables are permanently lost.
    """
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError(
            "MOTHERDUCK_TOKEN not set. Add it to .env: MOTHERDUCK_TOKEN=eyJ..."
        )

    _TABLE_KEYS: dict[str, list[str]] = {
        "dim_teams":          ["team_id"],
        "dim_players":        ["player_id"],
        "dim_games":          ["game_id"],
        "raw_possessions":    ["game_id", "possession_id"],
        "raw_fouls":          ["game_id", "action_number"],
        "raw_substitutions":  ["game_id", "action_number"],
        "raw_timeouts":       ["game_id", "action_number"],
        "kalshi_settled":     ["market_ticker"],
        "kalshi_market_map":  ["market_ticker"],
        "kalshi_ticks":       ["market_ticker", "ts"],
        "possession_feed":    ["game_id", "event_id"],
        "player_game_stats":  ["game_id", "player_id"],
    }

    mode = "FULL COPY (destructive)" if full else "merge (additive)"
    logger.info("Connecting to MotherDuck as db '%s' [%s]...", db_name, mode)
    md_conn = duckdb.connect(f"md:{db_name}?motherduck_token={token}")

    logger.info("Attaching local database...")
    md_conn.execute(f"ATTACH '{local_db_path}' AS local_db")

    md_tables = [r[0] for r in md_conn.execute("SHOW TABLES").fetchall()]

    for table in md_tables:
        local_exists = md_conn.execute(
            f"SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_catalog = 'local_db' AND table_schema = 'main' AND table_name = '{table}'"
        ).fetchone()[0] > 0

        if full:
            # Completely replace local table with MotherDuck copy
            md_conn.execute(
                f"CREATE OR REPLACE TABLE local_db.main.{table} AS SELECT * FROM main.{table}"
            )
            count = md_conn.execute(f"SELECT COUNT(*) FROM local_db.main.{table}").fetchone()[0]
            logger.info("%-30s replaced (%d rows)", table, count)

        else:
            # Merge: add columns/rows from cloud that aren't local

            if not local_exists:
                md_conn.execute(
                    f"CREATE TABLE local_db.main.{table} AS SELECT * FROM main.{table}"
                )
                count = md_conn.execute(f"SELECT COUNT(*) FROM local_db.main.{table}").fetchone()[0]
                logger.info("%-30s created locally (%d rows)", table, count)
                continue

            # Schema migration: add columns present in cloud but missing locally
            md_schema    = {r[0]: r[1] for r in md_conn.execute(f"DESCRIBE main.{table}").fetchall()}
            local_schema = {r[0] for r in md_conn.execute(f"DESCRIBE local_db.main.{table}").fetchall()}
            for col, dtype in md_schema.items():
                if col not in local_schema:
                    logger.info("  %-28s adding column %s (%s)", table, col, dtype)
                    md_conn.execute(f"ALTER TABLE local_db.main.{table} ADD COLUMN {col} {dtype}")

            # Insert rows from cloud not present locally
            keys = _TABLE_KEYS.get(table)
            if not keys:
                logger.warning("%-30s no key defined — skipping row sync", table)
                continue

            key_join = " AND ".join(f"l.{k} = r.{k}" for k in keys)
            col_list = ", ".join(md_schema.keys())

            before = md_conn.execute(f"SELECT COUNT(*) FROM local_db.main.{table}").fetchone()[0]
            md_conn.execute(f"""
                INSERT INTO local_db.main.{table} ({col_list})
                SELECT {col_list} FROM main.{table} r
                WHERE NOT EXISTS (
                    SELECT 1 FROM local_db.main.{table} l WHERE {key_join}
                )
            """)
            after = md_conn.execute(f"SELECT COUNT(*) FROM local_db.main.{table}").fetchone()[0]
            added = after - before
            if added > 0:
                logger.info("%-30s +%d new rows → %d total", table, added, after)
            else:
                logger.info("%-30s up to date (%d rows)", table, after)

    md_conn.close()
    logger.info("Pull from MotherDuck complete [%s].", mode)


# ---------------------------------------------------------------------------
# Feature schema pull
# ---------------------------------------------------------------------------

def pull_possession_flat(local_db_path: Path) -> None:
    """
    Pull features.possession_flat from MotherDuck into local DuckDB.

    This table lives in the `features` schema on MotherDuck (not `main`), so
    the standard additive sync doesn't cover it. Use this when you need the
    flat feature table locally for model training.

    Safe to re-run — uses CREATE OR REPLACE so local copy is always overwritten
    with the latest remote version.
    """
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set")

    logger.info("Connecting to MotherDuck to pull features.possession_flat …")
    md_conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={token}")

    row_count = md_conn.execute(
        "SELECT COUNT(*) FROM features.possession_flat"
    ).fetchone()[0]
    logger.info("Remote possession_flat: %d rows", row_count)

    df = md_conn.execute("SELECT * FROM features.possession_flat").df()
    md_conn.close()

    local_conn = duckdb.connect(str(local_db_path))
    local_conn.execute("CREATE SCHEMA IF NOT EXISTS features")
    local_conn.execute(
        "CREATE OR REPLACE TABLE features.possession_flat AS SELECT * FROM df"
    )
    local_count = local_conn.execute(
        "SELECT COUNT(*) FROM features.possession_flat"
    ).fetchone()[0]
    local_conn.close()

    logger.info("Pulled %d rows into local features.possession_flat", local_count)


def pull_features_schema(local_db_path: Path) -> None:
    """
    Pull lineup_ratings and player_ratings from MotherDuck features schema.

    Only pulls if the cloud has MORE rows than local (cloud is assumed authoritative
    when ahead). If local is equal or ahead, skips to avoid overwriting newer local data.
    Uses CREATE OR REPLACE when pulling.
    """
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set")

    tables = ["lineup_ratings", "player_ratings"]
    md_conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={token}")

    for table in tables:
        cloud_count = md_conn.execute(
            f"SELECT COUNT(*) FROM features.{table}"
        ).fetchone()[0]

        local_conn = duckdb.connect(str(local_db_path))
        try:
            local_count = local_conn.execute(
                f"SELECT COUNT(*) FROM features.{table}"
            ).fetchone()[0]
        except Exception:
            local_count = 0
        local_conn.close()

        if cloud_count <= local_count:
            logger.info(
                "features.%-25s cloud=%d local=%d — local is current, skipping",
                table, cloud_count, local_count,
            )
            continue

        logger.info(
            "features.%-25s cloud=%d > local=%d — pulling …",
            table, cloud_count, local_count,
        )
        df = md_conn.execute(f"SELECT * FROM features.{table}").df()
        local_conn = duckdb.connect(str(local_db_path))
        local_conn.execute("CREATE SCHEMA IF NOT EXISTS features")
        local_conn.execute(
            f"CREATE OR REPLACE TABLE features.{table} AS SELECT * FROM df"
        )
        new_count = local_conn.execute(
            f"SELECT COUNT(*) FROM features.{table}"
        ).fetchone()[0]
        local_conn.close()
        logger.info("features.%-25s pulled %d rows", table, new_count)

    md_conn.close()


def push_possession_flat(local_db_path: Path) -> None:
    """
    Push the local features.possession_flat to MotherDuck.

    Use after enriching possession_flat locally (e.g. adding lineup ratings).
    Overwrites the cloud table entirely — run validation before calling this.
    """
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set")

    local_conn = duckdb.connect(str(local_db_path), read_only=True)
    local_count = local_conn.execute(
        "SELECT COUNT(*) FROM features.possession_flat"
    ).fetchone()[0]
    local_cols = len(local_conn.execute("DESCRIBE features.possession_flat").fetchall())
    logger.info(
        "Pushing local possession_flat to MotherDuck: %d rows, %d columns …",
        local_count, local_cols,
    )
    df = local_conn.execute("SELECT * FROM features.possession_flat").df()
    local_conn.close()

    md_conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={token}")
    md_conn.execute("CREATE OR REPLACE TABLE features.possession_flat AS SELECT * FROM df")
    cloud_count = md_conn.execute(
        "SELECT COUNT(*) FROM features.possession_flat"
    ).fetchone()[0]
    md_conn.close()

    if cloud_count != local_count:
        raise RuntimeError(
            f"Push verification failed: local={local_count} cloud={cloud_count}"
        )
    logger.info("Pushed %d rows to MotherDuck features.possession_flat ✓", cloud_count)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load all data into DuckDB")
    parser.add_argument(
        "--sync-motherduck", action="store_true",
        help="After building local DB, push new rows to MotherDuck (additive)",
    )
    parser.add_argument(
        "--pull-motherduck", action="store_true",
        help="Pull new rows from MotherDuck into local DB (additive merge)",
    )
    parser.add_argument(
        "--pull-motherduck-full", action="store_true",
        help="DESTRUCTIVE: replace local DB tables entirely from MotherDuck",
    )
    parser.add_argument(
        "--pull-possession-flat", action="store_true",
        help="Pull features.possession_flat from MotherDuck into local features schema",
    )
    parser.add_argument(
        "--pull-features-schema", action="store_true",
        help="Pull lineup_ratings and player_ratings from MotherDuck (only if cloud is ahead)",
    )
    parser.add_argument(
        "--push-possession-flat", action="store_true",
        help="Push enriched local features.possession_flat up to MotherDuck",
    )
    parser.add_argument(
        "--db", default=str(DB_PATH),
        help=f"Path to local DuckDB file (default: {DB_PATH})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    load_dotenv()

    # Pull/push-only flags: skip the full local build when used standalone
    standalone_flags = any([
        args.pull_possession_flat, args.pull_features_schema, args.push_possession_flat,
    ])
    build_flags = any([
        args.sync_motherduck, args.pull_motherduck, args.pull_motherduck_full,
    ])
    if standalone_flags and not build_flags:
        if args.pull_possession_flat:
            pull_possession_flat(Path(args.db))
        if args.pull_features_schema:
            pull_features_schema(Path(args.db))
        if args.push_possession_flat:
            push_possession_flat(Path(args.db))
        return

    logger.info("Opening DuckDB at %s", args.db)
    conn = duckdb.connect(args.db)

    logger.info("=== Tier 0: Dimension tables ===")
    load_dim_teams(conn)
    load_dim_players(conn)
    load_dim_games(conn)

    logger.info("=== Tier 1: Raw event tables ===")
    load_raw_events(conn)

    logger.info("=== Tier 2: Kalshi tables ===")
    load_kalshi(conn)

    logger.info("=== Tier 3: Derived tables ===")
    build_possession_feed(conn)
    build_player_game_stats(conn)
    enrich_possession_feed(conn)

    conn.close()
    logger.info("Local DuckDB build complete: %s", args.db)

    if args.sync_motherduck:
        sync_to_motherduck(args.db)

    if args.pull_motherduck:
        sync_from_motherduck(args.db, full=False)

    if args.pull_motherduck_full:
        sync_from_motherduck(args.db, full=True)

    if args.pull_possession_flat:
        pull_possession_flat(Path(args.db))
    if args.pull_features_schema:
        pull_features_schema(Path(args.db))
    if args.push_possession_flat:
        push_possession_flat(Path(args.db))


if __name__ == "__main__":
    main()
