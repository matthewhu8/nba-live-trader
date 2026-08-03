"""
Backfill wall_clock_ts in features.possession_flat.

wall_clock_ts is the real-world UTC timestamp for each possession event.
It's required by the collapsed price movement predictor to ASOF-join
possession rows to Kalshi tick data.

How it's computed:
  1. Fetch PlayByPlayV3 for the game (uses existing PBP cache when available)
  2. Find "Start of Period" description strings → extract ET tip-off time per quarter
  3. Linearly interpolate real-world time for every event between period anchors
  4. Join computed timestamps to possession_flat on (game_id, period, game_clock_secs)
  5. UPDATE MotherDuck in batch — skip games already fully populated

The linear interpolation between period anchors accounts for stoppages (timeouts,
fouls, reviews) that make real time > game time. Typical accuracy: ±15–45 seconds,
which is sufficient for matching to Kalshi ticks that update every few seconds.

Usage:
    # Backfill all games with Kalshi tick coverage that are missing wall_clock_ts
    python -m data.ingestion.backfill_wall_clock_ts

    # Specific date
    python -m data.ingestion.backfill_wall_clock_ts --date 2026-03-25

    # Date range
    python -m data.ingestion.backfill_wall_clock_ts --since 2026-03-23

    # Single game
    python -m data.ingestion.backfill_wall_clock_ts --game 0022501059

    # Dry run: compute and print without writing
    python -m data.ingestion.backfill_wall_clock_ts --date 2026-03-25 --dry-run
"""

import argparse
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import pytz
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Reuse the same PBP cache as nba_api_client so we don't re-fetch
PBP_CACHE_DIR = Path("data/raw/pbp_cache")

# Regex to extract the time from period/OT start descriptions:
#   "Start of 1st Period (10:13 PM EST)"
#   "Start of 1st OT (12:12 AM EST)"
# Captures "10:13 PM" — we ignore the timezone label and always use America/New_York
_PERIOD_START_RE = re.compile(
    r"Start of .+? (?:Period|OT) \((\d{1,2}:\d{2} (?:AM|PM))",
    re.IGNORECASE,
)

ET = pytz.timezone("America/New_York")

# NBA period durations in game seconds
_PERIOD_GAME_SECS = {**{p: 720.0 for p in range(1, 5)}, **{p: 300.0 for p in range(5, 20)}}

# Fallback real/game time ratio when we can't compute from period anchors.
# Real NBA quarters average ~26-28 real minutes for 12 game minutes → ratio ~2.2
_FALLBACK_REAL_GAME_RATIO = 2.2


# ---------------------------------------------------------------------------
# MotherDuck connection (same pattern as post_game_pipeline)
# ---------------------------------------------------------------------------

def _md_connect() -> duckdb.DuckDBPyConnection:
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set — check .env or environment")
    conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={token}")
    conn.execute("SET memory_limit='256MB'")
    return conn


# ---------------------------------------------------------------------------
# Game discovery
# ---------------------------------------------------------------------------

def _games_needing_backfill(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, date]]:
    """
    Return (game_id, game_date) for all games that:
      1. Have Kalshi tick coverage (appear in kalshi_ticks market_tickers)
      2. Have possession_flat rows where wall_clock_ts IS NULL
    These are the games the predictor can't train on until we backfill.
    """
    # Extract game dates from ticker format: KXNBASPREAD-26MAR25DALDEN-DEN2 → 2026-03-25
    # We join via dim_games using parsed date + teams rather than game_id (which is null in ticks)
    df = conn.execute("""
        WITH tick_dates AS (
            SELECT DISTINCT
                '20' || SUBSTRING(SPLIT_PART(market_ticker, '-', 2), 1, 2) AS yr,
                SUBSTRING(SPLIT_PART(market_ticker, '-', 2), 3, 3)          AS mon,
                SUBSTRING(SPLIT_PART(market_ticker, '-', 2), 6, 2)          AS dy
            FROM kalshi_ticks
        ),
        tick_game_dates AS (
            SELECT DISTINCT
                CASE mon
                    WHEN 'JAN' THEN CAST(yr || '-01-' || dy AS DATE)
                    WHEN 'FEB' THEN CAST(yr || '-02-' || dy AS DATE)
                    WHEN 'MAR' THEN CAST(yr || '-03-' || dy AS DATE)
                    WHEN 'APR' THEN CAST(yr || '-04-' || dy AS DATE)
                    WHEN 'MAY' THEN CAST(yr || '-05-' || dy AS DATE)
                    WHEN 'JUN' THEN CAST(yr || '-06-' || dy AS DATE)
                    WHEN 'OCT' THEN CAST(yr || '-10-' || dy AS DATE)
                    WHEN 'NOV' THEN CAST(yr || '-11-' || dy AS DATE)
                    WHEN 'DEC' THEN CAST(yr || '-12-' || dy AS DATE)
                END AS game_date
            FROM tick_dates
        )
        SELECT DISTINCT dg.game_id, dg.game_date
        FROM dim_games dg
        JOIN tick_game_dates tgd ON dg.game_date = tgd.game_date
        JOIN features.possession_flat pf ON pf.game_id = dg.game_id
        WHERE pf.wall_clock_ts IS NULL
        ORDER BY dg.game_date, dg.game_id
    """).df()

    return [(row.game_id, row.game_date) for row in df.itertuples(index=False)]


def _games_for_date(
    game_date: date,
    conn: duckdb.DuckDBPyConnection,
) -> list[tuple[str, date]]:
    df = conn.execute(
        "SELECT game_id, game_date FROM dim_games WHERE game_date = ? ORDER BY game_id",
        [game_date.isoformat()],
    ).df()
    return [(row.game_id, row.game_date) for row in df.itertuples(index=False)]


def _games_since(
    since_date: date,
    conn: duckdb.DuckDBPyConnection,
) -> list[tuple[str, date]]:
    df = conn.execute(
        "SELECT game_id, game_date FROM dim_games WHERE game_date >= ? ORDER BY game_date, game_id",
        [since_date.isoformat()],
    ).df()
    return [(row.game_id, row.game_date) for row in df.itertuples(index=False)]


def _is_already_populated(game_id: str, conn: duckdb.DuckDBPyConnection) -> bool:
    """True if ALL possession_flat rows for this game already have wall_clock_ts."""
    result = conn.execute(
        """
        SELECT COUNT(*) = 0 AS all_populated
        FROM features.possession_flat
        WHERE game_id = ? AND wall_clock_ts IS NULL
        """,
        [game_id],
    ).fetchone()
    return bool(result[0]) if result else False


# ---------------------------------------------------------------------------
# PBP fetching (reuses cache from nba_api_client)
# ---------------------------------------------------------------------------

def _fetch_pbp(game_id: str) -> pd.DataFrame:
    """
    Return PlayByPlayV3 DataFrame for the game.
    Reads from local cache if available; otherwise fetches from nba_api.
    """
    cache_path = PBP_CACHE_DIR / f"{game_id}.parquet"
    if cache_path.exists():
        logger.debug("game %s: loading PBP from cache", game_id)
        return pd.read_parquet(cache_path)

    logger.info("game %s: fetching PBP from nba_api", game_id)
    time.sleep(0.6)  # same rate limit as nba_api_client
    from nba_api.stats.endpoints import playbyplayv3
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, end_period=10)
    df = pbp.get_data_frames()[0]
    PBP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Period anchor extraction
# ---------------------------------------------------------------------------

def _parse_clock_secs(clock_str: str) -> float:
    """Parse ISO 8601 game clock 'PT11M40.00S' → 700.0 seconds."""
    m = re.match(r"PT(\d+)M([\d.]+)S", clock_str or "")
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def _parse_period_start_et(
    time_str: str,
    game_date: date,
    prev_dt: Optional[datetime],
) -> Optional[datetime]:
    """
    Parse 'HH:MM AM/PM' into an ET-aware datetime on the correct date.

    Handles midnight crossover: if the parsed time is earlier than the
    previous period's time, assume the game crossed midnight and add 1 day.
    """
    try:
        naive = datetime.strptime(time_str.strip(), "%I:%M %p")
    except ValueError:
        return None

    # Determine the calendar date (may be game_date or game_date+1 for late PT games)
    base_date = game_date
    if prev_dt is not None:
        prev_naive = prev_dt.astimezone(ET).replace(tzinfo=None)
        # If this period's hour is earlier than the previous, we crossed midnight
        if naive.hour < prev_naive.hour or (naive.hour == 0 and prev_naive.hour == 23):
            base_date = (prev_dt.astimezone(ET).date() + timedelta(days=1))

    candidate = datetime(base_date.year, base_date.month, base_date.day,
                         naive.hour, naive.minute, naive.second)
    # Localize to ET (pytz handles DST — March games are EDT = UTC-4)
    return ET.localize(candidate)


def _extract_period_anchors(
    pbp: pd.DataFrame,
    game_date: date,
) -> dict[int, datetime]:
    """
    Return {period: start_datetime_UTC} for each period that has a
    "Start of Period" event with a parseable wall-clock time.
    """
    # Find rows where the description contains the period/OT start marker
    # Matches: "Start of 1st Period (10:13 PM EST)" and "Start of 1st OT (12:12 AM EST)"
    mask = pbp["description"].str.contains(
        r"Start of .+? (?:Period|OT) \(", na=False, regex=True
    )
    period_starts = pbp[mask].sort_values("period")

    anchors: dict[int, datetime] = {}
    prev_dt: Optional[datetime] = None

    for _, row in period_starts.iterrows():
        m = _PERIOD_START_RE.search(str(row.get("description", "")))
        if not m:
            continue
        period = int(row["period"])
        dt = _parse_period_start_et(m.group(1), game_date, prev_dt)
        if dt is None:
            continue
        anchors[period] = dt
        prev_dt = dt
        logger.debug("game: period %d anchor → %s", period, dt.isoformat())

    return anchors


# ---------------------------------------------------------------------------
# Wall-clock timestamp computation
# ---------------------------------------------------------------------------

def _compute_event_timestamps(
    pbp: pd.DataFrame,
    anchors: dict[int, datetime],
) -> dict[tuple[int, float], datetime]:
    """
    For every event in the PBP, compute an estimated wall_clock_ts using
    linear interpolation between period anchor times.

    Returns: {(period, clock_secs): wall_clock_ts_utc}
    """
    if not anchors:
        return {}

    # Compute real/game time ratio for each period we have both anchors
    ratios: list[float] = []
    periods_present = sorted(anchors.keys())

    for i in range(len(periods_present) - 1):
        p = periods_present[i]
        p_next = periods_present[i + 1]
        real_secs = (anchors[p_next] - anchors[p]).total_seconds()
        game_secs = _PERIOD_GAME_SECS.get(p, 720.0)
        if real_secs > 0:
            ratios.append(real_secs / game_secs)

    fallback_ratio = (sum(ratios) / len(ratios)) if ratios else _FALLBACK_REAL_GAME_RATIO

    result: dict[tuple[int, float], datetime] = {}

    pbp = pbp.copy()
    pbp["_clock_secs"] = pbp["clock"].apply(_parse_clock_secs)

    for period, group in pbp.groupby("period"):
        period = int(period)
        if period not in anchors:
            continue

        period_start = anchors[period]
        game_secs = _PERIOD_GAME_SECS.get(period, 720.0)

        # Use next period's start as the end anchor when available
        next_period = period + 1
        if next_period in anchors:
            real_duration_secs = (anchors[next_period] - period_start).total_seconds()
        else:
            real_duration_secs = fallback_ratio * game_secs

        for _, row in group.iterrows():
            clock = float(row["_clock_secs"])
            elapsed_game = game_secs - clock
            fraction = elapsed_game / game_secs if game_secs > 0 else 0.0
            fraction = max(0.0, min(1.0, fraction))

            ts = period_start + timedelta(seconds=fraction * real_duration_secs)
            # Convert to UTC
            result[(period, clock)] = ts.astimezone(pytz.utc)

    return result


# ---------------------------------------------------------------------------
# Core backfill function (importable by post_game_pipeline)
# ---------------------------------------------------------------------------

# Plausible NBA tip-off window in ET. Earliest regular tips are ~12:00 (holiday
# afternoon games); nothing legitimately tips at/after midnight ET.
_MIN_TIPOFF_HOUR_ET = 12
_MAX_TIPOFF_HOUR_ET = 23


def _validate_anchors(
    game_id: str,
    game_date: date,
    anchors: dict[int, datetime],
) -> bool:
    """
    Sanity-check computed period anchors before they are written.

    This guard exists because a wrong `game_date` silently produces timestamps
    with the correct time-of-day on the wrong calendar day. Downstream,
    pd.merge_asof never errors — it just matches the last (settled) tick — so the
    corruption is invisible until you audit prices. 36 of 71 games were written
    one day late this way, which contaminated Head B's training rows and every
    backtest measured through the tick join.

    Two invariants, both cheap:
      1. The period-1 anchor's ET calendar date must equal `game_date`.
         Catches the UTC-vs-ET date bug (games tipping >= 20:00 ET cross 00:00 UTC).
      2. The period-1 anchor's ET hour must be a plausible tip-off time.
         Catches anchors misparsed from an OT start line (e.g. "12:12 AM").

    Returns True if the anchors look sane; logs an error and returns False if not,
    so the caller skips the game rather than writing bad data.
    """
    first_period = min(anchors)
    anchor_et = anchors[first_period].astimezone(ET)

    if anchor_et.date() != game_date:
        logger.error(
            "game %s: period-%d anchor resolves to %s ET but dim_games.game_date is %s "
            "(off by %+d day(s)). This is the UTC-vs-ET date bug — refusing to write. "
            "Verify dim_games.game_date is the EASTERN game date, not a UTC-derived one.",
            game_id, first_period, anchor_et.isoformat(), game_date.isoformat(),
            (anchor_et.date() - game_date).days,
        )
        return False

    if not (_MIN_TIPOFF_HOUR_ET <= anchor_et.hour <= _MAX_TIPOFF_HOUR_ET):
        logger.error(
            "game %s: period-%d anchor is %s ET — implausible tip-off hour (%d). "
            "The anchor was likely parsed from the wrong PBP line (e.g. an OT start "
            "such as '12:12 AM'). Refusing to write.",
            game_id, first_period, anchor_et.isoformat(), anchor_et.hour,
        )
        return False

    return True


def _compute_timestamps_for_game(
    game_id: str,
    game_date: date,
    conn: duckdb.DuckDBPyConnection,
) -> Optional[pd.DataFrame]:
    """
    Compute wall_clock_ts for all possession_flat rows of one game.

    Returns a DataFrame with columns [game_id, period, game_clock_secs, wall_clock_ts],
    or None if the game cannot be processed (missing anchors, empty PBP, etc.).
    Does NOT write anything — callers handle the write.
    """
    if _is_already_populated(game_id, conn):
        logger.debug("game %s: wall_clock_ts already fully populated — skipping", game_id)
        return None

    try:
        pbp = _fetch_pbp(game_id)
    except Exception:
        logger.exception("game %s: PBP fetch failed — skipping", game_id)
        return None

    anchors = _extract_period_anchors(pbp, game_date)
    if not anchors:
        logger.warning(
            "game %s: no period start anchors found in PBP descriptions — "
            "cannot compute wall_clock_ts. Skipping.",
            game_id,
        )
        return None

    if not _validate_anchors(game_id, game_date, anchors):
        return None

    logger.debug("game %s: found anchors for periods %s", game_id, sorted(anchors.keys()))
    ts_map = _compute_event_timestamps(pbp, anchors)
    if not ts_map:
        return None

    # Load possession_flat rows for this game that still need wall_clock_ts
    pf = conn.execute(
        """
        SELECT event_id, period, game_clock_secs
        FROM features.possession_flat
        WHERE game_id = ? AND wall_clock_ts IS NULL
        """,
        [game_id],
    ).df()

    if pf.empty:
        return None

    # Join computed timestamps: match on (period, game_clock_secs).
    # Cast to float64 BEFORE rounding — float32 can't represent 58.9 exactly
    # (float32(58.9) = 58.900001...), so .round(2) on a float32 column gives
    # the wrong key. Upcasting first makes round() produce the correct value.
    pf["_key"] = list(zip(
        pf["period"].astype(int),
        pf["game_clock_secs"].astype(float).round(2),
    ))
    ts_map_rounded = {
        (int(p), round(c, 2)): ts
        for (p, c), ts in ts_map.items()
    }

    pf["wall_clock_ts"] = pf["_key"].map(ts_map_rounded)
    matched = pf["wall_clock_ts"].notna()

    n_matched = int(matched.sum())
    n_total = len(pf)

    if n_matched == 0:
        logger.warning(
            "game %s: 0 of %d possession_flat rows matched PBP events. "
            "Check period/clock_secs alignment.",
            game_id, n_total,
        )
        return None

    logger.info(
        "game %s: matched %d / %d possession_flat rows to computed timestamps",
        game_id, n_matched, n_total,
    )

    # Keep period + game_clock_secs for the JOIN — event_id is NULL for games
    # inserted by the nightly post_game_pipeline, so we can't rely on it.
    result = pf[matched][["period", "game_clock_secs", "wall_clock_ts"]].copy()
    result["game_id"] = game_id
    # Store as float64 so the CTAS JOIN can round-compare correctly
    result["game_clock_secs"] = result["game_clock_secs"].astype(float)
    return result[["game_id", "period", "game_clock_secs", "wall_clock_ts"]]


def _apply_timestamps(
    all_updates: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """
    Write wall_clock_ts values to MotherDuck using INSERT → CTAS pattern.

    MotherDuck does not support UPDATE FROM a registered DataFrame (raises
    CatalogException: "Remote catalog has changed").  Instead we:
      1. INSERT all new timestamps into features.wcts_backfill (a staging table)
      2. CTAS to atomically rebuild features.possession_flat with the new values
      3. DROP the staging table

    This is safe — possession_flat is rebuilt atomically by DuckDB/MotherDuck.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features.wcts_backfill (
            game_id       VARCHAR,
            period        INTEGER,
            game_clock_secs DOUBLE,
            wall_clock_ts  TIMESTAMPTZ
        )
    """)

    # Wipe and reload the staging table
    conn.execute("DELETE FROM features.wcts_backfill")
    conn.register("_wcts_new", all_updates)
    conn.execute("INSERT INTO features.wcts_backfill SELECT * FROM _wcts_new")
    conn.unregister("_wcts_new")

    logger.info(
        "Inserted %d rows into wcts_backfill — rebuilding possession_flat via CTAS",
        len(all_updates),
    )

    # Atomic rebuild: swap wall_clock_ts values where we have a match.
    # Join on (game_id, period, game_clock_secs) — not event_id, because
    # games inserted by the nightly post_game_pipeline have NULL event_id.
    # Cast both sides to DOUBLE and ROUND to 2 dp to handle float32 imprecision.
    conn.execute("""
        CREATE OR REPLACE TABLE features.possession_flat AS
        SELECT
            pf.* EXCLUDE (wall_clock_ts),
            COALESCE(wb.wall_clock_ts, pf.wall_clock_ts) AS wall_clock_ts
        FROM features.possession_flat pf
        LEFT JOIN features.wcts_backfill wb
               ON pf.game_id = wb.game_id
              AND pf.period = wb.period
              AND ROUND(CAST(pf.game_clock_secs AS DOUBLE), 2)
                = ROUND(wb.game_clock_secs, 2)
    """)

    logger.info("CTAS rebuild complete — dropping wcts_backfill")
    conn.execute("DROP TABLE features.wcts_backfill")


def backfill_game(
    game_id: str,
    game_date: date,
    conn: duckdb.DuckDBPyConnection,
    dry_run: bool = False,
) -> int:
    """
    Compute wall_clock_ts for one game and immediately write it to MotherDuck.

    This is the importable API used by post_game_pipeline for nightly runs
    (one game at a time). For bulk historical backfill use the CLI, which
    batches all games and does a single CTAS rebuild.

    Returns the number of rows updated (0 if already populated or no anchors found).
    Safe to call repeatedly — idempotency checked before doing any work.
    """
    result = _compute_timestamps_for_game(game_id, game_date, conn)
    if result is None:
        return 0

    n_matched = len(result)

    if dry_run:
        sample = result.head(10)
        print(f"\n--- DRY RUN: game {game_id} ({game_date}) ---")
        print(sample.to_string(index=False))
        return n_matched

    # Single-game write: still use INSERT+CTAS (same path as bulk backfill)
    _apply_timestamps(result, conn)
    logger.info("game %s: wrote wall_clock_ts for %d rows", game_id, n_matched)
    return n_matched


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill wall_clock_ts in features.possession_flat"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date",  help="Backfill all games on this date (YYYY-MM-DD)")
    group.add_argument("--since", help="Backfill all games on or after this date (YYYY-MM-DD)")
    group.add_argument("--game",  help="Backfill a single game by game_id")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute timestamps but do not write to MotherDuck",
    )
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
        if args.game:
            game_date_row = conn.execute(
                "SELECT game_date FROM dim_games WHERE game_id = ?", [args.game]
            ).fetchone()
            if not game_date_row:
                logger.error("game_id %s not found in dim_games", args.game)
                raise SystemExit(1)
            games = [(args.game, game_date_row[0])]

        elif args.date:
            games = _games_for_date(date.fromisoformat(args.date), conn)

        elif args.since:
            games = _games_since(date.fromisoformat(args.since), conn)

        else:
            # Default: all games with tick coverage that are missing wall_clock_ts
            games = _games_needing_backfill(conn)

        if not games:
            logger.info("No games to backfill.")
            raise SystemExit(0)

        logger.info(
            "Backfilling wall_clock_ts for %d game(s)%s",
            len(games),
            " [DRY RUN]" if args.dry_run else "",
        )

        # Compute timestamps for all games (read-only phase)
        all_frames: list[pd.DataFrame] = []
        for game_id, game_date in games:
            try:
                df = _compute_timestamps_for_game(game_id, game_date, conn)
                if df is not None:
                    all_frames.append(df)
            except Exception:
                logger.exception("game %s: unexpected error — skipping", game_id)

        if not all_frames:
            logger.info("No timestamps computed — nothing to write.")
            raise SystemExit(0)

        all_updates = pd.concat(all_frames, ignore_index=True)
        total_rows = len(all_updates)
        n_games = all_updates["game_id"].nunique()

        if args.dry_run:
            for gid in all_updates["game_id"].unique():
                sample = all_updates[all_updates["game_id"] == gid].head(3)
                print(f"\n--- DRY RUN: game {gid} ---")
                print(sample.to_string(index=False))
            logger.info(
                "Dry run complete. %d rows across %d games — nothing written.",
                total_rows, n_games,
            )
            raise SystemExit(0)

        # Single write: batch INSERT into staging table + one CTAS rebuild
        # (MotherDuck does not support UPDATE FROM registered DataFrames;
        #  INSERT + CTAS is the only reliable write path)
        logger.info(
            "Writing %d rows across %d games via INSERT+CTAS...",
            total_rows, n_games,
        )
        _apply_timestamps(all_updates, conn)
        logger.info(
            "Done. %d rows updated across %d games.",
            total_rows, n_games,
        )

    finally:
        try:
            conn.close()
        except Exception:
            pass
