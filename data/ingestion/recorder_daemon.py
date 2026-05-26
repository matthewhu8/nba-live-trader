"""
Kalshi recorder daemon — always-on process for Fly.io.

Runs 24/7. Each day it:
  1. Wakes at 7 AM ET (12:00 UTC EST / 11:00 UTC EDT)
  2. Fetches today's NBA schedule
  3. Arms a KalshiRecorder coroutine per game, starting 2 min before tipoff
  4. Waits for all games to complete
  5. Sleeps until tomorrow's schedule window

Per-game recorders connect only to that game's markets (filtered by team codes),
so multiple concurrent games each get their own isolated WebSocket session.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.ingestion.game_schedule import GameInfo, get_todays_games, is_edt, save_schedule
from data.ingestion.kalshi_historical_client import KalshiAuth, kalshi_auth_from_env
from data.ingestion.kalshi_recorder import KalshiRecorder

logger = logging.getLogger(__name__)

SCHEDULE_LEAD_SECONDS       = 120   # start recorder this many seconds before tipoff
SCHEDULE_FETCH_HOUR_UTC_EST = 12    # 12:00 UTC = 7 AM ET (EST)
SCHEDULE_FETCH_HOUR_UTC_EDT = 11    # 11:00 UTC = 7 AM ET (EDT)
MAX_GAME_HOURS              = 4.0   # games older than this are considered finished
PIPELINE_HOUR_UTC_EST       = 8     # 3 AM ET (EST) = 08:00 UTC
PIPELINE_HOUR_UTC_EDT       = 7     # 3 AM ET (EDT) = 07:00 UTC


def _schedule_fetch_utc(today: date) -> datetime:
    """Return today's 7 AM ET moment in UTC."""
    hour = SCHEDULE_FETCH_HOUR_UTC_EDT if is_edt(today) else SCHEDULE_FETCH_HOUR_UTC_EST
    return datetime(today.year, today.month, today.day, hour, 0, 0, tzinfo=timezone.utc)


def _pipeline_trigger_utc(d: date) -> datetime:
    """Return 3 AM ET for the given date in UTC."""
    hour = PIPELINE_HOUR_UTC_EDT if is_edt(d) else PIPELINE_HOUR_UTC_EST
    return datetime(d.year, d.month, d.day, hour, 0, 0, tzinfo=timezone.utc)


def _build_auth() -> KalshiAuth:
    return kalshi_auth_from_env()


async def _run_game(auth: KalshiAuth, game: GameInfo) -> None:
    """Record ticks for a single game. Launched at tipoff - SCHEDULE_LEAD_SECONDS."""
    team_codes = (game.visitor_team, game.home_team)
    recorder   = KalshiRecorder(auth=auth, team_codes=team_codes, game_id=game.game_id)
    logger.info(
        "Starting recorder for %s @ %s (tipoff %s)",
        game.visitor_team, game.home_team, game.tipoff_utc.isoformat(),
    )
    await recorder.run()
    logger.info("Recorder finished for %s @ %s", game.visitor_team, game.home_team)


async def _schedule_games(auth: KalshiAuth, games: list[GameInfo]) -> None:
    """
    Arm one coroutine per game, each sleeping until tipoff - SCHEDULE_LEAD_SECONDS,
    then run all concurrently.
    """
    now = datetime.now(tz=timezone.utc)

    async def _delayed_game(game: GameInfo) -> None:
        start_at = game.tipoff_utc - timedelta(seconds=SCHEDULE_LEAD_SECONDS)
        delay    = (start_at - now).total_seconds()
        if delay > 0:
            logger.info(
                "Waiting %.0f seconds before starting %s @ %s",
                delay, game.visitor_team, game.home_team,
            )
            await asyncio.sleep(delay)
        await _run_game(auth, game)

    await asyncio.gather(*[_delayed_game(g) for g in games])


async def _sleep_until(target: datetime) -> None:
    """Sleep until a specific UTC datetime, logging the wait."""
    now   = datetime.now(tz=timezone.utc)
    delta = (target - now).total_seconds()
    if delta > 0:
        logger.info(
            "Sleeping %.1f hours until schedule fetch at %s UTC",
            delta / 3600, target.strftime("%H:%M"),
        )
        await asyncio.sleep(delta)


CATCHUP_DAYS = 7   # look back this many days for unprocessed games on startup


def _run_pipeline_sync(game_date: date, games: list[GameInfo]) -> None:
    from data.ingestion.post_game_pipeline import run_post_game_pipeline
    run_post_game_pipeline(game_date, games)


def _run_pregame_prefill_sync(tomorrow: date) -> None:
    """
    Compute and insert features.pregame rows for tomorrow's games so
    load_pregame() finds real data at tip-off time instead of zeroed defaults.

    Runs at 3 AM ET after tonight's ratings have been updated (Phase 3 already
    wrote fresh player_ratings / lineup_ratings / team_ratings), so the pregame
    features are computed with the most current ratings available.
    """
    from data.ingestion.post_game_pipeline import (
        _md_connect, upsert_dim_games, build_pregame_features_phase,
    )
    games = get_todays_games(tomorrow)
    if not games:
        logger.info("Pregame prefill: no games found for %s", tomorrow.isoformat())
        return
    conn = _md_connect()
    try:
        # Ensure dim_games rows exist for tomorrow's games so the pregame
        # feature queries can join on game_id → team tricodes.
        upsert_dim_games(games, tomorrow)
    finally:
        conn.close()
    build_pregame_features_phase(games, tomorrow)
    logger.info(
        "Pregame prefill: inserted features for %d game(s) on %s",
        len(games), tomorrow.isoformat(),
    )


def _catchup_missed_games(lookback_days: int = CATCHUP_DAYS) -> None:
    """
    On daemon startup, check the last `lookback_days` days for games that are
    in dim_games but missing from possession_flat, and re-run the pipeline for
    each missed date.  Idempotent — safe to run every startup.
    """
    from data.ingestion.post_game_pipeline import _md_connect, _get_unprocessed_games
    today = date.today()
    conn = _md_connect()
    try:
        start = (today - timedelta(days=lookback_days)).isoformat()
        rows = conn.execute(
            "SELECT game_id, game_date, home_team, away_team FROM dim_games "
            "WHERE game_date >= ? ORDER BY game_date",
            [start],
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return

    all_ids = [r[0] for r in rows]
    unprocessed = set(_get_unprocessed_games(all_ids))
    if not unprocessed:
        logger.info("Catch-up: all games from last %d days already processed", lookback_days)
        return

    # Group unprocessed game_ids back by date
    from collections import defaultdict
    by_date: dict[date, list] = defaultdict(list)
    for game_id, game_date_raw, home_team, away_team in rows:
        if game_id not in unprocessed:
            continue
        gd = game_date_raw if isinstance(game_date_raw, date) else date.fromisoformat(str(game_date_raw))
        # Reconstruct a minimal GameInfo — only game_id, home_team, visitor_team used by pipeline
        by_date[gd].append(GameInfo(
            game_id      = game_id,
            home_team    = home_team,
            visitor_team = away_team,
            tipoff_et    = "",
            tipoff_utc   = datetime(gd.year, gd.month, gd.day, tzinfo=timezone.utc),
            game_status  = "3",
        ))

    logger.info(
        "Catch-up: found %d unprocessed game(s) across %d date(s) — backfilling",
        len(unprocessed), len(by_date),
    )
    for gd in sorted(by_date):
        logger.info("Catch-up: running pipeline for %s (%d games)", gd.isoformat(), len(by_date[gd]))
        try:
            _run_pipeline_sync(gd, by_date[gd])
        except Exception:
            logger.exception("Catch-up: pipeline failed for %s — continuing", gd.isoformat())


async def run_daemon() -> None:
    """Main daemon loop — runs indefinitely, one iteration per day."""
    load_dotenv()
    auth = _build_auth()

    logger.info("Kalshi recorder daemon started")

    # On startup, backfill any games missed while the daemon was down
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _catchup_missed_games)
    except Exception:
        logger.exception("Startup catch-up failed — continuing")

    while True:
        today    = date.today()
        fetch_at = _schedule_fetch_utc(today) # 7 AM ET
        now      = datetime.now(tz=timezone.utc)

        if now < fetch_at:
            # Before today's schedule window — sleep until 7 AM ET
            await _sleep_until(fetch_at)
            game_date = date.today()
        else:
            # Already past the schedule window (e.g. daemon deployed mid-afternoon).
            # Fetch immediately — games tonight may not have tipped yet.
            game_date = today
            logger.info(
                "Past schedule window — fetching today's games immediately (%s)",
                game_date.isoformat(),
            )

        logger.info("Fetching NBA schedule for %s", game_date.isoformat())
        games = get_todays_games(game_date)

        if not games:
            logger.info("No games today — sleeping until tomorrow's schedule window")
            tomorrow = game_date + timedelta(days=1)
            await _sleep_until(_schedule_fetch_utc(tomorrow))
            continue

        # Drop games that have already tipped off and settled — no point recording
        now = datetime.now(tz=timezone.utc)
        upcoming = [g for g in games if g.tipoff_utc > now - timedelta(hours=MAX_GAME_HOURS)]
        skipped  = len(games) - len(upcoming)
        if skipped:
            logger.info("Skipping %d already-finished game(s) for recording", skipped)

        if not upcoming:
            logger.info("All games for %s already finished — moving to tomorrow", game_date.isoformat())
            tomorrow = game_date + timedelta(days=1)
            await _sleep_until(_schedule_fetch_utc(tomorrow))
            continue

        save_schedule(upcoming, game_date)
        logger.info(
            "Arming %d game(s) — first tipoff %s UTC",
            len(upcoming), upcoming[0].tipoff_utc.strftime("%H:%M"),
        )

        await _schedule_games(auth, upcoming)
        logger.info("All games complete for %s", game_date.isoformat())

        # Sleep until 3 AM ET, then run post-game pipeline.
        # If games finish after 3 AM (rare), sleep is skipped and pipeline runs immediately.
        pipeline_at = _pipeline_trigger_utc(game_date)
        now = datetime.now(tz=timezone.utc)
        if now < pipeline_at:
            await asyncio.sleep((pipeline_at - now).total_seconds())

        # Pipeline: process ALL of tonight's games, not just the ones we recorded.
        # Re-fetch the full list so games that were already finished at 7 AM still
        # get their possession_flat + ratings rows updated.
        logger.info("Running post-game pipeline for %s", game_date.isoformat())
        all_games_tonight = get_todays_games(game_date) or games
        try:
            await loop.run_in_executor(None, _run_pipeline_sync, game_date, all_games_tonight)
        except Exception:
            logger.exception("Post-game pipeline uncaught exception — continuing")

        # Prefill pregame features for tomorrow so StartGame has real data at tip-off.
        tomorrow = game_date + timedelta(days=1)
        logger.info("Prefilling pregame features for %s", tomorrow.isoformat())
        try:
            await loop.run_in_executor(None, _run_pregame_prefill_sync, tomorrow)
        except Exception:
            logger.exception("Pregame prefill failed — continuing")

        await _sleep_until(_schedule_fetch_utc(tomorrow))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_task = None

    def _handle_signal() -> None:
        logger.info("Signal received — shutting down daemon")
        if main_task and not main_task.done():
            main_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    try:
        main_task = loop.create_task(run_daemon())
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
