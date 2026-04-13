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
from data.ingestion.kalshi_historical_client import KalshiAuth
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
    key_id      = os.environ.get("API_KEY_ID")
    pem_content = os.environ.get("PRIVATE_RSA_KEY_PEM")
    pem_path    = os.environ.get("PRIVATE_RSA_KEY")

    if not key_id or (not pem_content and not pem_path):
        raise RuntimeError(
            "API_KEY_ID and either PRIVATE_RSA_KEY_PEM or PRIVATE_RSA_KEY must be set"
        )
    return KalshiAuth(key_id=key_id, private_key_pem=pem_content, private_key_path=pem_path)


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


def _run_pipeline_sync(game_date: date, games: list[GameInfo]) -> None:
    # Deferred import: keeps cold start fast and avoids circular import risk
    from data.ingestion.post_game_pipeline import run_post_game_pipeline
    run_post_game_pipeline(game_date, games)


async def run_daemon() -> None:
    """Main daemon loop — runs indefinitely, one iteration per day."""
    load_dotenv()
    auth = _build_auth()

    logger.info("Kalshi recorder daemon started")

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
            logger.info("Skipping %d already-finished game(s)", skipped)

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

        logger.info("Running post-game pipeline for %s", game_date.isoformat())
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _run_pipeline_sync, game_date, upcoming)
        except Exception:
            logger.exception("Post-game pipeline uncaught exception — continuing")

        tomorrow = game_date + timedelta(days=1)
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
