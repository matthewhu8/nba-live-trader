"""
NBA game schedule fetcher.

Fetches today's games via nba_api ScoreboardV3 and displays tipoff times
in ET and UTC. Useful for knowing when to start the Kalshi recorder.

Usage:
    python data/ingestion/game_schedule.py              # today
    python data/ingestion/game_schedule.py --date 2026-03-25
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _nth_sunday(year: int, month: int, n: int) -> date:
    """Return the nth Sunday (1-indexed) of the given year/month."""
    first = date(year, month, 1)
    days_until_sunday = (6 - first.weekday()) % 7
    first_sunday = first + timedelta(days=days_until_sunday)
    return first_sunday + timedelta(weeks=n - 1)


def is_edt(d: date) -> bool:
    """
    Return True if date falls in EDT (clocks spring forward).
    EDT spans: second Sunday in March → first Sunday in November.
    """
    year = d.year
    edt_start = _nth_sunday(year, 3, 2)   # 2nd Sunday in March
    edt_end   = _nth_sunday(year, 11, 1)  # 1st Sunday in November
    return edt_start <= d < edt_end


@dataclass
class GameInfo:
    game_id:      str
    visitor_team: str
    home_team:    str
    tipoff_et:    str
    tipoff_utc:   datetime
    game_status:  str


def _format_et(tipoff_utc: datetime, game_date: date) -> str:
    et_offset_hours = -4 if is_edt(game_date) else -5
    et_time = tipoff_utc.astimezone(timezone(timedelta(hours=et_offset_hours)))
    suffix = "EDT" if et_offset_hours == -4 else "EST"
    return et_time.strftime("%-I:%M %p ") + suffix


def get_todays_games(game_date: date | None = None) -> list[GameInfo]:
    if game_date is None:
        game_date = date.today()
    try:
        from nba_api.stats.endpoints import scoreboardv3
        board = scoreboardv3.ScoreboardV3(
            game_date=game_date.strftime("%m/%d/%Y"),
            league_id="00",
        )
        raw_games = board.get_dict()["scoreboard"]["games"]
    except Exception as exc:
        logger.warning("nba_api ScoreboardV3 failed: %s", exc)
        return []

    games: list[GameInfo] = []
    for g in raw_games:
        game_id    = str(g.get("gameId", ""))
        gamecode   = str(g.get("gameCode", ""))
        status_id  = str(g.get("gameStatus", "1"))

        visitor_team, home_team = "???", "???"
        if "/" in gamecode:
            code_part = gamecode.split("/", 1)[1]
            if len(code_part) == 6:
                visitor_team = code_part[:3]
                home_team    = code_part[3:]

        tipoff_utc: datetime | None = None
        game_time_utc = g.get("gameTimeUTC", "")
        if game_time_utc:
            try:
                tipoff_utc = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
            except ValueError:
                pass

        if tipoff_utc is None:
            tipoff_utc = datetime(game_date.year, game_date.month, game_date.day, tzinfo=timezone.utc)
            tipoff_et_str = str(g.get("gameStatusText", ""))
        else:
            tipoff_et_str = _format_et(tipoff_utc, game_date)

        games.append(GameInfo(
            game_id      = game_id,
            visitor_team = visitor_team,
            home_team    = home_team,
            tipoff_et    = tipoff_et_str,
            tipoff_utc   = tipoff_utc,
            game_status  = status_id,
        ))

    games.sort(key=lambda g: g.tipoff_utc)
    return games


def display_schedule(games: list[GameInfo]) -> None:
    line = "=" * 55
    logger.info(line)
    logger.info("NBA GAMES TODAY — %d game%s", len(games), "s" if len(games) != 1 else "")
    logger.info(line)
    if not games:
        logger.info("  No games scheduled.")
        logger.info(line)
        return
    for g in games:
        utc_str = g.tipoff_utc.strftime("%H:%M UTC")
        logger.info("  %-14s  %s @ %s  (%s)", g.tipoff_et, g.visitor_team, g.home_team, utc_str)
    first_tip = games[0].tipoff_utc
    if games[0].game_status == "1":
        rec_start = first_tip - timedelta(minutes=10)
        rec_et    = _format_et(rec_start, first_tip.date())
        logger.info("-" * 55)
        logger.info(
            "Recommended recorder start: %s  (%s)",
            rec_start.strftime("%H:%M UTC"), rec_et,
        )
    logger.info(line)


def save_schedule(games: list[GameInfo], game_date: date) -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    out_path = logs_dir / f"game_schedule_{game_date.isoformat()}.json"
    payload = [
        {**asdict(g), "tipoff_utc": g.tipoff_utc.isoformat()}
        for g in games
    ]
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Schedule saved to %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Fetch today's NBA game schedule")
    parser.add_argument("--date", default=None, help="Date in YYYY-MM-DD format (default: today)")
    args = parser.parse_args()

    game_date = date.today()
    if args.date:
        try:
            game_date = date.fromisoformat(args.date)
        except ValueError:
            logger.error("Invalid date format: %s — expected YYYY-MM-DD", args.date)
            sys.exit(1)

    games = get_todays_games(game_date)
    display_schedule(games)
    if games:
        save_schedule(games, game_date)


if __name__ == "__main__":
    main()
