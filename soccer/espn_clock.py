"""
Live match-minute from ESPN's public scoreboard (no key required).

ESPN's hidden soccer scoreboard exposes a `displayClock` like "67'" or "90'+14'"
plus a state of pre/in/post. We match the game by the two FIFA team codes (which
line up with Kalshi's tickers — ENG/CRO, POR/COD, GHA/PAN, ...) and read the
absolute match minute, including stoppage time. This drives break timing more
precisely than the wall-clock estimate. If ESPN can't be reached or hasn't found
the game live, the runner falls back to `MatchClock`.

Results are cached briefly so a 15s poll loop doesn't hammer ESPN.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"

_BASE_RE = re.compile(r"(\d+)")
_EXTRA_RE = re.compile(r"\+\s*(\d+)")


def parse_espn_minute(status: dict) -> float | None:
    """
    Absolute match minute from an ESPN competition `status`, or None when the
    match is not currently in play. "90'+14'" -> 104, "67'" -> 67.
    """
    state = status.get("type", {}).get("state")
    if state != "in":
        return None

    clock = str(status.get("displayClock") or "")
    base_m = _BASE_RE.search(clock)
    if base_m is None:
        return None

    minute = int(base_m.group(1))
    extra_m = _EXTRA_RE.search(clock)
    if extra_m is not None:
        minute += int(extra_m.group(1))
    return float(minute)


@dataclass
class EspnClock:
    team_codes: tuple[str, str]
    league: str = "fifa.world"
    cache_ttl: float = 10.0
    timeout: float = 8.0
    last_state: str | None = None  # "pre"|"in"|"post"|None(not found) for matched event
    _cache: tuple[float, dict] | None = field(default=None, repr=False)

    def live_minute(self, now=None) -> float | None:
        data = self._scoreboard()
        self.last_state = None
        if data is None:
            return None

        a, b = (c.upper() for c in self.team_codes)
        for ev in data.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            abbrs = {
                c.get("team", {}).get("abbreviation", "").upper()
                for c in comp.get("competitors", [])
            }
            if {a, b} <= abbrs:
                status = comp.get("status", {})
                self.last_state = status.get("type", {}).get("state")
                return parse_espn_minute(status)
        return None

    def _scoreboard(self) -> dict | None:
        now = time.monotonic()
        if self._cache is not None and now - self._cache[0] < self.cache_ttl:
            return self._cache[1]
        url = SCOREBOARD_URL.format(league=self.league)
        try:
            resp = requests.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("ESPN scoreboard fetch failed: %s", exc)
            return None
        if resp.status_code != 200:
            logger.debug("ESPN scoreboard -> %d", resp.status_code)
            return None
        data = resp.json()
        self._cache = (now, data)
        return data
