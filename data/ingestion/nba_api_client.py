"""
NBA play-by-play ingestion for recent games.

Fetches PBP via nba_api.stats.endpoints.PlayByPlayV3, parses into
possession/foul/substitution/timeout events, and appends to raw parquets.

Usage:
    python -m data.ingestion.nba_api_client               # all games missing from parquets
    python -m data.ingestion.nba_api_client --since 2026-03-13
    python -m data.ingestion.nba_api_client --game 0022501039
"""

import argparse
import hashlib
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

RAW_DIR       = Path("data/raw")
PBP_CACHE_DIR = RAW_DIR / "pbp_cache"
POSS_PATH     = RAW_DIR / "possessions_202526.parquet"
FOUL_PATH     = RAW_DIR / "foul_events_202526.parquet"
SUB_PATH      = RAW_DIR / "substitution_events_202526.parquet"
TO_PATH       = RAW_DIR / "timeout_events_202526.parquet"

# Team nickname → tricode for parsing timeout descriptions
_NICKNAME_TO_TRICODE: dict[str, str] = {
    "hawks": "ATL", "celtics": "BOS", "nets": "BKN", "hornets": "CHA",
    "bulls": "CHI", "cavaliers": "CLE", "mavs": "DAL", "mavericks": "DAL",
    "nuggets": "DEN", "pistons": "DET", "warriors": "GSW", "rockets": "HOU",
    "pacers": "IND", "clippers": "LAC", "lakers": "LAL", "grizzlies": "MEM",
    "heat": "MIA", "bucks": "MIL", "timberwolves": "MIN", "wolves": "MIN",
    "pelicans": "NOP", "knicks": "NYK", "thunder": "OKC", "magic": "ORL",
    "76ers": "PHI", "sixers": "PHI", "suns": "PHX", "trail blazers": "POR",
    "blazers": "POR", "kings": "SAC", "spurs": "SAS", "raptors": "TOR",
    "jazz": "UTA", "wizards": "WAS",
}

# PyArrow schemas matching existing parquets exactly
_POSS_SCHEMA = pa.schema([
    pa.field("game_id",            pa.string()),
    pa.field("possession_id",      pa.int32()),
    pa.field("period",             pa.int32()),
    pa.field("game_clock_secs",    pa.float32()),
    pa.field("period_wall_clock",  pa.string()),   # not available from PBP; kept for schema compat
    pa.field("possessing_team",    pa.string()),
    pa.field("team_scored",        pa.string()),
    pa.field("outcome",            pa.string()),
    pa.field("home_score",         pa.int32()),
    pa.field("away_score",         pa.int32()),
    pa.field("points",             pa.int32()),
    pa.field("shot_value",         pa.int32()),
    pa.field("shot_x",             pa.int32()),
    pa.field("shot_y",             pa.int32()),
    pa.field("shot_distance",      pa.int32()),
    pa.field("shot_type",          pa.string()),
    pa.field("play_type",          pa.string()),
    pa.field("player_id",          pa.int64()),
    pa.field("player_name",        pa.string()),
    pa.field("home_lineup_id",     pa.string()),
    pa.field("away_lineup_id",     pa.string()),
])

_FOUL_SCHEMA = pa.schema([
    pa.field("game_id",         pa.string()),
    pa.field("action_number",   pa.int32()),
    pa.field("period",          pa.int32()),
    pa.field("game_clock_secs", pa.float32()),
    pa.field("team_tricode",    pa.string()),
    pa.field("player_id",       pa.int64()),
    pa.field("player_name",     pa.string()),
    pa.field("foul_type",       pa.string()),
    pa.field("description",     pa.string()),
])

_SUB_SCHEMA = pa.schema([
    pa.field("game_id",         pa.string()),
    pa.field("action_number",   pa.int32()),
    pa.field("period",          pa.int32()),
    pa.field("game_clock_secs", pa.float32()),
    pa.field("team_tricode",    pa.string()),
    pa.field("player_out_id",   pa.int64()),
    pa.field("player_out_name", pa.string()),
    pa.field("player_in_id",    pa.int64()),
    pa.field("player_in_name",  pa.string()),
    pa.field("home_lineup_id",  pa.string()),
    pa.field("away_lineup_id",  pa.string()),
])

_TO_SCHEMA = pa.schema([
    pa.field("game_id",                  pa.string()),
    pa.field("action_number",            pa.int32()),
    pa.field("period",                   pa.int32()),
    pa.field("game_clock_secs",          pa.float32()),
    pa.field("team_tricode",             pa.string()),
    pa.field("timeout_type",             pa.string()),
    pa.field("full_timeouts_remaining",  pa.int32()),
])


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_clock(clock_str: str) -> float:
    m = re.match(r"PT(\d+)M([\d.]+)S", clock_str or "")
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val) if val is not None and str(val).strip() not in ("", "nan") else default
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any) -> float | None:
    try:
        v = float(val)
        return None if pd.isna(v) else v
    except (ValueError, TypeError):
        return None


def _normalize(name: str) -> str:
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode().lower()


def _lineup_id(player_ids: frozenset[int]) -> str:
    key = ",".join(str(p) for p in sorted(player_ids))
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Lineup tracker (mirrors _LineupTracker in player_rapm.py)
# ---------------------------------------------------------------------------

class _LineupTracker:
    def __init__(self) -> None:
        self._home: set[int] = set()
        self._away: set[int] = set()
        self._locked = False

    def observe(self, player_id: int, location: str) -> None:
        if self._locked:
            return
        if location == "h" and len(self._home) < 5:
            self._home.add(player_id)
        elif location == "v" and len(self._away) < 5:
            self._away.add(player_id)

    def lock(self) -> None:
        self._locked = True

    def substitute(self, out_id: int, in_id: int, location: str) -> None:
        if not self._locked:
            self.lock()
        pool = self._home if location == "h" else self._away
        pool.discard(out_id)
        pool.add(in_id)

    def is_ready(self) -> bool:
        return len(self._home) == 5 and len(self._away) == 5

    def home_id(self) -> str:
        return _lineup_id(frozenset(self._home)) if self._home else ""

    def away_id(self) -> str:
        return _lineup_id(frozenset(self._away)) if self._away else ""


def _build_name_map(df: pd.DataFrame) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for _, row in df.iterrows():
        pid  = _safe_int(row.get("personId"))
        name = str(row.get("playerName", "") or "").strip()
        if pid and name:
            norm = _normalize(name)
            mapping[norm] = pid
            parts = norm.split()
            if len(parts) > 1:
                mapping[parts[-1]] = pid
    return mapping


def _parse_player_in(description: str, name_map: dict[str, int]) -> int:
    m = re.match(r"SUB:\s+(.+?)\s+FOR\s+", description or "", re.IGNORECASE)
    if not m:
        return -1
    name = m.group(1).strip()
    norm = _normalize(name)
    pid  = name_map.get(norm, -1)
    if pid == -1:
        pid = name_map.get(norm.split()[-1], -1)
    return pid


def _team_tricode_from_timeout(
    description: str,
    home_tri: str,
    away_tri: str,
) -> str:
    """Extract tricode from 'SUNS Timeout: Regular (Full 1 Short 0)'."""
    m = re.match(r"([A-Za-z\s\d]+?)\s+Timeout:", description or "")
    if not m:
        return ""
    name  = m.group(1).strip().lower()
    # Try direct static map
    tri = _NICKNAME_TO_TRICODE.get(name, "")
    if tri:
        return tri
    # Partial match restricted to this game's teams
    for nick, code in _NICKNAME_TO_TRICODE.items():
        if nick in name and code in (home_tri, away_tri):
            return code
    return ""


# ---------------------------------------------------------------------------
# Core game parser
# ---------------------------------------------------------------------------

def _parse_score(val: Any) -> int:
    try:
        return int(val) if val is not None and str(val).strip() not in ("", "nan") else -1
    except (ValueError, TypeError):
        return -1


def parse_game_to_events(
    game_id: str,
    pbp_df: pd.DataFrame,
    home_tri: str,
    away_tri: str,
) -> dict[str, pd.DataFrame]:
    """
    Parse raw PlayByPlayV3 DataFrame into 4 structured DataFrames.

    Returns dict: {"possessions": df, "fouls": df, "substitutions": df, "timeouts": df}
    """
    df = pbp_df.copy()
    df["_clock_secs"] = df["clock"].apply(_parse_clock)

    # Chronological order: period asc, time desc (more time remaining = earlier), action# asc
    df = (
        df.sort_values(["period", "_clock_secs", "actionNumber"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    name_map = _build_name_map(df)
    tracker  = _LineupTracker()

    # Starter inference: scan period-1 events until we have 10 players
    for _, row in df.iterrows():
        if tracker.is_ready():
            tracker.lock()
            break
        action_type = str(row.get("actionType", "") or "")
        period      = _safe_int(row.get("period"), 1)
        player_id   = _safe_int(row.get("personId"))
        location    = str(row.get("location", "") or "")

        if period > 1 or action_type == "Substitution":
            tracker.lock()
            break
        if player_id and location in ("h", "v"):
            tracker.observe(player_id, location)

    possessions:   list[dict] = []
    fouls:         list[dict] = []
    substitutions: list[dict] = []
    timeouts:      list[dict] = []

    # Running score — updated whenever scoreHome/scoreAway is populated
    home_score = 0
    away_score = 0
    possession_id = 1

    # Possession state
    possessing_loc    = "h"      # "h" or "v" — which team has the ball
    missed_shot_loc   = ""       # set when a missed shot is pending a rebound
    in_ft_seq         = False    # True while inside a FT sequence
    ft_total          = 0        # total FTs in current sequence
    ft_made           = 0        # FTs made so far in sequence
    ft_player_id      = 0
    ft_player_name    = ""
    ft_clock          = 0.0
    ft_period         = 1

    def _end_possession(
        outcome: str,
        clock: float,
        period: int,
        poss_loc: str,
        scored_loc: str = "",
        points: int = 0,
        shot_value: int = 0,
        shot_x: float | None = None,
        shot_y: float | None = None,
        shot_distance: float | None = None,
        shot_type: str = "",
        play_type: str = "",
        player_id: int = 0,
        player_name: str = "",
    ) -> None:
        nonlocal possession_id

        _PLAY_TYPE_MAP = {
            "made_shot": "Made Shot",
            "turnover":  "Turnover",
            "free_throw": "Free Throw",
            "stop":       "Stop",
        }
        possessions.append({
            "game_id":            game_id,
            "possession_id":      possession_id,
            "period":             period,
            "game_clock_secs":    clock,
            "period_wall_clock":  "",   # wall clock not available from PlayByPlayV3
            "possessing_team":    "home" if poss_loc == "h" else "away",
            "team_scored":     ("home" if scored_loc == "h" else "away") if scored_loc else "",
            "outcome":         outcome,
            "home_score":      home_score,
            "away_score":      away_score,
            "points":          points,
            "shot_value":      shot_value,
            "shot_x":          shot_x,
            "shot_y":          shot_y,
            "shot_distance":   shot_distance,
            "shot_type":       shot_type,
            "play_type":       play_type or _PLAY_TYPE_MAP.get(outcome, outcome),
            "player_id":       player_id or None,
            "player_name":     player_name,
            "home_lineup_id":  tracker.home_id(),
            "away_lineup_id":  tracker.away_id(),
        })
        possession_id += 1

    for _, row in df.iterrows():
        action_type = str(row.get("actionType", "") or "")
        sub_type    = str(row.get("subType",    "") or "")
        description = str(row.get("description", "") or "")
        location    = str(row.get("location",   "") or "")
        period      = _safe_int(row.get("period"), 1)
        action_num  = _safe_int(row.get("actionNumber"))
        clock       = float(row["_clock_secs"])
        player_id   = _safe_int(row.get("personId"))
        player_name = str(row.get("playerName", "") or "").strip()
        team_tri    = str(row.get("teamTricode", "") or "").strip()

        # Update running score from any event that has it
        hs = _parse_score(row.get("scoreHome"))
        as_ = _parse_score(row.get("scoreAway"))
        if hs >= 0:
            home_score = hs
        if as_ >= 0:
            away_score = as_

        # ---------------------------------------------------------------
        # Jump Ball → determines initial possession
        # ---------------------------------------------------------------
        if action_type == "Jump Ball" and not possessions:
            # location = team that won the tip
            if location in ("h", "v"):
                possessing_loc = location

        # ---------------------------------------------------------------
        # Period end → flush any pending possession as stop
        # ---------------------------------------------------------------
        elif action_type == "period" and sub_type == "end":
            if missed_shot_loc:
                _end_possession(
                    outcome="stop",
                    clock=clock,
                    period=period,
                    poss_loc=missed_shot_loc,
                    play_type="Stop",
                )
                missed_shot_loc = ""
            in_ft_seq = False

        # ---------------------------------------------------------------
        # Period start → reset FT/missed state, keep possessing team
        # ---------------------------------------------------------------
        elif action_type == "period" and sub_type in ("start", ""):
            missed_shot_loc = ""
            in_ft_seq       = False

        # ---------------------------------------------------------------
        # Substitution → update lineup tracker, record event
        # ---------------------------------------------------------------
        elif action_type == "Substitution":
            out_id   = player_id
            out_name = player_name
            in_id    = _parse_player_in(description, name_map)
            in_name  = ""
            if in_id != -1:
                # recover name from name_map reverse lookup is expensive; use description
                m = re.match(r"SUB:\s+(.+?)\s+FOR\s+", description, re.IGNORECASE)
                in_name = m.group(1).strip() if m else ""
                tracker.substitute(out_id, in_id, location)

            substitutions.append({
                "game_id":         game_id,
                "action_number":   action_num,
                "period":          period,
                "game_clock_secs": clock,
                "team_tricode":    team_tri,
                "player_out_id":   out_id or None,
                "player_out_name": out_name,
                "player_in_id":    in_id if in_id != -1 else None,
                "player_in_name":  in_name,
                "home_lineup_id":  tracker.home_id(),
                "away_lineup_id":  tracker.away_id(),
            })

        # ---------------------------------------------------------------
        # Foul → record event
        # ---------------------------------------------------------------
        elif action_type == "Foul":
            fouls.append({
                "game_id":         game_id,
                "action_number":   action_num,
                "period":          period,
                "game_clock_secs": clock,
                "team_tricode":    team_tri,
                "player_id":       player_id or None,
                "player_name":     player_name,
                "foul_type":       sub_type,
                "description":     description,
            })

        # ---------------------------------------------------------------
        # Timeout → record event
        # ---------------------------------------------------------------
        elif action_type == "Timeout":
            tri = _team_tricode_from_timeout(description, home_tri, away_tri)
            m_full = re.search(r"Full\s+(\d+)", description, re.IGNORECASE)
            full_remaining = int(m_full.group(1)) if m_full else -1
            timeouts.append({
                "game_id":                 game_id,
                "action_number":           action_num,
                "period":                  period,
                "game_clock_secs":         clock,
                "team_tricode":            tri,
                "timeout_type":            sub_type or "Regular",
                "full_timeouts_remaining": full_remaining,
            })

        # ---------------------------------------------------------------
        # Made Shot → end possession
        # ---------------------------------------------------------------
        elif action_type == "Made Shot":
            missed_shot_loc = ""
            in_ft_seq       = False
            shot_val        = _safe_int(row.get("shotValue"), 2)
            _end_possession(
                outcome="made_shot",
                clock=clock,
                period=period,
                poss_loc=location or possessing_loc,
                scored_loc=location or possessing_loc,
                points=shot_val,
                shot_value=shot_val,
                shot_x=_safe_float(row.get("xLegacy")),
                shot_y=_safe_float(row.get("yLegacy")),
                shot_distance=_safe_float(row.get("shotDistance")),
                shot_type=str(row.get("subType", "") or ""),
                play_type="Made Shot",
                player_id=player_id,
                player_name=player_name,
            )
            # Next possession belongs to other team
            possessing_loc = "v" if (location or possessing_loc) == "h" else "h"

        # ---------------------------------------------------------------
        # Missed Shot → wait for rebound
        # ---------------------------------------------------------------
        elif action_type == "Missed Shot":
            missed_shot_loc = location or possessing_loc

        # ---------------------------------------------------------------
        # Free Throw
        # ---------------------------------------------------------------
        elif action_type == "Free Throw":
            made_ft  = str(row.get("shotResult", "") or "").lower() == "made"
            # Technical FTs have no "N of M" — treat as 1 of 1
            if sub_type == "Free Throw Technical":
                ft_n, ft_m = 1, 1
            else:
                ft_match = re.search(r"(\d+)\s+of\s+(\d+)", description, re.IGNORECASE)
                if not ft_match:
                    continue
                ft_n = int(ft_match.group(1))
                ft_m = int(ft_match.group(2))

            if ft_n == 1:
                # Start of FT sequence
                in_ft_seq      = True
                ft_total       = ft_m
                ft_made        = 1 if made_ft else 0
                ft_player_id   = player_id
                ft_player_name = player_name
                ft_clock       = clock
                ft_period      = period
                ft_loc         = location or possessing_loc
            else:
                ft_made += 1 if made_ft else 0

            if ft_n == ft_m:
                # Last FT in sequence → end possession
                in_ft_seq = False
                _end_possession(
                    outcome="free_throw",
                    clock=ft_clock,
                    period=ft_period,
                    poss_loc=ft_loc,
                    scored_loc=ft_loc if ft_made > 0 else "",
                    points=ft_made,
                    shot_value=1,
                    play_type="Free Throw",
                    player_id=ft_player_id,
                    player_name=ft_player_name,
                )
                # Next possession to other team
                possessing_loc  = "v" if ft_loc == "h" else "h"
                missed_shot_loc = ""

        # ---------------------------------------------------------------
        # Turnover → end possession
        # ---------------------------------------------------------------
        elif action_type == "Turnover":
            missed_shot_loc = ""
            in_ft_seq       = False
            _end_possession(
                outcome="turnover",
                clock=clock,
                period=period,
                poss_loc=location or possessing_loc,
                play_type="Turnover",
                player_id=player_id,
                player_name=player_name,
            )
            possessing_loc = "v" if (location or possessing_loc) == "h" else "h"

        # ---------------------------------------------------------------
        # Violation → scoring (goaltending) or turnover (kicked ball, etc.)
        # ---------------------------------------------------------------
        elif action_type == "Violation":
            if sub_type == "Defensive Goaltending":
                # Basket counts — score already updated from scoreHome/scoreAway above.
                # The preceding Missed Shot pending is resolved as a made possession.
                _end_possession(
                    outcome="made_shot",
                    clock=clock,
                    period=period,
                    poss_loc=missed_shot_loc or possessing_loc,
                    scored_loc=missed_shot_loc or possessing_loc,
                    points=2,
                    shot_value=2,
                    play_type="Made Shot",
                )
                possessing_loc  = "v" if (missed_shot_loc or possessing_loc) == "h" else "h"
                missed_shot_loc = ""
            else:
                # Kicked ball, lane violation → turnover
                missed_shot_loc = ""
                _end_possession(
                    outcome="turnover",
                    clock=clock,
                    period=period,
                    poss_loc=location or possessing_loc,
                    play_type="Turnover",
                )
                possessing_loc = "v" if (location or possessing_loc) == "h" else "h"

        # ---------------------------------------------------------------
        # Rebound → may end possession (defensive) or not (offensive)
        # ---------------------------------------------------------------
        elif action_type == "Rebound":
            if not missed_shot_loc:
                continue
            # Determine if offensive or defensive
            reb_loc = location or possessing_loc
            if reb_loc != missed_shot_loc:
                # Defensive rebound → stop
                _end_possession(
                    outcome="stop",
                    clock=clock,
                    period=period,
                    poss_loc=missed_shot_loc,
                    play_type="Stop",
                )
                possessing_loc  = reb_loc
                missed_shot_loc = ""
            else:
                # Offensive rebound → same team keeps possession, clear missed_shot
                missed_shot_loc = ""

    return {
        "possessions":   pd.DataFrame(possessions),
        "fouls":         pd.DataFrame(fouls),
        "substitutions": pd.DataFrame(substitutions),
        "timeouts":      pd.DataFrame(timeouts),
    }


# ---------------------------------------------------------------------------
# Fetch and cache
# ---------------------------------------------------------------------------

def fetch_and_cache_game(game_id: str) -> pd.DataFrame:
    """Fetch PBP from nba_api and cache to disk. Returns cached if already exists."""
    from nba_api.stats.endpoints import playbyplayv3

    cache_path = PBP_CACHE_DIR / f"{game_id}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    logger.info("Fetching PBP: %s", game_id)
    time.sleep(0.6)  # nba_api rate limit
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, end_period=10)
    df  = pbp.get_data_frames()[0]
    PBP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("  Cached %d events", len(df))
    return df


# ---------------------------------------------------------------------------
# Parquet append helpers
# ---------------------------------------------------------------------------

def _df_to_pa_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Convert DataFrame to PyArrow table with schema coercion."""
    import numpy as np
    arrays: list[pa.Array] = []
    for field in schema:
        col = df[field.name] if field.name in df.columns else pd.Series([None] * len(df))
        # NaN → None for non-float types so PyArrow can handle nullable integers
        if pa.types.is_integer(field.type) or pa.types.is_boolean(field.type):
            values = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                      for v in col.tolist()]
        else:
            values = col.tolist()
        arrays.append(pa.array(values, type=field.type))
    return pa.table(dict(zip([f.name for f in schema], arrays)), schema=schema)


def _append_to_parquet(new_df: pd.DataFrame, path: Path, schema: pa.Schema) -> None:
    if new_df.empty:
        return
    if path.exists():
        existing = pq.read_table(path)
        new_table = _df_to_pa_table(new_df, schema)
        combined  = pa.concat_tables([existing, new_table])
        pq.write_table(combined, path)
    else:
        pq.write_table(_df_to_pa_table(new_df, schema), path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def update_raw_parquets(since_date: str | None = None, game_id: str | None = None) -> int:
    """
    Fetch PBP for games missing from possessions parquet and append to raw parquets.

    Args:
        since_date: Only process games on or after this date (YYYY-MM-DD).
        game_id:    Process a single specific game.

    Returns:
        Number of games successfully parsed.
    """
    conn = duckdb.connect("kalshi_trading.duckdb", read_only=True)
    games_df = conn.execute("SELECT game_id, game_date, home_team, away_team FROM dim_games").df()
    conn.close()

    # Games already in possessions parquet
    existing_game_ids: set[str] = set()
    if POSS_PATH.exists():
        poss_existing = pd.read_parquet(POSS_PATH, columns=["game_id"])
        existing_game_ids = set(poss_existing["game_id"].unique())

    if game_id:
        target_games = games_df[games_df["game_id"] == game_id]
    else:
        target_games = games_df[~games_df["game_id"].isin(existing_game_ids)]
        if since_date:
            target_games = target_games[target_games["game_date"] >= since_date]
        target_games = target_games.sort_values("game_date")

    if target_games.empty:
        logger.info("No games to process")
        return 0

    logger.info("Processing %d games...", len(target_games))

    added = 0
    for _, game_row in target_games.iterrows():
        gid      = str(game_row["game_id"])
        home_tri = str(game_row.get("home_team", "") or "")
        away_tri = str(game_row.get("away_team", "") or "")

        try:
            pbp_df = fetch_and_cache_game(gid)
            if pbp_df.empty:
                logger.warning("  %s: empty PBP — skipping", gid)
                continue

            events = parse_game_to_events(gid, pbp_df, home_tri, away_tri)
            poss   = events["possessions"]

            if poss.empty:
                logger.warning("  %s: no possessions parsed — skipping", gid)
                continue

            _append_to_parquet(poss,             POSS_PATH, _POSS_SCHEMA)
            _append_to_parquet(events["fouls"],   FOUL_PATH, _FOUL_SCHEMA)
            _append_to_parquet(events["substitutions"], SUB_PATH, _SUB_SCHEMA)
            _append_to_parquet(events["timeouts"],       TO_PATH,  _TO_SCHEMA)

            logger.info(
                "  %s (%s): %d poss, %d fouls, %d subs, %d timeouts",
                gid,
                str(game_row.get("game_date", "")),
                len(poss),
                len(events["fouls"]),
                len(events["substitutions"]),
                len(events["timeouts"]),
            )
            added += 1

        except Exception:
            logger.exception("  %s: failed", gid)

    logger.info("Done. Added %d games.", added)
    return added


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Fetch and parse NBA PBP for recent games")
    parser.add_argument("--since", default=None, help="Only process games on/after YYYY-MM-DD")
    parser.add_argument("--game",  default=None, help="Process a single game ID")
    args = parser.parse_args()

    update_raw_parquets(since_date=args.since, game_id=args.game)


if __name__ == "__main__":
    main()
