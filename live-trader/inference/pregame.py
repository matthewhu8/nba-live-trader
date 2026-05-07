"""
PregameLoader — loads static game-level features from MotherDuck at game start.

Runs once per game, before tip-off. Returns a dict of the 11 PREGAME_COLS
features. These are static for the entire game — never recomputed mid-game.

Also loads:
    - lineup_ratings: dict[lineup_id → net_rating]  (for all lineups in this game's teams)
    - player_apm:     dict[player_id → APM]          (for all players on both rosters)
    - star_players:   dict[player_id → tier]          (Tier 1/2/3, from context_features.py defs)
    - home_b2b / away_b2b: bool
    - pace_baseline:  float (season avg pace for this game's teams)

All of this is pre-game knowledge — point-in-time safe, no lookahead.
"""

import logging
import os
from typing import Any

import duckdb
from dotenv import load_dotenv

from models.features.star_players import STAR_PLAYERS

load_dotenv()

_MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")

NBA_TEAM_IDS: dict[str, int] = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}


async def load_pregame(
    game_id: str,
    fallback_home_team_id: int = 0,
    fallback_away_team_id: int = 0,
) -> dict[str, Any]:
    """
    Load all static game context needed by GameState at tip-off.
    Returns a flat dict with pregame feature floats plus lineup_ratings,
    player_apm, star_players, home/away team ids, b2b flags, and pace_baseline.

    Gracefully handles missing games (e.g. playoff games not yet in dim_games)
    by returning zeroed pregame features with has_pregame_data=0.0.
    """
    pregame_features: dict[str, float] = {}
    lineup_ratings: dict[str, float] = {}
    player_apm: dict[int, float] = {}
    b2b_data: dict[str, Any] = {
        "home_team_id": fallback_home_team_id,
        "away_team_id": fallback_away_team_id,
        "home_b2b": False,
        "away_b2b": False,
    }

    try:
        conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={_MOTHERDUCK_TOKEN}")
        try:
            pregame_features = _load_pregame_features(conn, game_id)
            lineup_ratings   = _load_lineup_ratings(conn, game_id)
            player_apm       = _load_player_apm(conn, game_id)
            b2b_data         = _load_b2b_and_team_ids(conn, game_id)
        except ValueError as exc:
            logging.warning(
                "[PREGAME] game=%s not found in dim_games — using fallback defaults: %s",
                game_id, exc,
            )
        finally:
            conn.close()
    except Exception as exc:
        logging.warning(
            "[PREGAME] MotherDuck connection failed — running without pregame data: %s",
            exc,
        )

    # Ensure pregame features have all expected keys (zeroed if missing)
    _pregame_col_names = [
        "team_net_rating_delta", "home_off_rating", "away_off_rating",
        "home_def_rating", "away_def_rating", "roster_rapm_gap",
        "missing_rapm_impact", "rest_advantage", "expected_pace", "form_delta",
    ]
    for col in _pregame_col_names:
        pregame_features.setdefault(col, 0.0)
    pregame_features.setdefault("has_pregame_data", 0.0)

    pace_baseline = pregame_features.get("expected_pace", 0.0) or 14.0

    logging.info(
        "[PREGAME] game=%s lineup_ratings=%d players=%d pregame_ok=%s pace=%.1f",
        game_id,
        len(lineup_ratings),
        len(player_apm),
        bool(pregame_features.get("has_pregame_data")),
        pace_baseline,
    )

    return {
        **pregame_features,
        "lineup_ratings": lineup_ratings,
        "player_apm":     player_apm,
        "star_players":   STAR_PLAYERS,
        "home_team_id":   b2b_data["home_team_id"],
        "away_team_id":   b2b_data["away_team_id"],
        "home_b2b":       b2b_data["home_b2b"],
        "away_b2b":       b2b_data["away_b2b"],
        "pace_baseline":  pace_baseline,
    }


def _load_pregame_features(conn: duckdb.DuckDBPyConnection, game_id: str) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT team_net_rating_delta, home_off_rating, away_off_rating, home_def_rating,
               away_def_rating, roster_rapm_gap, missing_rapm_impact, rest_advantage,
               expected_pace, form_delta
        FROM features.pregame
        WHERE game_id = ?
        """,
        [game_id],
    ).fetchall()

    col_names = [
        "team_net_rating_delta", "home_off_rating", "away_off_rating",
        "home_def_rating", "away_def_rating", "roster_rapm_gap",
        "missing_rapm_impact", "rest_advantage", "expected_pace", "form_delta",
    ]

    if not rows:
        return {col: 0.0 for col in col_names} | {"has_pregame_data": 0.0}

    row = rows[0]
    result = {col: float(val) if val is not None else 0.0 for col, val in zip(col_names, row)}
    result["has_pregame_data"] = 1.0
    return result


def _load_lineup_ratings(conn: duckdb.DuckDBPyConnection, game_id: str) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT lineup_id, net_rating
        FROM features.lineup_ratings
        WHERE as_of_game_id = (
            SELECT MAX(as_of_game_id) FROM features.lineup_ratings
            WHERE as_of_game_id <= ?
        )
        """,
        [game_id],
    ).fetchall()

    return {str(lineup_id): float(net_rating) for lineup_id, net_rating in rows if net_rating is not None}


def _load_player_apm(conn: duckdb.DuckDBPyConnection, game_id: str) -> dict[int, float]:
    rows = conn.execute(
        """
        SELECT player_id, adjusted_plus_minus
        FROM features.player_ratings
        WHERE as_of_game_id = (
            SELECT MAX(as_of_game_id) FROM features.player_ratings
            WHERE as_of_game_id <= ?
        )
        """,
        [game_id],
    ).fetchall()

    return {int(player_id): float(apm) for player_id, apm in rows if apm is not None}


def _load_b2b_and_team_ids(conn: duckdb.DuckDBPyConnection, game_id: str) -> dict:
    rows = conn.execute(
        """
        SELECT home_team, away_team, home_back_to_back, away_back_to_back
        FROM dim_games
        WHERE game_id = ?
        """,
        [game_id],
    ).fetchall()

    if not rows:
        raise ValueError(f"game_id={game_id!r} not found in dim_games")

    home_tricode, away_tricode, home_b2b, away_b2b = rows[0]

    if home_tricode not in NBA_TEAM_IDS:
        raise ValueError(
            f"Unknown home team tricode {home_tricode!r} for game {game_id}. "
            f"Known tricodes: {sorted(NBA_TEAM_IDS)}"
        )
    if away_tricode not in NBA_TEAM_IDS:
        raise ValueError(
            f"Unknown away team tricode {away_tricode!r} for game {game_id}. "
            f"Known tricodes: {sorted(NBA_TEAM_IDS)}"
        )

    return {
        "home_team_id": NBA_TEAM_IDS[home_tricode],
        "away_team_id": NBA_TEAM_IDS[away_tricode],
        "home_b2b":     bool(home_b2b),
        "away_b2b":     bool(away_b2b),
    }
