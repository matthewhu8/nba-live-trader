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

import os
from typing import Any

import duckdb
from dotenv import load_dotenv

load_dotenv()

_MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")


async def load_pregame(game_id: str) -> dict[str, Any]:
    """
    Load all static game context needed by GameState at tip-off.
    Returns a dict with keys: pregame_features, lineup_ratings, player_apm,
    star_players, home_b2b, away_b2b, pace_baseline.
    """
    conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={_MOTHERDUCK_TOKEN}")

    pregame_features = _load_pregame_features(conn, game_id)
    lineup_ratings   = _load_lineup_ratings(conn, game_id)
    player_apm       = _load_player_apm(conn, game_id)
    b2b_flags        = _load_b2b_flags(conn, game_id)

    conn.close()

    return {
        **pregame_features,
        "lineup_ratings": lineup_ratings,
        "player_apm":     player_apm,
        "home_b2b":       b2b_flags["home"],
        "away_b2b":       b2b_flags["away"],
    }


def _load_pregame_features(conn: duckdb.DuckDBPyConnection, game_id: str) -> dict[str, float]:
    # TODO: SELECT * FROM features.pregame WHERE game_id = ?
    # TODO: fill nulls with 0.0, set has_pregame_data flag
    return {}


def _load_lineup_ratings(conn: duckdb.DuckDBPyConnection, game_id: str) -> dict[str, float]:
    # TODO: SELECT lineup_id, net_rating FROM features.lineup_ratings
    #       WHERE as_of_game_id <= game_id AND game_id IN (games for these teams)
    #       Latest as_of per lineup_id
    return {}


def _load_player_apm(conn: duckdb.DuckDBPyConnection, game_id: str) -> dict[int, float]:
    # TODO: SELECT player_id, adjusted_plus_minus FROM features.player_ratings
    #       WHERE as_of_game_id = (latest before game_id)
    return {}


def _load_b2b_flags(conn: duckdb.DuckDBPyConnection, game_id: str) -> dict[str, bool]:
    # TODO: SELECT home_back_to_back, away_back_to_back FROM dim_games WHERE game_id = ?
    return {"home": False, "away": False}
