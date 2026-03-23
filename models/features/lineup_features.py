"""
Lineup-level features for the feature store.

All features are backward-looking: computed from data available before the possession.
Lineup ratings come from player_rapm + lineup_net_rating (as_of_game_id = current game).
Player APM features require player_ratings table (optional).
"""

import pandas as pd

from models.features.star_players import STAR_PLAYERS


def _parse_player_ids(raw: object) -> list[int]:
    """Parse player_ids stored as list, set, or comma-separated string."""
    if isinstance(raw, (list, set)):
        return [int(p) for p in raw if str(p).strip()]
    return [int(p.strip()) for p in str(raw).split(",") if p.strip()]


def _build_lineup_star_map(lineup_ratings: pd.DataFrame) -> dict[str, bool]:
    """
    Returns {lineup_id: has_star} using the player_ids column.
    Handles player_ids stored as list, set, or comma-separated string.
    """
    result: dict[str, bool] = {}
    for _, row in lineup_ratings.drop_duplicates("lineup_id").iterrows():
        pids = set(_parse_player_ids(row["player_ids"]))
        result[row["lineup_id"]] = any(pid in STAR_PLAYERS for pid in pids)
    return result


def add_lineup_features(
    poss_df: pd.DataFrame,
    lineup_ratings: pd.DataFrame,
    game_id: str,
    player_ratings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join lineup net ratings onto possessions and compute derived features.

    Adds columns:
      home_lineup_net_rating    float — blended rating for home 5-man unit
      away_lineup_net_rating    float
      lineup_net_rating_delta   float — home minus away (THE core signal)
      home_lineup_sample_size   int   — possessions_together for home lineup
      away_lineup_sample_size   int
      home_lineup_just_changed  bool  — True if home_lineup_id differs from previous row
      away_lineup_just_changed  bool
      home_star_on_court        bool  — any Tier 1 or Tier 2 star in current home lineup
      away_star_on_court        bool
      home_best_player_apm      float — max APM among home 5 (requires player_ratings)
      away_best_player_apm      float
      home_worst_player_apm     float — min APM (weakest defensive link)
      away_worst_player_apm     float
      home_apm_spread           float — max - min APM (star-dependence measure)
      away_apm_spread           float
      home_lineup_apm_sum       float — sum of 5 APMs (independent signal from net_rating)
      away_lineup_apm_sum       float
      home_off_court_best_apm   float — best APM among players NOT on court
      away_off_court_best_apm   float
      apm_delta                 float — home_apm_sum - away_apm_sum
    """
    # lineup_ratings is expected to already be pre-filtered to this game's as_of_game_id
    # (done by the builder via groupby for efficiency). Fall back to filtering if needed.
    if "as_of_game_id" in lineup_ratings.columns and lineup_ratings["as_of_game_id"].nunique() > 1:
        game_ratings = lineup_ratings[lineup_ratings["as_of_game_id"] == game_id].copy()
    else:
        game_ratings = lineup_ratings.copy()

    rating_map = game_ratings.set_index("lineup_id")["net_rating"].to_dict()
    sample_map = game_ratings.set_index("lineup_id")["possessions_together"].to_dict()

    poss_df["home_lineup_net_rating"]  = poss_df["home_lineup_id"].map(rating_map).fillna(0.0)
    poss_df["away_lineup_net_rating"]  = poss_df["away_lineup_id"].map(rating_map).fillna(0.0)
    poss_df["lineup_net_rating_delta"] = poss_df["home_lineup_net_rating"] - poss_df["away_lineup_net_rating"]
    poss_df["home_lineup_sample_size"] = poss_df["home_lineup_id"].map(sample_map).fillna(0).astype(int)
    poss_df["away_lineup_sample_size"] = poss_df["away_lineup_id"].map(sample_map).fillna(0).astype(int)

    # Lineup change flags: True if the lineup changed from the previous possession
    poss_df["home_lineup_just_changed"] = poss_df["home_lineup_id"] != poss_df["home_lineup_id"].shift(1)
    poss_df["away_lineup_just_changed"] = poss_df["away_lineup_id"] != poss_df["away_lineup_id"].shift(1)

    # First possession of the game has no "previous" — not a real change
    poss_df.loc[poss_df.index[0], "home_lineup_just_changed"] = False
    poss_df.loc[poss_df.index[0], "away_lineup_just_changed"] = False

    # Restrict to lineups that actually appear in the game (avoids iterating 22K rows).
    active_lineup_ids = set(poss_df["home_lineup_id"].tolist() + poss_df["away_lineup_id"].tolist())
    if "player_ids" in game_ratings.columns:
        active_ratings = game_ratings[game_ratings["lineup_id"].isin(active_lineup_ids)]
        star_map = _build_lineup_star_map(active_ratings)
        poss_df["home_star_on_court"] = poss_df["home_lineup_id"].map(star_map).fillna(False)
        poss_df["away_star_on_court"] = poss_df["away_lineup_id"].map(star_map).fillna(False)
    else:
        active_ratings = game_ratings[game_ratings["lineup_id"].isin(active_lineup_ids)]
        poss_df["home_star_on_court"] = False
        poss_df["away_star_on_court"] = False

    # ── Player APM features ───────────────────────────────────────────────────
    # Requires player_ratings table and player_ids column in lineup_ratings.
    if player_ratings is not None and "player_ids" in game_ratings.columns:
        # Build player_id → APM lookup for this game.
        # player_ratings is expected to already be pre-filtered to this game
        # (by the builder), but we filter defensively in case it isn't.
        if "as_of_game_id" in player_ratings.columns and player_ratings["as_of_game_id"].nunique() > 1:
            game_pr = player_ratings[player_ratings["as_of_game_id"] == game_id]
        else:
            game_pr = player_ratings
        game_player_apm: dict[int, float] = (
            game_pr.set_index("player_id")["adjusted_plus_minus"].to_dict()
        )

        # Build lineup_id → [player_ids] map for active lineups only (~20-50 per game)
        lineup_players: dict[str, list[int]] = {}
        for _, row in active_ratings.drop_duplicates("lineup_id").iterrows():
            lineup_players[row["lineup_id"]] = _parse_player_ids(row["player_ids"])

        # Collect all players who appeared in any lineup in this game (roster proxy)
        all_players_in_game: set[int] = set()
        for pids in lineup_players.values():
            all_players_in_game.update(pids)

        def _lineup_apm_stats(lineup_id: str) -> tuple[float, float, float, float]:
            """(best, worst, spread, sum) APM for a lineup."""
            pids  = lineup_players.get(lineup_id, [])
            apms  = [game_player_apm.get(pid, 0.0) for pid in pids]
            if not apms:
                return 0.0, 0.0, 0.0, 0.0
            return max(apms), min(apms), max(apms) - min(apms), sum(apms)

        def _off_court_best(lineup_id: str) -> float:
            """Best APM of players NOT in this lineup (bench quality)."""
            on_court = set(lineup_players.get(lineup_id, []))
            off      = all_players_in_game - on_court
            if not off:
                return 0.0
            return max(game_player_apm.get(pid, 0.0) for pid in off)

        home_stats = poss_df["home_lineup_id"].map(_lineup_apm_stats).tolist()
        away_stats = poss_df["away_lineup_id"].map(_lineup_apm_stats).tolist()

        poss_df["home_best_player_apm"]   = [s[0] for s in home_stats]
        poss_df["home_worst_player_apm"]  = [s[1] for s in home_stats]
        poss_df["home_apm_spread"]        = [s[2] for s in home_stats]
        poss_df["home_lineup_apm_sum"]    = [s[3] for s in home_stats]
        poss_df["away_best_player_apm"]   = [s[0] for s in away_stats]
        poss_df["away_worst_player_apm"]  = [s[1] for s in away_stats]
        poss_df["away_apm_spread"]        = [s[2] for s in away_stats]
        poss_df["away_lineup_apm_sum"]    = [s[3] for s in away_stats]
        poss_df["home_off_court_best_apm"] = poss_df["home_lineup_id"].map(_off_court_best)
        poss_df["away_off_court_best_apm"] = poss_df["away_lineup_id"].map(_off_court_best)
        poss_df["apm_delta"]              = (
            poss_df["home_lineup_apm_sum"] - poss_df["away_lineup_apm_sum"]
        )
    else:
        for col in [
            "home_best_player_apm", "away_best_player_apm",
            "home_worst_player_apm", "away_worst_player_apm",
            "home_apm_spread", "away_apm_spread",
            "home_lineup_apm_sum", "away_lineup_apm_sum",
            "home_off_court_best_apm", "away_off_court_best_apm",
            "apm_delta",
        ]:
            poss_df[col] = 0.0

    return poss_df
