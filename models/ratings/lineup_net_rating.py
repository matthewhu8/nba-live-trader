"""
Point-in-time lineup net ratings with Bayesian shrinkage.

For each (lineup_id, as_of_game_id), compute net rating by blending:
  1. Observed:  points-per-100 differential while this exact unit was on court
  2. Predicted: mean adjusted_plus_minus of the 5 individual players

Blending formula:
  w = possessions / (possessions + K)     where K=50 (shrinkage constant)
  net_rating = w × observed + (1-w) × predicted

Lineups with many possessions together trust their own observed history.
Lineups with few possessions lean on the individual player ratings.

Output: data/feature_store/lineup_ratings.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from models.ratings.player_rapm import (
    FEATURE_STORE_DIR,
    GAMES_PATH,
    POSSESSIONS_PATH,
    PBP_CACHE_DIR,
    build_lineup_map,
)

logger = logging.getLogger(__name__)

PLAYER_RATINGS_PATH = FEATURE_STORE_DIR / "player_ratings.parquet"
LINEUP_RATINGS_PATH = FEATURE_STORE_DIR / "lineup_ratings.parquet"

SHRINKAGE_K = 50  # possessions before we fully trust observed data

LINEUP_RATINGS_SCHEMA = pa.schema([
    pa.field("lineup_id",            pa.string()),
    pa.field("as_of_game_id",        pa.string()),
    pa.field("net_rating",           pa.float32()),
    pa.field("observed_net_rating",  pa.float32()),
    pa.field("predicted_net_rating", pa.float32()),
    pa.field("possessions_together", pa.int32()),
    pa.field("shrinkage_weight",     pa.float32()),
    pa.field("player_ids",           pa.string()),
])


# ---------------------------------------------------------------------------
# Lineup composition table
# ---------------------------------------------------------------------------

def build_lineup_player_df(lineup_map: dict[str, frozenset[int]]) -> pd.DataFrame:
    """
    Explode lineup_map into a long DataFrame: one row per (lineup_id, player_id).
    Used for vectorized joins against player ratings.
    """
    rows = [
        {"lineup_id": lid, "player_id": pid}
        for lid, players in lineup_map.items()
        for pid in players
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Observed lineup ratings (vectorized)
# ---------------------------------------------------------------------------

def compute_observed_ratings(
    possessions: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every (lineup_id, as_of_game_id) pair, compute the observed net rating
    from all possessions in games STRICTLY BEFORE as_of_game_id.

    Uses a cumulative approach: sort possessions chronologically, compute
    cumulative totals, then join against game ordering to get point-in-time values.

    Returns DataFrame: lineup_id, as_of_game_id, observed_net_rating, possessions_together
    """
    games_sorted  = games.sort_values("game_date").reset_index(drop=True)
    game_to_order = {gid: i for i, gid in enumerate(games_sorted["game_id"])}

    poss = possessions.copy()
    poss["game_order"] = poss["game_id"].map(game_to_order)
    poss = poss.dropna(subset=["game_order"]).copy()
    poss["game_order"] = poss["game_order"].astype(int)

    # Points contribution per possession: + if home lineup scored, - if away scored
    poss["home_pts"] = np.where(poss["team_scored"] == "home", poss["points"].astype(float), 0.0)
    poss["away_pts"] = np.where(poss["team_scored"] == "away", poss["points"].astype(float), 0.0)

    # Build two long-form tables (once for home lineup, once for away)
    # For home lineup: their net = home_pts - away_pts for that possession
    home_poss = poss[["game_order", "home_lineup_id", "home_pts", "away_pts"]].copy()
    home_poss = home_poss.rename(columns={"home_lineup_id": "lineup_id"})
    home_poss["net_pts"] = home_poss["home_pts"] - home_poss["away_pts"]

    away_poss = poss[["game_order", "away_lineup_id", "home_pts", "away_pts"]].copy()
    away_poss = away_poss.rename(columns={"away_lineup_id": "lineup_id"})
    away_poss["net_pts"] = away_poss["away_pts"] - away_poss["home_pts"]

    all_poss = pd.concat([
        home_poss[["game_order", "lineup_id", "net_pts"]],
        away_poss[["game_order", "lineup_id", "net_pts"]],
    ], ignore_index=True)

    # Sort by game order within each lineup, then cumsum
    all_poss = all_poss.sort_values(["lineup_id", "game_order"]).reset_index(drop=True)
    all_poss["cum_net"] = all_poss.groupby("lineup_id")["net_pts"].cumsum()
    all_poss["cum_n"]   = all_poss.groupby("lineup_id").cumcount() + 1

    # For each (lineup, game_order), the "prior" stats are the last cumsum
    # values where game_order < that game's order.
    # Strategy: take last row per (lineup, game_order) group, then for each
    # as_of_game_id G, look up the entry with game_order = G-1 (or max < G).
    last_per_game = (
        all_poss
        .groupby(["lineup_id", "game_order"])
        .agg(cum_net=("cum_net", "last"), cum_n=("cum_n", "last"))
        .reset_index()
    )

    # Cross-join: for each as_of game G, what stats were accumulated up to G-1?
    # Efficient approach: for each game G (0..929), merge on game_order < G
    # using a forward-fill / merge-asof approach.
    n_games = len(games_sorted)
    records: list[pd.DataFrame] = []

    for g_idx, game_row in games_sorted.iterrows():
        game_id = game_row["game_id"]

        # Prior data: all lineup stats with game_order < g_idx
        prior = last_per_game[last_per_game["game_order"] < g_idx]
        if prior.empty:
            continue

        # Keep only the most recent (highest game_order) row per lineup
        latest = prior.sort_values("game_order").groupby("lineup_id").last().reset_index()
        latest["as_of_game_id"]       = game_id
        latest["observed_net_rating"]  = latest["cum_net"] / latest["cum_n"].clip(lower=1) * 100
        latest["possessions_together"] = latest["cum_n"].astype(int)
        records.append(latest[["lineup_id", "as_of_game_id", "observed_net_rating", "possessions_together"]])

        if g_idx % 100 == 0:
            logger.info("  Observed ratings: game %d/%d (%d lineups with history)",
                        g_idx, n_games, len(latest))

    if not records:
        return pd.DataFrame(columns=["lineup_id", "as_of_game_id", "observed_net_rating", "possessions_together"])

    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Predicted lineup ratings (vectorized via join)
# ---------------------------------------------------------------------------

def compute_predicted_ratings(
    lineup_player_df: pd.DataFrame,
    player_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (lineup_id, as_of_game_id), compute mean adjusted_plus_minus of
    the 5 players.

    Uses a pivot approach: player ratings are pivoted to a (game × player) matrix,
    lineup membership is a (lineup × player) membership matrix, and matrix
    multiplication gives (lineup × game) predicted ratings in one shot.
    This avoids the 288M-row merge that a naive join would produce.
    """
    games     = player_ratings["as_of_game_id"].unique()
    players   = player_ratings["player_id"].unique()
    player_to_col = {pid: i for i, pid in enumerate(players)}
    game_to_row   = {gid: i for i, gid in enumerate(games)}
    lineup_ids    = lineup_player_df["lineup_id"].unique()
    lineup_to_row = {lid: i for i, lid in enumerate(lineup_ids)}

    n_games   = len(games)
    n_players = len(players)
    n_lineups = len(lineup_ids)

    logger.info("Predicted ratings: %d lineups × %d games × %d players",
                n_lineups, n_games, n_players)

    # Build player-ratings matrix: shape (n_games, n_players)
    ratings_matrix = np.zeros((n_games, n_players), dtype=np.float32)
    for _, row in player_ratings.iterrows():
        g = game_to_row.get(row["as_of_game_id"])
        p = player_to_col.get(row["player_id"])
        if g is not None and p is not None:
            ratings_matrix[g, p] = float(row["adjusted_plus_minus"])

    # Build lineup-membership matrix: shape (n_lineups, n_players)
    # Value = 1/5 so matmul gives mean over the 5 players
    membership_matrix = np.zeros((n_lineups, n_players), dtype=np.float32)
    for _, row in lineup_player_df.iterrows():
        l_idx = lineup_to_row.get(row["lineup_id"])
        p_idx = player_to_col.get(row["player_id"])
        if l_idx is not None and p_idx is not None:
            membership_matrix[l_idx, p_idx] = 1.0

    # Count players per lineup for mean (some lineups may have <5 known players)
    player_counts = membership_matrix.sum(axis=1, keepdims=True).clip(min=1)
    membership_matrix /= player_counts

    # (n_lineups, n_players) × (n_players, n_games) = (n_lineups, n_games)
    predicted_matrix = membership_matrix @ ratings_matrix.T  # shape: (n_lineups, n_games)

    # Unpack via numpy tile/repeat — avoids Python loop over 53M records
    lineup_arr = np.array(lineup_ids)
    games_arr  = np.array(games)

    return pd.DataFrame({
        "lineup_id":            np.repeat(lineup_arr, n_games),
        "as_of_game_id":        np.tile(games_arr, n_lineups),
        "predicted_net_rating": predicted_matrix.ravel().astype(float),
    })


# ---------------------------------------------------------------------------
# Final blend
# ---------------------------------------------------------------------------

def blend_ratings(
    observed_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    lineup_player_df: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Merge observed + predicted and apply Bayesian shrinkage blend."""
    # All lineup × game combinations come from predicted (every lineup, every game)
    merged = predicted_df.merge(observed_df, on=["lineup_id", "as_of_game_id"], how="left")
    merged["observed_net_rating"]  = merged["observed_net_rating"].fillna(0.0)
    merged["possessions_together"] = merged["possessions_together"].fillna(0).astype(int)

    n = merged["possessions_together"].astype(float)
    merged["shrinkage_weight"] = n / (n + SHRINKAGE_K)
    merged["net_rating"] = (
        merged["shrinkage_weight"] * merged["observed_net_rating"]
        + (1 - merged["shrinkage_weight"]) * merged["predicted_net_rating"]
    )

    # Add player_ids string
    player_ids_map = {
        lid: ",".join(str(p) for p in sorted(players))
        for lid, players in zip(
            lineup_player_df.groupby("lineup_id").groups.keys(),
            [g["player_id"].tolist() for _, g in lineup_player_df.groupby("lineup_id")],
        )
    }
    merged["player_ids"] = merged["lineup_id"].map(player_ids_map).fillna("")

    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

    if not PLAYER_RATINGS_PATH.exists():
        raise RuntimeError(f"{PLAYER_RATINGS_PATH} not found — run player_rapm.py first")

    possessions    = pq.read_table(POSSESSIONS_PATH).to_pandas()
    games          = pq.read_table(GAMES_PATH).to_pandas()
    player_ratings = pq.read_table(PLAYER_RATINGS_PATH).to_pandas()
    logger.info("Loaded %d possessions, %d games, %d player rating rows",
                len(possessions), len(games), len(player_ratings))

    lineup_map       = build_lineup_map(PBP_CACHE_DIR)
    lineup_player_df = build_lineup_player_df(lineup_map)

    # Filter to lineups that actually appear in possessions — the lineup_map
    # contains 57k+ lineups including many that appear in no scoring events.
    active_lineups = (
        set(possessions["home_lineup_id"].unique())
        | set(possessions["away_lineup_id"].unique())
    )
    lineup_player_df = lineup_player_df[lineup_player_df["lineup_id"].isin(active_lineups)]
    logger.info("Active lineups in possessions: %d (of %d total in map)",
                lineup_player_df["lineup_id"].nunique(), len(lineup_map))

    logger.info("Computing observed ratings...")
    observed_df = compute_observed_ratings(possessions, games)
    logger.info("Observed: %d (lineup, game) pairs", len(observed_df))

    logger.info("Computing predicted ratings (vectorized join)...")
    predicted_df = compute_predicted_ratings(lineup_player_df, player_ratings)
    logger.info("Predicted: %d (lineup, game) pairs", len(predicted_df))

    logger.info("Blending...")
    final_df = blend_ratings(observed_df, predicted_df, lineup_player_df, games)
    logger.info("Final lineup ratings: %d rows", len(final_df))

    # Write parquet
    out_cols = [f.name for f in LINEUP_RATINGS_SCHEMA]
    arrays = {}
    for f in LINEUP_RATINGS_SCHEMA:
        arrays[f.name] = pa.array(
            final_df[f.name].tolist() if f.name in final_df.columns else [None] * len(final_df),
            type=f.type,
        )
    table = pa.table(arrays, schema=LINEUP_RATINGS_SCHEMA)
    pq.write_table(table, LINEUP_RATINGS_PATH)
    logger.info("Done — lineup_ratings.parquet written to %s", LINEUP_RATINGS_PATH)

    n_lineups = final_df["lineup_id"].nunique()
    n_games   = final_df["as_of_game_id"].nunique()
    logger.info("  %d unique lineups × %d game snapshots", n_lineups, n_games)

    # Sanity check: top lineups by final game
    last_game = games.sort_values("game_date")["game_id"].iloc[-1]
    top = (
        final_df[final_df["as_of_game_id"] == last_game]
        .sort_values("net_rating", ascending=False)
        .head(5)
    )
    logger.info("Top 5 lineups (final snapshot):\n%s",
                top[["lineup_id", "net_rating", "possessions_together", "shrinkage_weight"]].to_string(index=False))


if __name__ == "__main__":
    main()
