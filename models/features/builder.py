"""
Feature store builder.

Reads raw parquet tables + rating tables, processes each game's possessions
through all feature modules, and writes the final feature store:

  data/feature_store/feature_rows.parquet

One row per scoring possession. All features backward-looking.
Target columns clearly prefixed with `target_` and separated.

Run after player_rapm.py and lineup_net_rating.py have completed.
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from models.features.lineup_features  import add_lineup_features
from models.features.momentum_features import add_momentum_features
from models.features.context_features  import add_context_features
from models.features.targets           import add_targets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR          = Path("data/raw")
FEATURE_STORE    = Path("data/feature_store")
FEATURE_ROWS_PATH = FEATURE_STORE / "feature_rows.parquet"

POSSESSIONS_PATH    = RAW_DIR / "possessions_202526.parquet"
GAMES_PATH          = RAW_DIR / "games_202526.parquet"
FOULS_PATH          = RAW_DIR / "foul_events_202526.parquet"
TIMEOUTS_PATH       = RAW_DIR / "timeout_events_202526.parquet"  # optional; built by nba_api_client
LINEUP_RATINGS_PATH = FEATURE_STORE / "lineup_ratings.parquet"
PLAYER_RATINGS_PATH = FEATURE_STORE / "player_ratings.parquet"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FEATURE_SCHEMA = pa.schema([
    # Identity
    pa.field("game_id",                    pa.string()),
    pa.field("possession_id",              pa.int32()),
    pa.field("period",                     pa.int32()),
    pa.field("game_clock_secs",            pa.float32()),
    pa.field("minutes_into_game",          pa.float32()),

    # Score state
    pa.field("home_score",                 pa.int32()),
    pa.field("away_score",                 pa.int32()),
    pa.field("score_diff",                 pa.int32()),
    pa.field("team_scored",                pa.string()),
    pa.field("points",                     pa.int32()),

    # Lineup state (googoogagag)
    pa.field("home_lineup_id",             pa.string()),
    pa.field("away_lineup_id",             pa.string()),
    pa.field("home_lineup_net_rating",     pa.float32()),
    pa.field("away_lineup_net_rating",     pa.float32()),
    pa.field("lineup_net_rating_delta",    pa.float32()),  # THE core signal
    pa.field("home_lineup_sample_size",    pa.int32()),
    pa.field("away_lineup_sample_size",    pa.int32()),
    pa.field("home_lineup_just_changed",   pa.bool_()),
    pa.field("away_lineup_just_changed",   pa.bool_()),

    # Momentum
    pa.field("home_points_last_5_poss",    pa.int32()),
    pa.field("away_points_last_5_poss",    pa.int32()),
    pa.field("home_points_last_10_poss",   pa.int32()),
    pa.field("away_points_last_10_poss",   pa.int32()),
    pa.field("current_run_team",           pa.string()),
    pa.field("current_run_length",         pa.int32()),
    pa.field("current_run_points",         pa.int32()),
    pa.field("pace_last_10_possessions",   pa.float32()),
    # Expanding within-game mean. `pace_season_baseline` is the legacy name kept
    # while possession_flat is mid-migration; `pace_game_to_date` is the real one.
    pa.field("pace_season_baseline",       pa.float32()),
    pa.field("pace_game_to_date",          pa.float32()),
    pa.field("home_scoring_sustainable",   pa.bool_()),
    pa.field("away_scoring_sustainable",   pa.bool_()),

    # Context
    pa.field("is_blowout",                 pa.bool_()),
    pa.field("is_garbage_time",            pa.bool_()),
    pa.field("home_key_foul_count",        pa.int32()),
    pa.field("away_key_foul_count",        pa.int32()),
    pa.field("home_max_player_fouls",      pa.int32()),
    pa.field("away_max_player_fouls",      pa.int32()),
    pa.field("home_player_in_trouble",     pa.bool_()),
    pa.field("away_player_in_trouble",     pa.bool_()),
    pa.field("home_trouble_star_tier",     pa.int32()),
    pa.field("away_trouble_star_tier",     pa.int32()),
    pa.field("home_star_on_court",         pa.bool_()),
    pa.field("away_star_on_court",         pa.bool_()),
    pa.field("home_back_to_back",          pa.bool_()),
    pa.field("away_back_to_back",          pa.bool_()),

    # Run shot composition
    pa.field("current_run_3pt_count",      pa.int32()),
    pa.field("current_run_3pt_pct",        pa.float32()),
    pa.field("current_run_paint_pct",      pa.float32()),

    # xPPP shot quality (continuous)
    pa.field("home_xPPP_last_5",           pa.float32()),
    pa.field("away_xPPP_last_5",           pa.float32()),
    pa.field("home_actual_vs_expected_PPP", pa.float32()),
    pa.field("away_actual_vs_expected_PPP", pa.float32()),
    # Raw xPPP differences, no longer np.sign()-collapsed.
    pa.field("home_shot_quality_trend",    pa.float32()),
    pa.field("away_shot_quality_trend",    pa.float32()),
    pa.field("home_xPPP_prev_5",           pa.float32()),
    pa.field("away_xPPP_prev_5",           pa.float32()),

    # Bonus state
    pa.field("home_in_bonus",              pa.bool_()),
    pa.field("away_in_bonus",              pa.bool_()),
    pa.field("home_fouls_until_bonus",     pa.int32()),
    pa.field("away_fouls_until_bonus",     pa.int32()),
    pa.field("both_teams_in_bonus",        pa.bool_()),

    # Score × time interactions
    pa.field("trailing_team_urgency",      pa.float32()),
    pa.field("comeback_probability_proxy", pa.float32()),
    pa.field("q4_close_game",              pa.bool_()),
    pa.field("garbage_time_risk",          pa.float32()),

    # Timeout features
    pa.field("possessions_since_last_timeout",     pa.int32()),
    pa.field("home_called_timeout_in_last_3_poss", pa.bool_()),
    pa.field("away_called_timeout_in_last_3_poss", pa.bool_()),
    pa.field("timeout_on_opponent_run",            pa.bool_()),
    pa.field("home_full_timeouts_remaining",        pa.int32()),
    pa.field("away_full_timeouts_remaining",        pa.int32()),

    # In-game star stats
    pa.field("home_star_points_this_game", pa.int32()),
    pa.field("away_star_points_this_game", pa.int32()),
    pa.field("home_star_foul_count",       pa.int32()),
    pa.field("away_star_foul_count",       pa.int32()),

    # Player APM features
    pa.field("home_best_player_apm",       pa.float32()),
    pa.field("away_best_player_apm",       pa.float32()),
    pa.field("home_worst_player_apm",      pa.float32()),
    pa.field("away_worst_player_apm",      pa.float32()),
    pa.field("home_apm_spread",            pa.float32()),
    pa.field("away_apm_spread",            pa.float32()),
    pa.field("home_lineup_apm_sum",        pa.float32()),
    pa.field("away_lineup_apm_sum",        pa.float32()),
    pa.field("home_off_court_best_apm",    pa.float32()),
    pa.field("away_off_court_best_apm",    pa.float32()),
    pa.field("apm_delta",                  pa.float32()),

    # Shot info (passthrough from raw)
    pa.field("shot_type",                  pa.string()),
    pa.field("shot_distance",              pa.int32()),
    pa.field("shot_value",                 pa.int32()),

    # Targets — TRAINING ONLY, DO NOT USE AS FEATURES
    pa.field("target_home_next_5_margin",  pa.int32()),
    pa.field("target_home_next_10_margin", pa.int32()),
    pa.field("target_meaningful_run_5_scoring", pa.bool_()),
    pa.field("target_meaningful_run_10",   pa.bool_()),
])


# ---------------------------------------------------------------------------
# Per-game processor
# ---------------------------------------------------------------------------

def process_game(
    game_id: str,
    poss_df: pd.DataFrame,
    foul_events: pd.DataFrame,
    games: pd.DataFrame,
    lineup_ratings: pd.DataFrame,
    player_ratings: pd.DataFrame | None = None,
    timeout_events: pd.DataFrame | None = None,
    player_tier_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """
    Run all feature modules on a single game's possessions.
    Returns a feature-row DataFrame for this game.
    """
    df = poss_df.sort_values("possession_id").reset_index(drop=True)

    # Lineup features (uses pre-computed ratings — no lookahead)
    df, lineup_player_map = add_lineup_features(df, lineup_ratings, game_id, player_ratings=player_ratings)

    # Momentum features (backward-looking rolling windows)
    df = add_momentum_features(df)

    # Context features (score state, foul counts, flags, timeout features, in-game star stats)
    df = add_context_features(
        df, foul_events, games, game_id,
        timeout_events=timeout_events,
        lineup_player_map=lineup_player_map,
        player_tier_map=player_tier_map,
    )

    # Target variables (forward-looking — for training only)
    df = add_targets(df)

    return df


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_feature_store(
    possessions: pd.DataFrame,
    games: pd.DataFrame,
    foul_events: pd.DataFrame,
    lineup_ratings: pd.DataFrame,
    player_ratings: pd.DataFrame | None = None,
    timeout_events: pd.DataFrame | None = None,
    player_tier_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Process all games and return the full feature store DataFrame."""
    games_sorted = games.sort_values("game_date")["game_id"].tolist()
    all_rows: list[pd.DataFrame] = []
    n_games = len(games_sorted)

    # Pre-group large tables by game_id to avoid O(n²) scanning per game
    logger.info("Pre-grouping lineup ratings by game (this may take a moment)...")
    lineup_by_game: dict[str, pd.DataFrame] = {
        gid: grp for gid, grp in lineup_ratings.groupby("as_of_game_id")
    }
    player_by_game: dict[str, pd.DataFrame] | None = None
    if player_ratings is not None:
        player_by_game = {
            gid: grp for gid, grp in player_ratings.groupby("as_of_game_id")
        }
    logger.info("Pre-grouping complete.")

    for i, game_id in enumerate(games_sorted):
        game_poss = possessions[possessions["game_id"] == game_id]
        if game_poss.empty:
            continue

        game_lineups = lineup_by_game.get(game_id, pd.DataFrame(columns=lineup_ratings.columns))
        game_players = player_by_game.get(game_id, pd.DataFrame()) if player_by_game else None

        try:
            feature_df = process_game(
                game_id, game_poss, foul_events, games, game_lineups,
                player_ratings=game_players,
                timeout_events=timeout_events,
                player_tier_map=player_tier_map,
            )
            all_rows.append(feature_df)
        except Exception as exc:
            logger.warning("[%s] Feature computation failed: %s", game_id, exc)
            continue

        if (i + 1) % 100 == 0:
            logger.info("[%d/%d] Processed %s — %d possessions",
                        i + 1, n_games, game_id, len(game_poss))

    return pd.concat(all_rows, ignore_index=True)


def write_feature_store(df: pd.DataFrame) -> None:
    """Write the feature store to parquet, enforcing the schema."""
    arrays: dict[str, pa.Array] = {}
    for field in FEATURE_SCHEMA:
        col = df.get(field.name)
        if col is None:
            logger.warning("Feature column '%s' missing — filling with nulls", field.name)
            arrays[field.name] = pa.nulls(len(df), type=field.type)
        else:
            try:
                arrays[field.name] = pa.array(col.tolist(), type=field.type)
            except Exception as exc:
                logger.error("Failed to cast '%s' to %s: %s", field.name, field.type, exc)
                arrays[field.name] = pa.nulls(len(df), type=field.type)

    table = pa.table(arrays, schema=FEATURE_SCHEMA)
    pq.write_table(table, FEATURE_ROWS_PATH, compression="snappy")
    logger.info("Feature store written: %d rows → %s", len(df), FEATURE_ROWS_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)

    logger.info("Loading input tables...")
    possessions    = pq.read_table(POSSESSIONS_PATH).to_pandas()
    games          = pq.read_table(GAMES_PATH).to_pandas()
    foul_events    = pq.read_table(FOULS_PATH).to_pandas()
    lineup_ratings = pq.read_table(LINEUP_RATINGS_PATH).to_pandas()
    player_ratings = pq.read_table(PLAYER_RATINGS_PATH).to_pandas() if PLAYER_RATINGS_PATH.exists() else None

    timeout_events: pd.DataFrame | None = None
    if TIMEOUTS_PATH.exists():
        timeout_events = pq.read_table(TIMEOUTS_PATH).to_pandas()
        logger.info("Timeout events: %d rows", len(timeout_events))
    else:
        logger.warning("timeout_events not found at %s — timeout features will be default. "
                       "Run: python -c \"from data.ingestion.nba_api_client import parse_timeouts_from_cache; "
                       "parse_timeouts_from_cache()\"", TIMEOUTS_PATH)

    logger.info(
        "Inputs: %d possessions, %d games, %d fouls, %d lineup-rating rows",
        len(possessions), len(games), len(foul_events), len(lineup_ratings),
    )

    import duckdb
    import os
    db_path = "kalshi_trading.duckdb"
    player_tier_map = {}
    if os.path.exists(db_path):
        con = duckdb.connect(db_path, read_only=True)
        try:
            tiers = con.execute("SELECT player_id, star_tier FROM main.dim_players WHERE star_tier IS NOT NULL").fetchall()
            player_tier_map = {int(p): int(t) for p, t in tiers}
            logger.info("Loaded player tier map with %d entries", len(player_tier_map))
        except Exception as e:
            logger.warning("Could not load dim_players from %s: %s", db_path, e)
        finally:
            con.close()

    logger.info("Building feature store...")
    feature_df = build_feature_store(
        possessions, games, foul_events, lineup_ratings,
        player_ratings=player_ratings,
        timeout_events=timeout_events,
        player_tier_map=player_tier_map,
    )

    logger.info("Feature store: %d rows, %d columns", len(feature_df), len(feature_df.columns))
    write_feature_store(feature_df)

    # Quick sanity stats
    logger.info("lineup_net_rating_delta: mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
                feature_df["lineup_net_rating_delta"].mean(),
                feature_df["lineup_net_rating_delta"].std(),
                feature_df["lineup_net_rating_delta"].min(),
                feature_df["lineup_net_rating_delta"].max())
    logger.info("Blowout possessions: %d (%.1f%%)",
                feature_df["is_blowout"].sum(),
                100 * feature_df["is_blowout"].mean())
    logger.info("Target run_10 rate: %.1f%%",
                100 * feature_df["target_meaningful_run_10"].mean())


if __name__ == "__main__":
    main()
