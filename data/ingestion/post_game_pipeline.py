"""
Stateless post-game NBA data pipeline.

Runs at 3 AM ET on the existing Fly.io recorder machine after all games finish.
Processes one game at a time — never holds multiple games in memory simultaneously.

Phases (in order — do NOT reorder):
  Phase 0: Upsert dim_games (all other tables FK on game_id)
  Phase 1: Parse PBP for tonight's games
  Phase 2: Build possession_flat rows (uses pre-tonight ASOF ratings — point-in-time correct)
  Phase 3: Update player_ratings + lineup_ratings (incorporates tonight's data for tomorrow)

Manual trigger:
  python -m data.ingestion.post_game_pipeline 2026-04-02
"""

import logging
import os
import time
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.linear_model import Ridge

from data.ingestion.game_schedule import GameInfo
from data.ingestion.nba_api_client import parse_game_to_events
from models.features.builder import process_game

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIDGE_ALPHA          = 1000
DECAY_HALFLIFE_GAMES = 80
SHRINKAGE_K          = 50    # possessions before fully trusting observed lineup data
FIXED_ALPHA          = 0.05  # EWMA floor — half-life ~20 games


# ---------------------------------------------------------------------------
# MotherDuck connection
# ---------------------------------------------------------------------------

def _md_connect() -> duckdb.DuckDBPyConnection:
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN environment variable not set")
    conn = duckdb.connect(f"md:kalshi_trading?motherduck_token={token}")
    conn.execute("SET memory_limit='256MB'")
    return conn


# ---------------------------------------------------------------------------
# Phase 0: Upsert dim_games
# ---------------------------------------------------------------------------

def _get_back_to_back_flags(
    team: str,
    game_date: date,
    conn: duckdb.DuckDBPyConnection,
) -> bool:
    """Check if team played the prior calendar day."""
    yesterday = (game_date - timedelta(days=1)).isoformat()
    result = conn.execute(
        """
        SELECT COUNT(*) FROM dim_games
        WHERE game_date = ?
          AND (home_team = ? OR away_team = ?)
        """,
        [yesterday, team, team],
    ).fetchone()
    return bool(result and result[0] > 0)


def upsert_dim_games(games: list[GameInfo], game_date: date) -> None:
    """
    Phase 0: Insert tonight's games into dim_games before any other phase runs.
    All other tables have FK dependencies on game_id.
    """
    conn = _md_connect()
    try:
        inserted = 0
        for game in games:
            home_b2b = _get_back_to_back_flags(game.home_team, game_date, conn)
            away_b2b = _get_back_to_back_flags(game.visitor_team, game_date, conn)
            conn.execute(
                """
                INSERT OR IGNORE INTO dim_games
                    (game_id, game_date, season, home_team, away_team,
                     home_back_to_back, away_back_to_back)
                VALUES (?, ?, '2025-26', ?, ?, ?, ?)
                """,
                [
                    game.game_id,
                    game_date.isoformat(),
                    game.home_team,
                    game.visitor_team,
                    home_b2b,
                    away_b2b,
                ],
            )
            inserted += 1
        logger.info("Phase 0: upserted %d games into dim_games", inserted)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 1: Fetch + parse PBP
# ---------------------------------------------------------------------------

def _fetch_pbp_stateless(
    game_id: str,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """
    Fetch play-by-play directly from nba_api with progressive backoff.
    No disk cache — ephemeral Fly.io filesystem makes caching pointless.
    Returns None if all retries fail (game not yet logged by nba_api).
    """
    from nba_api.stats.endpoints import playbyplayv3

    backoff_seconds = [0.6, 2.6, 4.6]
    for attempt in range(max_retries):
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
            df = pbp.get_data_frames()[0]
            if df.empty:
                logger.warning("game %s: PBP returned empty DataFrame", game_id)
                return None
            return df
        except Exception as exc:
            wait = backoff_seconds[attempt] if attempt < len(backoff_seconds) else 4.6
            logger.warning(
                "game %s: PBP fetch attempt %d/%d failed (%s) — retrying in %.1fs",
                game_id, attempt + 1, max_retries, exc, wait,
            )
            if attempt < max_retries - 1:
                time.sleep(wait)

    logger.warning("game %s: all %d PBP fetch attempts failed — skipping", game_id, max_retries)
    return None


def fetch_and_parse_all(games: list[GameInfo]) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Phase 1: Fetch + parse PBP for all tonight's games.
    Returns {game_id: {"possessions": df, "fouls": df, "timeouts": df}}.
    Only successfully parsed games are included.
    """
    parsed: dict[str, dict[str, pd.DataFrame]] = {}
    for game in games:
        pbp_df = _fetch_pbp_stateless(game.game_id)
        if pbp_df is None:
            continue
        try:
            events = parse_game_to_events(
                game.game_id, pbp_df, game.home_team, game.visitor_team
            )
            if events["possessions"].empty:
                logger.warning("game %s: parsed 0 possessions — skipping", game.game_id)
                continue
            parsed[game.game_id] = events
            logger.info(
                "game %s: parsed %d possessions",
                game.game_id, len(events["possessions"]),
            )
        except Exception:
            logger.exception("game %s: parse_game_to_events raised — skipping", game.game_id)

        time.sleep(1)  # nba_api rate limit courtesy

    logger.info("Phase 1: parsed %d/%d games", len(parsed), len(games))
    return parsed


# ---------------------------------------------------------------------------
# Phase 2: Build possession_flat
# ---------------------------------------------------------------------------

def _get_unprocessed_games(game_ids: list[str]) -> list[str]:
    """Return game_ids not already present in features.possession_flat."""
    if not game_ids:
        return []

    conn = _md_connect()
    try:
        placeholders = ", ".join("?" * len(game_ids))
        existing = conn.execute(
            f"SELECT DISTINCT game_id FROM features.possession_flat WHERE game_id IN ({placeholders})",
            game_ids,
        ).fetchall()
        existing_ids = {row[0] for row in existing}
        return [gid for gid in game_ids if gid not in existing_ids]
    except duckdb.CatalogException:
        # Table doesn't exist yet — all games are new
        return game_ids
    finally:
        conn.close()


def _fetch_asof_lineup_ratings(
    game_id: str,
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Fetch the most recent lineup_ratings snapshot on or before game_id,
    relabeling as_of_game_id to game_id so process_game()'s filter passes.
    """
    return conn.execute(
        """
        SELECT
            lr.lineup_id,
            ? AS as_of_game_id,
            lr.net_rating,
            lr.observed_net_rating,
            lr.predicted_net_rating,
            lr.possessions_together,
            lr.shrinkage_weight,
            lr.player_ids
        FROM features.lineup_ratings lr
        WHERE lr.as_of_game_id = (
            SELECT MAX(as_of_game_id) FROM features.lineup_ratings
            WHERE as_of_game_id <= ?
        )
        """,
        [game_id, game_id],
    ).df()


def _fetch_asof_player_ratings(
    game_id: str,
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame | None:
    """Same ASOF pattern for player_ratings. Returns None if empty."""
    df = conn.execute(
        """
        SELECT pr.player_id, pr.player_name, ? AS as_of_game_id, pr.adjusted_plus_minus, pr.games_fitted
        FROM features.player_ratings pr
        WHERE pr.as_of_game_id = (
            SELECT MAX(as_of_game_id) FROM features.player_ratings
            WHERE as_of_game_id <= ?
        )
        """,
        [game_id, game_id],
    ).df()
    return df if not df.empty else None


def _fetch_dim_games_context(
    game_date: date,
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Fetch ±7 days of dim_games context. Read once per pipeline run."""
    start = (game_date - timedelta(days=7)).isoformat()
    end   = (game_date + timedelta(days=7)).isoformat()
    return conn.execute(
        """
        SELECT game_id, game_date, home_team, away_team, home_back_to_back, away_back_to_back
        FROM dim_games
        WHERE game_date BETWEEN ? AND ?
        """,
        [start, end],
    ).df()


def _fetch_player_tier_map(conn: duckdb.DuckDBPyConnection) -> dict[int, int]:
    """Returns {player_id: star_tier} for all players with a tier assigned."""
    rows = conn.execute(
        "SELECT player_id, star_tier FROM dim_players WHERE star_tier IS NOT NULL"
    ).fetchall()
    return {int(row[0]): int(row[1]) for row in rows}


def _get_possession_flat_columns(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return the column names of features.possession_flat in order."""
    return [r[0] for r in conn.execute("DESCRIBE features.possession_flat").fetchall()]


def _insert_possession_flat(
    feature_df: pd.DataFrame,
    game_id: str,
    conn: duckdb.DuckDBPyConnection,
) -> int:
    """Insert new possession_flat rows. Returns count inserted.

    Aligns the feature DataFrame to the target table's columns before inserting,
    dropping extra source columns and filling missing ones with None.
    Uses plain INSERT — callers must ensure game_id isn't already present.
    """
    target_cols = _get_possession_flat_columns(conn)

    aligned = pd.DataFrame(index=feature_df.index)
    for col in target_cols:
        if col in feature_df.columns:
            aligned[col] = feature_df[col]
        else:
            aligned[col] = None

    conn.register("_feature_batch", aligned)
    conn.execute("INSERT INTO features.possession_flat SELECT * FROM _feature_batch")
    conn.unregister("_feature_batch")
    return len(aligned)


def build_possession_flat_phase(
    parsed: dict[str, dict[str, pd.DataFrame]],
    unprocessed: list[str],
    games_df: pd.DataFrame,
    tier_map: dict[int, int],
) -> None:
    """
    Phase 2: Build possession_flat rows for all unprocessed games.
    Uses ASOF ratings — point-in-time correct (ratings from BEFORE tonight).
    One game at a time to keep peak memory flat.
    """
    conn = _md_connect()
    try:
        total_inserted = 0
        for game_id in unprocessed:
            if game_id not in parsed:
                logger.warning("game %s: not in parsed dict — skipping possession_flat", game_id)
                continue
            try:
                lineup_ratings = _fetch_asof_lineup_ratings(game_id, conn)
                player_ratings = _fetch_asof_player_ratings(game_id, conn)
                events         = parsed[game_id]

                feature_df = process_game(
                    game_id,
                    events["possessions"],
                    events["fouls"],
                    games_df,
                    lineup_ratings,
                    player_ratings=player_ratings,
                    timeout_events=events.get("timeouts"),
                    player_tier_map=tier_map,
                )

                n_inserted = _insert_possession_flat(feature_df, game_id, conn)
                total_inserted += n_inserted
                logger.info("game %s: inserted %d possession_flat rows", game_id, n_inserted)

                del lineup_ratings, player_ratings, feature_df

            except Exception:
                logger.exception("game %s: possession_flat build failed — skipping", game_id)

        logger.info("Phase 2: inserted %d total possession_flat rows", total_inserted)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 3a: Per-game lineup metrics
# ---------------------------------------------------------------------------

def compute_lineup_game_ratings(
    parsed: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """
    For each game tonight, compute per-lineup net rating and possession count.
    Returns DataFrame: lineup_id, game_id, this_game_net_rating, poss_count
    """
    records: list[dict] = []
    for game_id, events in parsed.items():
        poss = events["possessions"]
        if poss.empty:
            continue

        home_pts = np.where(poss["team_scored"] == "home", poss["points"].astype(float), 0.0)
        away_pts = np.where(poss["team_scored"] == "away", poss["points"].astype(float), 0.0)

        # Home lineups: positive net = scored more than conceded
        home_df = pd.DataFrame({
            "lineup_id": poss["home_lineup_id"],
            "net_pts":   home_pts - away_pts,
            "game_id":   game_id,
        })
        away_df = pd.DataFrame({
            "lineup_id": poss["away_lineup_id"],
            "net_pts":   away_pts - home_pts,
            "game_id":   game_id,
        })

        combined = pd.concat([home_df, away_df], ignore_index=True)
        grouped  = combined.groupby(["lineup_id", "game_id"]).agg(
            net_pts    = ("net_pts", "sum"),
            poss_count = ("net_pts", "count"),
        ).reset_index()

        grouped["this_game_net_rating"] = (
            grouped["net_pts"] / grouped["poss_count"].clip(lower=1) * 100
        )
        records.append(grouped[["lineup_id", "game_id", "this_game_net_rating", "poss_count"]])

    if not records:
        return pd.DataFrame(columns=["lineup_id", "game_id", "this_game_net_rating", "poss_count"])
    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 3b: Player ratings via Ridge regression
# ---------------------------------------------------------------------------

def _build_lineup_player_map_from_md(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, frozenset[int]]:
    """
    Build {lineup_id: frozenset[player_ids]} from MotherDuck lineup_ratings.player_ids.
    Avoids needing PBP cache files (ephemeral on Fly.io).
    """
    rows = conn.execute(
        "SELECT DISTINCT lineup_id, player_ids FROM features.lineup_ratings WHERE player_ids != ''"
    ).fetchall()

    lineup_map: dict[str, frozenset[int]] = {}
    for lineup_id, player_ids_str in rows:
        try:
            pids = frozenset(int(p) for p in player_ids_str.split(",") if p.strip())
            lineup_map[lineup_id] = pids
        except (ValueError, AttributeError):
            continue
    return lineup_map


def compute_player_ratings_ridge(
    as_of_game_id: str,
    lineup_player_map: dict[str, frozenset[int]],
) -> pd.DataFrame:
    """
    Run one Ridge regression fit on all season possessions prior to as_of_game_id.
    Same algorithm as player_rapm.py — exponential decay weights + Ridge(alpha=1000).
    Returns a DataFrame with a single as_of_game_id row per player.
    """
    conn = _md_connect()
    try:
        poss = conn.execute(
            """
            SELECT game_id, home_lineup_id, away_lineup_id, team_scored, points
            FROM possession_feed
            WHERE game_id < ?
            ORDER BY game_id
            """,
            [as_of_game_id],
        ).df()

        games = conn.execute(
            "SELECT game_id, game_date FROM dim_games ORDER BY game_date"
        ).df()

        # Fetch player names from existing ratings (best-effort lookup)
        player_names_df = conn.execute(
            "SELECT DISTINCT player_id, player_name FROM features.player_ratings"
        ).df()
        player_names: dict[int, str] = dict(
            zip(player_names_df["player_id"].astype(int), player_names_df["player_name"])
        )
    finally:
        conn.close()

    if poss.empty:
        logger.warning("compute_player_ratings_ridge: no prior possessions found — skipping")
        return pd.DataFrame()

    # Build chronological game ordering
    games_sorted  = games.sort_values("game_date").reset_index(drop=True)
    game_to_order = {gid: i for i, gid in enumerate(games_sorted["game_id"])}

    poss["game_order"] = poss["game_id"].map(game_to_order)
    poss = poss.dropna(subset=["game_order"]).copy()
    poss["game_order"] = poss["game_order"].astype(int)

    # Build player index from lineup_player_map
    all_players   = sorted({p for players in lineup_player_map.values() for p in players})
    player_idx    = {pid: col for col, pid in enumerate(all_players)}
    idx_player    = {col: pid for pid, col in player_idx.items()}
    n_players     = len(player_idx)
    n             = len(poss)

    logger.info(
        "Building RAPM matrix: %d possessions × %d players (as_of=%s)",
        n, n_players, as_of_game_id,
    )

    X           = lil_matrix((n, n_players), dtype=np.float32)
    y           = np.zeros(n, dtype=np.float32)
    order_arr   = np.zeros(n, dtype=np.int32)
    max_order   = int(poss["game_order"].max()) if not poss.empty else 0

    for i, row in enumerate(poss.itertuples(index=False)):
        home_pl = lineup_player_map.get(row.home_lineup_id)
        away_pl = lineup_player_map.get(row.away_lineup_id)
        if home_pl is None or away_pl is None:
            continue

        if pd.isna(row.points):
            continue
        scored_home = row.team_scored == "home"
        pts         = float(row.points)

        for p in home_pl:
            col = player_idx.get(p)
            if col is not None:
                X[i, col] = 1.0 if scored_home else -1.0
        for p in away_pl:
            col = player_idx.get(p)
            if col is not None:
                X[i, col] = -1.0 if scored_home else 1.0

        y[i]         = pts
        order_arr[i] = int(row.game_order)

    X_csr   = X.tocsr()
    decay   = np.log(2) / DECAY_HALFLIFE_GAMES
    weights = np.exp(-decay * (max_order - order_arr)).astype(np.float32)

    reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    reg.fit(X_csr, y, sample_weight=weights)

    games_fitted = int(poss["game_id"].nunique())
    records = [
        {
            "player_id":           idx_player[col],
            "player_name":         player_names.get(idx_player[col], ""),
            "as_of_game_id":       as_of_game_id,
            "adjusted_plus_minus": float(reg.coef_[col]) * 100,
            "games_fitted":        games_fitted,
        }
        for col in range(n_players)
    ]

    logger.info(
        "Ridge regression done — %d player ratings for as_of=%s",
        len(records), as_of_game_id,
    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Phase 3c: Lineup EWMA update
# ---------------------------------------------------------------------------

def _parse_player_ids(player_ids_str: str) -> list[int]:
    if not player_ids_str:
        return []
    try:
        return [int(p) for p in player_ids_str.split(",") if p.strip()]
    except ValueError:
        return []


def apply_lineup_ewma(
    current_ratings: dict[str, dict],
    game_ratings: pd.DataFrame,
    updated_player_apm: dict[int, float],
    as_of_game_id: str,
) -> pd.DataFrame:
    """
    Compute updated lineup_ratings rows for all lineups that played tonight.

    current_ratings: {lineup_id: row dict} — most recent prior snapshot
    game_ratings: DataFrame with lineup_id, this_game_net_rating, poss_count columns
    updated_player_apm: {player_id: adjusted_plus_minus} — from tonight's Ridge fit
    """
    records: list[dict] = []

    for row in game_ratings.itertuples(index=False):
        lineup_id = row.lineup_id
        prior     = current_ratings.get(lineup_id)

        if prior is None:
            prev_observed  = 0.0
            prev_poss      = 0
            prev_games     = 0
            player_ids_str = ""
        else:
            prev_observed  = float(prior.get("observed_net_rating", 0.0) or 0.0)
            prev_poss      = int(prior.get("possessions_together", 0) or 0)
            games_together = prior.get("games_together")
            prev_games     = (
                int(games_together)
                if games_together is not None
                else prev_poss // 25
            )
            player_ids_str = str(prior.get("player_ids", "") or "")

        n_games     = prev_games + 1
        alpha       = max(FIXED_ALPHA, 1.0 / n_games)
        new_observed = alpha * float(row.this_game_net_rating) + (1 - alpha) * prev_observed
        new_poss     = prev_poss + int(row.poss_count)

        # Predicted = mean APM of the 5 players using tonight's updated ratings
        player_ids = _parse_player_ids(player_ids_str)
        apm_values = [updated_player_apm[pid] for pid in player_ids if pid in updated_player_apm]
        predicted  = float(np.mean(apm_values)) if apm_values else 0.0

        w          = new_poss / (new_poss + SHRINKAGE_K)
        net_rating = w * new_observed + (1 - w) * predicted

        records.append({
            "lineup_id":            lineup_id,
            "as_of_game_id":        as_of_game_id,
            "net_rating":           net_rating,
            "observed_net_rating":  new_observed,
            "predicted_net_rating": predicted,
            "possessions_together": new_poss,
            "shrinkage_weight":     w,
            "player_ids":           player_ids_str,
            "games_together":       n_games,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Phase 3: Orchestrate rating updates
# ---------------------------------------------------------------------------

def _fetch_current_lineup_ratings(
    lineup_ids: list[str],
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, dict]:
    """
    Fetch the most recent lineup_ratings row per lineup_id.
    Returns {lineup_id: row_as_dict}.
    """
    if not lineup_ids:
        return {}

    placeholders = ", ".join("?" * len(lineup_ids))
    rows = conn.execute(
        f"""
        SELECT lineup_id, observed_net_rating, predicted_net_rating,
               possessions_together, games_together, player_ids
        FROM features.lineup_ratings
        WHERE lineup_id IN ({placeholders})
          AND as_of_game_id = (SELECT MAX(as_of_game_id) FROM features.lineup_ratings)
        """,
        lineup_ids,
    ).fetchdf()

    return {
        row["lineup_id"]: row.to_dict()
        for _, row in rows.iterrows()
    }


def _insert_player_ratings(df: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> int:
    """Insert new player rating rows. Skips if as_of_game_id already exists."""
    if df.empty:
        return 0
    as_of = df["as_of_game_id"].iloc[0]
    already = conn.execute(
        "SELECT COUNT(*) FROM features.player_ratings WHERE as_of_game_id = ?", [as_of]
    ).fetchone()[0]
    if already > 0:
        logger.info("player_ratings as_of=%s already exists — skipping insert", as_of)
        return 0
    conn.register("_player_batch", df)
    conn.execute("INSERT INTO features.player_ratings BY NAME SELECT * FROM _player_batch")
    conn.unregister("_player_batch")
    return len(df)


def _insert_lineup_ratings(df: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> int:
    """Insert new lineup rating rows. Skips if as_of_game_id already exists."""
    if df.empty:
        return 0
    as_of = df["as_of_game_id"].iloc[0]
    already = conn.execute(
        "SELECT COUNT(*) FROM features.lineup_ratings WHERE as_of_game_id = ?", [as_of]
    ).fetchone()[0]
    if already > 0:
        logger.info("lineup_ratings as_of=%s already exists — skipping insert", as_of)
        return 0
    conn.register("_lineup_batch", df)
    conn.execute("INSERT INTO features.lineup_ratings BY NAME SELECT * FROM _lineup_batch")
    conn.unregister("_lineup_batch")
    return len(df)


def update_ratings_phase(
    parsed: dict[str, dict[str, pd.DataFrame]],
    as_of_game_id: str,
) -> None:
    """
    Phase 3: Update player_ratings and lineup_ratings with tonight's data.
    Must run AFTER Phase 2 (possession_flat) to preserve point-in-time correctness.
    """
    conn = _md_connect()
    try:
        lineup_player_map = _build_lineup_player_map_from_md(conn)
        logger.info("Phase 3: lineup_player_map has %d lineups", len(lineup_player_map))
    finally:
        conn.close()

    # 3a: Per-game lineup metrics
    game_ratings = compute_lineup_game_ratings(parsed)
    if game_ratings.empty:
        logger.warning("Phase 3: no lineup game ratings computed — skipping rating updates")
        return

    # 3b: Player ratings via Ridge regression
    player_ratings_df = compute_player_ratings_ridge(as_of_game_id, lineup_player_map)
    updated_player_apm: dict[int, float] = {}
    if not player_ratings_df.empty:
        updated_player_apm = dict(
            zip(
                player_ratings_df["player_id"].astype(int),
                player_ratings_df["adjusted_plus_minus"].astype(float),
            )
        )

    # 3c: Lineup EWMA update
    unique_lineup_ids = game_ratings["lineup_id"].unique().tolist()
    conn = _md_connect()
    try:
        current_lineup_ratings = _fetch_current_lineup_ratings(unique_lineup_ids, conn)
        new_lineup_df = apply_lineup_ewma(
            current_lineup_ratings, game_ratings, updated_player_apm, as_of_game_id
        )

        n_players = _insert_player_ratings(player_ratings_df, conn)
        n_lineups = _insert_lineup_ratings(new_lineup_df, conn)
        logger.info(
            "Phase 3: inserted %d player rating rows, %d lineup rating rows (as_of=%s)",
            n_players, n_lineups, as_of_game_id,
        )
    finally:
        conn.close()


def update_team_ratings_phase(parsed: dict[str, dict[str, pd.DataFrame]], as_of_game_id: str) -> None:
    """
    Phase 3d: Update features.team_ratings (EWMA + Shrinkage) with tonight's data.
    """
    from models.ratings.team_ratings import compute_game_team_stats, apply_team_ewma
    
    conn = _md_connect()
    try:
        # Dynamically compute running league average roughly safely
        league_avg_res = conn.execute("""
            SELECT AVG(pts * 100.0 / NULLIF(poss, 0)) FROM (
                SELECT SUM(points) as pts, COUNT(*) as poss
                FROM possession_feed WHERE possessing_team IN ('home', 'away')
                GROUP BY game_id, possessing_team
            )
        """).fetchone()
        league_avg_ortg = league_avg_res[0] if (league_avg_res and league_avg_res[0]) else 110.0

        for game_id, dfs in parsed.items():
            if "possession_feed" not in dfs: continue
            raw_stats = compute_game_team_stats(dfs["possession_feed"])
            if not raw_stats: continue
            
            # For each team (home/away)
            for side, tricode_col in [('home', 'home_team'), ('away', 'away_team')]:
                tricode_res = conn.execute(f"SELECT {tricode_col} FROM dim_games WHERE game_id='{game_id}'").fetchone()
                if not tricode_res: continue
                tricode = tricode_res[0]
                
                curr_state = conn.execute(f"""
                    SELECT games_played, ewma_off_rating, ewma_def_rating, last_5_net_ratings 
                    FROM features.team_ratings WHERE team_tricode='{tricode}' 
                    ORDER BY as_of_game_id DESC LIMIT 1
                """).df()
                state_dict = curr_state.iloc[0].to_dict() if not curr_state.empty else {}
                
                raw = raw_stats[side]
                updated = apply_team_ewma(state_dict, raw['ortg'], raw['drtg'], float(league_avg_ortg))
                
                # Cumulative pace average across this team's games up to tonight
                pace_res = conn.execute(f"""
                    SELECT AVG(pf.pace_season_baseline) 
                    FROM possession_feed pf JOIN dim_games g ON pf.game_id = g.game_id 
                    WHERE (g.home_team='{tricode}' OR g.away_team='{tricode}') 
                      AND pf.pace_season_baseline > 0 AND g.game_id <= '{game_id}'
                """).fetchone()
                pace = pace_res[0] if (pace_res and pace_res[0]) else 15.0
                
                # Replace logic ensures updates if ran twice
                conn.execute(f"""
                    INSERT OR REPLACE INTO features.team_ratings (
                        team_tricode, as_of_game_id, games_played, ewma_off_rating, ewma_def_rating, ewma_net_rating,
                        off_rating, def_rating, net_rating, avg_secs_per_poss, last_5_net_ratings
                    ) VALUES (
                        '{tricode}', '{as_of_game_id}', {updated['games_played']}, 
                        {updated['ewma_off_rating']}, {updated['ewma_def_rating']}, {updated['ewma_net_rating']},
                        {updated['off_rating']}, {updated['def_rating']}, {updated['net_rating']},
                        {pace}, '{updated['last_5_net_ratings']}'
                    )
                """)
        logger.info("Phase 3d: updated team_ratings for tonight's games")
    finally:
        conn.close()


def build_pregame_features_phase(games: list, game_date: date) -> None:
    """
    Phase 4: Compute 10 pregame features for tonight's games using purely ASOF state.
    """
    from models.features.pregame_features import compute_pregame_features
    import pandas as pd
    conn = _md_connect()
    try:
        inserted = 0
        for game in games:
            features = compute_pregame_features(
                conn=conn, game_id=game.game_id, game_date=str(game_date), 
                home_team=game.home_team, away_team=game.visitor_team
            )
            df = pd.DataFrame([features])
            conn.execute("INSERT OR REPLACE INTO features.pregame SELECT * FROM df")
            inserted += 1
        logger.info("Phase 4: inserted %d games into features.pregame", inserted)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_post_game_pipeline(game_date: date, games: list[GameInfo]) -> None:
    """
    Main entry point. Called by recorder_daemon at 3 AM ET.

    Phases:
      0  Upsert dim_games (game_ids must exist before other tables reference them)
      1  Parse PBP for tonight's games
      2  Build possession_flat (uses pre-tonight ASOF ratings — point-in-time correct)
      3  Update player_ratings + lineup_ratings (uses tonight's data, for tomorrow)
    """
    if not games:
        logger.info("run_post_game_pipeline: no games for %s — exiting", game_date.isoformat())
        return

    logger.info(
        "Post-game pipeline starting for %s (%d games)",
        game_date.isoformat(), len(games),
    )

    # Phase 0
    upsert_dim_games(games, game_date)

    # Phase 4 (Moved up to ensure pregame features are calculated before games finish)
    build_pregame_features_phase(games, game_date)

    # Phase 1
    parsed = fetch_and_parse_all(games)
    if not parsed:
        logger.warning("Post-game pipeline: no games parsed successfully — exiting")
        return

    # Phase 2
    unprocessed = _get_unprocessed_games(list(parsed.keys()))
    if not unprocessed:
        logger.info("Phase 2: all %d games already in possession_flat — skipping", len(parsed))
    else:
        conn = _md_connect()
        try:
            games_df = _fetch_dim_games_context(game_date, conn)
            tier_map = _fetch_player_tier_map(conn)
        finally:
            conn.close()
        build_possession_flat_phase(parsed, unprocessed, games_df, tier_map)

    # Phase 2b: Populate wall_clock_ts for tonight's games.
    # Required by the collapsed price movement predictor to ASOF-join to Kalshi ticks.
    # Runs immediately after possession_flat insert so new games are always populated.
    try:
        from data.ingestion.backfill_wall_clock_ts import backfill_game
        conn = _md_connect()
        try:
            for game in games:
                backfill_game(game.game_id, game.tipoff_utc.date(), conn)
        finally:
            conn.close()
    except Exception:
        logger.exception("Phase 2b: wall_clock_ts backfill failed — non-fatal, continuing")

    # Phase 3
    as_of_game_id = max(parsed.keys())
    update_ratings_phase(parsed, as_of_game_id)
    update_team_ratings_phase(parsed, as_of_game_id)

    logger.info(
        "Post-game pipeline complete for %s — as_of_game_id=%s",
        game_date.isoformat(), as_of_game_id,
    )


# ---------------------------------------------------------------------------
# Manual trigger
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import date

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    from dotenv import load_dotenv
    load_dotenv()

    from data.ingestion.game_schedule import get_todays_games

    target_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    games = get_todays_games(target_date)
    if not games:
        logger.info("No games found for %s", target_date.isoformat())
        sys.exit(0)

    run_post_game_pipeline(target_date, games)
