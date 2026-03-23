"""
Rolling point-in-time player ratings via regularized adjusted plus/minus (RAPM).

For each game G, fits a weighted ridge regression on all possessions from games
prior to G. Recent games are weighted more heavily via exponential decay.

The regression matrix:
  X[possession i, player j] = +1 if player j is on the team that scored
                              = -1 if player j is on the team that conceded
  y[i] = points scored on possession i

Ridge coefficients × 100 = player's net contribution per 100 possessions.

Output: data/feature_store/player_ratings.parquet
  player_id, player_name, as_of_game_id, adjusted_plus_minus, games_fitted

Run once after backfill. Takes ~10-20 minutes for 930 games.
"""

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PBP_CACHE_DIR       = Path("data/raw/pbp_cache")
POSSESSIONS_PATH    = Path("data/raw/possessions_202526.parquet")
GAMES_PATH          = Path("data/raw/games_202526.parquet")
FEATURE_STORE_DIR   = Path("data/feature_store")
PLAYER_RATINGS_PATH = FEATURE_STORE_DIR / "player_ratings.parquet"

# Exponential decay: weight = exp(-ln(2)/halflife * games_ago)
# Half-life of 80 games ≈ one month of NBA schedule
DECAY_HALFLIFE_GAMES = 80

# Ridge regularization. Higher = more shrinkage toward zero.
# 1000 is reasonable for ~100 possessions per player per season.
RIDGE_ALPHA = 1000

# Compute ratings at every Nth game; use most recent for games in between.
# Reduces compute from 930 regressions to 930/CHECKPOINT_INTERVAL.
CHECKPOINT_INTERVAL = 5

MIN_POSSESSIONS_TO_FIT = 200  # skip regression until we have enough data

PLAYER_RATINGS_SCHEMA = pa.schema([
    pa.field("player_id",           pa.int64()),
    pa.field("player_name",         pa.string()),
    pa.field("as_of_game_id",       pa.string()),
    pa.field("adjusted_plus_minus", pa.float32()),  # net pts per 100 possessions
    pa.field("games_fitted",        pa.int32()),
])

# ---------------------------------------------------------------------------
# Lineup reconstruction (minimal duplicate of nba_api_client logic)
# ---------------------------------------------------------------------------

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v) if v not in (None, "", "None") else default
    except (ValueError, TypeError):
        return default


def _normalize(name: str) -> str:
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode().lower()


def _lineup_id(player_ids: frozenset[int]) -> str:
    key = ",".join(str(p) for p in sorted(player_ids))
    return hashlib.md5(key.encode()).hexdigest()[:12]


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
    name  = m.group(1).strip()
    norm  = _normalize(name)
    pid   = name_map.get(norm, -1)
    if pid == -1:
        pid = name_map.get(norm.split()[-1], -1)
    return pid


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

    def home_players(self) -> frozenset[int]:
        return frozenset(self._home)

    def away_players(self) -> frozenset[int]:
        return frozenset(self._away)

    def home_id(self) -> str:
        return _lineup_id(self.home_players())

    def away_id(self) -> str:
        return _lineup_id(self.away_players())


def _process_pbp_for_lineups(df: pd.DataFrame) -> dict[str, frozenset[int]]:
    """Return all lineup_id → frozenset[player_id] seen in one game's PBP."""
    result: dict[str, frozenset[int]] = {}
    tracker  = _LineupTracker()
    name_map = _build_name_map(df)

    # Pre-pass: infer starters from first player events before any sub
    for _, row in df.iterrows():
        if tracker.is_ready():
            tracker.lock()
            break
        period      = _safe_int(row.get("period"), 1)
        action_type = str(row.get("actionType", "") or "")
        player_id   = _safe_int(row.get("personId"))
        location    = str(row.get("location", "") or "")

        if period > 1 or action_type == "Substitution":
            tracker.lock()
            break
        if player_id and location:
            tracker.observe(player_id, location)

    result[tracker.home_id()] = tracker.home_players()
    result[tracker.away_id()] = tracker.away_players()

    # Main pass: track substitutions
    for _, row in df.iterrows():
        if str(row.get("actionType", "")) != "Substitution":
            continue
        out_id   = _safe_int(row.get("personId"))
        location = str(row.get("location", "") or "")
        in_id    = _parse_player_in(str(row.get("description", "")), name_map)
        if in_id != -1:
            tracker.substitute(out_id, in_id, location)
        result[tracker.home_id()] = tracker.home_players()
        result[tracker.away_id()] = tracker.away_players()

    return result


def build_lineup_map(pbp_cache_dir: Path) -> dict[str, frozenset[int]]:
    """
    Read all PBP cache files and return lineup_id → frozenset[player_id].
    Starter compositions are only recoverable from raw PBP — not from the
    processed output tables — so we re-read the cache here.
    """
    lineup_map: dict[str, frozenset[int]] = {}
    files = sorted(pbp_cache_dir.glob("*.parquet"))
    logger.info("Building lineup map from %d PBP cache files...", len(files))

    for cache_file in files:
        df = pq.read_table(cache_file).to_pandas()
        lineup_map.update(_process_pbp_for_lineups(df))

    logger.info("Lineup map built: %d unique lineups", len(lineup_map))
    return lineup_map


# ---------------------------------------------------------------------------
# RAPM matrix construction
# ---------------------------------------------------------------------------

def _build_player_index(lineup_map: dict[str, frozenset[int]]) -> dict[int, int]:
    all_players = sorted({p for players in lineup_map.values() for p in players})
    return {pid: col for col, pid in enumerate(all_players)}


def build_rapm_matrix(
    possessions: pd.DataFrame,
    games: pd.DataFrame,
    lineup_map: dict[str, frozenset[int]],
) -> tuple[csr_matrix, np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    """
    Build the full RAPM regression matrix from all possessions.

    Returns:
        X          : sparse (n_possessions × n_players) indicator matrix
        y          : points vector (+ if home scored, - if away scored)
        game_order : integer game index per possession (for decay weighting)
        player_idx : player_id → column index
        idx_player : column index → player_id
    """
    # Sort games chronologically; assign 0-based order index
    games_sorted = games.sort_values("game_date").reset_index(drop=True)
    game_to_order = {gid: i for i, gid in enumerate(games_sorted["game_id"])}

    player_idx = _build_player_index(lineup_map)
    idx_player = {v: k for k, v in player_idx.items()}
    n_players  = len(player_idx)

    # Attach game order to possessions; drop any possession with unknown game
    poss = possessions.copy()
    poss["_order"] = poss["game_id"].map(game_to_order)
    poss = poss.dropna(subset=["_order"]).copy()
    poss["_order"] = poss["_order"].astype(int)

    n = len(poss)
    logger.info("Building RAPM matrix: %d possessions × %d players", n, n_players)

    X = lil_matrix((n, n_players), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    game_order_arr = np.zeros(n, dtype=np.int32)

    missing = 0
    for i, (_, row) in enumerate(poss.iterrows()):
        home_lid = row["home_lineup_id"]
        away_lid = row["away_lineup_id"]
        home_pl  = lineup_map.get(home_lid)
        away_pl  = lineup_map.get(away_lid)

        if home_pl is None or away_pl is None:
            missing += 1
            continue

        scored_home = row["team_scored"] == "home"
        pts = float(row["points"])

        for p in home_pl:
            col = player_idx.get(p)
            if col is not None:
                X[i, col] = 1.0 if scored_home else -1.0
        for p in away_pl:
            col = player_idx.get(p)
            if col is not None:
                X[i, col] = -1.0 if scored_home else 1.0

        y[i]              = pts
        game_order_arr[i] = int(row["_order"])

    if missing:
        logger.warning("%d possessions skipped — lineup not in map", missing)

    return X.tocsr(), y, game_order_arr, player_idx, idx_player


# ---------------------------------------------------------------------------
# Rolling regression
# ---------------------------------------------------------------------------

def compute_rolling_ratings(
    X: csr_matrix,
    y: np.ndarray,
    game_order_arr: np.ndarray,
    games: pd.DataFrame,
    possessions: pd.DataFrame,
    player_idx: dict[int, int],
    idx_player: dict[int, int],
) -> pd.DataFrame:
    """
    For each game G (in chronological order), fit Ridge on all possessions
    from games 0..G-1 with exponential decay weights.

    Ratings are computed at every CHECKPOINT_INTERVAL games to keep runtime
    manageable; games in between receive the most recently computed ratings.
    """
    games_sorted = games.sort_values("game_date").reset_index(drop=True)
    decay        = np.log(2) / DECAY_HALFLIFE_GAMES

    # Build player_name lookup from possessions
    player_names: dict[int, str] = {}
    for _, row in possessions.iterrows():
        pid  = int(row["player_id"])
        name = str(row["player_name"])
        if pid and name and pid not in player_names:
            player_names[pid] = name

    records: list[dict] = []
    last_coefs: np.ndarray | None = None
    last_games_fitted: int = 0

    n_games = len(games_sorted)
    logger.info("Computing rolling ratings for %d games (checkpoint every %d)...",
                n_games, CHECKPOINT_INTERVAL)

    for g_idx, game_row in games_sorted.iterrows():
        game_id    = game_row["game_id"]
        prior_mask = game_order_arr < g_idx

        # Only refit at checkpoints (or first time we have enough data)
        n_prior = int(prior_mask.sum())
        should_fit = (
            n_prior >= MIN_POSSESSIONS_TO_FIT
            and (g_idx % CHECKPOINT_INTERVAL == 0 or last_coefs is None)
        )

        if should_fit:
            X_prior = X[prior_mask]
            y_prior = y[prior_mask]
            orders_prior = game_order_arr[prior_mask]
            weights = np.exp(-decay * (g_idx - orders_prior)).astype(np.float32)

            reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
            reg.fit(X_prior, y_prior, sample_weight=weights)
            last_coefs        = reg.coef_
            last_games_fitted = n_prior // 100  # rough game count estimate

        if last_coefs is None:
            # Not enough data yet — emit zero ratings
            for col, pid in idx_player.items():
                records.append({
                    "player_id":           pid,
                    "player_name":         player_names.get(pid, ""),
                    "as_of_game_id":       game_id,
                    "adjusted_plus_minus": 0.0,
                    "games_fitted":        0,
                })
        else:
            for col, pid in idx_player.items():
                records.append({
                    "player_id":           pid,
                    "player_name":         player_names.get(pid, ""),
                    "as_of_game_id":       game_id,
                    # ×100: convert per-possession → per-100-possessions
                    "adjusted_plus_minus": float(last_coefs[col]) * 100,
                    "games_fitted":        last_games_fitted,
                })

        if g_idx % 100 == 0:
            logger.info("  [%d/%d] %s — %d prior possessions", g_idx, n_games, game_id, n_prior)

    return pd.DataFrame(records)


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

    possessions = pq.read_table(POSSESSIONS_PATH).to_pandas()
    games       = pq.read_table(GAMES_PATH).to_pandas()
    logger.info("Loaded %d possessions, %d games", len(possessions), len(games))

    lineup_map = build_lineup_map(PBP_CACHE_DIR)

    X, y, game_order_arr, player_idx, idx_player = build_rapm_matrix(
        possessions, games, lineup_map
    )

    ratings_df = compute_rolling_ratings(
        X, y, game_order_arr, games, possessions, player_idx, idx_player
    )

    logger.info("Writing %d rating rows to %s", len(ratings_df), PLAYER_RATINGS_PATH)
    arrays = {f.name: [] for f in PLAYER_RATINGS_SCHEMA}
    for _, row in ratings_df.iterrows():
        for f in PLAYER_RATINGS_SCHEMA:
            arrays[f.name].append(row.get(f.name))

    table = pa.table(
        {name: pa.array(vals, type=PLAYER_RATINGS_SCHEMA.field(name).type)
         for name, vals in arrays.items()},
        schema=PLAYER_RATINGS_SCHEMA,
    )
    pq.write_table(table, PLAYER_RATINGS_PATH)
    logger.info("Done — player_ratings.parquet written")

    n_players = ratings_df["player_id"].nunique()
    n_games   = ratings_df["as_of_game_id"].nunique()
    logger.info("  %d unique players × %d game snapshots", n_players, n_games)


if __name__ == "__main__":
    main()
