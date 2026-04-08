"""
Collapsed Price Movement Predictor.

Directly predicts Δ(yes_bid) over the next 120 seconds from basketball game state
+ Kalshi market microstructure + lag/velocity features across the last 3 possessions.

This collapses the old L1 (run classifier) + L2 (price movement) pipeline into one
end-to-end regressor. Instead of predicting "will a run happen?" as an intermediate
step, the model is trained directly on what we care about: will the Kalshi price move
enough to profit, given everything we can observe right now?

Direction: always from the home team's perspective (home spread-1 contract).
  - Positive Δ(yes_bid) = home team gaining ground (BUY YES signal)
  - Negative Δ(yes_bid) = away team gaining ground (BUY NO signal)
  - Entry signal threshold: abs(prediction) > 1.75¢ (maker fee floor at 100 contracts)

Data source: MotherDuck — features.possession_flat + kalshi_ticks
  - Kalshi tick data available from ~March 23, 2026 onward
  - Training set is intentionally small until more games are recorded
  - Run this monthly to retrain as tick data accumulates

Train/val split: 80/20 chronological (no fixed dates — all data is post-March 2026)

Usage:
    python models/kalshi_price_movement_predictor.py
"""

import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.features.kalshi_features import add_kalshi_features
from models.targets.kalshi_targets import add_kalshi_targets

logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = Path("models/saved/kalshi_price_movement_predictor.pkl")
TARGET_COL = "target_delta_bid"

# Minimum games required before the model is considered meaningful
MIN_GAMES_WARNING = 20

# ── Feature columns ──────────────────────────────────────────────────────────

# Basketball game state — same 58 features as run_predictor.py
# These capture the full game context at each possession.
BASKETBALL_FEATURES: list[str] = [
    # Score × time
    "score_diff",
    "period",
    "minutes_into_game",
    "trailing_team_urgency",
    "comeback_probability_proxy",
    "q4_close_game",
    "garbage_time_risk",
    # Shot info
    "shot_value",
    "shot_distance",
    # Momentum
    "home_points_last_5_poss",
    "away_points_last_5_poss",
    "home_points_last_10_poss",
    "away_points_last_10_poss",
    "current_run_team_encoded",
    "current_run_length",
    "current_run_points",
    "current_run_3pt_pct",
    "current_run_paint_pct",
    # Pace
    "pace_last_10_possessions",
    "pace_season_baseline",
    # Shot quality
    "home_scoring_sustainable",
    "away_scoring_sustainable",
    "home_xPPP_last_5",
    "away_xPPP_last_5",
    "home_actual_vs_expected_PPP",
    "away_actual_vs_expected_PPP",
    "home_shot_quality_trend",
    "away_shot_quality_trend",
    # Foul state
    "home_team_fouls_q",
    "away_team_fouls_q",
    "home_cum_fouls",
    "away_cum_fouls",
    "home_in_bonus",
    "away_in_bonus",
    "home_fouls_until_bonus",
    "away_fouls_until_bonus",
    "home_star_in_foul_trouble",
    "away_star_in_foul_trouble",
    "home_star_on_court",
    "away_star_on_court",
    # Event context
    "was_foul",
    "was_sub",
    "had_shooting_foul",
    "had_personal_foul",
    "home_sub_count",
    "away_sub_count",
    # Timeout signals
    "possessions_since_last_timeout",
    "home_called_timeout_in_last_3_poss",
    "away_called_timeout_in_last_3_poss",
    "home_full_timeouts_remaining",
    "away_full_timeouts_remaining",
    # Lineup signal
    "home_lineup_net_rating",
    "away_lineup_net_rating",
    "lineup_net_rating_delta",
    "home_lineup_sample_size",
    "away_lineup_sample_size",
    "home_lineup_just_changed",
    "away_lineup_just_changed",
]

# Kalshi microstructure at the moment of each possession
KALSHI_FEATURES: list[str] = [
    "yes_bid",
    "yes_ask",
    "spread",
    "trade_volume_60s",
    "time_since_last_trade_ms",
]

# Features whose trajectory across the last 3 possessions carries signal.
# Chosen because their rate of change matters: is the matchup edge growing?
# Is Kalshi already starting to move before we enter?
LAG_COLS: list[str] = [
    "lineup_net_rating_delta",       # matchup advantage growing or stable?
    "current_run_length",            # run accelerating?
    "current_run_points",            # run magnitude building?
    "score_diff",                    # game state trajectory
    "yes_bid",                       # Kalshi already repricing? (KEY new signal)
    "spread",                        # liquidity changing?
    "home_lineup_just_changed",      # how recent was the sub?
    "away_lineup_just_changed",
    "home_points_last_5_poss",
    "away_points_last_5_poss",
]

# Generated lag column names: prev1_*, prev2_*, prev3_* for each LAG_COL
LAG_FEATURE_COLS: list[str] = [
    f"prev{n}_{col}"
    for n in [1, 2, 3]
    for col in LAG_COLS
]

# Velocity = change between consecutive possessions.
# d_yes_bid is the most important: if Kalshi is already moving, edge is smaller.
VELOCITY_FEATURE_COLS: list[str] = [
    "d_lineup_net_rating_delta",
    "d_current_run_length",
    "d_score_diff",
    "d_yes_bid",
    "d_spread",
]

FEATURE_COLS: list[str] = (
    BASKETBALL_FEATURES + KALSHI_FEATURES + LAG_FEATURE_COLS + VELOCITY_FEATURE_COLS
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_possession_flat(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Pull possession_flat joined with game metadata from MotherDuck.
    Excludes blowouts and garbage time — model is off during these.
    """
    df = conn.execute("""
        SELECT
            pf.*,
            dg.game_date,
            dg.home_team,
            dg.away_team
        FROM kalshi_trading.features.possession_flat pf
        JOIN kalshi_trading.main.dim_games dg ON pf.game_id = dg.game_id
        WHERE NOT pf.is_blowout
          AND NOT pf.is_garbage_time
          AND pf.wall_clock_ts IS NOT NULL
        ORDER BY pf.game_id, pf.event_id
    """).df()
    logger.info("Loaded %d possession rows from possession_flat", len(df))
    return df




def _load_kalshi_ticks(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Pull all recorded Kalshi ticks from MotherDuck.
    Data available from ~March 23, 2026 onward.
    Note: game_id is not reliably populated in this table — we derive it by
    parsing the market_ticker and joining to dim_games.
    """
    df = conn.execute("""
        SELECT market_ticker, ts, yes_bid, yes_ask, volume
        FROM kalshi_trading.main.kalshi_ticks
        ORDER BY ts
    """).df()
    logger.info("Loaded %d Kalshi tick rows", len(df))
    return df


_MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def _parse_ticker(ticker: str) -> dict:
    """
    Parse a Kalshi NBA spread ticker into its components.

    Format: KXNBASPREAD-YYMONDDAWYAWAYHOME-TEAMSPREAD
    Example: KXNBASPREAD-26MAR25DALDEN-DEN2
      → game_date=2026-03-25, away=DAL, home=DEN, contract_team=DEN, spread_val=2
    """
    parts = ticker.split("-")
    if len(parts) != 3:
        return {}
    middle = parts[1]   # e.g. "26MAR25DALDEN"
    suffix = parts[2]   # e.g. "DEN2"

    year = "20" + middle[:2]
    month = _MONTH_MAP.get(middle[2:5], "01")
    day = middle[5:7]
    teams = middle[7:]  # e.g. "DALDEN"

    if len(teams) < 6 or not suffix[-1].isdigit():
        return {}

    return {
        "game_date_str": f"{year}-{month}-{day}",
        "away_team":     teams[:3],
        "home_team":     teams[3:6],
        "contract_team": suffix[:-1],
        "spread_val":    int(suffix[-1]),
    }


def _select_home_best_contract(
    ticks: pd.DataFrame,
    possessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each game, select the home team's contract with the lowest recorded
    spread number (1 preferred, then 2, then 3). Assigns game_id by joining
    parsed ticker fields to dim_games info from possessions.

    Not every game has a spread-1 contract — we take whatever is closest to
    moneyline for that game.
    """
    parsed = ticks["market_ticker"].apply(_parse_ticker).apply(pd.Series)
    ticks = pd.concat([ticks, parsed], axis=1)
    ticks = ticks.dropna(subset=["home_team", "contract_team", "spread_val"])

    # Keep only home team contracts
    ticks = ticks[ticks["contract_team"] == ticks["home_team"]].copy()

    # Build date + team → game_id lookup from possessions (already has game_date, home_team)
    game_lookup = (
        possessions[["game_id", "game_date", "home_team", "away_team"]]
        .drop_duplicates("game_id")
        .assign(game_date=lambda d: pd.to_datetime(d["game_date"]).dt.strftime("%Y-%m-%d"))
    )
    ticks = ticks.merge(
        game_lookup,
        left_on=["game_date_str", "home_team", "away_team"],
        right_on=["game_date", "home_team", "away_team"],
        how="left",
    ).drop(columns=["game_date", "game_date_str"], errors="ignore")

    unmatched = ticks["game_id"].isna().sum()
    if unmatched > 0:
        logger.warning(
            "%d ticks could not be matched to a game_id via ticker parsing "
            "(date/team mismatch). They will be dropped.",
            unmatched,
        )
    ticks = ticks.dropna(subset=["game_id"])

    # Per game, keep only the lowest spread_val contract
    min_spread = ticks.groupby("game_id")["spread_val"].min()
    ticks = ticks[ticks.apply(lambda r: r["spread_val"] == min_spread[r["game_id"]], axis=1)]

    logger.info(
        "Selected home best-contract ticks: %d rows across %d games",
        len(ticks), ticks["game_id"].nunique(),
    )
    return ticks


# ── Feature engineering ───────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features derived from raw possession_flat columns.
    Mirrors run_predictor._add_derived_features() exactly.
    """
    out = df.copy()

    elapsed_in_period = (720 - out["game_clock_secs"].clip(0, 720)) / 60
    out["minutes_into_game"] = (out["period"] - 1) * 12 + elapsed_in_period

    # Fill NULLs for games where foul/timeout tracking wasn't recorded
    for col in ("home_team_fouls_q", "away_team_fouls_q",
                "home_timeouts_used", "away_timeouts_used"):
        out[col] = out[col].fillna(0)

    out["home_in_bonus"] = (out["home_team_fouls_q"] >= 5).astype(int)
    out["away_in_bonus"] = (out["away_team_fouls_q"] >= 5).astype(int)
    out["home_fouls_until_bonus"] = (5 - out["home_team_fouls_q"]).clip(lower=0)
    out["away_fouls_until_bonus"] = (5 - out["away_team_fouls_q"]).clip(lower=0)

    out["home_full_timeouts_remaining"] = (4 - out["home_timeouts_used"]).clip(lower=0)
    out["away_full_timeouts_remaining"] = (4 - out["away_timeouts_used"]).clip(lower=0)

    out["current_run_team_encoded"] = (
        out["current_run_team"].map({"home": 1, "away": -1}).fillna(0)
    )

    out["home_star_in_foul_trouble"] = (out["home_trouble_star_tier"] > 0).astype(float)
    out["away_star_in_foul_trouble"] = (out["away_trouble_star_tier"] > 0).astype(float)

    return out


def _join_ticks_to_possessions(
    possessions: pd.DataFrame,
    ticks: pd.DataFrame,
) -> pd.DataFrame:
    """
    ASOF backward join: for each possession, attach the most recent Kalshi tick
    at or before that possession's wall_clock_ts.

    Uses merge_asof per game_id to avoid cross-game contamination.
    """
    possessions["wall_clock_ts"] = pd.to_datetime(possessions["wall_clock_ts"], utc=True)
    ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True)

    parts = []
    for game_id in possessions["game_id"].unique():
        poss_game = possessions[possessions["game_id"] == game_id].sort_values("wall_clock_ts")
        tick_game = ticks[ticks["game_id"] == game_id].sort_values("ts")

        if tick_game.empty:
            # No ticks for this game — possessions will have NaN kalshi columns
            parts.append(poss_game)
            continue

        merged = pd.merge_asof(
            poss_game,
            tick_game[["ts", "yes_bid", "yes_ask", "volume"]],
            left_on="wall_clock_ts",
            right_on="ts",
            direction="backward",
        )
        # merge_asof retains the right key column — drop it to avoid duplicate 'ts'
        # when we later rename wall_clock_ts → ts for kalshi_features
        merged = merged.drop(columns=["ts"], errors="ignore")
        parts.append(merged)

    result = pd.concat(parts, ignore_index=True)

    # Drop possessions with no Kalshi data yet — can't compute target or microstructure
    before = len(result)
    result = result.dropna(subset=["yes_bid"])
    dropped = before - len(result)
    if dropped > 0:
        logger.info("Dropped %d possessions with no tick data (no Kalshi coverage)", dropped)

    return result


def _add_lag_and_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-possession lag features (t-1, t-2, t-3) and velocity features
    for the columns in LAG_COLS. Computed within each game to prevent
    cross-game contamination.

    Rows at the start of a game (fewer than 3 prior possessions) get 0.0 fill
    for missing lags — XGBoost handles this gracefully.
    """
    out = df.copy()
    out = out.sort_values(["game_id", "event_id"]).reset_index(drop=True)

    # Cast all lag source columns to float so shifted/filled values are numeric.
    # Boolean columns (lineup_just_changed) come through as object dtype otherwise.
    for col in LAG_COLS:
        if col in out.columns:
            out[col] = out[col].astype(float)

    for col in LAG_COLS:
        if col not in out.columns:
            logger.warning("LAG_COL '%s' not found — filling lags with 0", col)
            for n in [1, 2, 3]:
                out[f"prev{n}_{col}"] = 0.0
            continue

        for n in [1, 2, 3]:
            out[f"prev{n}_{col}"] = (
                out.groupby("game_id")[col]
                .shift(n)
                .fillna(0)
            )

    # Velocity = change from t-1 to t (captures acceleration, not just level)
    velocity_map = {
        "d_lineup_net_rating_delta": "lineup_net_rating_delta",
        "d_current_run_length":      "current_run_length",
        "d_score_diff":              "score_diff",
        "d_yes_bid":                 "yes_bid",
        "d_spread":                  "spread",
    }
    for vel_col, base_col in velocity_map.items():
        if base_col in out.columns and f"prev1_{base_col}" in out.columns:
            out[vel_col] = out[base_col] - out[f"prev1_{base_col}"]
        else:
            out[vel_col] = 0.0

    return out


def _prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast booleans, fill missing values, and return a clean numeric matrix
    aligned to FEATURE_COLS.
    """
    out = df.copy()

    bool_cols = [
        "home_scoring_sustainable", "away_scoring_sustainable",
        "q4_close_game",
        "home_in_bonus", "away_in_bonus",
        "home_star_in_foul_trouble", "away_star_in_foul_trouble",
        "home_star_on_court", "away_star_on_court",
        "was_foul", "was_sub",
        "had_shooting_foul", "had_personal_foul",
        "home_called_timeout_in_last_3_poss", "away_called_timeout_in_last_3_poss",
        "home_lineup_just_changed", "away_lineup_just_changed",
    ]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].astype(float)

    available = [c for c in FEATURE_COLS if c in out.columns]
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing:
        logger.warning("Features missing from data (filling with 0): %s", missing)

    result = out[available].copy()
    for col in missing:
        result[col] = 0.0

    return result[FEATURE_COLS].fillna(0)


# ── Dataset construction ──────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    """
    Pull data from MotherDuck, join tick data to possessions, compute all features
    and targets. Returns a DataFrame ready for train/val split.

    NOTE: Kalshi tick data starts ~March 23, 2026. The resulting dataset will be
    small until more games are recorded. Model reliability improves significantly
    after 4+ weeks of tick coverage.
    """
    load_dotenv()
    conn = duckdb.connect("md:")

    possessions = _load_possession_flat(conn)
    ticks = _load_kalshi_ticks(conn)
    conn.close()

    if possessions.empty or ticks.empty:
        raise RuntimeError(
            "Missing data from MotherDuck. "
            "Check MOTHERDUCK_TOKEN and verify kalshi_ticks / possession_flat are populated."
        )

    # Filter ticks to the home team's best (lowest spread) contract per game.
    # game_id is not reliable in raw ticks — derived by parsing the ticker.
    ticks = _select_home_best_contract(ticks, possessions)
    if ticks.empty:
        raise RuntimeError(
            "No home-team contracts found after ticker parsing. "
            "Check _parse_ticker and verify ticker format matches KXNBASPREAD-YYMONDDAWYAWAYHOME-TEAMSPREAD."
        )

    n_games_with_ticks = ticks["game_id"].nunique()
    if n_games_with_ticks < MIN_GAMES_WARNING:
        logger.warning(
            "Only %d games have Kalshi tick data. Model needs %d+ games to be meaningful. "
            "Keep the recorder running — results will improve as data accumulates.",
            n_games_with_ticks, MIN_GAMES_WARNING,
        )

    # Compute derived basketball features
    possessions = _add_derived_features(possessions)

    # Join current Kalshi state to each possession (backward ASOF)
    df = _join_ticks_to_possessions(possessions, ticks)

    # Add Kalshi microstructure features (spread, rolling volume, time_since_trade)
    df = add_kalshi_features(df.rename(columns={"wall_clock_ts": "ts"}))
    df = df.rename(columns={"ts": "wall_clock_ts"})

    # Add lag and velocity features across possessions
    df = _add_lag_and_velocity_features(df)

    # Add forward-looking target: Δ(yes_bid) over next 120s
    # Reuses models/targets/kalshi_targets.py — same tradability filter
    df_with_ts = df.rename(columns={"wall_clock_ts": "ts"})
    df_with_ts = add_kalshi_targets(df_with_ts, window_seconds=120)
    df = df_with_ts.rename(columns={"ts": "wall_clock_ts"})

    # Normalize target column name (kalshi_targets uses window-suffixed name as fallback)
    if "target_delta_bid_120s" in df.columns and TARGET_COL not in df.columns:
        df = df.rename(columns={"target_delta_bid_120s": TARGET_COL})

    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    logger.info(
        "Dropped %d rows with null target (last 120s of each game). "
        "%d rows remaining.",
        before - len(df), len(df),
    )

    return df


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    model: XGBRegressor
    val_rmse: float
    baseline_rmse: float
    directional_accuracy: float
    profitable_signal_rate: float
    feature_importances: dict[str, float]
    n_train: int
    n_val: int
    n_games: int


def train() -> TrainResult:
    df = build_dataset()

    df["game_date"] = pd.to_datetime(df["game_date"])
    dates = sorted(df["game_date"].unique())
    n_games = df["game_id"].nunique()

    # 80/20 chronological split — no fixed dates since all data is post-March 2026.
    # As tick data accumulates over months, this split naturally expands.
    split_idx = int(len(dates) * 0.8)
    train_dates = set(dates[:split_idx])

    train_mask = df["game_date"].isin(train_dates)
    val_mask = ~train_mask

    X_all = _prepare_feature_matrix(df)
    y_all = df[TARGET_COL]

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]

    # Weight toward liquid moments (tight spread = real market activity)
    # Zero weight on untradable ticks so they don't pollute signal
    spread_train = df.loc[train_mask, "spread"].clip(lower=1.0)
    tradable_train = df.loc[train_mask, "is_tradable_target"].fillna(True)
    w_train = np.where(tradable_train, 1.0 / spread_train, 0.0)

    spread_val = df.loc[val_mask, "spread"].clip(lower=1.0)
    tradable_val = df.loc[val_mask, "is_tradable_target"].fillna(True)
    w_val = np.where(tradable_val, 1.0 / spread_val, 0.0)

    logger.info(
        "Split: %d games total, train=%d rows (%d games), val=%d rows (%d games)",
        n_games,
        len(X_train), df.loc[train_mask, "game_id"].nunique(),
        len(X_val),   df.loc[val_mask,   "game_id"].nunique(),
    )

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=100,
    )

    val_preds = model.predict(X_val)

    # Weighted RMSE — the primary metric
    val_rmse = float(np.sqrt(np.average((y_val - val_preds) ** 2, weights=w_val)))

    # Baseline: always predict 0 (no movement). Model must beat this.
    baseline_rmse = float(np.sqrt(np.average(y_val ** 2, weights=w_val)))

    # Directional accuracy on tradable, non-trivial moves (actual > 1¢ to avoid noise)
    tradable_idx = tradable_val.values & (np.abs(y_val.values) > 1.0)
    if tradable_idx.sum() > 0:
        dir_correct = np.sign(val_preds[tradable_idx]) == np.sign(y_val.values[tradable_idx])
        directional_accuracy = float(dir_correct.mean())
    else:
        directional_accuracy = 0.0

    # Profitable signal rate: predicted > fee floor AND directionally correct
    fee_floor = 1.75  # cents, maker fee at 100 contracts × 50¢
    high_conf = np.abs(val_preds) > fee_floor
    if high_conf.sum() > 0:
        correct_dir = np.sign(val_preds[high_conf]) == np.sign(y_val.values[high_conf])
        profitable_signal_rate = float(correct_dir.mean())
    else:
        profitable_signal_rate = 0.0

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))

    return TrainResult(
        model=model,
        val_rmse=val_rmse,
        baseline_rmse=baseline_rmse,
        directional_accuracy=directional_accuracy,
        profitable_signal_rate=profitable_signal_rate,
        feature_importances=importances,
        n_train=len(X_train),
        n_val=len(X_val),
        n_games=n_games,
    )


def save_model(model: XGBRegressor, path: Path = MODEL_OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: Path = MODEL_OUTPUT_PATH) -> XGBRegressor:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Inference wrapper ─────────────────────────────────────────────────────────

class KalshiPriceMovementPredictor:
    """
    Thin wrapper for inference. Assembles the full feature vector from a current
    possession dict + up to 3 prior possession dicts, then returns the expected
    Δ(yes_bid) over the next 120 seconds.

    Usage:
        predictor = KalshiPriceMovementPredictor.from_file()
        delta = predictor.predict(
            current_row=feat_dict,
            prev_rows=[prev1_dict, prev2_dict, prev3_dict],
            yes_bid=52,
            yes_ask=55,
        )
        if abs(delta) > 1.75:
            direction = "YES" if delta > 0 else "NO"
    """

    def __init__(self, model: XGBRegressor) -> None:
        self.model = model

    @classmethod
    def from_file(cls, path: Path = MODEL_OUTPUT_PATH) -> "KalshiPriceMovementPredictor":
        return cls(load_model(path))

    def predict(
        self,
        current_row: dict,
        prev_rows: list[dict],
        yes_bid: int,
        yes_ask: int,
    ) -> float:
        """
        Returns expected Δ(yes_bid) in cents over the next 120 seconds.

        Args:
            current_row:  Feature dict for the current possession.
            prev_rows:    List of feature dicts for prior possessions,
                          ordered [t-1, t-2, t-3]. Pass fewer if unavailable.
            yes_bid:      Current best bid on home spread-1 contract (cents).
            yes_ask:      Current best ask on home spread-1 contract (cents).
        """
        row = dict(current_row)
        row["yes_bid"] = yes_bid
        row["yes_ask"] = yes_ask
        row["spread"] = max(1, yes_ask - yes_bid)

        # Kalshi microstructure defaults — caller should provide if available
        row.setdefault("trade_volume_60s", 0.0)
        row.setdefault("time_since_last_trade_ms", 0.0)

        # Derived features (mirrors _add_derived_features)
        if "game_clock_secs" in row and "period" in row:
            elapsed = (720 - max(0, min(720, row["game_clock_secs"]))) / 60
            row["minutes_into_game"] = (row["period"] - 1) * 12 + elapsed
        row["home_in_bonus"] = int(row.get("home_team_fouls_q", 0) >= 5)
        row["away_in_bonus"] = int(row.get("away_team_fouls_q", 0) >= 5)
        row["home_fouls_until_bonus"] = max(0, 5 - row.get("home_team_fouls_q", 0))
        row["away_fouls_until_bonus"] = max(0, 5 - row.get("away_team_fouls_q", 0))
        row["home_full_timeouts_remaining"] = max(0, 4 - row.get("home_timeouts_used", 0))
        row["away_full_timeouts_remaining"] = max(0, 4 - row.get("away_timeouts_used", 0))
        raw_run_team = row.get("current_run_team", None)
        row["current_run_team_encoded"] = {"home": 1, "away": -1}.get(raw_run_team, 0)
        row["home_star_in_foul_trouble"] = float(row.get("home_trouble_star_tier", 0) > 0)
        row["away_star_in_foul_trouble"] = float(row.get("away_trouble_star_tier", 0) > 0)

        # Lag features from prior possession dicts
        for n, prev in enumerate(prev_rows[:3], start=1):
            for col in LAG_COLS:
                val = prev.get(col, 0.0)
                if col == "yes_bid":
                    val = prev.get("yes_bid", yes_bid)
                row[f"prev{n}_{col}"] = val if val is not None else 0.0

        # Fill missing lags if fewer than 3 prior rows provided
        for n in range(len(prev_rows) + 1, 4):
            for col in LAG_COLS:
                row.setdefault(f"prev{n}_{col}", 0.0)

        # Velocity features
        row["d_lineup_net_rating_delta"] = (
            row.get("lineup_net_rating_delta", 0.0) - row.get("prev1_lineup_net_rating_delta", 0.0)
        )
        row["d_current_run_length"] = (
            row.get("current_run_length", 0.0) - row.get("prev1_current_run_length", 0.0)
        )
        row["d_score_diff"] = row.get("score_diff", 0.0) - row.get("prev1_score_diff", 0.0)
        row["d_yes_bid"] = yes_bid - row.get("prev1_yes_bid", yes_bid)
        row["d_spread"] = row.get("spread", 0.0) - row.get("prev1_spread", 0.0)

        feature_vec = {col: row.get(col, 0.0) for col in FEATURE_COLS}
        X = pd.DataFrame([feature_vec], columns=FEATURE_COLS).fillna(0)
        return float(self.model.predict(X)[0])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result = train()
    save_model(result.model)

    top10 = sorted(result.feature_importances.items(), key=lambda x: -x[1])[:10]

    print("\n=== Collapsed Price Movement Predictor ===")
    print(f"  Games with tick data:   {result.n_games}")
    print(f"  Train rows:             {result.n_train:,}")
    print(f"  Val rows:               {result.n_val:,}")
    print(f"  Val RMSE (weighted):    {result.val_rmse:.4f} cents")
    print(f"  Baseline RMSE (pred=0): {result.baseline_rmse:.4f} cents")
    print(f"  Directional accuracy:   {result.directional_accuracy:.1%}  (on |actual| > 1¢)")
    print(f"  Profitable signal rate: {result.profitable_signal_rate:.1%}  (|pred| > 1.75¢, correct direction)")
    print(f"  Model saved:            {MODEL_OUTPUT_PATH}")
    print(f"\n  Top 10 features:")
    for feat, imp in top10:
        print(f"    {feat:<45} {imp:.4f}")

    if result.n_games < MIN_GAMES_WARNING:
        print(
            f"\n  ⚠ WARNING: Only {result.n_games} games of tick data. "
            f"Model needs {MIN_GAMES_WARNING}+ games to be meaningful. "
            "Keep the recorder running."
        )
